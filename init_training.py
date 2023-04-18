# related third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pytz

# local application / library specific imports 
import modules.data_handler as dh
import modules.indicators as ind

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.activations import linear
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse
from tensorflow.keras.metrics import mae
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

from sklearn.preprocessing import MinMaxScaler


# todo: model tester module
# todo: model db
# todo: model trainer module
# todo: index in dataframes in nummern umwandeln
# todo: mehr tage forecasten
# todo: grid search für die modelle
# todo: cuda zum laufen bringen


# set the parameters:
time_frame = 10
epochs = 30000
hidden_layers = 10
simulations = 10

neurons_first_layer = 200
neurons_hidden_loop = 100
neurons_last_layer = 50
neurons_dense = 3


# specify the ticker that should be downloaded and the starting day
ticker = "EURUSD=X"
start = "2015-01-01"

# get the ticker data from the data_handler module (either yfinance or csv)
data_df = (dh.
        DataHandler(
            ticker=ticker,
            start_date=start
        )
        .get_data())


# maybe delete below section because only the close price should be predicted
#----------------------------------------
# drop the volumn column if it doe's not contain values
if data_df["Volume"].sum()==0:
    data_df.drop("Volume", axis=1, inplace=True)

# drop adj close
data_df.drop("Adj Close", axis=1, inplace=True)
data_df.drop("Open", axis=1, inplace=True)

# ---------------------------------------

# convert DataFrame into scaled array (reshape was needed for close price only)
scaler = MinMaxScaler(feature_range=(0,1))
data_arr = scaler.fit_transform(np.array(data_df[:]))

# split the array into the training and testing chunks and returns two arrays
def ts_split(ts, window=4):
    X, y = [], []
    for i in range(len(ts) - window):
        X.append(ts[i:i+window])
        y.append(ts[i+window])
    
    return np.array(X), np.array(y)

# split the data with the prev. defined function
X, y = ts_split(data_arr, window=time_frame)


# build the model
model = Sequential()
model.add(LSTM(neurons_first_layer, input_shape=[X.shape[1], X.shape[2]], return_sequences=True))
for x in range(hidden_layers):
    model.add(LSTM(neurons_hidden_loop, return_sequences=True))
model.add(LSTM(neurons_last_layer, return_sequences=False))
model.add(Dense(neurons_dense, activation=linear))

print(model.summary())


# prepare model for training
model.compile(
    loss=mse,
    optimizer=Adam()
)

# create callback to save model with lowest loss
checkpoint = ModelCheckpoint(
    filepath="bestmodel_new_more_layer_more.h5",
    monitor="loss",
    save_best_only = True,
    verbose = True
)

# custom learning rate schedule
def lr_schedule(epoch, lr):
    initial_lr = 0.1
    decay_rate = 0.5
    update_frequency = 100  # Update learning rate every 20 epochs

    if epoch % update_frequency == 0 and epoch != 0:
        return initial_lr * np.power(decay_rate, epoch // update_frequency)
    else:
        return lr

lr_callback = LearningRateScheduler(schedule=lr_schedule, verbose=1)

# early stopping callback
early_stopper = EarlyStopping(
    monitor="loss",
    patience = 1000,
    verbose=1,
    mode="min",
    restore_best_weights=True
)


# train model
hist = model.fit(
    X,
    y,
    epochs=epochs,
    verbose=1,
    callbacks=[
        checkpoint,
        early_stopper
    ],
    validation_split = 0.3
)

# shape after time_frames is for amount of predicitons - 1 because only close
X_last = data_arr[len(data_arr)-time_frame:].reshape(1, time_frame, 3)

def predict_future(X, model, horizon=5):
    X_in = X
    predictions = []
    for _ in range(horizon):
        y_out = list(model.predict(np.array(X_in)))
        predictions.append(list(y_out[0]))
        np.append(X_in[0], list(y_out[0]))
        X_in = [X_in[0][1:]]
    return scaler.inverse_transform(predictions)



predictions = pd.DataFrame(predict_future(X_last, model, horizon=simulations))
predictions.columns = data_df.columns

# set the index of the dataframe to the upcoming week
today = datetime.date.today()

# if today is saturday, add 2 delta days, if sunday add 1
if today.weekday() == 5:
    delta_days = 2
elif today.weekday() == 6:
    delta_days = 1
else:
    delta_days = 0

# get the date from monday
monday = today + datetime.timedelta(days=delta_days)

# create range to friday
date_range = pd.date_range(start=monday, periods=len(predictions), freq="D")

# set the index to the dates
predictions.index = date_range


# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

# First subplot: plot loss and val_loss
ax1.plot(hist.history["loss"][100:], label="loss")
ax1.plot(hist.history["val_loss"], label="val_loss")
ax1.legend(loc=0)
ax1.set_title('Loss and Val Loss')

# Second subplot: plot data_df and predictions
ax2.plot(data_df[len(data_df)-20:])
ax2.plot(predictions)
ax2.set_title('Data and Predictions')

# Save the figure as a PNG image
fig.savefig(
    f"result_img/training_{epochs}_epochs_{hidden_layers}_layers_{time_frame}_time_frame_{neurons_first_layer + neurons_hidden_loop * hidden_layers + neurons_last_layer + neurons_dense}_neurons.png",
)
