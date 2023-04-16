import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Number of GPUs available: {len(gpus)}")
    for gpu in gpus:
        print(f"GPU details: {gpu}")
else:
    print("No GPU found.")
