from datetime import datetime as dt

import yfinance as yf
import pandas as pd
import json

# noch einzubauen:
# daten werden nur an die csv files "appended" - nicht komplett neu geladen
# wenn der Zeithorizont reduziert wird (start), dann wird auch nicht komplett
# neu geladen, sondern nur vom CSV entsprechend ausgegeben


class DataHandler():
    """
    A class used to download data from yfinance, but if data was already downloaded
    less than 5 days ago, loads data from stored csv file
    ...
    Attributes:
    ticker: str
        the ticker symbol from yfinance
    start_date: str
        the data from which you want to download the data
    end_date: datetime.date object
        will always default to today
    """
    # define the date format that should be used within the class
    date_format = "%Y-%m-%d"

    # init with ticker and start date - don't specify end date
    def __init__(self, ticker="EURUSD=X", start_date="2020-01-01", end_date=dt.today().date()):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self._log_data = {
            self.ticker: {
                "start_date": self.start_date,
                "end_date": dt.strftime(self.end_date, DataHandler.date_format)
            }
        }

    # load the data from yfinance if any of the parameters in get_data is false
    def load_data_from_yf(self):
        data = yf.download(self.ticker, self.start_date, self.end_date)
        try:
            with open("./database/log_data/log.json", "r") as f:
                log_hist = json.load(f)
                log_hist[self.ticker] = self._log_data[self.ticker]
            with open("./database/log_data/log.json", "w") as f:
                json.dump(log_hist, f, indent=4)
        except FileNotFoundError:
            with open("./database/log_data/log.json", "w") as f:
                json.dump(self._log_data, f, indent=4)
        data.to_csv(f"./database/ticker_data/{self.ticker}_latest_ticker.csv")
        return data

    # load data from yfinance or .csv depending on the parameters
    def get_data(self):
        try:    
            with open("./database/log_data/log.json", "r") as f:
                log_data = json.load(f)
            
            old_end = dt.strptime(log_data[self.ticker]["end_date"], DataHandler.date_format).date()
            new_end = self.end_date

            old_start = dt.strptime(log_data[self.ticker]["start_date"], DataHandler.date_format).date()
            new_start = dt.strptime(self.start_date, DataHandler.date_format).date()

            if (int((new_end - old_end).days))>=5 or (old_start != new_start):
                print("data was loaded from yfinance - parameters dont match")
                data = self.load_data_from_yf()
            else:
                print("data was loaded from csv")
                data = pd.read_csv(
                    f"./database/ticker_data/{self.ticker}_latest_ticker.csv",
                    parse_dates=["Date"],
                    index_col="Date"
                )
            return data
            
        except (FileNotFoundError, KeyError) as e:
            print(f"data was loaded from yfinance - cannot open file: error {e}")
            data = self.load_data_from_yf()
        
            return data

    


if __name__=="__main__":
    dh = DataHandler()
    data = dh.get_data()
    print(data)