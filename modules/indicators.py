import pandas as pd
import numpy as np

class IndicatorCreator():
    """
    this class ads indicator data to dataframes which carry stock data
    """

    def __init__(self, df=None):
        assert isinstance(df, pd.core.frame.DataFrame), "df muss ein DataFrame sein"

        self.df = df

    def ichimoku(self, periods_tenkan=9, periods_kijun=26, periods_senkou=52):
        # conversion line (tenkan-sen) and basisline (kijun-sen)
        tenkan_period_high = self.df.High.rolling(window=periods_tenkan).max()
        tenkan_period_low = self.df.Low.rolling(window=periods_tenkan).min()
        self.df["tenkan_sen"] = (tenkan_period_high + tenkan_period_low) / 2

        kijun_period_high = self.df.High.rolling(window=periods_kijun).max()
        kijun_period_low = self.df.Low.rolling(window=periods_kijun).min()
        self.df["kijun_sen"] = (kijun_period_high + kijun_period_low) / 2

        # senkou span a
        self.df["senkou_span_a"] = ((self.df["tenkan_sen"] + self.df["kijun_sen"])/2).shift(26)

        # senkou span b
        senkou_period_high = self.df.High.rolling(window=periods_senkou).max()
        senkou_period_low = self.df.Low.rolling(window=periods_senkou).min()
        self.df["senkou_span_b"] = ((senkou_period_high + senkou_period_low)/2).shift(26)

        # chikou span 
        self.df["chikou_span"] = df.Close.shift(-26)

        return self.df


if __name__ == "__main__":
    test = IndicatorCreator()
