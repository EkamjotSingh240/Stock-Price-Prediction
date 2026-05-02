import pandas as pd
import numpy as np


def create_features(df):
    """
    Generate machine learning features for stock data.
    """

    df = df.copy()

    # -------------------------
    # Price Features
    # -------------------------

    df["Price_Range"] = df["High"] - df["Low"]

    df["Price_Change"] = df["Close"] - df["Open"]

    # -------------------------
    # Return Features
    # -------------------------

    df["Daily_Return"] = df["Close"].pct_change()

    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # -------------------------
    # Lag Features
    # -------------------------

    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_2"] = df["Close"].shift(2)
    df["Lag_3"] = df["Close"].shift(3)

    # -------------------------
    # Rolling Statistics
    # -------------------------

    df["Rolling_Mean_5"] = df["Close"].rolling(window=5).mean()
    df["Rolling_Mean_10"] = df["Close"].rolling(window=10).mean()

    df["Rolling_STD_5"] = df["Close"].rolling(window=5).std()
    df["Rolling_STD_10"] = df["Close"].rolling(window=10).std()

    # -------------------------
    # Volume Features
    # -------------------------

    df["Volume_Change"] = df["Volume"].pct_change()

    df["Volume_MA_5"] = df["Volume"].rolling(window=5).mean()

    # -------------------------
    # Drop NA created by shifts
    # -------------------------

    # df.dropna(inplace=True)

    return df


def create_features_multiple_stocks(stock_data_dict):

    feature_data = {}

    for ticker, df in stock_data_dict.items():

        print(f"Creating features for {ticker}")

        feature_data[ticker] = create_features(df)

    return feature_data