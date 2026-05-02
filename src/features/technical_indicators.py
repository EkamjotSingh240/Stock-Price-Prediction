import pandas as pd
import ta


def add_technical_indicators(df):
    """
    Add technical indicators to stock dataset.
    """

    df = df.copy()

    # -------------------------
    # Moving Averages
    # -------------------------

    df["MA10"] = df["Close"].rolling(window=10).mean()

    df["MA50"] = df["Close"].rolling(window=50).mean()

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # -------------------------
    # RSI
    # -------------------------

    close_prices = df["Close"].squeeze()

    df["RSI"] = ta.momentum.RSIIndicator(
        close=close_prices,
        window=14
    ).rsi()

    # -------------------------
    # MACD
    # -------------------------

    macd = ta.trend.MACD(close=close_prices)

    df["MACD"] = macd.macd()

    df["MACD_signal"] = macd.macd_signal()

    df["MACD_diff"] = macd.macd_diff()

    # -------------------------
    # Bollinger Bands
    # -------------------------

    bollinger = ta.volatility.BollingerBands(close=close_prices)

    df["BB_upper"] = bollinger.bollinger_hband()

    df["BB_lower"] = bollinger.bollinger_lband()

    df["BB_width"] = df["BB_upper"] - df["BB_lower"]

    # -------------------------
    # Volatility
    # -------------------------

    df["Volatility"] = df["Close"].pct_change().rolling(window=10).std()

    # -------------------------
    # Drop NA rows
    # -------------------------

    # df.dropna(inplace=True)

    return df

def add_indicators_multiple_stocks(stock_data_dict):

    indicator_data = {}

    for ticker, df in stock_data_dict.items():

        print(f"Adding indicators for {ticker}")

        indicator_data[ticker] = add_technical_indicators(df)

    return indicator_data