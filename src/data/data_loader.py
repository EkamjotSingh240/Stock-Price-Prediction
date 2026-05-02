import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker, start="2018-01-01"):
    ticker = ticker.upper().strip()
    if not ticker:
        return None

    df = yf.download(ticker, start=start)
    if df is None or df.empty:
        return None

    df.reset_index(inplace=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(f"data/raw/{ticker}_stock_data.csv", index=False)
    return df

def fetch_multiple_stocks(tickers, start="2018-01-01"):
    """
    Fetch data for multiple stocks.

    Parameters:
    tickers (list): List of stock tickers

    Returns:
    dict of DataFrames
    """

    data = {}

    for ticker in tickers:
        df = fetch_stock_data(ticker, start=start)
        if df is not None:
            data[ticker] = df

    return data


def fetch_usd_inr_rate(default=83.0):
    """
    Latest USD/INR spot from Yahoo (USDINR=X) for converting US stock prices to INR for display.
    """
    try:
        df = yf.download("USDINR=X", period="5d", progress=False)
        if df is None or df.empty:
            return default
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        rate = float(df["Close"].dropna().iloc[-1])
        if rate <= 0:
            return default
        return rate
    except Exception:
        return default
