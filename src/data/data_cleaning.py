import pandas as pd

def clean_stock_data(df):
    """
    Clean stock dataset before feature engineering.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw stock dataset

    Returns
    -------
    pandas.DataFrame
        Cleaned dataset
    """

    df = df.copy()

    # -------------------------
    # Convert Date column
    # -------------------------
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    # -------------------------
    # Sort data by date
    # -------------------------
    df = df.sort_values("Date")

    # -------------------------
    # Remove duplicate rows
    # -------------------------
    df = df.drop_duplicates()

    # -------------------------
    # Handle missing values
    # -------------------------
    df = df.dropna()

    # -------------------------
    # Remove invalid price rows
    # (price cannot be <= 0)
    # -------------------------
    price_columns = ["Open", "High", "Low", "Close"]

    for col in price_columns:
        if col in df.columns:
            df = df[df[col] > 0]

    # -------------------------
    # Remove extreme outliers
    # using IQR method
    # -------------------------
    if "Close" in df.columns:

        Q1 = df["Close"].quantile(0.25)
        Q3 = df["Close"].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df["Close"] >= lower) & (df["Close"] <= upper)]

    # -------------------------
    # Reset index
    # -------------------------
    df.reset_index(drop=True, inplace=True)

    return df


def clean_multiple_stocks(stock_data_dict):
    """
    Clean multiple stock datasets.

    Parameters
    ----------
    stock_data_dict : dict
        Dictionary with ticker as key and DataFrame as value

    Returns
    -------
    dict
        Dictionary of cleaned DataFrames
    """

    cleaned_data = {}

    for ticker, df in stock_data_dict.items():

        print(f"Cleaning data for {ticker}")

        cleaned_df = clean_stock_data(df)

        cleaned_data[ticker] = cleaned_df

    return cleaned_data