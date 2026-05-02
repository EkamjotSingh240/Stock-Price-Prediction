from src.data.data_loader import fetch_multiple_stocks
from src.data.data_cleaning import clean_multiple_stocks
from src.features.feature_engineering import create_features_multiple_stocks
from src.features.technical_indicators import add_indicators_multiple_stocks
from src.models.train_model import train_multiple_stocks


def run_training_pipeline(tickers):
    raw_data = fetch_multiple_stocks(tickers)
    if not raw_data:
        raise ValueError("No stock data fetched for training.")

    cleaned_data = clean_multiple_stocks(raw_data)
    feature_data = create_features_multiple_stocks(cleaned_data)
    indicator_data = add_indicators_multiple_stocks(feature_data)

    # Drop NA rows created by lag/rolling indicators before training.
    indicator_data = {ticker: df.dropna().copy() for ticker, df in indicator_data.items()}
    return train_multiple_stocks(indicator_data)
