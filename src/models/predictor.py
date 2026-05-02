import joblib
import json
import os
import pandas as pd
from src.models.train_model import FEATURE_COLUMNS

def load_ticker_model(ticker, model_dir="models"):
    model_path = os.path.join(model_dir, f"{ticker}_rf.pkl")
    metadata_path = os.path.join(model_dir, f"{ticker}_rf_metadata.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found for {ticker}. Train first using the training pipeline."
        )

    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    return joblib.load(model_path), metadata

def predict_next_day(model, df):
    latest_features = df[FEATURE_COLUMNS].iloc[-1:]
    prediction = model.predict(latest_features)[0]
    return prediction

def forecast_multiple_days(model, df, days=7):
    temp_df = df.copy()
    predictions = []
    for _ in range(days):
        latest_features = temp_df[FEATURE_COLUMNS].iloc[-1:]
        predicted_price = model.predict(latest_features)[0]
        predictions.append(round(float(predicted_price), 2))

        # Build the next synthetic row with conservative drift.
        new_row = temp_df.iloc[-1:].copy()
        current_close = float(new_row["Close"].iloc[0])
        avg_return = float(temp_df["Daily_Return"].tail(10).fillna(0).mean())
        implied_close = current_close * (1 + avg_return)
        predicted_price = (0.6 * predicted_price) + (0.4 * implied_close)
        new_row["Close"] = predicted_price
        new_row["Open"] = current_close
        new_row["High"] = max(float(new_row["High"].iloc[0]), predicted_price)
        new_row["Low"] = min(float(new_row["Low"].iloc[0]), predicted_price)
        new_row["Daily_Return"] = (
            (predicted_price - current_close) / current_close if current_close else 0
        )
        new_row["Log_Return"] = new_row["Daily_Return"]
        new_row["Lag_1"] = current_close
        new_row["Lag_2"] = float(temp_df["Close"].iloc[-2]) if len(temp_df) > 1 else current_close
        new_row["Lag_3"] = float(temp_df["Close"].iloc[-3]) if len(temp_df) > 2 else current_close
        temp_df = pd.concat([temp_df, new_row], ignore_index=True)

    return predictions

def predict_multiple_stocks(models, stock_data_dict):
    results = {}
    for ticker, df in stock_data_dict.items():
        model = models.get(ticker)
        if model is None:
            continue
        prediction = predict_next_day(model, df)
        results[ticker] = prediction
    return results