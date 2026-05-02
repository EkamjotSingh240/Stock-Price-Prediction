import joblib
import json
import os
from datetime import datetime
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURE_COLUMNS = [
    "Price_Range",
    "Price_Change",
    "Daily_Return",
    "Log_Return",
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Rolling_Mean_5",
    "Rolling_Mean_10",
    "Rolling_STD_5",
    "Rolling_STD_10",
    "Volume_Change",
    "MA10",
    "MA50",
    "EMA20",
    "RSI",
    "MACD",
    "BB_width",
    "Volatility",
]

def _prepare_training_data(df):
    df = df.copy()
    df["Target"] = df["Close"].shift(-1)
    df = df.dropna().copy()
    X = df[FEATURE_COLUMNS]
    y = df["Target"]
    return X, y

def _directional_accuracy(y_true, y_pred):
    if len(y_true) <= 1:
        return 0.0
    true_delta = np.diff(y_true)
    pred_delta = np.diff(y_pred)
    return float((np.sign(true_delta) == np.sign(pred_delta)).mean() * 100)

def _calculate_metrics(y_test, predictions):
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)
    mape = float(np.mean(np.abs((y_test - predictions) / y_test)) * 100)
    directional_accuracy = _directional_accuracy(
        np.asarray(y_test), np.asarray(predictions)
    )
    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "mape": round(float(mape), 4),
        "directional_accuracy": round(float(directional_accuracy), 2),
    }

def rolling_backtest(df, n_splits=5, test_size=30):
    X, y = _prepare_training_data(df)
    if len(X) < (test_size * (n_splits + 1)):
        return None

    metrics = []
    for split in range(n_splits):
        train_end = len(X) - test_size * (n_splits - split)
        test_end = train_end + test_size
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]

        model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics.append(_calculate_metrics(y_test, predictions))

    aggregate = {
        key: round(float(np.mean([m[key] for m in metrics])), 4) for key in metrics[0]
    }
    return {"folds": metrics, "aggregate": aggregate}

def train_model(df):
    X, y = _prepare_training_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = _calculate_metrics(y_test, predictions)
    return model, metrics

def _save_artifacts(model, ticker, metrics, backtest=None, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{ticker}_rf.pkl")
    metadata_path = os.path.join(model_dir, f"{ticker}_rf_metadata.json")

    joblib.dump(model, model_path)
    metadata = {
        "ticker": ticker,
        "model_type": "RandomForestRegressor",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "backtest": backtest,
    }
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    return model_path, metadata_path, metadata

def train_and_save_ticker_model(ticker, df, model_dir="models"):
    model, metrics = train_model(df)
    backtest = rolling_backtest(df)
    model_path, metadata_path, metadata = _save_artifacts(
        model, ticker, metrics, backtest=backtest, model_dir=model_dir
    )
    return {
        "model": model,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }

def train_multiple_stocks(stock_data_dict):
    training_results = {}
    for ticker, df in stock_data_dict.items():
        print(f"Training model for {ticker}")
        training_results[ticker] = train_and_save_ticker_model(ticker, df)
    return training_results