import sys
import os
import plotly
import plotly.graph_objs as go
import json
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.data_loader import fetch_multiple_stocks, fetch_usd_inr_rate
from src.data.data_cleaning import clean_multiple_stocks

from src.features.feature_engineering import create_features_multiple_stocks
from src.features.technical_indicators import add_indicators_multiple_stocks

from src.models.predictor import load_ticker_model, predict_next_day, forecast_multiple_days

from src.analytics.trend_detection import detect_trend
from src.analytics.buy_sell_signal import generate_signal
from src.analytics.risk_analysis import calculate_risk

ALLOWED_TICKERS = {"AAPL", "MSFT", "TSLA", "GOOGL"}


def _scale_metrics_to_inr(metrics, rate):
    out = dict(metrics) if metrics else {}
    for key in ("mae", "rmse"):
        if key in out and out[key] is not None:
            try:
                out[key] = round(float(out[key]) * rate, 4)
            except (TypeError, ValueError):
                pass
    return out


def run_pipeline(stock):
    stock = stock.upper().strip()
    if stock not in ALLOWED_TICKERS:
        raise ValueError(f"Unsupported ticker: {stock}")
    stocks = [stock]
    raw_data = fetch_multiple_stocks(stocks)
    if stock not in raw_data:
        raise ValueError(f"Unable to fetch data for {stock}.")

    cleaned_data = clean_multiple_stocks(raw_data)
    feature_data = create_features_multiple_stocks(cleaned_data)
    indicator_data = add_indicators_multiple_stocks(feature_data)
    df = indicator_data[stock].dropna().copy()

    model, metadata = load_ticker_model(stock)
    prediction = round(float(predict_next_day(model, df)), 2)
    future_prices = forecast_multiple_days(model, df, days=7)

    trend = detect_trend(df)
    signal = generate_signal(df)
    risk = calculate_risk(df)
    metrics = metadata.get("metrics", {})

    accuracy = metrics.get("directional_accuracy")
    if accuracy is None:
        accuracy = 0.0

    usd_inr = fetch_usd_inr_rate()
    df_inr = df.copy()
    for col in ("Open", "High", "Low", "Close"):
        df_inr[col] = df_inr[col].astype(float) * usd_inr

    prediction_inr = round(prediction * usd_inr, 2)
    future_prices_inr = [round(float(p) * usd_inr, 2) for p in future_prices]
    metrics_inr = _scale_metrics_to_inr(metrics, usd_inr)

    fig = go.Figure(data=[go.Candlestick(
        x=df_inr["Date"],
        open=df_inr["Open"],
        high=df_inr["High"],
        low=df_inr["Low"],
        close=df_inr["Close"],
    )])

    fig.update_layout(
        title=f"{stock} Price (INR)",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
    )

    graph = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return {
        "prediction": prediction_inr,
        "trend": trend,
        "signal": signal,
        "risk": risk,
        "accuracy": round(float(accuracy), 2),
        "metrics": metrics_inr,
        "future_prices": future_prices_inr,
        "graph": graph,
        "usd_inr_rate": round(float(usd_inr), 4),
    }
