import sys
import os
import plotly
import plotly.graph_objs as go
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.data_loader import fetch_multiple_stocks
from src.data.data_cleaning import clean_multiple_stocks

from src.features.feature_engineering import create_features_multiple_stocks
from src.features.technical_indicators import add_indicators_multiple_stocks

from src.models.predictor import load_ticker_model, predict_next_day, forecast_multiple_days

from src.analytics.trend_detection import detect_trend
from src.analytics.buy_sell_signal import generate_signal
from src.analytics.risk_analysis import calculate_risk

ALLOWED_TICKERS = {"AAPL", "MSFT", "TSLA", "GOOGL"}


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

    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
    )])

    fig.update_layout(
        title=f"{stock} Price (USD)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
    )

    graph = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return {
        "prediction": prediction,
        "trend": trend,
        "signal": signal,
        "risk": risk,
        "accuracy": round(float(accuracy), 2),
        "metrics": metrics,
        "future_prices": future_prices,
        "graph": graph,
    }
