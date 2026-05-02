# Stock Price Prediction Dashboard

This project forecasts next-day stock prices for selected tickers and serves analytics through a Flask dashboard.

## Features

- End-to-end data pipeline: fetch -> clean -> features -> indicators
- RandomForest model per ticker (`models/<TICKER>_rf.pkl`)
- Saved training metadata with metrics and rolling backtest
- Flask dashboard with candlestick chart, trend/signal/risk analysis
- Prediction history in SQLite and downloadable PDF report

## Tech Stack

- Python, Flask, scikit-learn, pandas, numpy
- yfinance for market data
- plotly for visualization
- ta for technical indicators
- reportlab for PDF reports

## Quick Start

1) Install dependencies:

```bash
pip install -r requirements.txt
```

2) Train models offline (required before predictions):

```bash
python train_models.py --tickers AAPL MSFT TSLA GOOGL
```

3) Run the web app:

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

## Notes

- `FLASK_DEBUG=true` enables debug mode; default is disabled.
- Set `FLASK_SECRET_KEY` for production use.
- Run tests with:

```bash
pytest -q
```