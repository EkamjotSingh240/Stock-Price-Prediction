import pandas as pd

from src.data.data_cleaning import clean_stock_data
from src.features.feature_engineering import create_features
from src.features.technical_indicators import add_technical_indicators
from src.models.train_model import FEATURE_COLUMNS


def test_clean_stock_data_removes_invalid_prices_and_sorts():
    df = pd.DataFrame(
        {
            "Date": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "Open": [100, -1, 102],
            "High": [102, 1, 104],
            "Low": [99, 1, 101],
            "Close": [101, 0, 103],
            "Volume": [1000, 1000, 1000],
        }
    )

    cleaned = clean_stock_data(df)

    assert len(cleaned) == 2
    assert cleaned["Date"].is_monotonic_increasing
    assert (cleaned[["Open", "High", "Low", "Close"]] > 0).all().all()


def test_feature_contract_contains_training_columns():
    rows = 120
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=rows),
            "Open": [100 + i * 0.5 for i in range(rows)],
            "High": [101 + i * 0.5 for i in range(rows)],
            "Low": [99 + i * 0.5 for i in range(rows)],
            "Close": [100 + i * 0.5 for i in range(rows)],
            "Volume": [1000 + i * 10 for i in range(rows)],
        }
    )

    features = create_features(df)
    indicators = add_technical_indicators(features).dropna()

    missing = [col for col in FEATURE_COLUMNS if col not in indicators.columns]
    assert not missing, f"Missing expected feature columns: {missing}"
