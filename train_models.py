import argparse
from src.pipeline.train_pipeline import run_training_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train stock models per ticker.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "TSLA", "GOOGL"],
        help="Tickers to train.",
    )
    args = parser.parse_args()
    results = run_training_pipeline([ticker.upper() for ticker in args.tickers])

    print("Training complete.")
    for ticker, result in results.items():
        model_path = result["model_path"]
        metadata_path = result["metadata_path"]
        print(f"{ticker}: model={model_path}, metadata={metadata_path}")


if __name__ == "__main__":
    main()
