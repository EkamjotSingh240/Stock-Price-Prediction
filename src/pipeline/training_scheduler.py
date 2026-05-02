import os
import threading
import time
from datetime import datetime, timedelta, timezone

from src.pipeline.train_pipeline import run_training_pipeline


class TrainingScheduler:
    def __init__(self, tickers, interval_hours=24):
        self.tickers = sorted(set([ticker.upper() for ticker in tickers]))
        self.interval_hours = interval_hours
        self._status = {
            ticker: {"state": "unknown", "updated_at": None, "message": ""}
            for ticker in self.tickers
        }
        self._lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()

    def _set_status(self, ticker, state, message=""):
        with self._lock:
            self._status[ticker] = {
                "state": state,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "message": message,
            }

    def mark_ready_if_model_exists(self, ticker, model_dir="models"):
        model_path = os.path.join(model_dir, f"{ticker}_rf.pkl")
        if os.path.exists(model_path):
            self._set_status(ticker, "ready", "Model already present.")

    def ensure_model_async(self, ticker):
        ticker = ticker.upper()
        with self._lock:
            current = self._status.get(ticker, {}).get("state")
            if current in {"queued", "training", "ready"}:
                return
            self._status[ticker] = {
                "state": "queued",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "message": "Training queued.",
            }
        threading.Thread(target=self._train_single_ticker, args=(ticker,), daemon=True).start()

    def _train_single_ticker(self, ticker):
        self._set_status(ticker, "training", "Training started.")
        try:
            run_training_pipeline([ticker])
            self._set_status(ticker, "ready", "Training completed.")
        except Exception as exc:
            self._set_status(ticker, "failed", f"Training failed: {exc}")

    def run_daily_training(self):
        for ticker in self.tickers:
            self._set_status(ticker, "training", "Scheduled retraining in progress.")
        try:
            run_training_pipeline(self.tickers)
            for ticker in self.tickers:
                self._set_status(ticker, "ready", "Scheduled retraining completed.")
        except Exception as exc:
            for ticker in self.tickers:
                self._set_status(ticker, "failed", f"Scheduled retraining failed: {exc}")

    def _loop(self):
        while not self._stop_event.is_set():
            self.run_daily_training()
            next_run = datetime.now(timezone.utc) + timedelta(hours=self.interval_hours)
            while datetime.now(timezone.utc) < next_run:
                if self._stop_event.wait(timeout=5):
                    return

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def get_status(self, ticker=None):
        with self._lock:
            if ticker:
                return self._status.get(ticker.upper(), {})
            return dict(self._status)
