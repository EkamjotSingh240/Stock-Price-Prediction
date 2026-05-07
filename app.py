from flask import (
    Flask,
    render_template,
    request,
    send_file,
    redirect,
    after_this_request,
    session,
)
from reportlab.pdfgen import canvas
from datetime import datetime
import sqlite3
import sys
import os
import tempfile
import secrets

sys.path.append(os.path.abspath("."))

from src.pipeline.main_pipeline import run_pipeline
from src.pipeline.training_scheduler import TrainingScheduler

ALLOWED_TICKERS = {"AAPL", "MSFT", "TSLA", "GOOGL"}
TRAINING_SCHEDULER = TrainingScheduler(tickers=ALLOWED_TICKERS, interval_hours=24)

def init_db():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock TEXT,
            price TEXT,
            accuracy TEXT,
            trend TEXT,
            date TEXT
        )
        """
    )
    conn.commit()
    conn.close()

def _ensure_csrf_token():
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_hex(16)
        session["csrf_token"] = token
    return token

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "change-this-in-production")
    init_db()
    for ticker in ALLOWED_TICKERS:
        TRAINING_SCHEDULER.mark_ready_if_model_exists(ticker)
    TRAINING_SCHEDULER.start()

    @app.route("/")
    def home():
        return render_template(
            "index.html",
            csrf_token=_ensure_csrf_token(),
            error=None,
            model_status=None,
            status_stock=None,
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        csrf_token = request.form.get("csrf_token")
        if csrf_token != session.get("csrf_token"):
            return render_template(
                "index.html",
                csrf_token=_ensure_csrf_token(),
                error="Session expired. Please try again.",
                model_status=None,
                status_stock=None,
            )

        stock = request.form.get("stock", "").upper().strip()
        if stock not in ALLOWED_TICKERS:
            return render_template(
                "index.html",
                csrf_token=_ensure_csrf_token(),
                error="Please select a valid stock ticker.",
                model_status=None,
                status_stock=None,
            )

        try:
            result = run_pipeline(stock)
        except FileNotFoundError:
            TRAINING_SCHEDULER.ensure_model_async(stock)
            return render_template(
                "index.html",
                csrf_token=_ensure_csrf_token(),
                error=(
                    f"Model for {stock} is being prepared. "
                    "Training has started in background; retry in a minute."
                ),
                model_status=TRAINING_SCHEDULER.get_status(stock),
                status_stock=stock,
            )
        except Exception as exc:
            return render_template(
                "index.html",
                csrf_token=_ensure_csrf_token(),
                error=f"Prediction unavailable: {exc}",
                model_status=TRAINING_SCHEDULER.get_status(stock),
                status_stock=stock,
            )

        conn = sqlite3.connect("history.db")
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO predictions(stock,price,accuracy,trend,date)
            VALUES(?,?,?,?,?)
            """,
            (
                stock,
                str(result["prediction"]),
                str(result["accuracy"]),
                result["trend"],
                datetime.now().strftime("%d-%m-%Y %H:%M"),
            ),
        )
        conn.commit()
        conn.close()

        return render_template(
            "result.html",
            stock=stock,
            prediction=result["prediction"],
            trend=result["trend"],
            signal=result["signal"],
            risk=result["risk"],
            graph=result["graph"],
            accuracy=result["accuracy"],
            metrics=result.get("metrics", {}),
            future_prices=result["future_prices"],
        )

    @app.route("/model_status")
    def model_status():
        stock = request.args.get("stock")
        return TRAINING_SCHEDULER.get_status(stock)

    @app.route("/download/<stock>")
    def download(stock):
        stock = stock.upper().strip()
        if stock not in ALLOWED_TICKERS:
            return redirect("/")

        try:
            result = run_pipeline(stock)
        except Exception:
            return redirect("/")

        reports_dir = os.path.join("reports")
        os.makedirs(reports_dir, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=f"{stock}_", suffix=".pdf", dir=reports_dir)
        os.close(fd)
        report_canvas = canvas.Canvas(temp_path)

        report_canvas.setFont("Helvetica-Bold", 16)
        report_canvas.drawString(150, 800, "AI Stock Prediction Report")
        report_canvas.setFont("Helvetica", 12)
        report_canvas.drawString(50, 760, f"Stock Name: {stock}")
        report_canvas.drawString(50, 740, f"Predicted Price: USD {result['prediction']}")
        report_canvas.drawString(50, 720, f"Directional Accuracy: {result['accuracy']} %")
        report_canvas.drawString(50, 690, f"Trend: {result['trend']}")
        report_canvas.drawString(50, 670, f"Signal: {result['signal']}")
        report_canvas.drawString(50, 650, f"Risk Level: {result['risk']}")
        report_canvas.drawString(50, 620, "7-Day Forecast:")

        y_pos = 600
        for i, price in enumerate(result["future_prices"], start=1):
            report_canvas.drawString(70, y_pos, f"Day {i}: USD {price}")
            y_pos -= 20

        report_canvas.drawString(50, 430, f"Generated On: {datetime.now().strftime('%d-%m-%Y %H:%M')}")
        report_canvas.drawString(50, 410, "Project: AI-Based Stock Price Prediction System")
        report_canvas.save()

        @after_this_request
        def cleanup_file(response):
            try:
                os.remove(temp_path)
            except OSError:
                pass
            return response

        return send_file(temp_path, as_attachment=True, download_name=f"{stock}_report.pdf")

    @app.route("/history")
    def history():
        conn = sqlite3.connect("history.db")
        c = conn.cursor()
        c.execute("SELECT * FROM predictions ORDER BY id DESC")
        data = c.fetchall()
        conn.close()
        return render_template("history.html", data=data, csrf_token=_ensure_csrf_token())

    @app.route("/clear_history", methods=["POST"])
    def clear_history():
        csrf_token = request.form.get("csrf_token")
        if csrf_token != session.get("csrf_token"):
            return redirect("/history")

        conn = sqlite3.connect("history.db")
        c = conn.cursor()
        c.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()
        return redirect("/history")

    return app


app = create_app()

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)
