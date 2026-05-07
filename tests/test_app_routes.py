import app as app_module


def _get_csrf_token(client):
    client.get("/")
    with client.session_transaction() as session:
        return session["csrf_token"]


def test_predict_route_success(monkeypatch):
    test_app = app_module.create_app()
    client = test_app.test_client()
    token = _get_csrf_token(client)

    def fake_run_pipeline(stock):
        return {
            "prediction": 150.25,
            "trend": "Bullish",
            "signal": "BUY",
            "risk": "Low Risk",
            "accuracy": 61.4,
            "metrics": {"mae": 1.2, "rmse": 1.8, "r2": 0.9, "mape": 2.1},
            "future_prices": [151, 152, 153, 154, 155, 156, 157],
            "graph": '{"data":[],"layout":{}}',
        }

    monkeypatch.setattr(app_module, "run_pipeline", fake_run_pipeline)
    response = client.post("/predict", data={"stock": "AAPL", "csrf_token": token})
    assert response.status_code == 200
    assert b"AAPL Analysis" in response.data

def test_predict_route_missing_model_queues_training(monkeypatch):
    test_app = app_module.create_app()
    client = test_app.test_client()
    token = _get_csrf_token(client)

    def fake_run_pipeline(_stock):
        raise FileNotFoundError("Model missing")

    calls = {"queued": False}

    def fake_ensure_model_async(stock):
        calls["queued"] = stock == "AAPL"

    monkeypatch.setattr(app_module, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(app_module.TRAINING_SCHEDULER, "ensure_model_async", fake_ensure_model_async)
    monkeypatch.setattr(
        app_module.TRAINING_SCHEDULER,
        "get_status",
        lambda _ticker=None: {"state": "queued", "message": "Training queued."},
    )

    response = client.post("/predict", data={"stock": "AAPL", "csrf_token": token})
    assert response.status_code == 200
    assert b"Model for AAPL is being prepared" in response.data
    assert b"refreshModelStatus" in response.data
    assert calls["queued"]

def test_model_status_endpoint(monkeypatch):
    test_app = app_module.create_app()
    client = test_app.test_client()

    monkeypatch.setattr(
        app_module.TRAINING_SCHEDULER,
        "get_status",
        lambda _ticker=None: {"state": "ready", "message": "ok"},
    )

    response = client.get("/model_status?stock=AAPL")
    assert response.status_code == 200
    assert response.json["state"] == "ready"


def test_clear_history_requires_post_and_csrf():
    test_app = app_module.create_app()
    client = test_app.test_client()
    token = _get_csrf_token(client)

    get_response = client.get("/clear_history")
    assert get_response.status_code == 405

    post_response = client.post("/clear_history", data={"csrf_token": token})
    assert post_response.status_code == 302
