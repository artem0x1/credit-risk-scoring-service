import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAYLOAD_PATH = PROJECT_ROOT / "data" / "processed" / "inference_api_example.json"


def load_test_payload():
    with open(PAYLOAD_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload["request"]


def test_predict_success():
    request_payload = load_test_payload()

    response = client.post(
        "/predict",
        json=request_payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "client_id" in data
    assert "model" in data
    assert "probability_of_default" in data
    assert "threshold" in data
    assert "risk_grade" in data
    assert "decision" in data

    assert data["model"] == "LightGBM"
    assert 0 <= data["probability_of_default"] <= 1
    assert 0 <= data["threshold"] <= 1
    assert data["risk_grade"] in ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
    assert data["decision"] in ["APPROVE", "REJECT"]


def test_predict_missing_features():
    bad_payload = {
        "client_id": 123,
        "features": {
            "some_wrong_feature": 1
        }
    }

    response = client.post(
        "/predict",
        json=bad_payload,
    )

    assert response.status_code == 400
    assert "Missing required features" in response.json()["detail"]