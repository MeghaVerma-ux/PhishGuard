
import pytest
from fastapi.testclient import TestClient
import app

client = TestClient(app.app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200


def test_homepage():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_sample_endpoint():
    response = client.get("/sample")

    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert "features" in response.json()
    assert isinstance(response.json()["features"], dict)


def test_valid_prediction():
    sample_response = client.get("/sample")
    sample_data = sample_response.json()

    response = client.post("/predict", json=sample_data)

    assert response.status_code == 200

    result = response.json()
    assert "prediction" in result
    assert "confidence" in result
    assert "model" in result
    assert 0 <= result["confidence"] <= 100


def test_invalid_prediction():
    response = client.post("/predict", json={"features": {}})

    assert response.status_code == 422


def test_feature_importance():
    response = client.get("/feature-importance")

    assert response.status_code == 200

    result = response.json()
    assert "features" in result
    assert len(result["features"]) > 0
