from fastapi.testclient import TestClient


def test_predict_endpoint(test_image: bytes, test_client: TestClient):
    response = test_client.post("/predict", files={'file': ("test_image.png", test_image)})
    assert response.status_code == 200


def test_metrics_endpoint(test_client: TestClient):
    response = test_client.get("/metrics")
    assert response.status_code == 200
