from fastapi.testclient import TestClient

from painting_estimation.api.main import APP


def test_predict_endpoint(test_image: bytes):
    client = TestClient(APP)
    response = client.post("/predict", files={'file': ("test_image.png", test_image)})
    assert response.status_code == 200
