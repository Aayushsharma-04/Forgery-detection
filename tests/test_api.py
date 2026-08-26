from fastapi.testclient import TestClient
from app.main import app



def test_health():

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_predict_rejects_invalid_file():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
    )
        assert response.status_code == 400