from fastapi.testclient import TestClient

from api.main import app


def test_app_starts_and_loads_real_models():
    client = TestClient(app)
    response = client.get("/openapi.json")
    assert response.status_code == 200
