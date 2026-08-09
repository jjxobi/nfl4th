from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint_returns_basic_info():
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "nfl4th API"
    assert isinstance(body["latest_season"], int)
    assert isinstance(body["n_coaches"], int)
    assert body["n_coaches"] > 0


def test_cors_allows_the_configured_origin():
    response = client.get("/", headers={"Origin": "http://localhost:4321"})

    assert response.headers.get("access-control-allow-origin") == "http://localhost:4321"


def test_cors_rejects_an_unlisted_origin():
    response = client.get("/", headers={"Origin": "https://not-allowed.example.com"})

    assert "access-control-allow-origin" not in response.headers
