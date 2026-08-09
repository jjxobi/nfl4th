from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_list_coaches_returns_known_coaches_with_valid_rates():
    response = client.get("/coaches")

    assert response.status_code == 200
    coaches = response.json()
    assert len(coaches) > 0
    first = coaches[0]
    assert "coach" in first
    assert first["n_decisions"] > 0
    assert 0.0 <= first["shrinkage_weight"] <= 1.0
    assert isinstance(first["last_season"], int)


def test_get_coach_returns_full_profile_for_a_known_coach():
    coach_name = client.get("/coaches").json()[0]["coach"]

    response = client.get(f"/coaches/{coach_name}")

    assert response.status_code == 200
    body = response.json()
    assert body["coach"] == coach_name
    assert "observed" in body["punt"]
    assert "expected" in body["punt"]
    assert "shrunk" in body["punt"]
    assert isinstance(body["last_season"], int)


def test_get_coach_returns_404_for_unknown_coach():
    response = client.get("/coaches/Definitely Not A Real Coach")

    assert response.status_code == 404
