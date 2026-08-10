from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_findings_returns_all_five_sections():
    response = client.get("/findings")

    assert response.status_code == 200
    body = response.json()
    assert "situational_splits" in body
    assert "coach_bucket_leaderboard" in body
    assert "league_trend" in body
    assert "conversion_by_distance" in body
    assert "outcome_by_context" in body
    assert len(body["situational_splits"]) > 0
    assert len(body["league_trend"]) > 0
    assert len(body["conversion_by_distance"]) > 0
    assert len(body["outcome_by_context"]) > 0
