from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

_SITUATION = {
    "ydstogo": 2,
    "yardline_100": 40,
    "score_differential": 0,
    "game_seconds_remaining": 1800,
    "qtr": 2,
    "posteam_timeouts_remaining": 3,
    "defteam_timeouts_remaining": 3,
    "is_home": True,
}


def test_predict_returns_probabilities_for_a_known_coach():
    coach_name = client.get("/coaches").json()[0]["coach"]

    response = client.post("/predict", json={"coach": coach_name, **_SITUATION})

    assert response.status_code == 200
    body = response.json()
    for key in ("predicted", "coach_career_average", "league_baseline"):
        probs = body[key]
        total = probs["punt"] + probs["field_goal"] + probs["go_for_it"]
        assert abs(total - 1.0) < 0.01


def test_predict_returns_404_for_unknown_coach():
    response = client.post("/predict", json={"coach": "Definitely Not A Real Coach", **_SITUATION})

    assert response.status_code == 404


def test_predict_returns_422_for_out_of_bounds_situation():
    coach_name = client.get("/coaches").json()[0]["coach"]
    situation = {**_SITUATION, "ydstogo": 1000}

    response = client.post("/predict", json={"coach": coach_name, **situation})

    assert response.status_code == 422


def test_predict_moves_with_the_situation_not_just_the_coach():
    coach_name = client.get("/coaches").json()[0]["coach"]

    short_yardage_near_goal_line = {
        **_SITUATION,
        "ydstogo": 1,
        "yardline_100": 2,
        "score_differential": 0,
    }
    long_yardage_own_territory = {
        **_SITUATION,
        "ydstogo": 15,
        "yardline_100": 90,
        "score_differential": 0,
    }

    go_response = client.post("/predict", json={"coach": coach_name, **short_yardage_near_goal_line})
    punt_response = client.post("/predict", json={"coach": coach_name, **long_yardage_own_territory})

    assert go_response.status_code == 200
    assert punt_response.status_code == 200

    go_probs = go_response.json()["predicted"]
    punt_probs = punt_response.json()["predicted"]

    assert max(go_probs, key=go_probs.get) == "go_for_it"
    assert max(punt_probs, key=punt_probs.get) == "punt"
