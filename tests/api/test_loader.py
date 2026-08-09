from api.loader import LoadedModels


def test_loaded_models_loads_real_committed_artifacts():
    loaded = LoadedModels()

    assert loaded.baseline_model is not None
    assert loaded.baseline_no_coach_model is not None
    assert loaded.embedding_model is not None
    assert len(loaded.vocab) > 1
    assert loaded.report.index.name == "coach"
    assert len(loaded.report) > 0
    assert isinstance(loaded.latest_season, int)
