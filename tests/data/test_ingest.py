from pathlib import Path

import pandas as pd

from nfl4th.data.ingest import _load_cached_seasons


def _fake_fetch(calls):
    def fetch(seasons):
        calls.append(list(seasons))
        return pd.DataFrame(
            {
                "season": [s for s in seasons for _ in range(2)],
                "week": [1, 2] * len(seasons),
                "extra": ["x"] * (2 * len(seasons)),
            }
        )

    return fetch


def test_fetches_and_caches_missing_seasons(tmp_path: Path):
    calls = []
    result = _load_cached_seasons([2020, 2021], "pbp", _fake_fetch(calls), tmp_path)

    assert calls == [[2020, 2021]]
    assert (tmp_path / "pbp" / "2020.parquet").exists()
    assert (tmp_path / "pbp" / "2021.parquet").exists()
    assert sorted(result["season"].unique().tolist()) == [2020, 2021]


def test_reuses_cached_seasons_without_refetching(tmp_path: Path):
    calls = []
    _load_cached_seasons([2020], "pbp", _fake_fetch(calls), tmp_path)

    calls.clear()
    _load_cached_seasons([2020], "pbp", _fake_fetch(calls), tmp_path)

    assert calls == []


def test_only_fetches_missing_seasons(tmp_path: Path):
    calls = []
    _load_cached_seasons([2020], "pbp", _fake_fetch(calls), tmp_path)

    calls.clear()
    _load_cached_seasons([2020, 2021], "pbp", _fake_fetch(calls), tmp_path)

    assert calls == [[2021]]


def test_columns_restricts_both_cached_and_freshly_fetched_reads(tmp_path: Path):
    calls = []
    _load_cached_seasons([2020], "pbp", _fake_fetch(calls), tmp_path, columns=["season", "week"])

    cached_on_disk = pd.read_parquet(tmp_path / "pbp" / "2020.parquet")
    assert "extra" in cached_on_disk.columns

    calls.clear()
    result = _load_cached_seasons(
        [2020, 2021], "pbp", _fake_fetch(calls), tmp_path, columns=["season", "week"]
    )

    assert calls == [[2021]]
    assert list(result.columns) == ["season", "week"]
