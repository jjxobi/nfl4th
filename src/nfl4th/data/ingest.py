from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

DEFAULT_CACHE_DIR = Path("data/raw")


def _load_cached_seasons(
    seasons: list[int],
    subdir: str,
    fetch_fn: Callable[[list[int]], pd.DataFrame],
    cache_dir: Path,
) -> pd.DataFrame:
    cache_subdir = cache_dir / subdir
    cache_subdir.mkdir(parents=True, exist_ok=True)

    frames = []
    missing_seasons = []
    for season in seasons:
        season_path = cache_subdir / f"{season}.parquet"
        if season_path.exists():
            frames.append(pd.read_parquet(season_path))
        else:
            missing_seasons.append(season)

    if missing_seasons:
        fetched = fetch_fn(missing_seasons)
        for season in missing_seasons:
            season_df = fetched[fetched["season"] == season]
            season_df.to_parquet(cache_subdir / f"{season}.parquet")
            frames.append(season_df)

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["season", "week"])
        .reset_index(drop=True)
    )


def load_pbp(seasons: list[int], cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    return _load_cached_seasons(seasons, "pbp", nfl.import_pbp_data, cache_dir)


def load_schedules(seasons: list[int], cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    return _load_cached_seasons(seasons, "schedules", nfl.import_schedules, cache_dir)
