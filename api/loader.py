from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nfl4th.models.baseline import load_baseline
from nfl4th.models.conversion import load_conversion_model
from nfl4th.models.embedding_model import load_embedding_model

DEFAULT_MODEL_DIR = Path(__file__).parent / "models"


class LoadedModels:
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR):
        self.baseline_model = load_baseline(model_dir / "baseline")
        self.baseline_no_coach_model = load_baseline(model_dir / "baseline_no_coach")
        self.embedding_model, self.vocab = load_embedding_model(model_dir / "embedding")
        self.report = pd.read_csv(model_dir / "coach_tendency_report.csv").set_index("coach")
        metadata = json.loads((model_dir / "metadata.json").read_text())
        self.latest_season = metadata["latest_season"]
        self.conversion_model = load_conversion_model(model_dir / "conversion")
        self.findings = json.loads((model_dir / "findings.json").read_text())
