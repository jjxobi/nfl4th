from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from nfl4th.features.build_features import DECISION_CLASSES, SITUATIONAL_FEATURES

UNKNOWN_COACH = "<unknown>"


class CoachVocab:
    def __init__(self, coach_names: list[str]):
        unique_names = sorted(set(coach_names))
        self.coach_to_index = {UNKNOWN_COACH: 0}
        for name in unique_names:
            self.coach_to_index[name] = len(self.coach_to_index)

    def __len__(self) -> int:
        return len(self.coach_to_index)

    def encode(self, name: str) -> int:
        return self.coach_to_index.get(name, 0)

    def encode_series(self, names: pd.Series) -> torch.Tensor:
        return torch.tensor([self.encode(n) for n in names], dtype=torch.long)


class TendencyEmbeddingModel(nn.Module):
    def __init__(self, n_coaches: int, n_situational: int, embedding_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.coach_embedding = nn.Embedding(n_coaches, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim + n_situational, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(DECISION_CLASSES)),
        )
        # Situational features are on wildly different raw scales (season is
        # ~2010-2024, game_seconds_remaining is 0-3600, is_home is 0/1), which
        # stalls gradient descent for a plain linear layer. These buffers hold
        # the training-set mean/std so every forward pass normalizes the same
        # way a coach's identity gets a fixed vocabulary index.
        self.register_buffer("feature_mean", torch.zeros(n_situational))
        self.register_buffer("feature_std", torch.ones(n_situational))

    def normalize(self, situational: torch.Tensor) -> torch.Tensor:
        return (situational - self.feature_mean) / self.feature_std

    def forward(self, coach_idx: torch.Tensor, situational: torch.Tensor) -> torch.Tensor:
        embedded = self.coach_embedding(coach_idx)
        combined = torch.cat([embedded, self.normalize(situational)], dim=1)
        return self.classifier(combined)

    def mean_coach_embedding(self) -> torch.Tensor:
        # Index 0 is the reserved unknown-coach slot and never receives a
        # training signal, so it is excluded from the mean.
        return self.coach_embedding.weight[1:].mean(dim=0)


def _to_tensors(df: pd.DataFrame, vocab: CoachVocab) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    label_index = {label: i for i, label in enumerate(DECISION_CLASSES)}
    coach_idx = vocab.encode_series(df["coach"])
    situational = torch.tensor(df[SITUATIONAL_FEATURES].to_numpy(dtype="float32"))
    labels = torch.tensor(df["decision"].map(label_index).to_numpy(), dtype=torch.long)
    return coach_idx, situational, labels


def train_embedding_model(
    train_df: pd.DataFrame,
    epochs: int = 30,
    lr: float = 0.01,
    batch_size: int = 256,
    seed: int = 42,
) -> tuple[TendencyEmbeddingModel, CoachVocab]:
    torch.manual_seed(seed)
    vocab = CoachVocab(train_df["coach"].tolist())
    coach_idx, situational, labels = _to_tensors(train_df, vocab)

    model = TendencyEmbeddingModel(len(vocab), len(SITUATIONAL_FEATURES))
    feature_std = situational.std(dim=0)
    feature_std = torch.where(feature_std < 1e-6, torch.ones_like(feature_std), feature_std)
    model.feature_mean.copy_(situational.mean(dim=0))
    model.feature_std.copy_(feature_std)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n_rows = coach_idx.shape[0]
    model.train()
    for _ in range(epochs):
        # Mini-batching gives far more gradient steps per epoch than one
        # full-batch step, which matters once train_df has tens of thousands
        # of rows instead of a handful in a test fixture.
        permutation = torch.randperm(n_rows)
        for start in range(0, n_rows, batch_size):
            batch = permutation[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(coach_idx[batch], situational[batch])
            loss = loss_fn(logits, labels[batch])
            loss.backward()
            optimizer.step()

    return model, vocab


def save_embedding_model(model: TendencyEmbeddingModel, vocab: CoachVocab, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    # Saving just the known coach names (not the index dict directly) lets
    # loading rebuild the vocab through the normal constructor, which is
    # deterministic given the same names.
    coach_names = [name for name in vocab.coach_to_index if name != UNKNOWN_COACH]
    (path / "vocab.json").write_text(json.dumps(coach_names))


def load_embedding_model(path: Path) -> tuple[TendencyEmbeddingModel, CoachVocab]:
    coach_names = json.loads((path / "vocab.json").read_text())
    vocab = CoachVocab(coach_names)
    model = TendencyEmbeddingModel(n_coaches=len(vocab), n_situational=len(SITUATIONAL_FEATURES))
    model.load_state_dict(torch.load(path / "model.pt", weights_only=True))
    model.eval()
    return model, vocab


def predict_with_coldstart(
    model: TendencyEmbeddingModel, vocab: CoachVocab, df: pd.DataFrame
) -> pd.DataFrame:
    model.eval()
    situational = torch.tensor(df[SITUATIONAL_FEATURES].to_numpy(dtype="float32"))
    mean_embedding = model.mean_coach_embedding().detach()

    embeddings = []
    for name in df["coach"]:
        idx = vocab.encode(name)
        if idx == 0:
            embeddings.append(mean_embedding)
        else:
            embeddings.append(model.coach_embedding.weight[idx].detach())
    embedding_batch = torch.stack(embeddings)

    with torch.no_grad():
        combined = torch.cat([embedding_batch, model.normalize(situational)], dim=1)
        probs = torch.softmax(model.classifier(combined), dim=1)

    return pd.DataFrame(probs.numpy(), columns=DECISION_CLASSES, index=df.index)
