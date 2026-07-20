"""Runtime and training helpers for one-model compact Featurewise MLP."""
from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:  # pragma: no cover - optional runtime dependency
    HAS_TORCH = False

from .common import clamp_float
from .compact_featurewise_schema import (
    COMPACT_MODEL_GROUP_KEYS,
    build_compact_feature_scaler,
    build_compact_score_design_vector,
    build_compact_vector,
    compact_action_logs_from_multipliers,
    expand_compact_group_multipliers,
    normalise_compact_group_bounds,
)

DEFAULT_HIDDEN = 16
DEFAULT_DROPOUT = 0.2
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_EPOCHS = 1000
DEFAULT_PATIENCE = 50
# Fallback prediction grid size. Explicit candidate_log_multipliers_by_group from testing overrides this.
DEFAULT_CANDIDATE_POINTS = 30
DEFAULT_SEED = 42


if HAS_TORCH:
    class CompactFeaturewiseMLP(nn.Module):
        """One compact score model over shared descriptors and three joint actions."""

        def __init__(self, input_dim: int, hidden: int = DEFAULT_HIDDEN, dropout: float = DEFAULT_DROPOUT):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, max(4, hidden // 2)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(4, hidden // 2), 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)
else:
    class CompactFeaturewiseMLP:  # pragma: no cover - fallback when torch is unavailable
        def __init__(self, *args: Any, **kwargs: Any):
            raise RuntimeError("PyTorch is required for Compact Featurewise MLP.")


def train_compact_featurewise_mlp_model(
    *,
    training_data: list[dict[str, Any]],
    save_dir: Path,
    group_bounds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not HAS_TORCH:
        return {"trained": False, "error": "PyTorch not available"}
    if len(training_data) < 5:
        return {"trained": False, "error": f"Need at least 5 training samples (got {len(training_data)})"}

    # Keep compact MLP training reproducible between the platform and the standalone notebook.
    torch.manual_seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)

    bounds = normalise_compact_group_bounds(group_bounds)
    prepared = _prepare_compact_mlp_dataset(training_data=training_data, bounds=bounds)
    if not prepared["ok"]:
        return {"trained": False, "error": str(prepared["error"])}

    X = prepared["X"]
    Y = prepared["Y"]
    train_idx = prepared["train_idx"]
    val_idx = prepared["val_idx"]
    train_projects = prepared["train_projects"]
    val_projects = prepared["val_projects"]
    feature_scaler = prepared["feature_scaler"]
    n = int(prepared["training_samples"])

    model = CompactFeaturewiseMLP(input_dim=X.shape[1], hidden=DEFAULT_HIDDEN, dropout=DEFAULT_DROPOUT)
    optimizer = optim.Adam(model.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY)
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(DEFAULT_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(X[train_idx])
        loss = ((pred - Y[train_idx]) ** 2).mean()
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_loss = ((model(X[val_idx]) - Y[val_idx]) ** 2).mean().item()
        val_losses.append(float(val_loss))

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = float(val_loss)
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= DEFAULT_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    import time as _time
    timestamp = _time.strftime("%Y%m%d_%H%M%S")
    model_dir = save_dir / "compact_featurewise_mlp"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"compact_featurewise_{timestamp}.pt"
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_type": "compact_featurewise_mlp",
        "input_dim": int(X.shape[1]),
        "hidden": DEFAULT_HIDDEN,
        "dropout": DEFAULT_DROPOUT,
        "learning_rate": DEFAULT_LR,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "max_epochs": DEFAULT_EPOCHS,
        "early_stopping_patience": DEFAULT_PATIENCE,
        "candidate_points": DEFAULT_CANDIDATE_POINTS,
        "seed": DEFAULT_SEED,
        "training_samples": int(n),
        "validation_split_level": "project",
        "train_split": int(len(train_idx)),
        "val_split": int(len(val_idx)),
        "train_project_count": len(train_projects),
        "val_project_count": len(val_projects),
        "train_projects": train_projects,
        "val_projects": val_projects,
        "feature_scaler": feature_scaler,
        "feature_standardization": "train_projects",
        "log_multiplier_bounds": {key: [float(bounds[key][0]), float(bounds[key][1])] for key in COMPACT_MODEL_GROUP_KEYS},
    }
    torch.save(checkpoint, model_path)

    metadata = {
        "model_type": "compact_featurewise_mlp",
        "score_key": "relative_quality_score",
        "input_dim": int(X.shape[1]),
        "hidden": DEFAULT_HIDDEN,
        "dropout": DEFAULT_DROPOUT,
        "training_samples": int(n),
        "validation_split_level": "project",
        "train_split": int(len(train_idx)),
        "val_split": int(len(val_idx)),
        "train_project_count": len(train_projects),
        "val_project_count": len(val_projects),
        "train_projects": train_projects,
        "val_projects": val_projects,
        "feature_scaler": feature_scaler,
        "feature_standardization": "train_projects",
        "epochs_trained": len(train_losses),
        "max_epochs": DEFAULT_EPOCHS,
        "best_epoch": best_epoch,
        "early_stopping_patience": DEFAULT_PATIENCE,
        "best_val_loss": best_val_loss,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "learning_rate": DEFAULT_LR,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "candidate_points": DEFAULT_CANDIDATE_POINTS,
        "seed": DEFAULT_SEED,
        "log_multiplier_bounds": checkpoint["log_multiplier_bounds"],
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }
    metadata_path = model_dir / f"compact_featurewise_{timestamp}_metadata.json"
    metadata_path.write_text(__import__("json").dumps(metadata, indent=2), encoding="utf-8")

    return {
        "trained": True,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        **metadata,
    }


def _prepare_compact_mlp_dataset(
    *,
    training_data: list[dict[str, Any]],
    bounds: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """Prepare train/validation tensors before the MLP optimizer runs.

    The project split happens before descriptor standardization. The scaler is
    fitted only from training-project rows, then applied to validation rows.
    """
    valid_entries: list[dict[str, Any]] = []
    scores: list[float] = []
    project_names: list[str] = []

    for entry in training_data:
        features = entry.get("features")
        score = entry.get("relative_quality_score")
        selected = entry.get("selected_multipliers")
        project_name = entry.get("project_name")
        if (
            not isinstance(features, dict)
            or not isinstance(selected, dict)
            or not isinstance(score, (int, float))
            or not isinstance(project_name, str)
            or not project_name.strip()
        ):
            continue
        action_logs = compact_action_logs_from_multipliers(selected, bounds=bounds)
        if action_logs is None:
            continue
        valid_entries.append(
            {
                "features": dict(features),
                "x_features": dict(features),
                "action_logs": action_logs,
                "project_name": project_name.strip(),
            }
        )
        scores.append(float(score))
        project_names.append(project_name.strip())

    if not scores:
        return {"ok": False, "error": "No valid compact MLP score-training rows available"}
    if len(set(project_names)) < 2:
        return {"ok": False, "error": "Compact MLP project-level validation needs at least 2 projects"}

    train_idx, val_idx, train_projects, val_projects = _project_level_split_indices(project_names)
    train_index_set = set(int(idx) for idx in train_idx.tolist())
    scaler_rows = [valid_entries[idx] for idx in sorted(train_index_set)]
    feature_scaler = build_compact_feature_scaler(scaler_rows)

    design_rows = [
        build_compact_score_design_vector(
            build_compact_vector(entry["features"], feature_scaler),
            entry["action_logs"],
        ).astype(np.float32)
        for entry in valid_entries
    ]

    return {
        "ok": True,
        "X": torch.tensor(np.array(design_rows), dtype=torch.float32),
        "Y": torch.tensor(np.array(scores), dtype=torch.float32),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_projects": train_projects,
        "val_projects": val_projects,
        "feature_scaler": feature_scaler,
        "training_samples": len(scores),
    }


def _project_level_split_indices(project_names: list[str]) -> tuple["torch.Tensor", "torch.Tensor", list[str], list[str]]:
    """Split rows by project so validation measures unseen-project behaviour."""
    unique_projects = sorted(set(project_names))
    if len(unique_projects) < 2:
        raise ValueError("Project-level validation needs at least 2 projects")

    project_perm = torch.randperm(len(unique_projects))
    split = max(1, int(0.8 * len(unique_projects)))
    if split >= len(unique_projects):
        split = len(unique_projects) - 1

    train_projects = [unique_projects[int(i)] for i in project_perm[:split]]
    val_projects = [unique_projects[int(i)] for i in project_perm[split:]]
    train_set = set(train_projects)
    val_set = set(val_projects)
    train_idx = [idx for idx, name in enumerate(project_names) if name in train_set]
    val_idx = [idx for idx, name in enumerate(project_names) if name in val_set]
    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx, dtype=torch.long),
        sorted(train_projects),
        sorted(val_projects),
    )


def predict_compact_featurewise_mlp_multipliers(
    *,
    shared_models_dir: Path,
    features: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for compact featurewise MLP prediction.")
    model_path = _latest_model_path(shared_models_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Compact featurewise MLP model not found: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    return predict_compact_featurewise_mlp_from_checkpoint(
        checkpoint=checkpoint,
        features=features,
        candidate_log_multipliers_by_group=(params or {}).get("candidate_log_multipliers_by_group"),
    )


def predict_compact_featurewise_mlp_from_checkpoint(
    *,
    checkpoint: dict[str, Any],
    features: dict[str, Any],
    candidate_log_multipliers_by_group: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bounds = normalise_compact_group_bounds(checkpoint.get("log_multiplier_bounds"))
    model = CompactFeaturewiseMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden=int(checkpoint.get("hidden", DEFAULT_HIDDEN)),
        dropout=float(checkpoint.get("dropout", DEFAULT_DROPOUT)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    candidates_by_group = _candidate_logs_by_group(
        candidate_points=int(checkpoint.get("candidate_points", DEFAULT_CANDIDATE_POINTS)),
        bounds=bounds,
        source=candidate_log_multipliers_by_group,
    )
    combos = list(itertools.product(*(candidates_by_group[group] for group in COMPACT_MODEL_GROUP_KEYS)))
    feature_scaler = checkpoint.get("feature_scaler")
    if not isinstance(feature_scaler, dict):
        raise RuntimeError("Compact MLP checkpoint is missing feature_scaler. Retrain the MLP with standardized descriptors.")
    x = build_compact_vector(features, feature_scaler)
    batch = np.stack([build_compact_score_design_vector(x, np.array(combo, dtype=np.float64)).astype(np.float32) for combo in combos], axis=0)
    with torch.no_grad():
        scores = model(torch.tensor(batch, dtype=torch.float32)).cpu().numpy().tolist()

    spread = float(max(scores) - min(scores)) if scores else 0.0
    if spread < 1e-6 or not scores:
        selected_logs = np.zeros(len(COMPACT_MODEL_GROUP_KEYS), dtype=np.float64)
        has_signal = False
    else:
        selected_logs = np.array(combos[int(np.argmax(scores))], dtype=np.float64)
        has_signal = True

    group_multipliers: dict[str, float] = {}
    group_log_multipliers: dict[str, float] = {}
    for index, group in enumerate(COMPACT_MODEL_GROUP_KEYS):
        lo, hi = bounds[group]
        mult = clamp_float(float(math.exp(float(selected_logs[index]))), lo, hi)
        group_multipliers[group] = mult
        group_log_multipliers[group] = float(math.log(max(mult, 1e-9)))

    selected_multipliers, selected_log_multipliers = expand_compact_group_multipliers(group_multipliers)
    candidate_score_checks = _candidate_checks_by_group(candidates_by_group, combos, scores, group_log_multipliers)
    return {
        "selected_preset": "compact_featurewise_mlp",
        "yhat_scores": selected_multipliers,
        "selected_multipliers": selected_multipliers,
        "selected_multipliers_raw": dict(selected_multipliers),
        "selected_log_multipliers": selected_log_multipliers,
        "selected_log_multipliers_raw": dict(selected_log_multipliers),
        "group_multipliers": group_multipliers,
        "group_log_multipliers": group_log_multipliers,
        "exploration_mode": "greedy",
        "model_type": "compact_featurewise_mlp",
        "candidate_points": int(checkpoint.get("candidate_points", DEFAULT_CANDIDATE_POINTS)),
        "has_signal": has_signal,
        "score_spreads": {group: spread for group in COMPACT_MODEL_GROUP_KEYS},
        "candidate_score_checks": candidate_score_checks,
        "n_runs": int((checkpoint.get("training_samples") or 0) or 0),
    }


def _latest_model_path(shared_models_dir: Path) -> Path:
    model_dir = shared_models_dir / "compact_featurewise_mlp"
    versioned = sorted(model_dir.glob("compact_featurewise_*.pt"), key=lambda p: p.stem, reverse=True)
    if versioned:
        return versioned[0]
    return model_dir / "compact_featurewise.pt"


def _candidate_logs_by_group(
    *,
    candidate_points: int,
    bounds: dict[str, tuple[float, float]],
    source: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for group in COMPACT_MODEL_GROUP_KEYS:
        lo, hi = bounds[group]
        raw = source.get(group) if isinstance(source, dict) else None
        if isinstance(raw, list) and raw:
            values = [clamp_float(float(value), math.log(lo), math.log(hi)) for value in raw if isinstance(value, (int, float))]
            out[group] = np.array(values or [0.0], dtype=np.float64)
        else:
            out[group] = np.linspace(math.log(lo), math.log(hi), int(max(5, candidate_points)), dtype=np.float64)
    return out


def _candidate_checks_by_group(
    candidates_by_group: dict[str, np.ndarray],
    combos: list[tuple[float, ...]],
    scores: list[float],
    selected_logs: dict[str, float],
) -> dict[str, list[dict[str, Any]]]:
    checks: dict[str, list[dict[str, Any]]] = {}
    score_by_combo = {tuple(float(value) for value in combo): float(scores[index]) for index, combo in enumerate(combos)}
    selected_combo_logs: list[float] = []
    for group in COMPACT_MODEL_GROUP_KEYS:
        selected_log = float(selected_logs[group])
        candidates = candidates_by_group[group]
        if len(candidates):
            selected_combo_logs.append(float(candidates[int(np.argmin(np.abs(candidates - selected_log)))]))
        else:
            selected_combo_logs.append(selected_log)

    for group_index, group in enumerate(COMPACT_MODEL_GROUP_KEYS):
        rows: list[dict[str, Any]] = []
        selected_index = int(np.argmin(np.abs(candidates_by_group[group] - selected_combo_logs[group_index]))) if len(candidates_by_group[group]) else -1
        for candidate_index, candidate_log in enumerate(candidates_by_group[group]):
            # Compact models choose one joint combination. For a per-group chart,
            # vary only this group and keep the other selected group values fixed.
            combo = list(selected_combo_logs)
            combo[group_index] = float(candidate_log)
            score = float(score_by_combo[tuple(combo)])
            rows.append(
                {
                    "candidate_log_multiplier": float(candidate_log),
                    "candidate_multiplier": float(math.exp(float(candidate_log))),
                    "predicted_score": score,
                    "selected": candidate_index == selected_index,
                }
            )
        checks[group] = rows
    return checks


__all__ = [
    "CompactFeaturewiseMLP",
    "predict_compact_featurewise_mlp_from_checkpoint",
    "predict_compact_featurewise_mlp_multipliers",
    "train_compact_featurewise_mlp_model",
]
