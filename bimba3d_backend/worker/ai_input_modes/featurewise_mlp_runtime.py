"""Neural contextual learner.

NON_COMPACT_FEATUREWISE: legacy group-wise path kept for comparison.

Architecture:
    Input(N features) â†’ Dense(64, ReLU, Dropout 0.3, BatchNorm)
                      â†’ Dense(32, ReLU, Dropout 0.3, BatchNorm)
                      â†’ Dense(3, linear) + 1.0 offset
                      â†’ Clamp to group bounds

Training (offline from pipeline data):
    - Input: context features (same as ridge model)
    - Target: multiplier that was used in each run
    - Weight: |score| (higher score = more important sample)
    - For negative score runs: target is mirrored (2.0 - multiplier_used)
    - Loss: weighted MSE
    - Optimizer: Adam with weight_decay=1e-3
    - Epochs: 200 with early stopping

Prediction:
    - Greedy: single forward pass (dropout off) â†’ deterministic
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Any, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .common import clamp_float
from .featurewise_ridge_regression_runtime import (
    build_context_vector,
    _build_featurewise_vector,
    _build_compact_vector,
    _build_score_design_vector,
    GROUP_KEYS,
    GROUP_BOUNDS,
    PARAMETER_GROUPS,
    MODE_CONTEXT_DIMS,
    FEATUREWISE_GROUP_DIMS,
    normalize_context_mode,
)
from bimba3d_backend.worker.ai_input_modes.featurewise_ridge_helpers import normalise_group_bounds

# Default MLP hyperparameters
DEFAULT_HIDDEN_1 = 64
DEFAULT_HIDDEN_2 = 32
DEFAULT_DROPOUT = 0.3
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 200
DEFAULT_PATIENCE = 20  # early stopping patience
# Fallback prediction grid size. Explicit candidate_log_multipliers_by_group from testing overrides this.
DEFAULT_CANDIDATE_POINTS = 30


def _get_input_dim(mode: str) -> int:
    """Get input dimension for the given mode."""
    mode = normalize_context_mode(mode)
    if mode == "exif_compact_featurewise":
        # For featurewise, we concatenate all group features (without duplicate intercepts)
        # geometry(9) + appearance(6) + densification(8) - 2 duplicate intercepts = 21
        # Actually simpler: just use the compact vector (10 dims)
        return 10
    return MODE_CONTEXT_DIMS.get(mode, 10)


class MultiplierMLP(nn.Module):
    """Simple MLP that predicts 3 group multiplier deltas."""

    def __init__(self, input_dim: int, hidden1: int = DEFAULT_HIDDEN_1,
                 hidden2: int = DEFAULT_HIDDEN_2, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 3),  # 3 outputs: geometry, appearance, densification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeaturewiseGroupMLP(nn.Module):
    """Tiny MLP for a single parameter group (featurewise)."""

    def __init__(self, input_dim: int, hidden: int = 8, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeaturewiseMLP(nn.Module):
    """Three separate tiny MLPs, one per parameter group, scoring score for a candidate action."""

    def __init__(self, geo_dim: int = 9, app_dim: int = 6, den_dim: int = 8,
                 hidden: int = 8, dropout: float = 0.2):
        super().__init__()
        self.geometry_net = FeaturewiseGroupMLP(geo_dim, hidden, dropout)
        self.appearance_net = FeaturewiseGroupMLP(app_dim, hidden, dropout)
        self.densification_net = FeaturewiseGroupMLP(den_dim, hidden, dropout)

    def forward(self, x_geo: torch.Tensor, x_app: torch.Tensor, x_den: torch.Tensor) -> torch.Tensor:
        geo = self.geometry_net(x_geo)
        app = self.appearance_net(x_app)
        den = self.densification_net(x_den)
        return torch.cat([geo, app, den], dim=1)  # (batch, 3)


def _extract_group_action_log(
    entry: dict[str, Any],
    group_key: str,
    group_bounds: dict[str, tuple[float, float]] | None = None,
) -> float | None:
    bounds = normalise_group_bounds(group_bounds)
    selected = entry.get("selected_multipliers") or entry.get("yhat_scores")
    if isinstance(selected, dict):
        direct_mult = selected.get(group_key)
        if isinstance(direct_mult, (int, float)) and np.isfinite(float(direct_mult)) and float(direct_mult) > 0:
            lo, hi = bounds[group_key]
            mult = clamp_float(float(direct_mult), lo, hi)
            return float(np.log(max(mult, 1e-9)))

        vals = [
            float(selected[k])
            for k in PARAMETER_GROUPS[group_key]
            if isinstance(selected.get(k), (int, float)) and float(selected[k]) > 0
        ]
        if vals:
            lo, hi = bounds[group_key]
            mult = clamp_float(float(np.mean(vals)), lo, hi)
            return float(np.log(max(mult, 1e-9)))

    selected_logs = entry.get("selected_log_multipliers") if isinstance(entry.get("selected_log_multipliers"), dict) else {}
    direct_log = selected_logs.get(group_key)
    if isinstance(direct_log, (int, float)) and np.isfinite(float(direct_log)):
        lo, hi = bounds[group_key]
        return float(clamp_float(float(direct_log), float(np.log(lo)), float(np.log(hi))))
    for member_key in PARAMETER_GROUPS[group_key]:
        value = selected_logs.get(member_key)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            lo, hi = bounds[group_key]
            return float(clamp_float(float(value), float(np.log(lo)), float(np.log(hi))))
    return None


def _build_featurewise_score_tensor(features: dict[str, Any], group_key: str, action_log: float) -> np.ndarray:
    x = _build_featurewise_vector(features, group_key)
    return _build_score_design_vector(x, float(action_log)).astype(np.float32)


def _score_group_candidates(
    model: FeaturewiseMLP,
    *,
    features: dict[str, Any],
    candidate_points: int,
    candidate_log_multipliers_by_group: dict[str, Any] | None = None,
    group_bounds: dict[str, tuple[float, float]] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float], dict[str, list[float]], dict[str, list[float]]]:
    bounds = normalise_group_bounds(group_bounds)
    candidate_logs: dict[str, np.ndarray] = {}
    group_inputs: dict[str, torch.Tensor] = {}

    for group_key in GROUP_KEYS:
        lo_mult, hi_mult = bounds[group_key]
        raw_logs = (candidate_log_multipliers_by_group or {}).get(group_key) if isinstance(candidate_log_multipliers_by_group, dict) else None
        if isinstance(raw_logs, list) and raw_logs:
            logs = np.array(
                [clamp_float(float(value), float(np.log(lo_mult)), float(np.log(hi_mult))) for value in raw_logs if isinstance(value, (int, float))],
                dtype=np.float32,
            )
        else:
            logs = np.linspace(np.log(lo_mult), np.log(hi_mult), int(max(5, candidate_points)), dtype=np.float32)
        candidate_logs[group_key] = logs
        batch = np.stack([_build_featurewise_score_tensor(features, group_key, float(a)) for a in logs], axis=0)
        group_inputs[group_key] = torch.tensor(batch, dtype=torch.float32)

    model.eval()

    with torch.no_grad():
        pred = model(
            group_inputs["geometry_lr_mult"],
            group_inputs["appearance_lr_mult"],
            group_inputs["densification_mult"],
        ).cpu().numpy()

    score_lists = {
        "geometry_lr_mult": pred[:, 0].tolist(),
        "appearance_lr_mult": pred[:, 1].tolist(),
        "densification_mult": pred[:, 2].tolist(),
    }

    group_multipliers: dict[str, float] = {}
    group_log_multipliers: dict[str, float] = {}
    score_spreads: dict[str, float] = {}
    has_signal: dict[str, float] = {}
    SIGNAL_THRESHOLD = 1e-6

    for group_key in GROUP_KEYS:
        scores = score_lists[group_key]
        spread = float(max(scores) - min(scores)) if scores else 0.0
        score_spreads[group_key] = spread
        if spread < SIGNAL_THRESHOLD:
            selected_log = 0.0
            has_signal[group_key] = 0.0
        else:
            idx = int(np.argmax(scores))
            selected_log = float(candidate_logs[group_key][idx])
            has_signal[group_key] = 1.0
        lo_mult, hi_mult = bounds[group_key]
        selected_mult = clamp_float(float(np.exp(selected_log)), lo_mult, hi_mult)
        group_multipliers[group_key] = selected_mult
        group_log_multipliers[group_key] = float(np.log(selected_mult))

    candidate_logs_lists = {k: [float(v) for v in vals.tolist()] for k, vals in candidate_logs.items()}
    return group_multipliers, group_log_multipliers, score_spreads, has_signal, score_lists, candidate_logs_lists


def _model_path(shared_models_dir: Path, mode: str) -> Path:
    """Get path to saved MLP model."""
    return shared_models_dir / "featurewise_mlp" / f"{mode}.pt"


def _metadata_path(shared_models_dir: Path, mode: str) -> Path:
    """Get path to model metadata JSON."""
    return shared_models_dir / "featurewise_mlp" / f"{mode}_metadata.json"


def _filter_topk_positive(training_data: list[dict], topk: int) -> list[dict]:
    """Filter training data to keep only top-k positive score runs per project.

    Groups runs by project_name, keeps only runs with score > 0,
    then takes the top-k by score for each project.
    """
    from collections import defaultdict

    # Group by project
    by_project: dict[str, list[dict]] = defaultdict(list)
    for entry in training_data:
        project = entry.get("project_name", "unknown")
        score = float(entry.get("relative_score", 0.0))
        if score > 0:
            by_project[project].append(entry)

    # Take top-k per project
    filtered = []
    for project, runs in by_project.items():
        runs.sort(key=lambda e: float(e.get("relative_score", 0.0)), reverse=True)
        filtered.extend(runs[:topk])

    return filtered


def train_neural_model(
    training_data: list[dict],
    mode: str,
    save_dir: Path,
    *,
    hidden1: int = DEFAULT_HIDDEN_1,
    hidden2: int = DEFAULT_HIDDEN_2,
    dropout: float = DEFAULT_DROPOUT,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    topk_per_project: int = 0,
) -> dict[str, Any]:
    """Train MLP model offline from pipeline data.

    Args:
        training_data: List of dicts with 'features', 'relative_score', 'yhat_scores', 'project_name'
        mode: AI input mode (determines context vector construction)
        save_dir: Directory to save model (shared_models/)
        hidden1, hidden2, dropout, weight_decay, lr, epochs, patience: hyperparams
        topk_per_project: If > 0, only use top-k positive score runs per project
                          (cleaner ground truth, fewer samples). 0 = use all runs.

    Returns:
        Training result metadata
    """
    if not HAS_TORCH:
        return {"error": "PyTorch not available", "trained": False}

    # Apply topk filtering if requested
    if topk_per_project > 0:
        training_data = _filter_topk_positive(training_data, topk_per_project)

    if len(training_data) < 5:
        return {"error": f"Need at least 5 training samples (got {len(training_data)} after filtering)", "trained": False}

    # Build training tensors
    X_list = []
    Y_list = []
    W_list = []

    use_mirroring = (topk_per_project == 0)  # Only mirror negatives in "all runs" mode

    for entry in training_data:
        features = entry.get("features", {})
        score = float(entry.get("relative_score", 0.0))
        yhat = entry.get("yhat_scores", {})

        # Build context vector
        if mode == "exif_compact_featurewise" or mode == "exif_compact":
            x = _build_compact_vector(features)
        else:
            x = build_context_vector(features, mode)

        # Get the multipliers that were used in this run
        geo_mult = np.mean([yhat.get(k, 1.0) for k in PARAMETER_GROUPS["geometry_lr_mult"]])
        app_mult = np.mean([yhat.get(k, 1.0) for k in PARAMETER_GROUPS["appearance_lr_mult"]])
        den_mult = np.mean([yhat.get(k, 1.0) for k in PARAMETER_GROUPS["densification_mult"]])

        if score >= 0:
            # Positive score: target = the multiplier that worked
            target = np.array([geo_mult, app_mult, den_mult], dtype=np.float32)
        elif use_mirroring:
            # Negative score with mirroring: target = mirror around 1.0
            target = np.array([
                2.0 - geo_mult,
                2.0 - app_mult,
                2.0 - den_mult,
            ], dtype=np.float32)
        else:
            # topk mode: skip negative runs entirely (shouldn't reach here after filtering)
            continue

        # Clamp targets to valid bounds
        target[0] = clamp_float(target[0], *GROUP_BOUNDS["geometry_lr_mult"])
        target[1] = clamp_float(target[1], *GROUP_BOUNDS["appearance_lr_mult"])
        target[2] = clamp_float(target[2], *GROUP_BOUNDS["densification_mult"])

        # Weight by |score| (stronger signal = more important)
        weight = max(abs(score), 0.01)  # minimum weight to avoid zero

        X_list.append(x.astype(np.float32))
        Y_list.append(target)
        W_list.append(weight)

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)
    W = torch.tensor(np.array(W_list), dtype=torch.float32)

    # Normalize weights to mean=1
    W = W / W.mean()

    input_dim = X.shape[1]

    # Split into train/val (80/20)
    n = len(X)
    perm = torch.randperm(n)
    split = max(1, int(0.8 * n))
    train_idx = perm[:split]
    val_idx = perm[split:]

    X_train, Y_train, W_train = X[train_idx], Y[train_idx], W[train_idx]
    X_val, Y_val, W_val = X[val_idx], Y[val_idx], W[val_idx]

    # Create model
    model = MultiplierMLP(input_dim, hidden1, hidden2, dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        # Weighted MSE loss
        loss = (W_train.unsqueeze(1) * (pred - Y_train) ** 2).mean()
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.item()))

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val) if len(X_val) > 0 else torch.zeros(0, 3)
            if len(X_val) > 0:
                val_loss = (W_val.unsqueeze(1) * (val_pred - Y_val) ** 2).mean().item()
            else:
                val_loss = float(loss.item())
        val_losses.append(val_loss)

        # Early stopping
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model â€” use timestamped filename so repeated training doesn't overwrite
    import time as _time
    _ts = _time.strftime("%Y%m%d_%H%M%S")
    model_dir = save_dir / "featurewise_mlp"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = model_dir / f"{mode}_{_ts}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden1": hidden1,
        "hidden2": hidden2,
        "dropout": dropout,
        "mode": mode,
    }, model_save_path)

    # Save metadata alongside the versioned .pt file
    metadata = {
        "model_type": "featurewise_mlp",
        "mode": mode,
        "input_dim": input_dim,
        "hidden1": hidden1,
        "hidden2": hidden2,
        "dropout": dropout,
        "training_samples": n,
        "train_split": split,
        "val_split": n - split,
        "epochs_trained": len(train_losses),
        "best_val_loss": best_val_loss,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "weight_decay": weight_decay,
        "learning_rate": lr,
    }
    metadata_save_path = model_dir / f"{mode}_{_ts}_metadata.json"
    metadata_save_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "trained": True,
        "model_path": str(model_save_path),
        "metadata_path": str(metadata_save_path),
        "input_dim": input_dim,
        "training_samples": n,
        "epochs_trained": len(train_losses),
        "best_val_loss": best_val_loss,
        "final_train_loss": train_losses[-1] if train_losses else None,
        **metadata,
    }


def predict_neural_multipliers(
    shared_models_dir: Path,
    mode: str,
    features: dict[str, Any],
) -> dict[str, Any] | None:
    """Predict multipliers using trained MLP.

    Args:
        shared_models_dir: Path to shared_models directory
        mode: AI input mode
        features: Extracted feature dictionary

    Returns:
        Selection result with predicted multipliers.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for neural multiplier prediction.")

    # Load most recently trained versioned model (newest timestamp first)
    model_dir_path = shared_models_dir / "featurewise_mlp"
    versioned = sorted(model_dir_path.glob(f"{mode}_*.pt"), key=lambda p: p.stem, reverse=True)
    if versioned:
        model_path = versioned[0]
    else:
        model_path = _model_path(shared_models_dir, mode)
    if not model_path.exists():
        raise FileNotFoundError(f"Neural model not found: {model_path}")

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    input_dim = checkpoint["input_dim"]
    hidden1 = checkpoint.get("hidden1", DEFAULT_HIDDEN_1)
    hidden2 = checkpoint.get("hidden2", DEFAULT_HIDDEN_2)
    dropout = checkpoint.get("dropout", DEFAULT_DROPOUT)

    model = MultiplierMLP(input_dim, hidden1, hidden2, dropout)
    model.load_state_dict(checkpoint["state_dict"])

    # Build context vector
    if mode in ("exif_compact_featurewise", "exif_compact"):
        x = _build_compact_vector(features)
    else:
        x = build_context_vector(features, mode)

    x_tensor = torch.tensor(x.astype(np.float32)).unsqueeze(0)  # batch dim

    model.eval()
    with torch.no_grad():
        raw_output = model(x_tensor).squeeze(0).numpy()
    pred_mean = raw_output
    pred_std = np.zeros(3)

    # Clamp to bounds
    geo_mult = clamp_float(float(raw_output[0]), *GROUP_BOUNDS["geometry_lr_mult"])
    app_mult = clamp_float(float(raw_output[1]), *GROUP_BOUNDS["appearance_lr_mult"])
    den_mult = clamp_float(float(raw_output[2]), *GROUP_BOUNDS["densification_mult"])

    group_multipliers = {
        "geometry_lr_mult": geo_mult,
        "appearance_lr_mult": app_mult,
        "densification_mult": den_mult,
    }

    # Expand to 8 individual multipliers
    multipliers: dict[str, float] = {}
    for group_key, member_keys in PARAMETER_GROUPS.items():
        group_mult = group_multipliers[group_key]
        for member_key in member_keys:
            multipliers[member_key] = group_mult

    return {
        "selected_preset": "featurewise_mlp",
        "yhat_scores": multipliers,
        "group_multipliers": group_multipliers,
        "context_vector": x.tolist(),
        "context_norm": float(np.linalg.norm(x)),
        "exploration_mode": "greedy",
        "pred_mean": pred_mean.tolist(),
        "pred_std": pred_std.tolist(),
        "model_type": "featurewise_mlp",
    }


def train_featurewise_neural_model(
    training_data: list[dict],
    save_dir: Path,
    *,
    hidden: int = 8,
    dropout: float = 0.2,
    weight_decay: float = 1e-3,
    lr: float = 1e-3,
    epochs: int = 1000,
    patience: int = 50,
    topk_per_project: int = 0,
    group_bounds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the Featurewise MLP quality model from pipeline data.

    Each group gets its own small MLP with only its relevant features:
      geometry:      9 inputs â†’ 8 â†’ 4 â†’ 1 output  (117 params)
      appearance:    6 inputs â†’ 8 â†’ 4 â†’ 1 output  (93 params)
      densification: 8 inputs â†’ 8 â†’ 4 â†’ 1 output  (109 params)
      Total: ~319 parameters

    Args:
        training_data: List of dicts with features, selected multipliers, quality score, and project metadata.
        save_dir: Directory to save model (shared_models/)
        topk_per_project: If > 0, only use top-k positive score runs per project
    """
    if not HAS_TORCH:
        return {"error": "PyTorch not available", "trained": False}

    if len(training_data) < 5:
        return {"error": f"Need at least 5 training samples (got {len(training_data)})", "trained": False}

    score_key = "relative_quality_score"
    bounds = normalise_group_bounds(group_bounds)

    # Build per-group score-design tensors.
    X_geo_list, X_app_list, X_den_list = [], [], []
    Y_list = []

    for entry in training_data:
        features = entry.get("features", {})
        score_raw = entry.get(score_key)
        if not isinstance(score_raw, (int, float)):
            continue
        score = float(score_raw)

        geo_log = _extract_group_action_log(entry, "geometry_lr_mult", bounds)
        app_log = _extract_group_action_log(entry, "appearance_lr_mult", bounds)
        den_log = _extract_group_action_log(entry, "densification_mult", bounds)
        if geo_log is None or app_log is None or den_log is None:
            continue

        X_geo_list.append(_build_featurewise_score_tensor(features, "geometry_lr_mult", geo_log))
        X_app_list.append(_build_featurewise_score_tensor(features, "appearance_lr_mult", app_log))
        X_den_list.append(_build_featurewise_score_tensor(features, "densification_mult", den_log))
        Y_list.append(np.array([score, score, score], dtype=np.float32))

    if not Y_list:
        return {"error": "No valid score-training rows available", "trained": False}

    X_geo = torch.tensor(np.array(X_geo_list), dtype=torch.float32)
    X_app = torch.tensor(np.array(X_app_list), dtype=torch.float32)
    X_den = torch.tensor(np.array(X_den_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)

    n = len(Y)
    perm = torch.randperm(n)
    split = max(1, int(0.8 * n))
    train_idx, val_idx = perm[:split], perm[split:]

    model = FeaturewiseMLP(
        geo_dim=X_geo.shape[1], app_dim=X_app.shape[1], den_dim=X_den.shape[1],
        hidden=hidden, dropout=dropout,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_geo[train_idx], X_app[train_idx], X_den[train_idx])
        loss = ((pred - Y[train_idx]) ** 2).mean()
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            if len(val_idx) > 0:
                val_pred = model(X_geo[val_idx], X_app[val_idx], X_den[val_idx])
                val_loss = ((val_pred - Y[val_idx]) ** 2).mean().item()
            else:
                val_loss = float(loss.item())
        val_losses.append(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save latest quality-only featurewise score model with timestamped version.
    import time as _time
    _ts = _time.strftime("%Y%m%d_%H%M%S")
    model_dir = save_dir / "featurewise_mlp"
    model_dir.mkdir(parents=True, exist_ok=True)
    mode_name = "featurewise"
    model_save_path = model_dir / f"{mode_name}_{_ts}.pt"

    torch.save({
        "state_dict": model.state_dict(),
        "model_type": "featurewise_mlp",
        "geo_dim": int(X_geo.shape[1]),
        "app_dim": int(X_app.shape[1]),
        "den_dim": int(X_den.shape[1]),
        "hidden": hidden,
        "dropout": dropout,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "max_epochs": epochs,
        "early_stopping_patience": patience,
        "candidate_points": DEFAULT_CANDIDATE_POINTS,
        "log_multiplier_bounds": {key: [float(bounds[key][0]), float(bounds[key][1])] for key in GROUP_KEYS},
    }, model_save_path)

    metadata = {
        "model_type": "featurewise_mlp",
        "score_key": score_key,
        "geo_dim": int(X_geo.shape[1]),
        "app_dim": int(X_app.shape[1]),
        "den_dim": int(X_den.shape[1]),
        "hidden": hidden,
        "dropout": dropout,
        "training_samples": n,
        "epochs_trained": len(train_losses),
        "max_epochs": epochs,
        "best_epoch": best_epoch,
        "early_stopping_patience": patience,
        "best_val_loss": best_val_loss,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "candidate_points": DEFAULT_CANDIDATE_POINTS,
        "log_multiplier_bounds": {key: [float(bounds[key][0]), float(bounds[key][1])] for key in GROUP_KEYS},
    }
    metadata_path = model_dir / f"{mode_name}_{_ts}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "trained": True,
        "model_path": str(model_save_path),
        "metadata_path": str(metadata_path),
        "training_samples": n,
        "epochs_trained": len(train_losses),
        "best_val_loss": best_val_loss,
        "final_train_loss": train_losses[-1] if train_losses else None,
        **metadata,
    }


def predict_featurewise_neural_multipliers(
    shared_models_dir: Path,
    mode_name: str,
    features: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Predict multipliers by scanning bounded candidate actions with score MLP."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for featurewise neural multiplier prediction.")

    # Load the most recently trained versioned featurewise model.
    model_dir_path = shared_models_dir / "featurewise_mlp"
    versioned = sorted(model_dir_path.glob(f"{mode_name}_*.pt"), key=lambda p: p.stem, reverse=True)
    if versioned:
        model_path = versioned[0]
    else:
        model_path = shared_models_dir / "featurewise_mlp" / f"{mode_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Featurewise neural model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    geo_dim = checkpoint["geo_dim"]
    app_dim = checkpoint["app_dim"]
    den_dim = checkpoint["den_dim"]
    hidden = checkpoint.get("hidden", 8)
    dropout_val = checkpoint.get("dropout", 0.2)
    candidate_points = int(checkpoint.get("candidate_points", DEFAULT_CANDIDATE_POINTS))
    group_bounds = normalise_group_bounds(checkpoint.get("log_multiplier_bounds"))

    model = FeaturewiseMLP(geo_dim, app_dim, den_dim, hidden, dropout_val)
    model.load_state_dict(checkpoint["state_dict"])

    group_multipliers, group_log_multipliers, score_spreads, has_signal_map, score_lists, candidate_logs_lists = _score_group_candidates(
        model,
        features=features,
        candidate_points=candidate_points,
        candidate_log_multipliers_by_group=(params or {}).get("candidate_log_multipliers_by_group"),
        group_bounds=group_bounds,
    )

    candidate_score_checks: dict[str, list[dict[str, Any]]] = {}
    for group_key in GROUP_KEYS:
        group_logs = list(candidate_logs_lists.get(group_key) or [])
        group_scores = list(score_lists.get(group_key) or [])
        selected_log = float(group_log_multipliers.get(group_key, 0.0))
        selected_index = (
            int(np.argmin(np.abs(np.array(group_logs, dtype=np.float64) - selected_log)))
            if group_logs
            else -1
        )
        checks: list[dict[str, Any]] = []
        for idx, cand_log in enumerate(group_logs):
            score = float(group_scores[idx]) if idx < len(group_scores) else 0.0
            checks.append(
                {
                    "candidate_log_multiplier": float(cand_log),
                    "candidate_multiplier": float(np.exp(float(cand_log))),
                    "predicted_score": score,
                    "selected": idx == selected_index,
                }
            )
        candidate_score_checks[group_key] = checks

    multipliers: dict[str, float] = {}
    log_multipliers: dict[str, float] = {}
    for group_key, member_keys in PARAMETER_GROUPS.items():
        for member_key in member_keys:
            multipliers[member_key] = group_multipliers[group_key]
            log_multipliers[member_key] = group_log_multipliers[group_key]

    return {
        "selected_preset": "featurewise_mlp",
        "yhat_scores": multipliers,
        "selected_multipliers": multipliers,
        "selected_multipliers_raw": dict(multipliers),
        "selected_log_multipliers": log_multipliers,
        "selected_log_multipliers_raw": dict(log_multipliers),
        "group_multipliers": group_multipliers,
        "exploration_mode": "greedy",
        "model_type": "featurewise_mlp",
        "candidate_points": candidate_points,
        "has_signal": all(bool(has_signal_map[g]) for g in GROUP_KEYS),
        "score_spreads": score_spreads,
        "scores": score_lists,
        "candidate_score_checks": candidate_score_checks,
        "n_runs": int((checkpoint.get("training_samples") or 0) or 0),
    }

