"""Candidate pairing helpers for compact quality-model inference."""
from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from .compact_featurewise_schema import COMPACT_MODEL_GROUP_KEYS

PAIRING_LATIN_HYPERCUBE = "latin_hypercube_style_candidate_grid"
PAIRING_FULL_COMBINATION = "full_combination_grid"
VALID_PAIRING_MODES = {PAIRING_LATIN_HYPERCUBE, PAIRING_FULL_COMBINATION}


def normalise_candidate_pairing_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    aliases = {
        "latin": PAIRING_LATIN_HYPERCUBE,
        "latin_hypercube": PAIRING_LATIN_HYPERCUBE,
        "latin_hypercube_grid": PAIRING_LATIN_HYPERCUBE,
        "latin_hypercube_style": PAIRING_LATIN_HYPERCUBE,
        "latin_hypercube_style_candidate_grid": PAIRING_LATIN_HYPERCUBE,
        "same_index": PAIRING_LATIN_HYPERCUBE,
        "same_index_triplets": PAIRING_LATIN_HYPERCUBE,
        "zip": PAIRING_LATIN_HYPERCUBE,
        "zipped": PAIRING_LATIN_HYPERCUBE,
        "cartesian": PAIRING_FULL_COMBINATION,
        "cartesian_product": PAIRING_FULL_COMBINATION,
        "full_grid": PAIRING_FULL_COMBINATION,
        "full_combination": PAIRING_FULL_COMBINATION,
        "full_combination_grid": PAIRING_FULL_COMBINATION,
    }
    return aliases.get(mode, PAIRING_LATIN_HYPERCUBE)


def build_candidate_combinations(
    candidates_by_group: dict[str, np.ndarray],
    pairing_mode: Any,
) -> list[tuple[float, ...]]:
    mode = normalise_candidate_pairing_mode(pairing_mode)
    group_values = [list(candidates_by_group.get(group, [])) for group in COMPACT_MODEL_GROUP_KEYS]
    if mode == PAIRING_LATIN_HYPERCUBE:
        count = min((len(values) for values in group_values), default=0)
        return [
            tuple(float(group_values[group_index][index]) for group_index in range(len(COMPACT_MODEL_GROUP_KEYS)))
            for index in range(count)
        ]
    return [
        tuple(float(value) for value in combo)
        for combo in itertools.product(*group_values)
    ]


def build_candidate_score_checks(
    candidates_by_group: dict[str, np.ndarray],
    combos: list[tuple[float, ...]],
    scores: list[float],
    selected_logs: dict[str, float],
    pairing_mode: Any,
) -> dict[str, list[dict[str, Any]]]:
    mode = normalise_candidate_pairing_mode(pairing_mode)
    if mode == PAIRING_LATIN_HYPERCUBE:
        return _same_index_score_checks(candidates_by_group, combos, scores, selected_logs)
    return _full_combination_score_checks(candidates_by_group, combos, scores, selected_logs)


def _same_index_score_checks(
    candidates_by_group: dict[str, np.ndarray],
    combos: list[tuple[float, ...]],
    scores: list[float],
    selected_logs: dict[str, float],
) -> dict[str, list[dict[str, Any]]]:
    selected_index = _selected_combo_index(combos, selected_logs)
    checks: dict[str, list[dict[str, Any]]] = {}
    for group_index, group in enumerate(COMPACT_MODEL_GROUP_KEYS):
        rows: list[dict[str, Any]] = []
        for candidate_index, combo in enumerate(combos):
            candidate_log = float(combo[group_index])
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "candidate_log_multiplier": candidate_log,
                    "candidate_multiplier": float(math.exp(candidate_log)),
                    "predicted_score": float(scores[candidate_index]) if candidate_index < len(scores) else 0.0,
                    "selected": candidate_index == selected_index,
                    "candidate_pairing_mode": PAIRING_LATIN_HYPERCUBE,
                }
            )
        checks[group] = rows
    return checks


def _full_combination_score_checks(
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
            combo = list(selected_combo_logs)
            combo[group_index] = float(candidate_log)
            score = float(score_by_combo[tuple(combo)])
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "candidate_log_multiplier": float(candidate_log),
                    "candidate_multiplier": float(math.exp(float(candidate_log))),
                    "predicted_score": score,
                    "selected": candidate_index == selected_index,
                    "candidate_pairing_mode": PAIRING_FULL_COMBINATION,
                }
            )
        checks[group] = rows
    return checks


def _selected_combo_index(combos: list[tuple[float, ...]], selected_logs: dict[str, float]) -> int:
    if not combos:
        return -1
    target = tuple(float(selected_logs[group]) for group in COMPACT_MODEL_GROUP_KEYS)
    distances = [
        sum(abs(float(combo[index]) - target[index]) for index in range(len(target)))
        for combo in combos
    ]
    return int(np.argmin(np.array(distances, dtype=np.float64)))
