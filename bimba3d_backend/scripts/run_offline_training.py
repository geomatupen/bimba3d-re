#!/usr/bin/env python3
"""Orchestrate offline training for the Featurewise Ridge quality model.

Steps:
  1. build_offline_dataset  â€” scan train-dir, extract one row per run
  2. train_offline_models   â€” project-level split, train quality model

Usage:
    python -m bimba3d_backend.scripts.run_offline_training \\
        --train-dir "E:\\Thesis\\PipelineProjects\\New_First_Training_Pipeline" \\
        --out-dir   bimba3d_backend/data/_offline_training

Optional: pass extra args through:
    python -m bimba3d_backend.scripts.run_offline_training \\
        --train-dir "E:\\Thesis\\..." \\
        --out-dir   bimba3d_backend/data/_offline_training \\
        --cv-folds 5 --cv-repeats 3 --exclude-baseline
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run(cmd: list[str]) -> int:
    logger.info("Running: %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, check=False)
    return int(result.returncode)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Offline training pipeline: build dataset â†’ train quality model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        nargs="+",
        required=True,
        help="Pipeline project root directories containing completed runs",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("bimba3d_backend/data/_offline_training"),
        help="Root output directory (dataset + models will be written here)",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Project-level CV folds for lambda selection")
    parser.add_argument("--cv-repeats", type=int, default=3, help="Number of independent CV reshuffles")
    parser.add_argument("--cv-seed", type=int, default=11, help="Base random seed for repeated CV")
    parser.add_argument("--candidate-points", type=int, default=30, help="Fallback prediction grid size per group when no explicit test grid is provided")
    parser.add_argument(
        "--lambda-candidates",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    )
    parser.add_argument(
        "--exclude-baseline",
        action="store_true",
        help="Exclude baseline (run_jitter_only) runs from training",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    dataset_path = out_dir / "offline_dataset_v1.json"
    models_dir = out_dir / "models"

    python = sys.executable

    # â”€â”€ Step 1: build dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    build_cmd = [
        python, "-m", "bimba3d_backend.scripts.build_offline_dataset",
        "--out", str(dataset_path),
    ]
    for td in args.train_dir:
        build_cmd += ["--train-dir", str(td)]
    if args.verbose:
        build_cmd.append("--verbose")

    rc = run(build_cmd)
    if rc != 0:
        logger.error("build_offline_dataset failed (exit %d)", rc)
        return rc

    # â”€â”€ Step 2: train models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    train_cmd = [
        python, "-m", "bimba3d_backend.scripts.train_offline_models",
        "--dataset", str(dataset_path),
        "--out-dir", str(models_dir),
        "--cv-folds", str(args.cv_folds),
        "--cv-repeats", str(args.cv_repeats),
        "--cv-seed", str(args.cv_seed),
        "--candidate-points", str(args.candidate_points),
        "--lambda-candidates", *[str(l) for l in args.lambda_candidates],
    ]
    if args.exclude_baseline:
        train_cmd.append("--exclude-baseline")
    if args.verbose:
        train_cmd.append("--verbose")

    rc = run(train_cmd)
    if rc != 0:
        logger.error("train_offline_models failed (exit %d)", rc)
        return rc

    logger.info("Offline training complete.")
    logger.info("  Dataset : %s", dataset_path)
    logger.info("  Models  : %s", models_dir)
    logger.info("  Quality model    : %s", models_dir / "quality_model.json")
    logger.info("  Split report     : %s", models_dir / "split_report.json")
    logger.info("  Training report  : %s", models_dir / "training_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

