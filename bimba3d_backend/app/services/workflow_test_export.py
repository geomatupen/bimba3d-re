"""Export helpers for testing workflow pipelines."""
from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi.responses import StreamingResponse

from bimba3d_backend.app.services import model_registry
from bimba3d_backend.app.services import pipeline_learning_rows
from bimba3d_backend.app.services import training_pipeline_storage


async def export_current_test(pipeline_id: str) -> StreamingResponse:
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
    if not pipeline:
        raise FileNotFoundError("Pipeline not found")

    config = pipeline.get("config", {}) if isinstance(pipeline.get("config"), dict) else {}
    pipeline_type = str(config.get("pipeline_type") or pipeline.get("pipeline_type") or "offline_data").strip().lower()
    if pipeline_type != "test":
        raise ValueError("Export current test is only available for test pipelines.")

    selected_model_ids = _selected_model_ids(config)
    export_warnings: list[str] = []
    model_exports, source_training_pipeline_ids = _collect_model_exports(selected_model_ids, export_warnings)
    offline_datasets = _collect_offline_datasets(
        pipeline_id=pipeline_id,
        pipeline=pipeline,
        config=config,
        source_pipeline_ids=source_training_pipeline_ids,
        export_warnings=export_warnings,
    )
    preview_key, latest_preview = _latest_prediction_preview(pipeline)
    if latest_preview is None:
        export_warnings.append("No prediction preview found. Run Predict Multipliers first to include current predictions.")
    if not selected_model_ids:
        export_warnings.append("No selected source models found in pipeline config.")

    predictions_payload = {
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline.get("name"),
        "latest_prediction_preview_key": preview_key or None,
        "preview": latest_preview,
    }
    checks_by_model, candidate_pairs_records = _candidate_check_payloads(preview_key, latest_preview, export_warnings)
    ai_pipeline_log_payload = _pipeline_log_payload(pipeline_id, pipeline, pipeline_type)

    manifest = {
        "export_type": "test_pipeline_reference_bundle",
        "exported_at": _utc_now(),
        "pipeline": {
            "pipeline_id": pipeline_id,
            "pipeline_name": pipeline.get("name"),
            "status": pipeline.get("status"),
            "pipeline_type": pipeline_type,
        },
        "selected_model_ids": selected_model_ids,
        "offline_dataset_count": len(offline_datasets),
        "prediction_preview_key": preview_key or None,
        "export_mode": "compact",
        "warnings": export_warnings,
        "contents": {
            "manifest": "manifest.json",
            "ai_pipeline_log": "pipeline/ai_pipeline_log.json",
            "pipeline_config": "pipeline/pipeline_config.json",
            "predictions": "predictions/predicted_multipliers_by_project.json",
            "checks_by_model_dir": "predictions/intermediate_checks/by_model/",
            "candidate_pairs": "predictions/intermediate_checks/candidate_pairs_by_project_model.json",
            "model_records": "models/model_records.json",
            "source_pipelines": "offline_datasets/source_training_pipelines.json",
            "offline_datasets_dir": "offline_datasets/",
            "model_files_dir": "models/files/",
        },
    }

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        archive.writestr("pipeline/ai_pipeline_log.json", json.dumps(ai_pipeline_log_payload, indent=2))
        archive.writestr("pipeline/pipeline_config.json", json.dumps(config, indent=2))
        archive.writestr("predictions/predicted_multipliers_by_project.json", json.dumps(predictions_payload, indent=2))
        archive.writestr(
            "predictions/intermediate_checks/candidate_pairs_by_project_model.json",
            json.dumps(candidate_pairs_records, indent=2),
        )
        archive.writestr("models/model_records.json", json.dumps(model_exports, indent=2))
        archive.writestr(
            "offline_datasets/source_training_pipelines.json",
            json.dumps(_offline_dataset_manifest(offline_datasets), indent=2),
        )
        _write_candidate_checks(archive, checks_by_model)
        _write_offline_datasets(archive, offline_datasets)
        _write_model_files(archive, model_exports)

    zip_buffer.seek(0)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"test_pipeline_export_{pipeline_id}_{stamp}.zip"
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _selected_model_ids(config: dict[str, Any]) -> list[str]:
    raw_ids = config.get("source_model_ids") if isinstance(config.get("source_model_ids"), list) else []
    selected = [str(model_id).strip() for model_id in raw_ids if str(model_id or "").strip()]
    if not selected and config.get("source_model_id"):
        selected.append(str(config.get("source_model_id")).strip())
    return selected


def _collect_model_exports(
    selected_model_ids: list[str],
    export_warnings: list[str],
) -> tuple[list[dict[str, Any]], set[str]]:
    model_exports: list[dict[str, Any]] = []
    source_training_pipeline_ids: set[str] = set()
    for model_id in selected_model_ids:
        record = model_registry.resolve_reusable_model(model_id)
        if not record:
            model_exports.append({"model_id": model_id, "found": False})
            export_warnings.append(f"Model not found in registry: {model_id}")
            continue

        model_dir = _model_dir(record)
        provenance = _read_model_json(model_dir, "provenance.json")
        metadata = _read_model_json(model_dir, "metadata.json")
        lineage = _read_model_json(model_dir, "lineage.json")
        source_pid = str((provenance or {}).get("pipeline_id") or "").strip()
        if source_pid:
            source_training_pipeline_ids.add(source_pid)
        model_exports.append(
            {
                "model_id": model_id,
                "found": True,
                "record": record,
                "provenance": provenance,
                "metadata": metadata,
                "lineage": lineage,
            }
        )
    return model_exports, source_training_pipeline_ids


def _model_dir(record: dict[str, Any]) -> Path | None:
    paths = record.get("paths") if isinstance(record.get("paths"), dict) else {}
    raw = paths.get("model_dir") if isinstance(paths, dict) else None
    return Path(raw) if isinstance(raw, str) and raw.strip() else None


def _read_model_json(model_dir: Path | None, name: str) -> dict[str, Any] | None:
    if not model_dir:
        return None
    path = model_dir / name
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _collect_offline_datasets(
    *,
    pipeline_id: str,
    pipeline: dict[str, Any],
    config: dict[str, Any],
    source_pipeline_ids: set[str],
    export_warnings: list[str],
) -> list[dict[str, Any]]:
    offline_datasets: list[dict[str, Any]] = []
    for source_pid in sorted(source_pipeline_ids):
        src_pipeline = training_pipeline_storage.get_pipeline(source_pid)
        if not src_pipeline:
            export_warnings.append(f"Source training pipeline not found: {source_pid}")
            continue
        src_folder = Path(str((src_pipeline.get("config") or {}).get("pipeline_folder") or "").strip())
        dataset_file = src_folder / "_offline_training" / "offline_dataset.json"
        if dataset_file.exists():
            offline_datasets.append(
                {
                    "pipeline_id": source_pid,
                    "pipeline_name": src_pipeline.get("name"),
                    "path": dataset_file,
                }
            )
        else:
            export_warnings.append(f"Offline dataset missing for source pipeline: {source_pid}")

    if not offline_datasets:
        local_dataset = Path(str(config.get("pipeline_folder") or "").strip()) / "_offline_training" / "offline_dataset.json"
        if local_dataset.exists():
            offline_datasets.append(
                {
                    "pipeline_id": pipeline_id,
                    "pipeline_name": pipeline.get("name"),
                    "path": local_dataset,
                }
            )
    return offline_datasets


def _latest_prediction_preview(pipeline: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    previews = dict(pipeline.get("prediction_previews") or {})
    latest_key = str(pipeline.get("latest_prediction_preview_key") or "").strip()
    latest_preview = previews.get(latest_key) if latest_key else None
    if latest_preview is None and previews:
        latest_key, latest_preview = sorted(
            previews.items(),
            key=lambda item: str((item[1] or {}).get("generated_at") or ""),
            reverse=True,
        )[0]
    return latest_key, latest_preview if isinstance(latest_preview, dict) else None


def _candidate_check_payloads(
    preview_key: str,
    latest_preview: dict[str, Any] | None,
    export_warnings: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    checks_by_model: dict[str, list[dict[str, Any]]] = {}
    candidate_pairs_records: list[dict[str, Any]] = []
    if not latest_preview:
        return checks_by_model, candidate_pairs_records

    preview_has_candidate_checks = False
    for row in latest_preview.get("results") or []:
        if not isinstance(row, dict):
            continue
        model_key = str(row.get("model_id") or "no_model")
        checks_by_model.setdefault(model_key, []).append(
            {
                "preview_key": preview_key,
                "generated_at": latest_preview.get("generated_at"),
                "row": row,
            }
        )
        candidate_checks = row.get("candidate_score_checks") if isinstance(row.get("candidate_score_checks"), dict) else {}
        if candidate_checks:
            preview_has_candidate_checks = True
            candidate_pairs_records.append(
                {
                    "preview_key": preview_key,
                    "generated_at": latest_preview.get("generated_at"),
                    "project_id": row.get("project_id"),
                    "project_name": row.get("project_name"),
                    "model_id": row.get("model_id"),
                    "mode": row.get("mode"),
                    "selected_preset": row.get("selected_preset"),
                    "candidate_score_checks": candidate_checks,
                }
            )

    if (latest_preview.get("results") or []) and not preview_has_candidate_checks:
        export_warnings.append("The latest preview does not contain candidate_score_checks. Re-run Predict Multipliers to capture candidate multiplier-score pairs.")
    return checks_by_model, candidate_pairs_records


def _pipeline_log_payload(pipeline_id: str, pipeline: dict[str, Any], pipeline_type: str) -> dict[str, Any]:
    payload = {
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline.get("name"),
        "pipeline_type": pipeline_type,
        "status": pipeline.get("status"),
        "created_at": pipeline.get("created_at"),
        "started_at": pipeline.get("started_at"),
        "completed_at": pipeline.get("completed_at"),
        "total_runs": int(pipeline.get("total_runs") or 0),
        "completed_runs": int(pipeline.get("completed_runs") or 0),
        "failed_runs": int(pipeline.get("failed_runs") or 0),
        "hard_cap_runs": int(pipeline.get("hard_cap_runs") or 0),
        "pending_runs": int(pipeline.get("pending_runs") or 0),
        "mean_relative_score": pipeline.get("mean_relative_score"),
        "best_relative_score": pipeline.get("best_relative_score"),
        "success_rate": pipeline.get("success_rate"),
        "runs": list(pipeline.get("runs") or []),
    }
    try:
        learning_rows = pipeline_learning_rows.collect_pipeline_learning_rows(pipeline_id).get("rows") or []
        testing_records = _testing_records(learning_rows)
        payload["testing_records"] = testing_records
        payload["testing_records_count"] = len(testing_records)
        payload["testing_records_source"] = "learning_rows"
    except Exception as exc:
        payload["testing_records_error"] = str(exc)
    return payload


def _testing_records(learning_rows: list[Any]) -> list[dict[str, Any]]:
    baseline_by_project: dict[str, dict[str, Any]] = {}
    for row in learning_rows:
        if isinstance(row, dict) and bool(row.get("is_baseline_row")):
            project_key = str(row.get("project_name") or "")
            if project_key and project_key not in baseline_by_project:
                baseline_by_project[project_key] = row

    records: list[dict[str, Any]] = []
    for row in learning_rows:
        if not isinstance(row, dict) or bool(row.get("is_baseline_row")):
            continue
        project_key = str(row.get("project_name") or "")
        base = baseline_by_project.get(project_key, {})
        records.append(
            {
                "project_name": project_key,
                "run_id": row.get("run_id"),
                "run_name": row.get("run_name"),
                "model_id": row.get("model_id"),
                "mode": row.get("ai_input_mode"),
                "selected_preset": row.get("selected_preset"),
                "score": row.get("score"),
                "relative_quality_score": row.get("relative_quality_score"),
                "convergence_score": row.get("convergence_score"),
                "run_final_psnr": row.get("final_psnr"),
                "run_final_ssim": row.get("final_ssim"),
                "run_final_lpips": row.get("final_lpips"),
                "run_final_loss": row.get("final_loss") or row.get("loss_at_reference_step_run"),
                "base_final_psnr": base.get("final_psnr"),
                "base_final_ssim": base.get("final_ssim"),
                "base_final_lpips": base.get("final_lpips"),
                "base_final_loss": base.get("final_loss") or row.get("loss_at_reference_step_base"),
                "selected_multipliers": row.get("selected_multipliers"),
                "selected_log_multipliers": row.get("selected_log_multipliers"),
            }
        )
    return records


def _offline_dataset_manifest(offline_datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "pipeline_id": str(item.get("pipeline_id") or ""),
            "pipeline_name": item.get("pipeline_name"),
            "offline_dataset_file": f"offline_datasets/{str(item.get('pipeline_id') or 'unknown_pipeline')}_offline_dataset.json",
        }
        for item in offline_datasets
    ]


def _write_candidate_checks(archive: zipfile.ZipFile, checks_by_model: dict[str, list[dict[str, Any]]]) -> None:
    for model_key, rows in checks_by_model.items():
        safe_model = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in model_key) or "no_model"
        archive.writestr(
            f"predictions/intermediate_checks/by_model/{safe_model}.json",
            json.dumps(rows, indent=2),
        )


def _write_offline_datasets(archive: zipfile.ZipFile, offline_datasets: list[dict[str, Any]]) -> None:
    for dataset in offline_datasets:
        path = dataset.get("path")
        if isinstance(path, Path) and path.exists():
            source_pid = str(dataset.get("pipeline_id") or "unknown_pipeline")
            archive.write(path, arcname=f"offline_datasets/{source_pid}_offline_dataset.json")


def _write_model_files(archive: zipfile.ZipFile, model_exports: list[dict[str, Any]]) -> None:
    for entry in model_exports:
        if not entry.get("found"):
            continue
        record = entry.get("record") if isinstance(entry.get("record"), dict) else {}
        model_id = str(record.get("model_id") or "unknown_model")
        paths = record.get("paths") if isinstance(record.get("paths"), dict) else {}

        artifact_path = Path(paths["artifact"]) if isinstance(paths.get("artifact"), str) and paths.get("artifact") else None
        if artifact_path and artifact_path.exists() and artifact_path.is_file():
            archive.write(artifact_path, arcname=f"models/files/{model_id}/{artifact_path.name}")

        model_dir = Path(paths["model_dir"]) if isinstance(paths.get("model_dir"), str) and paths.get("model_dir") else None
        if model_dir and model_dir.exists() and model_dir.is_dir():
            for meta_name in ("metadata.json", "provenance.json", "lineage.json", "model.json"):
                candidate = model_dir / meta_name
                if candidate.exists() and candidate.is_file():
                    archive.write(candidate, arcname=f"models/files/{model_id}/{meta_name}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


