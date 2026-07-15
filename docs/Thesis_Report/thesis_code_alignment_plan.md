# Thesis Code Alignment Plan

This document is the persistent plan for aligning the Bimba3D codebase with the thesis report, "AI-Guided Optimization for 3D Gaussian Splatting on Drone Images." The report is the source of truth for naming, workflow structure, and model logic.

## Guiding Decisions

- Follow the report terminology in file names, function names, model metadata, API payloads, and frontend labels.
- Do not preserve backward compatibility for old names.
- Do not remove frontend controls that let the user configure thesis workflow values, such as the number of exploration runs. The report uses 12 exploration runs, but the UI should keep this configurable.
- Ask before deleting experiment artifacts, model registry data, generated datasets, or pipeline output folders.
- Treat the learning workflow as offline preparation followed by offline model training and later testing/evaluation.

## Thesis Workflow Shape

The UI and backend should expose three thesis stages:

1. Offline data preparation
   - Create project-level baseline runs.
   - Create exploratory runs using bounded log-space hyperparameter multipliers.
   - Extract project descriptors from images, EXIF/XMP, flight/acquisition indicators, COLMAP outputs, and scene proxies.
   - Build the offline dataset rows: project context, tested action, relative quality score, convergence information, and run metadata.

2. Model training
   - Train the Featurewise Ridge Regression model from the prepared offline dataset.
   - Train the Featurewise MLP comparison model from the same prepared offline dataset.
   - Save model metadata, selected lambda, metrics, feature scalers, and candidate-scoring configuration.
   - Save/register trained models for later testing without a separate elevation step.

3. Testing and evaluation pipeline
   - Run default baseline reconstruction on test projects.
   - Run predicted settings from Featurewise Ridge Regression.
   - Run predicted settings from Featurewise MLP.
   - Compare against default settings using reconstruction quality, convergence behaviour, selected multipliers, and visual/evaluation outputs.

## Gaussian Hard-Cap Handling

- If an exploratory run reaches the configured gaussian hard cap, mark the run as `partial_completed` rather than failed.
- Keep the batch moving to the next run after writing the run status, metadata, and analytics snapshot.
- Store a fixed penalty score of `-1.0` for the partial run and mark it with `quality_score_source = gaussian_cap_penalty`.
- Exclude gaussian-cap penalty rows from normalization so a cap event does not distort valid relative quality scores.
- Keep the partial row visible in the Training Data Rows table with a clear remark, because the selected multipliers are still useful as a negative example.

## Backend Renames

Rename report-relevant modules:

- `bimba3d_backend/worker/ai_input_modes/contextual_continuous_learner.py`
  -> `bimba3d_backend/worker/ai_input_modes/featurewise_ridge_regression.py`

- `bimba3d_backend/worker/ai_input_modes/neural_contextual_learner.py`
  -> `bimba3d_backend/worker/ai_input_modes/featurewise_mlp.py`

Move shared score/score calculation:

- Move `compute_score_summary` out of `learner.py`.
- New target module:
  - `bimba3d_backend/worker/ai_input_modes/relative_quality_score.py`
  - or `bimba3d_backend/worker/ai_input_modes/score_scoring.py`

Preferred report-aligned function names:

- `select_contextual_continuous`
  -> `select_featurewise_ridge_multipliers`

- `load_offline_quality_model`
  -> `load_featurewise_ridge_quality_model`

- `load_offline_convergence_model`
  -> `load_featurewise_ridge_convergence_model`

- `train_featurewise_neural_model`
  -> `train_featurewise_mlp_model`

- `predict_featurewise_neural_multipliers`
  -> `predict_featurewise_mlp_multipliers`

- `FeaturewiseMLP`
  can remain if used as the class name, but metadata should say `featurewise_mlp`.

## Internal Mode and Strategy Names

Replace old values:

- `exif_compact_featurewise` -> `featurewise_ridge`
- `neural_featurewise` -> `featurewise_mlp`
- `contextual_continuous` -> `featurewise_ridge_regression`
- `neural_contextual` -> `featurewise_mlp`
- `ridge_score_optimizer` -> `featurewise_ridge_regression`
- `neural_featurewise_score_mlp` -> `featurewise_mlp`

No backward-compatible aliases should be kept after the cleanup.

## Approved Deletions

The user approved deleting or replacing these old paths, after moving any still-needed logic:

- Delete `bimba3d_backend/worker/ai_input_modes/continuous_learner.py`.
  - Reason: old context-free continuous bandit is not part of the thesis workflow.

- Delete `bimba3d_backend/worker/ai_input_modes/learner.py` after moving `compute_score_summary`.
  - Reason: old preset-bias learner and broken online update path are not part of the thesis workflow.

- Remove old contextual-continuous tests and replace them with report-named tests:
  - `bimba3d_backend/tests/test_contextual_continuous_learner.py`
  - `bimba3d_backend/tests/test_contextual_continuous_integration.py`
  - `bimba3d_backend/tests/test_contextual_continuous_score_optimizer.py`

Potential frontend removals that still need explicit confirmation during implementation:

- Remove visible/manual `preset_bias` option from AI model-selection UI.
- Remove visible/manual `continuous_bandit_linear` option from AI model-selection UI.

These old options are not part of the thesis model comparison, but confirm before removing if they are still useful for manual experiments.

## Files That Should Not Be Deleted Without Fresh Approval

- `bimba3d_backend/data/models/...`
- `bimba3d_backend/data/_offline_training/...`
- `bimba3d_backend/data/_offline_training_for_best_lambda_search/...`
- `temp_extract/...`
- Existing pipeline output folders.

These are experiment/model artifacts. Code can stop supporting old names, but artifact deletion is a separate decision.

## UI Restructure Plan

The current `TrainingPipelinePage` combines concerns that the thesis treats as separate stages. Split the frontend experience into clearer workflow areas.

### Proposed Navigation

Add or rename top-level pages/routes around:

- Offline Data Preparation
- Model Training
- Testing and Evaluation
- Model Registry
- Pipeline History / Details

### Offline Data Preparation UI

Purpose:

- Build the dataset used by the learning models.

Expected controls:

- Dataset directory scan and project selection.
- Project-level COLMAP reuse/source selection.
- Baseline run configuration.
- Exploration run count per project, default can remain configurable.
- Log-space multiplier controls:
  - Geometry bounds: `[0.5, 2.0]`
  - Appearance bounds: `[0.5, 2.0]`
  - Densification bounds: `[0.7, 1.428571]`
  - The number of intervals is the configured Phase 2 exploration run count, not a hardcoded thesis example value.
  - Generate New Values samples one value inside each bounded log-space interval for each multiplier group.
  - Shuffle Values only reorders the already generated values independently per group; it does not create new values.
  - Save Schedule for Processing writes the previewed values to the pipeline state. Resume and retry must reuse those saved values exactly.
- Max steps, eval interval, image max size, storage options.
- Thermal/cooldown options.

Expected outputs:

- Prepared run folders.
- Baseline run IDs.
- Exploration run analytics.
- Offline dataset JSON.
- Dataset summary:
  - project count
  - baseline rows
  - exploration rows
  - feature availability
  - missing-value indicators
  - score fields present

### Model Training UI

Purpose:

- Train thesis models from a prepared offline dataset.

Expected controls:

- Select prepared dataset JSON.
- Train Featurewise Ridge Regression.
- Train Featurewise MLP.
- Ridge lambda candidate list.
- Project-level CV folds and repeats.
- Candidate grid count.
- MLP hyperparameters:
  - hidden layers `8 -> 4`
  - dropout `0.2`
  - learning rate `0.001`
  - weight decay `0.001`
  - max epochs `1000`
  - early stopping patience `50`

Expected outputs:

- Ridge quality model.
- Ridge convergence model if retained for analysis.
- MLP comparison model.
- Training reports.
- Metrics and model comparison summary.
- Coefficient/feature contribution export for Ridge interpretation.

### Testing and Evaluation UI

Purpose:

- Evaluate report models against default Gaussian Splatting settings.

Expected controls:

- Select test projects.
- Select trained Featurewise Ridge model.
- Select trained Featurewise MLP model.
- Run default baseline.
- Run Ridge prediction.
- Run MLP prediction.
- Keep the existing pipeline configurability for steps, storage, thermal, and project order.

Expected outputs:

- Per-project default vs Ridge vs MLP comparison.
- Selected multipliers by group.
- Candidate score curves.
- Relative quality score.
- Loss at 5000.
- Visual/evaluation assets.
- Exportable table for thesis results.

## Backend Workflow Changes

### Offline Data Preparation

- Keep `build_offline_dataset.py`, but align comments and metadata with the report.
- Ensure it stores:
  - `x_features`
  - `selected_multipliers`
  - `selected_log_multipliers`
  - `score_quality`
  - `score_convergence`
  - `loss_at_reference_step_run`
  - `loss_at_reference_step_base`
  - baseline/exploration markers

### Model Training

- Keep `train_offline_models.py`, but rename report-facing concepts.
- Output model metadata with:
  - `model_family: featurewise_ridge_regression`
  - `mode: featurewise_ridge`
  - selected lambda
  - project-level CV settings
  - per-group metrics
  - feature scalers
  - candidate-points configuration
  - coefficient/theta summaries

### MLP Training

- Move MLP training/prediction into `featurewise_mlp.py`.
- Output metadata with:
  - `model_family: featurewise_mlp`
  - `mode: featurewise_mlp`
  - training/validation split
  - epochs trained
  - best validation loss
  - architecture dimensions
  - parameter count

### Testing

- Test pipeline should resolve only the report models:
  - `featurewise_ridge`
  - `featurewise_mlp`
- Remove old strategy dispatch branches for `preset_bias`, `continuous_bandit_linear`, `contextual_continuous`, and `neural_contextual` once equivalent report paths are wired.

## Tests to Replace/Add

New tests:

- `test_featurewise_ridge_regression.py`
  - candidate scoring
  - bounds respected
  - neutral fallback on flat score surface
  - model loading
  - feature scaler usage

- `test_featurewise_mlp.py`
  - training tensor shapes
  - model metadata
  - prediction bounds
  - candidate scoring output

- `test_relative_quality_score.py`
  - baseline-relative quality calculation
  - loss-at-5000 convergence score
  - missing baseline behavior

- `test_workflow.py`
  - offline preparation -> model training -> testing dispatch at a smoke-test level

Frontend checks:

- TypeScript build.
- Verify old names no longer appear in user-facing labels.
- Verify offline preparation and model training are separate UI flows.

## Open Questions

1. Should `score_convergence` remain as a separately trained model, or only as an evaluation metric? The report focuses final multiplier selection on quality score prediction, while convergence is also discussed for evaluation.

2. Should the default exploration run count in the UI remain `5` and simply be configurable, or should the default value be changed to `12` while still allowing the user to edit it?

3. Should old `preset_bias` and `continuous_bandit_linear` UI options be removed completely from manual project tabs, or only hidden from thesis pipeline pages?

## Full Codebase Restructuring Plan

This section extends the thesis alignment into a broader cleanup plan. The goal is to remove old LiteGS and previous-experiment code, reduce very large files, and keep the working gsplat reconstruction path stable.

### Restructuring Principles

- Keep gsplat behavior as stable as possible.
- Remove LiteGS completely from backend and frontend.
- Do not delete Windows installer/runtime code yet.
- Do not delete dot-prefixed folders or files, such as `.git`, `.venv`, `.vscode`, `.pytest_cache`, `.codex`, or `.agents`.
- Remove generated/temporary files created by earlier AI work only after they are confirmed unnecessary.
- Prefer small, cohesive modules over very large files, but avoid excessive fragmentation.
- Add comments only where they clarify non-obvious workflow, model math, or file layout decisions.
- Keep project-level, pipeline-level, training-pipeline, and testing-pipeline concepts explicit.

## Change Levels And Execution Order

The restructuring should be performed in levels. UI flow comes first because it defines how the backend should be shaped. Backend/API changes should follow the UI workflow, and thesis model renames/logic cleanup should happen after the workflow boundaries are clear.

### Level 1: UI Workflow Restructure

Goal:

- Make the application flow match the report:
  - offline data preparation
  - model training
  - testing/evaluation

Rules:

- Do not delete frontend controls without explicit approval.
- If a control is not part of the active thesis workflow, first relocate it to an advanced/settings area or hide it behind a clearly named section.
- Keep all important values configurable from the frontend:
  - exploration run count
  - baseline run settings
  - max steps
  - eval interval
  - image max size
  - storage options
  - thermal/cooldown options
  - train/test project selections
  - model selection
  - Ridge lambda candidates
  - MLP hyperparameters
  - candidate grid size
- Use the report flow to decide placement, not removal.

UI pages should become:

1. Offline Data Preparation
   - Dataset/project selection.
   - Baseline configuration.
   - Exploration configuration.
   - Log-space multiplier bounds.
   - Run count per project.
   - Storage and thermal settings.
   - Build/export offline dataset.

2. Model Training
   - Select offline dataset.
   - Train Featurewise Ridge Regression.
   - Train Featurewise MLP comparison model.
   - Configure Ridge CV/lambda grid.
   - Configure MLP training parameters.
   - Show model training reports and saved model records.
   - Treat model training as a pipeline stage, not as a manual elevation step inside a data preparation pipeline.
   - Save trained models outside the data preparation pipeline so they are already available to testing workflows.

3. Testing and Evaluation
   - Select test projects.
   - Select trained Ridge and MLP models.
   - Configure default baseline, Ridge-predicted, and MLP-predicted test runs.
   - Show comparison outputs and thesis result tables.

4. All Pipelines
   - Browse data preparation, model training, and testing pipelines in one page.
   - Show stage labels:
     - `DATA` for offline data preparation pipelines.
     - `TRAIN` for model training pipelines.
     - `TEST` for testing/evaluation pipelines.
   - Search, filter, and sort all pipeline records.
   - Inspect model metadata, lineage, metrics, and source dataset from model training pipeline outputs.

Pipeline type naming target:

- `training_data`: prepares offline training data through baseline and exploration runs. Current backend `train` records should be treated as `training_data` until backend rename is complete.
- `model_training`: trains Featurewise Ridge Regression and Featurewise MLP models from prepared datasets.
- `test`: runs default and model-predicted evaluation pipelines.

Inner detail page plan:

1. Prepare Offline Data detail
   - Target pipeline type: `training_data`.
   - Keep EXIF in this detail page as a clearly named `EXIF` tab.
   - Main tabs/sections:
     - Overview
     - Projects & Runs
     - Prepared Dataset
     - Logs
     - Configuration
     - EXIF
   - Remove model elevation controls and test prediction controls from this page.

2. Train Models detail
   - Target pipeline type: `model_training`.
   - Model cards should come from the model-training creation flow, not unrelated/manual model registry entries.
   - Clicking a trained model card should open details for that trained model or its model-training pipeline output.
   - Main tabs/sections:
     - Overview
     - Training Data
     - Model Details
     - Metrics
     - Logs
     - Artifacts
   - Ridge details should show lambda grid, selected lambda, CV settings, coefficients, feature scaling, candidate bounds, and no-signal fallback diagnostics.
   - MLP details should show architecture, dropout, learning rate, weight decay, epochs, early stopping, curves, checkpoints, and per-group metrics.
   - Do not require an elevation action for models trained through this workflow.

3. Run Tests detail
   - Target pipeline type: `test`.
   - Main tabs/sections:
     - Overview
     - Projects & Runs
     - Models Used
     - Predictions
     - Results
     - Logs
     - Configuration
   - Rename old â€œElevated Modelsâ€ wording to selected/used models.
   - Remove offline dataset build/rebuild and model-training controls from this page.

What changes in existing UI:

- `TrainingPipelinePage.tsx` should no longer be the single page for everything.
- It should be split into the above workflow pages or replaced gradually by those pages.
- The dashboard should use Research Workflow as the entry point for pipeline work.
- The dashboard should not keep a separate direct Training Pipeline button once workflow pages expose create actions.
- The dashboard should not keep a separate Pipelines tab once the workflow has an All Pipelines page.
- `PipelineDetailsPage.tsx` should remain the operational detail/history page, but its panels should be split into components.
- `ProcessTab.tsx` should keep project-level manual controls, but controls should be grouped more clearly.
- LiteGS controls should be removed only after confirmation during implementation; if removing is already confirmed, delete from UI and backend together.

### Level 2: API And Backend Workflow Boundaries

Goal:

- Create backend services/routes that match the UI stages.

Backend services should be shaped around:

1. Offline data preparation
   - scan datasets
   - create project folders
   - run baseline/exploration pipelines
   - build offline dataset
   - summarize dataset readiness

2. Model training
   - train Featurewise Ridge Regression
   - train Featurewise MLP
   - write model metadata/reports
   - save/register trained models as first-class model pipeline outputs
   - avoid requiring a separate elevate step when models are trained from the offline dataset workflow

3. Testing/evaluation
   - create default/Ridge/MLP test runs
   - collect result summaries
   - export comparison tables

This level should avoid large behavioral changes to gsplat training. It mainly moves orchestration and API logic into clearer modules.

### Level 3: Thesis Report Required Model Changes

Goal:

- Rename and align model code with the report once the workflow boundaries are clear.

Changes:

- Rename old contextual/continuous/neural terms to:
  - Featurewise Ridge Regression
  - Featurewise MLP
  - relative quality score
  - bounded log-space candidate scoring
- Remove old non-report learner paths after shared score scoring is moved.
- Update metadata, model family names, and frontend labels.

### Level 4: LiteGS Removal

Goal:

- Remove LiteGS after the gsplat-only UI/backend workflow is clear.

Rules:

- Remove code paths and controls.
- Do not delete existing LiteGS output artifacts without fresh approval.
- Keep output discovery stable for gsplat.

### Level 5: File Size Reduction And Professional Cleanup

Goal:

- Split very large files into a small number of meaningful modules/components.
- Add comments only where helpful.
- Remove generated/debug files after approval.

Primary targets:

- `ProcessTab.tsx`
- `PipelineDetailsPage.tsx`
- `TrainingPipelinePage.tsx`
- `projects.py`
- `training_pipeline.py`
- `entrypoint.py`

### Level 6: Generated File Cleanup

Goal:

- Remove previous AI-bot debug/generated files.

Rules:

- Do not delete dot-prefixed files/folders.
- Do not delete Windows code yet.
- Do not delete experiment/model artifacts without fresh approval.

## LiteGS Removal Plan

LiteGS is not part of the thesis workflow and can be removed.

### Backend Files To Delete

- `bimba3d_backend/worker/engines/litegs_engine.py`
- `bimba3d_backend/worker/litegs_watch.py`
- `bimba3d_backend/worker/patches/litegs_fused_ssim_force_cuda.patch`
- `bimba3d_backend/worker/patches/litegs_gaussian_raster_sync.patch`

### Backend References To Remove

- In `bimba3d_backend/worker/engines/registry.py`:
  - remove `run_litegs_training`
  - set `SUPPORTED_ENGINES = {"gsplat"}`
  - set `ENGINE_LABELS = {"gsplat": "Gaussian Splatting"}`
  - set `ENGINE_RUNNERS = {"gsplat": run_gsplat_training}`

- In `bimba3d_backend/worker/entrypoint.py`:
  - remove `_find_latest_litegs_checkpoint`
  - remove `_patch_litegs_opacity_decay`
  - remove `_export_litegs_outputs`
  - remove LiteGS items from the `context` dict passed to engines
  - simplify engine validation to force/use `gsplat`
  - keep engine-scoped output directories if useful, but only support `outputs/engines/gsplat`

- In `bimba3d_backend/app/models/project.py`:
  - remove LiteGS-specific request fields:
    - `litegs_target_primitives`
    - `litegs_alpha_shrink`
  - update comments saying engine is `"gsplat"` only

- In `bimba3d_backend/app/services/colmap.py`:
  - remove cleanup of `litegs_target_primitives`
  - remove cleanup of `litegs_alpha_shrink`

- In `bimba3d_backend/app/services/gsplat.py`:
  - remove LiteGS params from forwarded payloads

- In `bimba3d_backend/app/services/resume.py`:
  - remove LiteGS checkpoint/model lookup paths
  - keep gsplat checkpoint discovery

- In `bimba3d_backend/app/api/projects.py`:
  - remove `litegs` from engine validation
  - remove LiteGS artifact checks:
    - `outputs/engines/litegs/splats.splat`
    - `outputs/engines/litegs/splats.ply`
    - `outputs/engines/litegs/metadata.json`
  - simplify viewer/model discovery to gsplat paths

### Frontend LiteGS Cleanup

- In `bimba3d_frontend/src/components/tabs/ProcessTab.tsx`:
  - change `TrainingEngine = "gsplat" | "litegs"` to a gsplat-only type or remove engine selection state entirely
  - remove LiteGS option from engine select
  - remove `litegsTargetPrimitives` state
  - remove `litegsAlphaShrink` state
  - remove LiteGS help text
  - remove LiteGS-only controls block
  - remove `litegs_target_primitives` and `litegs_alpha_shrink` from request payloads
  - simplify `engine === "gsplat"` conditional branches because gsplat is the only engine

- In viewer/comparison components:
  - keep the generic `engine` display only if it is useful for existing output layout
  - remove UI paths or labels that allow choosing LiteGS
  - keep reading `files.engines.gsplat` if backend still returns engine-scoped output bundles

### Data/Artifact Note

Do not delete existing run output folders under `outputs/engines/litegs` without fresh approval. The code can stop producing/reading LiteGS outputs, but old artifacts are user data.

## Generated And Temporary File Cleanup Plan

The following look like temporary, generated, or one-off AI/debug artifacts. They are candidates for deletion after confirming they are not needed for thesis evidence.

### Root-Level Candidates

- `filter_claude.py`
- `filter_message.py`
- `filter_msg.bat`
- `analyze_all_cameras.py`
- `audit_all_pipeline_exif.py`
- `comprehensive_camera_scan.py`
- `check_multipliers.py`
- `complete_end_to_end_test.py`
- `test_group_multiplier_flow.py`
- `test_offline_dataset_endpoint.py`
- `40.8.0`
- `No`
- `windows_log_train.txt`

Keep for now:

- `README.md`
- `docker-compose.yml`
- `compatibility-matrix.json`
- `start_backend.bat`
- `windows_install.text`
- all installer/Windows-related files

### Temporary Folders

Candidates for deletion only after confirming there is no thesis evidence inside:

- `temp_custom_projects/`
- `temp_extract/`
- `__pycache__/`
- any nested `__pycache__/`

Do not delete yet without review:

- `bimba3d_backend/data/_offline_training/`
- `bimba3d_backend/data/_offline_training_for_best_lambda_search/`
- `bimba3d_backend/data/models/`
- `docs/Thesis_Report/`
- `docs/training_and_learning/`

Reason: these may contain experiment datasets, trained model artifacts, or thesis source material.

### Old Documentation Candidates

These may describe previous AI-bot designs rather than final thesis implementation. Do not delete immediately; archive or review first:

- `docs/early_stop_hybrid_behavior.html`
- `docs/core_ai_optimization_tests_report.html`
- `docs/model_reuse_and_checkpoint_reference.html`
- `docs/exif_ai_modes_and_training.html`
- `docs/ai_optimization_input_modes_plan.html`
- `docs/ai_optimization_input_modes_plan.docx`
- `docs/AI_Modes_Reference.rtf`
- `docs/md/CONTEXTUAL_CONTINUOUS_*`
- `docs/md/BATCH_MODEL_UPDATE.md`
- `docs/md/LEARNER_MODEL_ELEVATION.md`
- `docs/md/MODEL_LINEAGE_COMPARISON.md`
- `docs/md/REMAINING_ISSUES_PLAN.md`

Preferred action:

- Move useful thesis-compatible content into the thesis report or final docs.
- Delete old design docs once their useful content is migrated.

## Backend Module Restructure

## Frontend Status Checkpoint

Frontend restructuring is now mostly implemented and should be treated as the UI contract for the backend pass.

Done:

- Home page is a high-level entry page with three paths:
  - Projects
  - Comparison
  - Research Workflow
- `/projects` is the dedicated project list page.
- `/workflow` is the Research Workflow page with four stage cards:
  - Prepare Offline Data
  - Training Data
  - Train Models
  - Run Tests
- `/training-data` exists as a separate workflow page.
- Workflow detail pages are split by stage:
  - Prepare Offline Data detail
  - Train Models detail
  - Run Tests detail
- Pipeline detail pages keep logs visible.
- Runtime controls are inside opened pipeline pages, not on summary cards.
- Pipeline cards and workflow cards are clickable and show navigation affordance.
- Breadcrumbs are active across workflow and project pages.
- Prepare Offline Data detail includes:
  - overview
  - projects and runs
  - training data rows
  - prepared dataset
  - configuration
  - EXIF
  - logs
- EXIF UI is reduced to Mode 3: EXIF + Flight Plan + Scene.
- Prepared Dataset avoids raw `[object Object]` table columns and uses modal/detail handling for object values.
- Log-space schedule UI shows three horizontal rows for geometry, appearance, and densification, with selected values highlighted.
- Relative score distribution charts are visible where current backend data supports them.
- Testing overview shows latest prediction preview, candidate score surface, observed score distribution, models used, export action, logs, and retry failed where available.
- Train Models page opens an in-page Create New modal.
- Train Models modal validates selected Training Data before showing model options.
- Train Models list supports search and sort.
- Model detail page remains registry-only. Backend must register trained models before this page can show newly trained artifacts.
- Project page has been restructured while keeping the old visual/tab behavior for kept tabs:
  - Images
  - Process
  - Test Results
  - Logs
  - Sessions
  - Models
- Project Logs now shows processing logs only.
- Project Test Results is a first-class tab, not inside logs.
- Project Models is read-only and links users to Train Models workflow.
- Active project Process config is gsplat-only in the frontend.
- LiteGS frontend options and payload fields are removed from the active new project path.
- Early-stop frontend controls are removed from active project Process config.
- Project Process config includes `Update training data` and `training_data_target_id`.
- Comparison is now top-level at `/comparison` and reuses the old run-comparison implementation.

Frontend items intentionally left for later cleanup:

- Remove or redirect legacy `/project/:id` and old `ProjectDetail.tsx` after backend routing is settled.
- Remove old `/model-training/new` / `ModelTrainingBuilderPage.tsx` after model-training backend endpoints support the in-page flow fully.
- Decide whether legacy `/comparison/:id` is still needed after the comparison hub is accepted.
- Delete remaining frontend legacy files in the same cleanup pass as backend legacy removal.
- Keep old combined pipeline page available until backend APIs are renamed and split.

### Current Problem Areas

- `bimba3d_backend/worker/entrypoint.py` is very large and mixes:
  - status updates
  - engine dispatch
  - LiteGS helpers
  - eval history collection
  - preview materialization
  - AI-learning result persistence

- `bimba3d_backend/app/api/projects.py` is very large and mixes:
  - project CRUD
  - run dispatch
  - model registry integration
  - output discovery
  - viewer artifact resolution
  - comparison summaries

- `bimba3d_backend/app/api/training_pipeline.py` is very large and mixes:
  - pipeline CRUD
  - model training endpoints
  - offline dataset management
  - model elevation
  - pipeline run summaries

### Proposed Backend Files To Create

## Backend Restructure Plan From Current Frontend Contract

This section supersedes the older backend phase order where it conflicts. The backend should be refactored by creating report-named modules first, then moving active code into them, and only then deleting legacy modules. Avoid frontend-local or pipeline-local special cases that will disappear later.

### Step 0: Report Naming And New Module Shells

- Create report-named backend modules before deleting old ones.
- Keep old files temporarily as import callers/adapters only while moving code.
- Do not keep backward-compatible strategy names after the migration is complete.
- Rename files, functions, metadata fields, route payload names, and logs to match the thesis terminology:
  - `training_data` for prepared final datasets
  - `offline_data_preparation` for baseline/exploration data collection
  - `model_training` for Ridge/MLP training
  - `testing_evaluation` or `testing_pipeline` for default/Ridge/MLP testing
  - `featurewise_ridge_regression`
  - `featurewise_mlp`
  - `relative_quality_score`
- Backend log messages should use the same terminology so logs match the UI and thesis report.

### Step 1: Data Ownership And Directory Structure

Avoid duplicate data by defining one owner for each artifact type. Other views should reference that owner by id/path, not copy the payload.

Implementation checkpoint:

- Added `DATA_ROOT = DATA_DIR.parent` in backend config as the shared-data root.
- Added `app/services/workflow_paths.py` with canonical target directories for:
  - offline data preparation
  - Training Data
  - model training
  - testing
  - shared workflow models
- Added `app/schemas/workflow_data.py` with report-aligned manifest schemas:
  - `TrainingDataRow`
  - `TrainingDataManifest`
  - `FixedLogSpaceSchedule`
  - `CandidateScorePoint`
  - `TestingCandidateCurve`
  - `WorkflowModelManifest`
- Added `app/services/training_data_registry.py` as the first real shared-data service:
  - creates reusable Training Data manifests
  - stores rows in the canonical workflow Training Data directory
  - lists manifests newest first
  - replaces row files atomically
  - marks manifests failed with explicit errors
- Added focused tests in `tests/test_training_data_registry.py`.
- Added `app/api/training_data.py` as the first `/api/...` workflow route:
  - `GET /api/workflow/training-data`
  - `GET /api/workflow/training-data?usable_only=true`
  - `POST /api/workflow/training-data`
  - `GET /api/workflow/training-data/{training_data_id}`
  - `GET /api/workflow/training-data/{training_data_id}/validity`
  - `GET /api/workflow/training-data/by-source-pipeline/{source_pipeline_id}`
  - `GET /api/workflow/training-data/{training_data_id}/rows`
  - `POST /api/workflow/training-data/{training_data_id}/build-from-learning-rows`
  - `POST /api/workflow/training-data/{training_data_id}/build-from-pipeline/{pipeline_id}`
- Registered the new router in `app/main.py` without changing legacy direct routes.
- Added focused API tests in `tests/test_training_data_api.py`.
- Added `app/services/training_data_builder.py`:
  - converts existing pipeline learning rows into `TrainingDataRow`
  - keeps `x_features`, `selected_multipliers`, `selected_log_multipliers`, relative/score fields, and 5000-step loss fields
  - writes converted rows through the Training Data registry
  - fails explicitly and marks the manifest failed when required row fields are missing
- Added focused tests in `tests/test_training_data_builder.py`.
- Added `app/services/pipeline_learning_rows.py`:
  - owns pipeline-level learning-row collection for completed offline-data pipelines
  - lets new Training Data APIs build from a pipeline id without importing the legacy API endpoint
  - preserves existing learning-table fields and adds `project_id`, `x_features`, and `selected_log_multipliers` for Training Data builds
  - keeps the fixed log-space schedule metadata from the pipeline config in the response
- Added `app/services/workflow_model_registry.py`:
  - registers trained thesis model artifacts as shared workflow models
  - validates that artifact, metadata, and report files exist before registration
  - writes per-model manifests under the canonical workflow model directory
  - maintains a shared model index
  - reads/list models newest first
- Added focused tests in `tests/test_workflow_model_registry.py`.
- Added `app/api/models.py` as the new read-only shared workflow model API:
  - `GET /api/models`
  - `GET /api/models?source_pipeline_id={pipeline_id}`
  - `GET /api/models?source_training_data_id={training_data_id}`
  - `GET /api/models/{model_id}`
- Registered the new router in `app/main.py`.
- Added focused tests in `tests/test_models_api.py`.
- Added `app/services/model_training.py`:
  - consumes reusable Training Data ids
  - filters baseline rows and optional phase/run selections
  - trains Featurewise Ridge Regression using the current working Ridge trainer logic
  - calls the current MLP trainer for Featurewise MLP while report-named modules are still being created
  - writes model artifacts under the canonical workflow model directory
  - registers completed models through `workflow_model_registry.py`
  - raises clear errors when selected Training Data has no usable rows
- Added focused tests in `tests/test_model_training_service.py`.
- Added `app/api/model_training.py` as the new workflow model-training API:
  - `GET /api/workflow/model-training/training-data-sources`
  - `GET /api/workflow/model-training/training-data-sources?usable_only=false`
  - `GET /api/workflow/model-training/summary`
  - `POST /api/workflow/model-training/train`
  - consumes `model_family`, `model_name`, `source_training_data_id`, optional Ridge lambda, candidate points, phase/run filters
  - returns the registered shared workflow model manifest
  - returns structured errors for missing Training Data, invalid inputs, and training failures
- Training Data validity checkpoint:
  - `training_data_registry.py` now has `is_usable_manifest(...)` and `list_usable_manifests(...)`.
  - A Training Data artifact is usable for model training only when:
    - status is `ready`
    - `schema_valid` is true
    - `row_count` is greater than 0
  - Frontend model-training setup should use `/api/workflow/model-training/training-data-sources` by default so invalid/empty datasets are not selectable.
  - If the UI shows all datasets, it should call `/api/workflow/training-data/{training_data_id}/validity` and keep model options disabled until `usable_for_model_training=true`.
- Added `app/services/workflow_summaries.py`:
  - read-only model-training summary for overview cards/tables
  - read-only testing-pipeline summary for overview cards/tables
  - read-only offline-data-preparation summary for overview cards/tables
  - fixed log-space schedule summary for restart/stop/resume visibility
- Model-training overview endpoint:
  - `GET /api/workflow/model-training/summary`
  - returns total models, Ridge count, MLP count, latest model, total training samples, and per-model summary rows
  - per-model rows include selected lambda, MLP validation/train-loss fields when available, best model step/epoch when available, and raw metrics
- Testing overview endpoint:
  - `GET /api/workflow/pipelines/{pipeline_id}/testing-summary`
  - returns total test projects, models tested, completed/failed run counts, relative-score stats, per-model completion status, and latest prediction-preview candidate counts
  - currently reads from pipeline state and prediction preview records only; deeper metric-delta summaries should be added when testing result rows are normalized.
- Offline Data Preparation overview endpoints:
  - `GET /api/workflow/pipelines/{pipeline_id}/fixed-log-space-schedule`
  - `GET /api/workflow/pipelines/{pipeline_id}/offline-data-summary`
  - fixed schedule response exposes restart version/token/timestamp, current index, last picked value, and next value per group
  - offline summary response exposes total projects, total/completed/failed runs, learning row counts, baseline/non-baseline counts, mean/best relative score, and multiplier-score distribution for geometry/appearance/densification
  - frontend Prepare Offline Data overview should use these endpoints instead of parsing raw pipeline config/learning rows directly.
- EXIF workflow routes:
  - `POST /api/workflow/pipelines/{pipeline_id}/test-exif/start`
  - `GET /api/workflow/pipelines/{pipeline_id}/test-exif/progress/{task_id}`
  - `GET /api/workflow/pipelines/{pipeline_id}/test-exif/results`
  - `POST /api/workflow/pipelines/{pipeline_id}/test-exif/stop/{task_id}`
  - These are currently the same streaming EXIF router mounted under the new workflow prefix.
  - Later cleanup should reduce the EXIF worker to Mode 3: EXIF + Flight Plan + Scene only, matching the current frontend/report flow.
- Testing prediction/export endpoints:
  - `GET /api/workflow/pipelines/{pipeline_id}/prediction-preview`
  - `GET /api/workflow/pipelines/{pipeline_id}/prediction-preview?preview_key={key}`
  - `GET /api/workflow/pipelines/{pipeline_id}/testing-candidate-curves`
  - `GET /api/workflow/pipelines/{pipeline_id}/testing-candidate-curves?preview_key={key}`
  - `POST /api/workflow/pipelines/{pipeline_id}/predict-multipliers`
  - `GET /api/workflow/pipelines/{pipeline_id}/export-current-test`
  - Prediction preview read is normalized in `workflow_pipeline_service.py`.
  - Candidate curves are normalized in `workflow_summaries.py` for notebook-style test graphs:
    - one curve per project/model preview row
    - points grouped by multiplier group
    - selected point preserved when present
    - highest point inferred from predicted relative score/score when no selected flag is present
  - Frontend testing charts should use `testing-candidate-curves` instead of parsing raw prediction preview blobs directly.
  - Predict multipliers and export current test are explicit temporary bridges to the current legacy implementation until their internals are extracted.
- Registered the model-training router in `app/main.py`.
- Added focused tests in `tests/test_model_training_api.py`.
- Thesis alignment note:
  - The new service/API now follows the thesis workflow shape: reusable Training Data -> Featurewise Ridge Regression or Featurewise MLP -> shared registered model.
  - The Ridge/MLP mathematical internals still call the existing working trainer functions while restructuring is in progress.
  - Next extraction step is to move those internals into report-named modules `featurewise_ridge_regression.py` and `featurewise_mlp.py` without changing model behavior.
- Added report-named model modules:
  - `worker/ai_input_modes/feature_schema.py`
  - `worker/ai_input_modes/featurewise_ridge_regression.py`
  - `worker/ai_input_modes/featurewise_mlp.py`
- Updated `app/services/model_training.py` to import through the report-named Ridge/MLP modules.
- Added focused tests in `tests/test_featurewise_report_modules.py`.
- Added `worker/ai_input_modes/relative_quality_score.py`:
  - new report-named entry point for the relative quality/convergence score summary
  - wraps the current known-good `compute_score_summary` implementation while migration is in progress
- Updated `contextual_continuous_learner.py` to import scoring through `relative_quality_score.py`.
- Added focused tests in `tests/test_relative_quality_score.py`.
- Added `app/services/workflow_pipeline_service.py`:
  - normalizes existing pipeline records into workflow stage summaries
  - maps legacy `train` pipelines to `offline_data_preparation`
  - maps legacy `test` pipelines to `testing_pipeline`
  - exposes fixed log-space schedule metadata from pipeline config in a consistent detail shape
  - exposes worker logs from both project-level and run-level `processing.log` files
  - reads learning rows through `pipeline_learning_rows.py`
  - owns simple workflow pipeline actions for start, pause, resume, and stop
  - preserves current orchestrator start/stop calls and worker stop behavior
  - owns creation preparation for workflow pipelines:
    - maps `offline_data_preparation` to current storage `pipeline_type=train`
    - maps `testing_pipeline` to current storage `pipeline_type=test`
    - canonicalizes testing model ids
    - initializes restart metadata
    - generates fixed pre-generated multiplier schedules at creation
  - owns workflow pipeline config updates:
    - rejects config saves while the pipeline is running
    - preserves `pipeline_folder`
    - preserves restart version/token/timestamp
    - preserves fixed pre-generated multiplier schedules and current multiplier index
    - recalculates total run count
    - marks completed pipelines stopped/resumable when only pass count increases
  - owns workflow pipeline creation-support helpers:
    - scans a dataset directory for image-containing dataset folders
    - batch-creates project configs from selected datasets
    - reports already existing projects without recreating them
- Added `app/api/workflow_pipelines.py` as the new workflow pipeline read API:
  - `GET /api/workflow/pipelines`
  - `POST /api/workflow/pipelines`
  - `POST /api/workflow/pipelines/scan-directory`
  - `POST /api/workflow/pipelines/batch-create-projects`
  - `GET /api/workflow/pipelines?stage=offline_data_preparation`
  - `GET /api/workflow/pipelines?stage=testing_pipeline`
  - `GET /api/workflow/pipelines/{pipeline_id}`
  - `PUT /api/workflow/pipelines/{pipeline_id}/config`
  - `GET /api/workflow/pipelines/{pipeline_id}/learning-rows`
  - `GET /api/workflow/pipelines/{pipeline_id}/worker-logs`
  - `GET /api/workflow/pipelines/{pipeline_id}/fixed-log-space-schedule`
  - `GET /api/workflow/pipelines/{pipeline_id}/offline-data-summary`
  - `POST /api/workflow/pipelines/{pipeline_id}/test-exif/start`
  - `GET /api/workflow/pipelines/{pipeline_id}/test-exif/progress/{task_id}`
  - `GET /api/workflow/pipelines/{pipeline_id}/test-exif/results`
  - `POST /api/workflow/pipelines/{pipeline_id}/test-exif/stop/{task_id}`
  - `GET /api/workflow/pipelines/{pipeline_id}/testing-summary`
  - `GET /api/workflow/pipelines/{pipeline_id}/prediction-preview`
  - `GET /api/workflow/pipelines/{pipeline_id}/testing-candidate-curves`
  - `POST /api/workflow/pipelines/{pipeline_id}/predict-multipliers`
  - `GET /api/workflow/pipelines/{pipeline_id}/export-current-test`
  - `POST /api/workflow/pipelines/{pipeline_id}/start`
  - `POST /api/workflow/pipelines/{pipeline_id}/pause`
  - `POST /api/workflow/pipelines/{pipeline_id}/resume`
  - `POST /api/workflow/pipelines/{pipeline_id}/stop`
  - `POST /api/workflow/pipelines/{pipeline_id}/restart`
  - `POST /api/workflow/pipelines/{pipeline_id}/retry-failed`
- Restart and retry-failed are currently explicit temporary bridges to the legacy implementation:
  - keep this bridge until dedicated service extraction tests cover project-folder cleanup
  - fixed log-space multiplier schedule preservation
  - retry snapshot preservation
  - restart version/token/seed metadata
  - worker stop/start behavior
- Frontend migration note:
  - Workflow pages should eventually call `/api/workflow/pipelines/...` for list/detail/logs/actions instead of `/training-pipeline/...`.
  - Create Offline Data and Run Tests should eventually post to `/api/workflow/pipelines` with `workflow_stage=offline_data_preparation` or `workflow_stage=testing_pipeline`.
  - Dataset scan should move from `/training-pipeline/scan-directory` to `/api/workflow/pipelines/scan-directory`.
  - Batch project creation should move from `/training-pipeline/batch-create-projects` to `/api/workflow/pipelines/batch-create-projects`.
  - The backend still stores offline-data-preparation pipelines with legacy `pipeline_type=train` internally; UI should use `workflow_stage` labels from the new API.
  - `model_training` is intentionally rejected by `/api/workflow/pipelines`; model training must use `/api/workflow/model-training`.
  - Pipeline config saves should migrate to `PUT /api/workflow/pipelines/{pipeline_id}/config`.
  - New config update behavior preserves fixed multiplier schedules. Frontend should tell users that changed project/phase/config structure still requires Restart to regenerate fixed values.
  - No frontend update has been made in this backend pass.
  - When frontend routes are migrated, log any API response mismatch or missing field in this memory document before adding compatibility fields.
- Registered the workflow pipeline router in `app/main.py`.
- Added focused tests in `tests/test_workflow_pipelines_api.py`.
- These are scaffolding only. Existing backend storage remains untouched until services are migrated.

Legacy code/endpoint candidates now marked for later review, not deletion yet:

- `GET /training-pipeline/{pipeline_id}/learning-table`
  - Legacy route name for aggregated pipeline learning rows.
  - It now delegates to `app/services/pipeline_learning_rows.py`.
  - Keep until frontend calls move fully to `/api/workflow/training-data` and any future `/api/workflow/...` summary endpoints.
  - Old unreachable collector body/helper functions in `app/api/training_pipeline.py` are marked for later deletion after verification.
- `GET /training-pipeline/list` and `GET /training-pipeline/{pipeline_id}`
  - Legacy direct pipeline listing/detail routes.
  - New read path is `/api/workflow/pipelines` and `/api/workflow/pipelines/{pipeline_id}`.
  - Keep legacy routes until frontend workflow pages are migrated to the new `/api/...` endpoints.
- `POST /training-pipeline/create`
  - Legacy direct pipeline creation route.
  - New create path is `POST /api/workflow/pipelines`.
  - New request can use `workflow_stage=offline_data_preparation` or `workflow_stage=testing_pipeline`.
  - Current storage still receives `pipeline_type=train` for offline data preparation until orchestrator/storage terminology is fully renamed.
- `POST /training-pipeline/scan-directory`
  - Legacy direct dataset scan route.
  - New path is `POST /api/workflow/pipelines/scan-directory`.
- `POST /training-pipeline/batch-create-projects`
  - Legacy direct batch project creation route.
  - New path is `POST /api/workflow/pipelines/batch-create-projects`.
- `PUT /training-pipeline/{pipeline_id}/config`
  - Legacy direct config update route.
  - New path is `PUT /api/workflow/pipelines/{pipeline_id}/config`.
  - New path preserves fixed multiplier schedule and current index until explicit restart.
- `GET /training-pipeline/{pipeline_id}/worker-logs`
  - Legacy direct worker-log route.
  - New read path is `/api/workflow/pipelines/{pipeline_id}/worker-logs`.
  - Keep until frontend workflow detail pages use the new endpoint.
- `POST /training-pipeline/{pipeline_id}/start`
- `POST /training-pipeline/{pipeline_id}/pause`
- `POST /training-pipeline/{pipeline_id}/resume`
- `POST /training-pipeline/{pipeline_id}/stop`
  - Legacy direct action routes.
  - New paths are `/api/workflow/pipelines/{pipeline_id}/start|pause|resume|stop`.
  - Keep until frontend workflow controls are migrated to the new `/api/...` action routes.
- `POST /training-pipeline/{pipeline_id}/restart`
- `POST /training-pipeline/{pipeline_id}/retry-failed`
  - Legacy direct action routes.
  - New `/api/workflow/pipelines/{pipeline_id}/restart` and `/api/workflow/pipelines/{pipeline_id}/retry-failed` routes exist, but currently bridge into the legacy implementation.
  - Full extraction is still needed because these actions clean project state and must preserve fixed log-space multiplier/retry snapshots exactly.
- `POST /training-pipeline/{pipeline_id}/predict-multipliers`
  - Legacy direct prediction-preview route.
  - New path is `POST /api/workflow/pipelines/{pipeline_id}/predict-multipliers`.
  - New read path for saved previews is `GET /api/workflow/pipelines/{pipeline_id}/prediction-preview`.
  - Current predict action still bridges to legacy implementation.
- `GET /training-pipeline/{pipeline_id}/export-current-test`
  - Legacy direct export route.
  - New path is `GET /api/workflow/pipelines/{pipeline_id}/export-current-test`.
  - Current export action still bridges to legacy implementation.
- `POST /training-pipeline/{pipeline_id}/test-exif/start`
- `GET /training-pipeline/{pipeline_id}/test-exif/progress/{task_id}`
- `GET /training-pipeline/{pipeline_id}/test-exif/results`
- `POST /training-pipeline/{pipeline_id}/test-exif/stop/{task_id}`
  - Legacy direct EXIF streaming routes.
  - New paths are the same under `/api/workflow/pipelines/...`.
- `POST /training-pipeline/{pipeline_id}/build-offline-dataset`
  - Old build path that writes `_offline_training/offline_dataset.json` under pipeline folders.
  - Replace with workflow Training Data registry build/update.
  - New API now has `POST /api/workflow/training-data/{training_data_id}/build-from-learning-rows`; remaining work is to collect rows from pipeline/project services instead of requiring rows in the request.
- `GET /training-pipeline/{pipeline_id}/offline-dataset`
  - Old dataset read path tied to pipeline-local storage.
  - Replace with `/api/workflow/training-data/by-source-pipeline/{pipeline_id}` to find reusable Training Data artifacts for a pipeline.
  - Then use `/api/workflow/training-data/{training_data_id}/rows` to read rows.
- `POST /training-pipeline/{pipeline_id}/train-model`
  - Old model training path tied to Training Data pipeline ids and pipeline-local `trained_models`.
  - Replace with model-training API that consumes shared Training Data ids and registers shared workflow models.
  - The active training logic has started moving into `app/services/model_training.py`; keep the old endpoint until the new API is wired and frontend calls are migrated.
- `GET /training-pipeline/{pipeline_id}/trained-models`
  - Temporary visibility bridge only.
  - Replace with `/api/models?source_pipeline_id={pipeline_id}` or `/api/models?source_training_data_id={training_data_id}`.
  - Remove after frontend Train Models page lists only shared workflow registry models.
- Old model elevation endpoint/action.
  - Remove after model training automatically registers shared workflow models.
- Existing `app/services/model_registry.py`
  - Keep for existing elevated/project reusable models while migration is in progress.
  - Later either merge it into the workflow model registry or keep a thin compatibility reader for old model records only.
  - New Ridge/MLP model-training outputs should use `workflow_model_registry.py`.
- `GET /projects/models`
  - Old model-list API used by the current frontend.
  - Replace with `/api/models` after model-training registration is wired and frontend API calls are migrated.
- `scripts/train_offline_models.py`
  - Keep while report-named Ridge module wraps its known-good math.
  - Later move Ridge internals fully into `worker/ai_input_modes/featurewise_ridge_regression.py`.
- `worker/ai_input_modes/neural_contextual_learner.py`
  - Keep while report-named MLP module wraps its known-good math.
  - Later move MLP internals fully into `worker/ai_input_modes/featurewise_mlp.py`.
- `worker/ai_input_modes/learner.py`
  - Old scorer owner.
  - Keep until all callers use `relative_quality_score.py` and the old preset-bias learning code is no longer referenced.

Canonical backend storage targets:

- `data/projects/{project_id}/`
  - Owns project source images, COLMAP artifacts, reconstruction runs, per-run logs, viewer outputs, and project-level test-result rows.
  - Does not own shared trained models.
  - Does not own final workflow-level Training Data datasets.

- `data/workflow/offline_data_preparation/{pipeline_id}/`
  - Owns pipeline run plan, selected projects, baseline/exploration run references, fixed log-space multiplier schedule, restart metadata, and extraction manifests.
  - Stores references to project/run outputs, not full duplicate copies of project outputs.
  - May store lightweight derived extraction reports when they are pipeline-specific.

- `data/workflow/training_data/{training_data_id}/`
  - Owns final prepared training dataset metadata and row index.
  - Rows should reference project/run ids and carry only the normalized report fields needed for model training:
    - `x_features`
    - `selected_multipliers`
    - `selected_log_multipliers`
    - `relative_quality_score`
    - `score_quality` temporarily until renamed
    - `score_convergence` if retained
    - `loss_at_reference_step_run`
    - `loss_at_reference_step_base`
    - baseline/default references
    - source offline-data-preparation pipeline id
  - Do not copy raw project outputs or entire run folders.
  - Expose validity metadata:
    - `row_count`
    - `schema_valid`
    - `feature_schema`
    - `source_pipeline_id`
    - `last_built_at`
    - `status`
    - `errors`

- `data/workflow/models/{model_id}/`
  - Owns trained Ridge/MLP model artifacts, metadata, training report, source Training Data id, and metrics.
  - This should be the shared model registry source used by `/projects/models` and `/model-training/models/:id`.
  - Model training completion must register the model here automatically.
  - No elevate action should be required.

- `data/workflow/testing/{pipeline_id}/`
  - Owns testing pipeline plan, selected test projects, selected model ids, default/Ridge/MLP run references, prediction previews, candidate score curves, comparison summaries, export manifests, and test logs.
  - Stores references to project/run outputs instead of copying outputs.

Data-source rules:

- One artifact type has one owner.
- Pipeline and project screens can read the same source, but should not maintain separate copies of the same rows.
- If a row is generated from a project run, store `project_id`, `run_id`, `pipeline_id`, `stage`, and `artifact_version`.
- If a dataset is rebuilt, create/update a manifest with a dataset version/hash so model training can record exactly which version it used.
- If a pipeline is stopped/resumed, never regenerate fixed data such as log-space schedules.
- If a pipeline is restarted after config/project-count changes, regenerate fixed schedules only through an explicit restart path and record seed/version/config hash.

### Step 2: API Route Standardization

- All backend API routes should live under `/api/...`.
- Frontend application routes such as `/workflow`, `/projects`, and `/model-training` must remain frontend routes only.
- Plan target route groups:
  - `/api/projects/...`
  - `/api/workflow/offline-data-preparation/...`
  - `/api/workflow/training-data/...`
  - `/api/workflow/model-training/...`
  - `/api/workflow/testing/...`
  - `/api/models/...`
  - `/api/comparison/...`
- During migration, update frontend API client calls in one coordinated pass rather than mixing old and new API paths across components.
- Do not create duplicate endpoints with old and new names long-term. Temporary adapters should be marked clearly and deleted in the cleanup phase.

### Step 3: Split Large Backend Files

Break large files by responsibility, not by tiny helper count.

Targets:

- `app/api/training_pipeline.py`
  - Move pipeline CRUD and status into a workflow/pipeline API module.
  - Move offline data preparation endpoints into `app/api/offline_data_preparation.py`.
  - Move Training Data dataset endpoints into `app/api/training_data.py`.
  - Move model training endpoints into `app/api/model_training.py`.
  - Move testing/export endpoints into `app/api/testing_pipeline.py`.

- `app/api/projects.py`
  - Keep project CRUD and project detail APIs.
  - Move model registry APIs into `app/api/models.py`.
  - Move comparison APIs into `app/api/comparison.py`.
  - Move run artifact discovery into services.
  - Keep project processing routes, but delegate config normalization and payload building to services.

- `worker/entrypoint.py`
  - Keep the worker orchestration entrypoint thin.
  - Move gsplat run configuration building, log parsing, eval collection, output discovery, and learning-row persistence into focused worker/service helpers.

Suggested service modules:

- `app/services/workflow_storage.py`
- `app/services/offline_data_preparation.py`
- `app/services/training_data_registry.py`
- `app/services/model_training.py`
- `app/services/workflow_model_registry.py`
- `app/services/testing_evaluation.py`
- `app/services/comparison_service.py`
- `app/services/gsplat_artifacts.py`
- `app/services/error_responses.py`

### Step 4: gsplat Baseline Safety

The thesis compares default baseline gsplat runs against predicted Ridge/MLP settings. Baseline behaviour must stay as close as possible to the current working gsplat path.

Rules:

- Do not modify `worker/gsplat_upstream/simple_trainer.py` unless there is a direct bug or required config bridge.
- Treat `gsplat_upstream/simple_trainer.py` as upstream/reference code.
- Keep custom integration code outside upstream code where possible:
  - wrapper config builder
  - command builder
  - run metadata writer
  - eval collector
  - output/artifact discovery
- Baseline runs should use default gsplat parameters except values explicitly configurable from the frontend.
- Predicted runs should differ only by the selected multiplier-derived parameters and other frontend-selected settings.
- Any change that affects baseline default training must be documented in a small `gsplat_changes.md` note with reason and affected parameters.
- Keep logs that show requested config, resolved config, and applied config so baseline/predicted differences are auditable.

### Step 5: Error Handling And No Silent Fallbacks

- Replace silent fallback behavior with explicit errors where the thesis workflow requires a valid artifact.
- Prefer typed exceptions or structured error responses with:
  - `code`
  - `message`
  - `details`
  - `action`
  - `correlation_id` or pipeline/run id where available
- Examples that should fail clearly:
  - missing Training Data rows during model training
  - missing model artifact during testing
  - unsupported old mode/strategy name
  - invalid feature schema
  - missing fixed log-space schedule on resume
  - missing baseline run for relative scoring
- Fallbacks may be appropriate only where they preserve a non-thesis operational workflow. Flag these before implementation:
  - viewer preview artifact alternatives
  - reading old run logs for display-only history
  - handling missing optional EXIF fields with explicit missing-value indicators
- Do not silently replace missing Ridge/MLP models with neutral multipliers during thesis testing. If neutral multipliers are used for a planned diagnostic mode, expose it as a named mode and log it clearly.

### Step 6: Comments, Standards, And Maintainability

- Add comments only for non-obvious workflow boundaries, scoring math, storage ownership, and restart/seed behaviour.
- Use typed Pydantic request/response models for API boundaries.
- Use dataclasses or typed dicts for internal manifests where Pydantic is too heavy.
- Centralize constants:
  - multiplier group keys
  - parameter-to-group mapping
  - bounds
  - feature schema
  - report model names
- Keep functions small enough to test, but avoid too many tiny files.
- Add migration notes in code comments only where old data may still be read.
- Avoid writing raw dictionaries deep in multiple places; build manifests through helper functions.
- Keep Windows scripts and Windows-specific paths until the user requests that cleanup.

### Step 7: Backend Migration Order

1. Create report-named constants/schema modules.
2. Create new storage/registry service skeletons.
3. Move relative score calculation into `relative_quality_score.py`.
4. Create Training Data registry and dataset manifest endpoints.
5. Update model training to consume Training Data registry records.
6. Register trained Ridge/MLP models into the shared workflow model registry.
7. Update testing pipeline to consume shared model registry ids only.
8. Expose normalized overview summaries for frontend pages:
   - Training Data validity and summary
   - Model training summary
   - Testing comparison summary
   - log-space schedule state
   - candidate score curves
9. Standardize routes under `/api/...` and update frontend API client calls.
10. Split large API/worker files after service boundaries are stable.
11. Remove LiteGS backend code after gsplat-only workflow is verified.
12. Delete old learner modules and old strategy names.

### Backend Verification Plan

- Unit tests:
  - feature schema and group mapping
  - relative quality score
  - fixed log-space schedule generation and resume/restart behaviour
  - Training Data registry manifest validation
  - Ridge training metadata and lambda selection
  - MLP training metadata including best step/epoch
  - model registry registration at training completion
  - testing candidate score curve generation
- API tests:
  - `/api/workflow/training-data`
  - `/api/workflow/model-training`
  - `/api/workflow/testing`
  - `/api/models`
  - project Process `update_training_data`
- Smoke tests:
  - prepare small Training Data from existing completed runs
  - train Ridge
  - train MLP
  - confirm both appear in `/api/models`
  - run test prediction preview without launching long reconstruction
  - verify baseline config is unchanged except frontend-selected values

Under `bimba3d_backend/worker/ai_input_modes/`:

- `relative_quality_score.py`
  - `compute_relative_quality_score`
  - `compute_convergence_score_at_5000`
  - `compute_score_summary`

- `featurewise_ridge_regression.py`
  - report-aligned Ridge model loading, candidate scoring, prediction, and update-free evaluation helpers

- `featurewise_mlp.py`
  - report-aligned MLP training and prediction

- `feature_schema.py`
  - group keys, parameter groups, feature lists, bounds, and feature scaling helpers shared by Ridge and MLP

Under `bimba3d_backend/app/services/`:

- `offline_data_preparation.py`
  - scan pipeline/project run folders
  - build offline dataset payloads
  - summarize dataset readiness

- `model_training.py`
  - call Ridge trainer
  - call MLP trainer
  - collect training report metadata

- `testing_pipeline.py`
  - prepare default/Ridge/MLP test run configs
  - summarize default vs predicted results

- `gsplat_outputs.py`
  - discover gsplat outputs
  - resolve viewer model paths
  - collect previews/eval outputs

Under `bimba3d_backend/app/api/`:

- `offline_data_preparation.py`
  - API routes for offline dataset preparation

- `model_registry.py`
  - API routes for Ridge/MLP training and model records

- `testing_pipeline.py`
  - API routes for testing/evaluation pipeline workflows

Keep existing route files initially, but gradually move logic out of them. The first implementation can import service functions from the new modules while preserving route behavior.

### Backend Functions To Rename Or Move

From `contextual_continuous_learner.py` to `featurewise_ridge_regression.py`:

- `_default_score_optimizer_model` -> `_default_featurewise_ridge_model`
- `_load_offline_model` -> `_load_featurewise_ridge_payload`
- `load_offline_quality_model` -> `load_featurewise_ridge_quality_model`
- `load_offline_convergence_model` -> `load_featurewise_ridge_convergence_model`
- `_build_featurewise_vector_scaled` -> `build_scaled_featurewise_vector`
- `_build_score_design_vector` -> `build_ridge_design_vector`
- `_select_group_action_from_score_model` -> `select_group_action_from_ridge_model`
- `_build_updates` -> `build_parameter_updates_from_multipliers`
- `select_contextual_continuous` -> `select_featurewise_ridge_multipliers`
- `update_from_run_contextual_continuous` -> remove unless still needed for compare-only result summaries
- `record_run_penalty_contextual_continuous` -> remove

From `neural_contextual_learner.py` to `featurewise_mlp.py`:

- `FeaturewiseGroupMLP` can remain
- `FeaturewiseMLP` can remain
- `train_featurewise_neural_model` -> `train_featurewise_mlp_model`
- `predict_featurewise_neural_multipliers` -> `predict_featurewise_mlp_multipliers`
- `_score_group_candidates` -> `score_mlp_group_candidates`
- `_build_featurewise_score_tensor` -> `build_mlp_candidate_tensor`
- remove older direct multiplier MLP functions if they are not used by the report:
  - `MultiplierMLP`
  - `train_neural_model`
  - `predict_neural_multipliers`

From `learner.py`:

- move `compute_score_summary` to `relative_quality_score.py`
- delete old preset-bias functions:
  - `select_preset`
  - `update_from_run`
  - `record_run_penalty`

From `continuous_learner.py`:

- delete the whole file.

## Frontend Restructure

### Current Problem Areas

- `ProcessTab.tsx` is approximately 370k characters and should be split.
- `PipelineDetailsPage.tsx` is approximately 180k characters and should be split.
- `TrainingPipelinePage.tsx` mixes offline preparation, training pipeline setup, test pipeline setup, and model selection.

### Proposed Frontend Routes

Update `App.tsx` to expose:

- `/offline-data-preparation`
- `/model-training`
- `/testing-pipeline`
- `/pipelines/:id`
- `/models`
- existing project detail/viewer routes as needed

The current `/training-pipeline` can become a redirect or be renamed after the new pages are ready.

### Proposed Frontend Files To Create

Under `bimba3d_frontend/src/pages/`:

- `OfflineDataPreparationPage.tsx`
- `ModelTrainingPage.tsx`
- `TestingPipelinePage.tsx`
- `ModelRegistryPage.tsx` if replacing the tab-only registry

Under `bimba3d_frontend/src/components/workflow/`:

- `DatasetDirectoryPicker.tsx`
- `ProjectSelectionTable.tsx`
- `OfflinePreparationConfig.tsx`
- `ExplorationSchedulePanel.tsx`
- `ModelTrainingConfig.tsx`
- `ModelTrainingResults.tsx`
- `TestingPipelineConfig.tsx`
- `EvaluationSummaryTable.tsx`

Under `bimba3d_frontend/src/components/process/`:

- `TrainingRunConfigPanel.tsx`
- `GsplatParameterControls.tsx`
- `CoreAiOptimizationControls.tsx`
- `RunStorageControls.tsx`
- `RunActionButtons.tsx`

Under `bimba3d_frontend/src/components/pipelineDetails/`:

- `PipelineHeader.tsx`
- `PipelineProgressPanel.tsx`
- `PipelineRunTable.tsx`
- `PipelineModelTrainingPanel.tsx`
- `PipelineEvaluationPanel.tsx`

Under `bimba3d_frontend/src/api/`:

- `workflow.ts`
  - typed API helpers for offline preparation, model training, and testing

### Frontend Files To Refactor

- `TrainingPipelinePage.tsx`
  - split into offline preparation and testing-specific pages
  - keep configurable exploration run count
  - remove LiteGS fields
  - replace old model names with report names

- `PipelineDetailsPage.tsx`
  - move sections into pipeline detail components
  - keep existing behavior while reducing file size

- `ProcessTab.tsx`
  - remove LiteGS
  - split high-risk pieces gradually:
    - config state/defaults
    - payload building
    - validation
    - gsplat parameter controls
    - AI optimization controls
    - viewer/output panels

- `SettingsTab.tsx`
  - remove old selector options if confirmed
  - show only report models for thesis AI optimization

- `ModelRegistryTab.tsx`
  - update types to `featurewise_ridge` and `featurewise_mlp`

## Project/Pipeline Logic Model

The code should make this hierarchy clear:

- Project level
  - A single image dataset and its outputs.
  - Can run baseline/default or predicted gsplat training.
  - Stores run-level analytics and outputs.

- Pipeline level
  - A coordinated workflow over many projects.
  - Owns shared configuration and shared model/output folders.

- Offline data preparation pipeline
  - Produces baseline and exploration runs.
  - Produces offline dataset JSON.
  - Does not train Ridge/MLP models during each run.

- Model training workflow
  - Consumes the offline dataset.
  - Produces Featurewise Ridge and Featurewise MLP models.

- Testing pipeline
  - Consumes trained models.
  - Runs default, Ridge-predicted, and MLP-predicted configurations on test projects.
  - Produces comparison summaries.

## Suggested Implementation Phases

Historical note: these older phases are kept for traceability, but the active backend order is now the "Backend Restructure Plan From Current Frontend Contract" above. In particular, do not start backend work by deleting LiteGS. Create the new report-named workflow/data/model structure first, verify gsplat behaviour, then remove legacy code.

### Phase 1: Create Report-Named Backend Structure

- Create report-named schema/constants modules.
- Create workflow storage and registry service skeletons.
- Keep existing backend files active while new services are introduced.
- Do not delete LiteGS backend code yet.

### Phase 2: Move Relative Score Calculation

- Create `relative_quality_score.py`.
- Move `compute_score_summary`.
- Update imports.
- Add tests for report score fields.
- Delete `learner.py` and `continuous_learner.py` only after all active imports are moved.

### Phase 3: Training Data And Model Registry

- Create `feature_schema.py`.
- Add Training Data registry and manifest services.
- Update model training to consume Training Data registry records.
- Register trained Ridge/MLP models in the shared model registry automatically.

### Phase 4: Rename Report Models And Testing

- Rename Ridge module/function names.
- Rename MLP module/function names.
- Update resolver, model registry, pipeline orchestration, and testing pipeline to use report names only.
- Remove old mode names after the report-named paths are verified.

### Phase 5: Standardize API Routes

- Move API routes under `/api/...`.
- Update frontend API client calls in one coordinated pass.
- Remove old direct route adapters after verification.

### Phase 6: Split Large Files And Remove Legacy Code

- Split large API/worker files through service boundaries.
- Remove LiteGS backend code after gsplat-only workflow is verified.
- Remove old combined frontend/backend legacy routes and files after confirmation.

Current frontend restructuring notes:

- Workflow pipeline cards should link to a new workflow detail route instead of the old combined detail page.
- Keep the old `PipelineDetailsPage.tsx` untouched until the replacement detail pages cover the required behavior.
- New detail route: `/workflow/pipelines/:id`.
- New detail pages:
  - Prepare Offline Data detail:
    - Overview
    - Projects & Runs
    - Prepared Dataset
    - Logs
    - Configuration
    - EXIF
  - Train Models detail:
    - Overview
    - Training Data
    - Model Details
    - Metrics
    - Logs
    - Artifacts
  - Run Tests detail:
    - Overview
    - Projects & Runs
    - Models Used
    - Predictions
    - Results
    - Logs
    - Configuration
- Shared workflow detail components should handle the common header, tabs, progress summary, project/run list, logs, and raw configuration display.
- Logging must remain visible in each workflow detail page.
- The Train Models detail is the UI contract for first-class saved model outputs. These models should not need a later elevate step.
- Existing backend `train` pipeline type is treated as `training_data` in the UI until the backend rename is completed.
- New overview pages should preserve the compact operational layout from the old detail page: status, progress, learning stats, time, live progress, activity logs, and errors.
- Projects & Runs should support filtering runs by one or more selected projects.
- The testing Predictions tab should reuse the prepared-row table UI instead of a small duplicate preview table.
- Existing registered model artifacts are clickable through `/model-training/models/:id`; future model-training pipeline cards should link to `/workflow/pipelines/:id`.
- `/model-training/new` is legacy. The active Train Models page uses an in-page Create New modal. Remove or redirect `/model-training/new` after backend model-training endpoints are finalized.
- Prepare Offline Data should expose `Training Data Rows` as its own primary tab. Avoid the old `AI Learning Table` wording because the workflow is now offline data preparation, not online training.
- EXIF should remain named `EXIF`, not grouped under `Advanced`.
- The Prepare Offline Data overview should stay compact and should not include the prepared dataset panel at the bottom.
- In Projects & Runs, the left side should be only a project selector list with checkboxes and project names; detailed progress belongs in the run table on the right.
- Runs should default to project grouping, with an optional time sort.
- Data-heavy lists/tables should use compact local typography instead of shrinking the whole platform globally.
- Truncated table/list values should use hover titles consistently.
- Object-valued dataset cells such as features and multipliers should open a detail modal instead of rendering `[object Object]`.
- Prepared Dataset should retain Build/Rebuild Dataset and Refresh actions.
- Prepared Dataset should use the old concise row table shape: phase, project, run, mode, baseline, score metrics, scores, multipliers button, and features button. Raw object fields such as `x_features`, `selected_multipliers`, and `selected_log_multipliers` should not become automatic table columns.
- `selected_multipliers` should be grouped as geometry, appearance, and densification in the UI. Parameter-level values should come from the prepared row's parameter multiplier data where available.
- EXIF should retain extraction controls, stop control, live progress, grouped result tables, and expandable feature values from the old details page.
- EXIF result display should only show Mode 3: EXIF + Flight Plan + Scene, because the older EXIF-only and EXIF + Flight Plan modes have been removed from the workflow.
- Breadcrumbs should appear from the workflow page onward and use one shared style: Home icon, chevron separators, linked ancestors, current page highlighted.
- Detail URLs should preserve the originating workflow section:
  - `/offline-data-preparation/pipelines/:id`
  - `/model-training/pipelines/:id`
  - `/testing-pipeline/pipelines/:id`
  - `/all-pipelines/pipelines/:id`
- `/workflow/pipelines/:id` can remain as a generic fallback, but stage list pages should link to their stage-specific detail paths.
- Runtime pipeline controls should be shared through one modest component and reused inside opened pipeline detail pages. Pipeline cards should stay read-only/navigation-only and should not show Stop/Resume/Restart controls.
- The Research Workflow page should show the four active workflow cards in a single row on wide screens: Prepare Offline Data, Training Data, Train Models, and Run Tests. The all-pipelines/search/filter workspace appears below those cards, not as a separate navigation card.
- Workflow stage cards should remain fully clickable but include a visible `Open` affordance with an arrow so users understand the cards navigate to deeper pages.
- Testing pipeline `Models Used` cards should handle long model names with wrapping/clamping and hover titles. The backend should later provide explicit per-model test progress, for example `{ model_id, completed_projects, total_projects, failed_projects, status }`, so the UI does not have to infer `1/5 done` from raw run records.
- Testing detail should not depend on a separate `config.results` blob for the main result view. The test result view is the recorded `Training Data Rows` table: project, model id, selected multipliers, scores, score terms, and report values. A later backend pass can add structured comparison summaries above that table.
- `Export Current Test` is an important testing workflow action and should remain available inside the opened Run Tests detail page.
- Train Models needs an overview page comparable to Prepare Offline Data and Run Tests, but focused on model-training outputs rather than run exploration.
- Train Models overview design plan:
  - Top summary cards: pipeline status, selected/source training dataset, total trained models, completed/failed model jobs, best validation/report metric, elapsed time.
  - Model summary strip/table: Ridge and MLP rows with status, dataset source, runs used, selected lambda or architecture, training/validation metrics, artifact path, and created time.
  - Lineage block: source `training_data` pipeline, prepared dataset file, feature set, multiplier groups, and model output directory.
  - Logs/activity block: latest training job events and errors, with the same logging visibility as other pipeline pages.
- Chart plan for Train Models overview:
  - Do not move old offline-data charts blindly. First define the chart question.
  - Dataset coverage chart: rows by project and phase, to confirm training data balance.
  - Score distribution chart: score/score distribution by phase or project, to show whether training targets have usable spread.
  - Model comparison chart: Ridge vs MLP validation/test metric bars once backend returns comparable metrics.
  - Prediction/multiplier chart: predicted multiplier ranges by group (geometry, appearance, densification) for trained models.
  - Training curve chart: only for MLP, showing train/validation loss by epoch when history is available.
  - Keep charts compact and operational; tables remain the source of exact values.
- Run Tests also needs a first-class overview page with test charts, data, and metrics. It should not reuse model-training charts except where a shared visual component is appropriate.
- Run Tests overview design plan:
  - Top summary cards: pipeline status, selected models, completed/failed test runs, default baseline count, predicted-run count, best model by report metric, elapsed time.
  - Model progress cards/table: one row/card per selected model with long-name handling, active status, completed projects, failed projects, total projects, and current model marker.
  - Result summary section: compare Default vs Ridge vs MLP per project and overall. Show whether each model improves or worsens quality relative to the default baseline.
  - Test-row table entry point: link or embedded compact preview of `Training Data Rows`; the exact values remain in the full table tab.
  - Export/action strip: keep `Export Current Test`, retry failed runs if still supported, and refresh.
- Chart plan for Run Tests overview:
  - Overall comparison chart: Default, Ridge, and MLP bars for aggregate score/score/quality metrics.
  - Per-project comparison chart: grouped bars or compact heatmap showing each project across Default, Ridge, and MLP.
  - Improvement chart: delta from default baseline per model and project, using green/red encoding for better/worse.
  - Completion chart: completed/failed/pending tests by model, matching the `1/5 done` model cards.
  - Multiplier group chart: predicted/tested multiplier ranges by geometry, appearance, and densification groups.
  - Failure/error summary chart: failed runs by model or project when failures exist.
  - Keep charts compact and paired with tables/tooltips because exact thesis/report values must remain inspectable.
- Backend contract needed for Run Tests charts:
  - Return structured per-project/per-model comparison rows with default baseline metrics and predicted-model metrics.
  - Return per-model progress and failure counts directly.
  - Return aggregate metric summaries for Default, Ridge, and MLP.
  - Return multiplier group summaries for selected/tested multipliers.
- Individual model-training detail pages should show training-specific charts and metrics:
  - Training/validation loss curves for MLP.
  - Lambda/CV selection chart for Ridge when available.
  - Feature/group coefficient or importance summaries.
  - Dataset lineage and coverage used by that model.
  - Artifact metadata, model config, and logs.
- Items still to port/check from the previous combined pipeline UI:
  - Testing detail needs the old pre-run model preview flow: model selector, `Test Model` action, per-project prediction rows, signal/no-signal status, multiplier preview, and feature-value modal.
  - Testing detail likely still needs `Export Current Test` and retry-failed-run actions if they remain part of the active workflow.
  - Model-training detail should bring forward old model metrics cards where relevant: runs used, score mean, selected lambda, score scale, theta norms, file metadata, and raw model JSON/debug view.

### Phase 5: Split Large Files

- Split `ProcessTab.tsx` into a few focused components and hooks.
- Split `PipelineDetailsPage.tsx` into focused components.
- Keep component count moderate.

### Phase 5A: Project-Level Restructuring Plan

Project-level UI should mirror the pipeline-level structure but keep names and responsibilities distinct:

- Pipeline pages answer: how a multi-project workflow is configured, running, producing datasets/models/tests.
- Project pages answer: what happened inside one reconstruction project/session, what data and outputs exist, and how to run or inspect processing for that project.

Naming target:

- Project page files/components should use `Project...` names, not `Pipeline...` names.
- Reusable neutral components can be shared only when they are truly generic:
  - Good shared names: `Breadcrumbs`, `StatusBadge`, `ProgressSummary`, `RunLogPanel`, `MetricCard`, `ActionToolbar`.
  - Avoid reusing pipeline-named components inside project pages.
- New project detail folder target:
  - `src/pages/projectDetails/ProjectDetailPage.tsx`
  - `src/components/projectDetails/ProjectDetailShell.tsx`
  - `src/components/projectDetails/ProjectOverviewPanel.tsx`
  - `src/components/projectDetails/ProjectImagesPanel.tsx`
  - `src/components/projectDetails/ProjectProcessingPanel.tsx`
  - `src/components/projectDetails/ProjectSessionsPanel.tsx`
  - `src/components/projectDetails/ProjectModelsPanel.tsx`
  - `src/components/projectDetails/ProjectComparisonPanel.tsx`
  - `src/components/projectDetails/ProjectLogsPanel.tsx`
  - `src/components/projectDetails/ProjectSettingsPanel.tsx`

Current project surfaces:

- `ProjectDetail.tsx` is the active tabbed project page used by `/project/:id`.
- `Project.tsx` is an older status page style and appears unused by `App.tsx`; keep until the replacement project detail is complete, then ask before deleting.
- `Projects.tsx` and `Dashboard.tsx` both contain project-list management behavior; later decide whether one should become canonical.
- `CreateProject.tsx` is active and should keep storage-root selection.
- `ProcessTab.tsx` is the largest project-level file and should be split carefully. It contains runtime processing controls, configuration, viewer/preview, telemetry/downloads, resume/stop, and LiteGS frontend controls.

Project detail page plan:

1. Project Overview
   - Same visual language as workflow/pipeline pages: blue header, back link, breadcrumbs, status chip, refresh.
   - Summary cards: status, progress, current stage, current step/max steps, output availability, sessions, created/elapsed time.
   - Live preview panel when preview exists.
   - Latest activity/log summary.
   - Metrics summary after completion: loss, PSNR/SSIM/LPIPS when available, Gaussian count, runtime, export status.
   - Project overview charts:
     - Training loss over steps when telemetry exists.
     - PSNR/SSIM/LPIPS over eval checkpoints when available.
     - Gaussian count over steps when available.
     - Stage duration/progress breakdown.
     - Keep exact metrics in tables/tooltips; charts are for scanability.

2. Images
   - Reuse current `ImagesTab` behavior first.
   - Later split upload/list/preview pieces if the file is large.
   - Preserve image upload and existing project image management.

3. Processing
   - Replace monolithic `ProcessTab.tsx` with focused parts:
     - `ProjectProcessControls`
     - `ProjectProcessConfig`
     - `ProjectTrainingConfig`
     - `ProjectDensificationConfig`
     - `ProjectEarlyStopConfig`
     - `ProjectRunSelector`
     - `ProjectPreviewViewer`
     - `ProjectTelemetryDownloads`
   - Keep stop/resume/restart/start controls project-level, not pipeline-level.
   - Preserve logging and telemetry downloads.
   - Keep configuration frontend-driven where currently available.

4. Sessions
   - Keep existing session list/selection behavior.
   - Make session cards compact and navigable.
   - Link sessions to metrics, outputs, logs, and comparisons.

5. Models
   - Project level should not train models.
   - Keep a lightweight `Models` tab only as a read-only reference to available trained models from the Train Models workflow.
   - The tab can show model name/id, type, source training-data output, created time, status, and a link/button to the Train Models page or specific model detail page for changes.
   - Any model creation, retraining, deletion, or model configuration changes should happen in the model-training workflow, not inside a project.

6. Comparison
   - Do not bury comparison inside project detail as a major workflow.
   - Comparison is large enough to become a top-level workspace beside Projects and Research Workflow.
   - Project detail can show a compact comparison preview or link to related comparisons, but the full comparison flow should live at top level.
   - Top-level comparison should handle project-to-project, run-to-run, baseline-vs-optimized, and model-test comparison views where applicable.
   - This is distinct from Run Tests pipeline result charts, which summarize one testing pipeline.

7. Logs
   - Project logs should remain available.
   - Rename old project-level `AI Learning Table` wording where it appears in logs to offline/prepared/training-data language when the logs tab is refactored.

8. Settings
   - Do not create a separate settings workflow inside the new project detail page.
   - Pipeline setup remains pipeline-owned.
   - If a project was created from or linked to a pipeline, show a link back to that pipeline instead of duplicating pipeline settings in the project page.

Project-level training/config wording:

- No model training should happen at project level.
- Any old project/process config wording that says `train` but only contributes data for offline model learning should be renamed to `Update Training Data` or `Update Offline Training Data`.
- During testing, configuration may still run a reconstruction/training process for a project, but that is not model training. UI wording should distinguish:
  - `Run Reconstruction`
  - `Update Training Data`
  - `Export Outputs`
  - not `Train Model`

Prepared training data storage/ownership:

- Keep two related but distinct data concepts:
  - Pipeline run data stays inside the Prepare Offline Data pipeline. It is the audit trail: runs, selected multipliers, scores/scores, EXIF rows, logs, per-run training rows, and pipeline-specific metadata. The pipeline detail `Training Data Rows` tab shows this.
  - Prepared training datasets live outside individual pipeline folders as reusable artifacts. They are the product consumed by Train Models: cleaned/exported rows, stable dataset id, version, source pipeline id, source project/run ids, feature schema, multiplier schema, row count, timestamps, and file path.
- A Prepare Offline Data pipeline should generate/update a reusable prepared dataset artifact from its internal pipeline run data.
- The reusable dataset must link back to the source pipeline for traceability.
- Project-level actions may update/add rows to prepared training data when appropriate, but they should not own model training.
- Model-training pipelines should consume selected prepared-data artifacts from the shared area.
- UI target:
  - Prepare Offline Data pipeline detail shows pipeline run data and exposes `Generate/Update Prepared Dataset`.
  - Workflow can show a prepared-data section/list for reusable datasets.
  - Train Models selects from prepared-data artifacts, not raw pipeline folders.
- Backend later needs a clean prepared-data registry or metadata index so UI can list datasets independently of pipelines.

Migration/cleanup policy:

- While restructuring, create new focused files/components first and keep old files working.
- Do not delete old project or pipeline files until the replacement flow covers the required behavior and the deletion list is reviewed.
- Remove old files altogether near the end after explicit approval.

Project restructuring implementation checkpoint:

- Added a non-breaking preview route `/project-v2/:id` for the new project detail structure.
- Kept the active `/project/:id` old project page untouched.
- Added new project files:
  - `components/projectDetails/ProjectDetailShell.tsx`
  - `components/projectDetails/ProjectOverviewPanel.tsx`
  - `components/projectDetails/ProjectModelsReferencePanel.tsx`
  - `components/projectDetails/ProjectProcessingPanel.tsx`
  - `pages/projectDetails/ProjectDetailPage.tsx`
- The new project Models tab is read-only and links to Train Models for changes.
- The new Processing tab has a first focused implementation with gsplat-only project reconstruction controls: run reconstruction, stop, resume, restart, basic reconstruction settings, and basic COLMAP settings.
- The new Processing tab does not include LiteGS and does not use model-training wording. The old processing tab remains available through the old project view until the replacement covers all required advanced behavior.
- Added a `New View` link from the old `/project/:id` header to `/project-v2/:id` for side-by-side comparison during migration.
- Added session/run selection to the new project processing panel using `/projects/:id/runs`. The panel can target an existing session or start a named new session.
- Added a `New view` action to project cards in the main Projects page so the new structure can be tested from the normal UI.
- Removed the Settings tab from the new project detail page. Pipeline settings should remain on the pipeline workflow; linked projects show a source-pipeline link when `pipeline_id` is available.
- Extended the new project Processing tab with frontend-configurable image resize, COLMAP matching/camera settings, sparse reconstruction source selection, manual sparse merge selection/building, and merge report viewing. These preserve important old-page behavior without adding LiteGS back into the new component.
- Added a new `Outputs` tab to the `/project-v2/:id` project detail route:
  - Loads project-level or selected-run files from `/projects/:id/files`.
  - Shows gsplat final/best model downloads, PNG previews, model snapshots, sparse `points.bin`/`sparse.json` downloads, and compact run telemetry.
  - Supports run selection and telemetry JSON export from `/projects/:id/telemetry`.
  - Does not surface LiteGS outputs in the new frontend path.
- Replaced the new project Overview placeholder chart section with real lightweight telemetry charts:
  - Loss vs step from `/projects/:id/telemetry`.
  - PSNR vs eval step from eval telemetry.
  - Gaussian count vs eval step from eval telemetry.
  - The overview selects the current run when available, otherwise the latest/completed run.

Correction after UI review:

- The project detail restructuring should preserve the existing tab UI exactly for tabs that remain.
- The active new route `/project-v2/:id` has been changed back to the existing project-page visual structure and existing tab components.
- Active tabs now keep the original order and behavior for the kept tabs:
  - Images
  - Process
  - Logs
  - Sessions
  - Models
- `Overview` is not needed because the existing `Process` tab already serves as the project overview/control surface.
- `Outputs` should not be a separate active tab unless explicitly requested; output/preview/telemetry behavior belongs inside the existing Process tab UI during restructuring.
- `Comparison` and `Settings` should be removed from project-level navigation because comparison is planned as top-level and pipeline settings remain pipeline-owned.
- The previously added redesigned project components are inactive and should be reviewed before deletion:
  - `ProjectDetailShell.tsx`
  - `ProjectOverviewPanel.tsx`
  - `ProjectOutputsPanel.tsx`
  - `ProjectProcessingPanel.tsx`
  - `ProjectModelsReferencePanel.tsx` is now active as the read-only project Models tab and should not be deleted.
- Exact-UI project refactor checkpoint:
  - Added `ProjectPageHeader.tsx` by extracting the existing project header markup/classes unchanged.
  - Added `ProjectTabsNav.tsx` by extracting the existing tab bar markup/classes unchanged.
  - Updated `/project-v2/:id` to use those extracted components while keeping existing tab order and existing tab implementations for Images, Process, Logs, Sessions, and Models.
  - Build passes after the extraction.
- Project Logs cleanup checkpoint:
  - `LogsTab.tsx` now shows processing logs only.
  - Removed project-level AI logs, AI log charts, AI learning table view, and AI log PDF/JSON download controls from the Logs tab.
  - Kept regular processing log refresh, auto-scroll, and text download.
  - Reduced log action button sizing to compact `text-xs` controls.
- Project test-results correction:
  - The project-level learning/result table should not live inside Logs.
  - Restored it as a main project tab named `Test Results`.
  - `Test Results` uses `/projects/:id/ai-learning-table` for now, but UI wording treats it as per-project testing results because project-level work selects a model and tests/configures reconstruction rather than training a model.
  - Project Logs remains processing logs only.
- Project breadcrumbs:
  - Added workflow-style breadcrumbs to the project header.
  - Project pages opened from workflow project panels carry `from=workflow-projects`.
  - Project pages opened from a pipeline carry `returnToPipeline=:id` and breadcrumb/back target points to `/workflow/pipelines/:id`.
- Main project cards now open `/project-v2/:id` so the restructured project page is the default visible project page.
- Project Models tab checkpoint:
  - Project-level Models is read-only.
  - It shows available models from the model registry and links users to the Train Models workflow for model changes.
  - No model training should happen at project level.
- Project Process config checkpoint:
  - Active project Process config now keeps the engine fixed to `gsplat` in the frontend.
  - LiteGS is removed from frontend options, state, persisted training config, and process payloads.
  - Early-stop controls are removed from the active project Process config and active process payloads.
  - Added `Update training data` checkbox in Config > Training.
  - When enabled, it sends `update_training_data: true` and `training_data_target_id`.
  - The dropdown currently offers the current project test-results target and existing workflow pipelines whose stage/type resolves to `training_data`.
  - Backend refactor should consume these fields by writing or queuing project test results into the selected prepared-training-data target.
  - Remaining early-stop mentions in old comparison/report/upload surfaces are legacy/report labels, not active project Process config; review them before deletion or cleanup.
- Workflow-level training data checkpoint:
  - The Research Workflow page has four first-class stage cards in one row on wide screens:
    - `Prepare Offline Data`
    - `Training Data`
    - `Train Models`
    - `Run Tests`
  - `Training Data` is a separate page at `/training-data`.
  - The Training Data page lets users select a data preparation pipeline target before inspecting or building prepared training data.
  - Training Data page control order is target selection first, then Build Dataset and Refresh.
  - The data-preparation pipeline detail `Prepared Dataset` tab also requires/selects the training-data target before build.
  - Frontend build requests now include `training_data_target_pipeline_id` for the selected target.
  - Current backend still writes the generated offline dataset into the selected pipeline folder. During backend restructuring, move generated training data into a workflow-level/shared training-data store and use the selected target id as the registry key.
  - Model-training setup must consume the workflow-level/shared Training Data registry, not raw Prepare Offline Data pipelines.
  - The current model-training builder now selects from Training Data targets as the frontend placeholder until backend dataset registry APIs are finalized.
  - Create Model Training Pipeline now includes an actual Train Model modal:
    - model type selector: Featurewise Ridge Regression or Featurewise MLP
    - model name input
    - Ridge auto-select lambda option
    - optional manual Ridge lambda input
    - train action posts to `/training-pipeline/:trainingDataTargetId/train-model`
  - Train Models list now includes models returned by each Training Data target's `/trained-models` endpoint, in addition to globally registered models.
  - Train Models `Create New` no longer navigates away from the page.
  - `Create New` opens an in-page modal where Training Data is selected first, then model inputs are shown.
  - After successful training, the modal shows success stats and can be closed.
  - The trained model list supports search and sorting by newest, oldest, and name.
  - Newly trained models are loaded into the list and appear first under the default newest sort.
  - Model training modal now validates the selected Training Data before showing model settings:
    - calls `/training-pipeline/:id/offline-dataset`
    - requires at least one dataset row
    - shows a warning and disables Train Model when the dataset is missing or empty
  - Backend follow-up: the future Training Data registry/list endpoint should expose dataset validity (`row_count`, `schema_valid`, `feature_schema`, `last_built_at`, `status`) so invalid/empty datasets can be hidden or clearly disabled before selection.

Home and navigation checkpoint:

- Home page `/` is now a high-level entry page rather than the project list.
- The hero/banner and project status counts remain on the home page.
- Home now also shows pipeline counts for data, train, and test pipelines.
- Removed the top hero action buttons from home:
  - Refresh
  - Research Workflow
  - New Project
- Home guides users into three large path cards:
  - `Projects` -> `/projects`
  - `Comparison` -> `/comparison`
  - `Research Workflow` -> `/workflow`
- Added `/projects` as the dedicated projects page.
- `/projects` reuses the existing banner, counts, project search/sort, and project list behavior from the old homepage.
- Added `/comparison` as a top-level comparison hub page.
- Existing `/comparison/:id` remains the legacy comparison detail/status page.
- `/comparison` now reuses the cleaned existing run-comparison implementation from the old project Comparison tab:
  - select left/right projects
  - select left/right completed runs
  - compare metrics, AI settings, parameter values, charts, preview snapshots
  - export comparison PDF
- Pipeline detail actions now include Retry Failed when `failed_runs > 0`.
- Pipeline worker logs now collect both project-level `processing.log` and run-level `runs/:run_id/processing.log` files.

Remaining frontend items to review:

- Comparison is now surfaced as a top-level page using the old project comparison implementation. Remaining comparison work:
  - connect test-pipeline comparison outputs more directly if needed
  - decide whether to remove the legacy project-level Comparison tab with the old `/project/:id` page
- Marked for later frontend/backend legacy cleanup pass:
  - legacy `/project/:id` route and old `ProjectDetail.tsx`, including old Comparison and Settings tabs
  - old `/model-training/new` route and `ModelTrainingBuilderPage.tsx` placeholder
  - old upload comparison mode in `Upload.tsx`
  - legacy comparison detail route `/comparison/:id` if no longer needed after comparison hub is fully accepted
- Deleted inactive generated project-detail files after approval:
  - `ProjectDetailShell.tsx`
  - `ProjectOverviewPanel.tsx`
  - `ProjectOutputsPanel.tsx`
  - `ProjectProcessingPanel.tsx`

Remaining overview-summary implementation plan:

- Model training pipeline overview should summarize trained models, not project reconstruction runs.
- Add model-training overview cards:
  - total trained models
  - Ridge model count
  - MLP model count
  - latest trained model
  - latest training row/sample count
  - best validation/loss metric where available
  - selected/auto lambda for Ridge
- Add model-training overview charts/tables:
  - models trained over time
  - training samples per model
  - Ridge lambda search / selected lambda summary
  - MLP train/validation loss if available in model config
  - compact latest-model table with model name, type, rows used, score mean, lambda/best val loss, trained time
- Testing pipeline overview should summarize model test performance and comparison outputs.
- Add testing overview cards:
  - total test projects
  - models tested
  - completed model/project tests
  - failed tests
  - best model by score/quality score when available
  - average PSNR/SSIM/LPIPS deltas vs baseline/default when available
- Add testing overview charts/tables:
  - per-model completion status (`done/total`)
  - model ranking table by score/quality and metric deltas
  - project/model result heatmap-style table
  - PSNR/SSIM/LPIPS/loss delta charts by model
  - failed project/model combinations with error text
- Reuse existing data sources first:
  - pipeline `trained_models`
  - testing learning/result rows
  - prediction results
  - export-current-test bundle metadata where useful later
- Keep chart implementation lightweight in frontend for now; backend refactor should later expose normalized summary endpoints for model-training and testing pipelines.

Current overview/chart implementation checkpoint:

- Prepare Offline Data detail overview now shows:
  - fixed pre-generated log-space multiplier schedule from pipeline config
  - current multiplier index, last picked value, and next value per group
  - relative score distribution by project for geometry, appearance, and densification log-space multipliers
- Run Tests detail overview now shows:
  - model/test prediction overview from the latest saved prediction preview
  - candidate score surface table/chart by multiplier group when `candidate_score_checks` exists
  - observed test score distribution by project for geometry, appearance, and densification log-space multipliers
  - fixed log-space schedule panel when the pipeline config exposes one
- Train Models detail overview now summarizes existing trained model metadata:
  - total outputs
  - Ridge count
  - MLP count
  - latest model
  - rows/samples used
  - Ridge selected lambda and lambda-search count
  - MLP best validation loss, final train loss, and best model step/epoch when available

Backend follow-up required for these overview panels:

- Expose a normalized `model_training_summary` for Train Models pipelines.
- For MLP models, persist and return the best model step/epoch explicitly:
  - `best_model_step` or `best_epoch`
  - `best_val_loss`
  - `final_train_loss`
  - `epochs_trained`
  - `training_samples`
  - `total_parameters`
  - source Training Data id
- For Ridge models, persist and return:
  - selected lambda
  - lambda search candidates with validation error
  - training row count
  - score mean
  - objective/score key
  - model artifact path
- Model-training outputs should no longer remain as pipeline-local-only model records.
  - Every successfully trained Ridge or MLP model must be registered as a shared workflow model in the same model registry used by `/projects/models`.
  - `/model-training/models/:id` should load from that shared registry only.
  - The current frontend may list models from `/training-pipeline/:id/trained-models` as a temporary visibility bridge, but those records will show "Model artifact was not found" on the detail page until backend registration is implemented.
  - Backend should save model artifact path, metadata path, source Training Data id, source pipeline id, trained_at, model type, and model metrics into the shared registry at training completion.
  - No separate "elevate model" action should be required after model training.
- Expose the fixed log-space schedule as first-class pipeline state, not only an opaque config value:
  - random seed used to generate values
  - restart version/token
  - generated values by multiplier group
  - current index
  - last picked index/value
  - next index/value
  - timestamp/config hash for the run/restart that fixed the values
- Stop/resume must never regenerate the fixed values. Restart may regenerate only when config, project list, or requested run count changes; this must update the seed/version metadata visibly.
- Test pipelines should expose notebook-style candidate exploration data in a stable shape:
  - project id/name
  - model id/name
  - candidate group
  - each candidate log multiplier and real multiplier
  - predicted score/relative score
  - selected candidate flag
  - highest candidate point per group
  - total candidate count, commonly 30 for the random/log-space exploration graph
- Relative score/score graph terminology should prefer `relative score` in UI labels, but backend can keep `score` internally until the scoring module is renamed.

LiteGS frontend removal plan:

- Remove LiteGS from frontend now, backend later.
- In `ProcessTab.tsx`:
  - Change `TrainingEngine` from `"gsplat" | "litegs"` to `"gsplat"`.
  - Remove LiteGS option from the engine select.
  - Remove `litegs_target_primitives` and `litegs_alpha_shrink` state, defaults, info text, UI controls, reset/load paths, storage payload fields, and process payload fields.
  - Remove any LiteGS-only conditional panels.
  - Default engine should be `gsplat` and remain configurable only if another supported frontend engine exists later.
- Keep backend-compatible behavior during frontend removal by simply no longer sending LiteGS-specific fields from UI.
- Search after removal should return no frontend references to `litegs`, `LiteGS`, or `litegs_*`.

### Phase 6: Cleanup Generated Files

- Delete approved temporary/root debug files.
- Review docs and experiment artifacts.
- Keep Windows code.
- Keep dot-prefixed files/folders.

### Phase 7: Verification

- Backend:
  - run featurewise Ridge tests
  - run featurewise MLP tests
  - run relative quality score tests
  - run API smoke tests for offline preparation/model training/testing

- Frontend:
  - TypeScript build
  - lint if available
  - verify workflow pages load

- Manual smoke:
  - prepare tiny offline dataset
  - train Ridge/MLP models
  - launch a test pipeline config without running a long reconstruction

### Frontend-to-New-Backend Connection Checkpoint

Status after the latest wiring pass:

- Do not delete old frontend or backend files yet.
  - User wants the new frontend code connected to the new backend APIs first.
  - User will test the connected flow.
  - Legacy frontend/backend code deletion happens later in one approved pass.
- New workflow surfaces now call new `/api/...` endpoints:
  - workflow pipeline lists and detail:
    - `/api/workflow/pipelines`
    - `/api/workflow/pipelines/{pipeline_id}`
  - pipeline actions:
    - `/api/workflow/pipelines/{pipeline_id}/start`
    - `/api/workflow/pipelines/{pipeline_id}/pause`
    - `/api/workflow/pipelines/{pipeline_id}/resume`
    - `/api/workflow/pipelines/{pipeline_id}/stop`
    - `/api/workflow/pipelines/{pipeline_id}/restart`
    - `/api/workflow/pipelines/{pipeline_id}/retry-failed`
  - worker logs:
    - `/api/workflow/pipelines/{pipeline_id}/worker-logs`
  - learning rows / relative-score table:
    - `/api/workflow/pipelines/{pipeline_id}/learning-rows`
  - EXIF test tools:
    - `/api/workflow/pipelines/{pipeline_id}/test-exif/...`
  - current test export:
    - `/api/workflow/pipelines/{pipeline_id}/export-current-test`
  - pipeline create/edit/scan:
    - `/api/workflow/pipelines`
    - `/api/workflow/pipelines/{pipeline_id}/config`
    - `/api/workflow/pipelines/scan-directory`
- Training Data page/panel now uses reusable Training Data artifacts:
  - Selects source offline data preparation pipelines from `/api/workflow/pipelines?stage=offline_data_preparation`.
  - Finds existing final Training Data artifacts with `/api/workflow/training-data/by-source-pipeline/{pipeline_id}`.
  - Creates final Training Data artifacts with `POST /api/workflow/training-data` when needed.
  - Builds final Training Data from a selected offline data pipeline with `/api/workflow/training-data/{training_data_id}/build-from-pipeline/{pipeline_id}`.
  - Reads rows with `/api/workflow/training-data/{training_data_id}/rows`.
- Train Models page now uses the new report-aligned model training API:
  - Lists valid Training Data choices from `/api/workflow/model-training/training-data-sources`.
  - Validates a selected Training Data artifact with `/api/workflow/training-data/{training_data_id}/validity`.
  - Hides model options until the selected Training Data is usable for model training.
  - Trains Ridge/MLP with `POST /api/workflow/model-training/train`.
  - Lists registered trained models from `/api/models`.
  - Model detail page opens a single model from `/api/models/{model_id}`.
- Project model reference panel now reads shared trained models from `/api/models` and remains read-only.
- Dashboard/home pipeline counts now read from `/api/workflow/pipelines?limit=100` and split counts by `workflow_stage`.
- Live progress/log visibility fix:
  - Pipeline overview live panel now polls the active project status every 3 seconds while the pipeline is running.
  - The live panel now supports `config.projects` entries that are plain project-id strings as well as project objects.
  - The live panel remains visible with a "Waiting for live project status..." message while the first status response is loading.
  - Pipeline logs panel now auto-refreshes every 5 seconds while open.
- Resume live-progress fix:
  - Training/testing orchestrator now writes `active_run` into the pipeline record when a run starts and clears it when the run finishes.
  - `active_run` contains project name, run id, phase, pass, run number, test model id, status, and start time.
  - The workflow pipeline detail API now returns `active_run`.
  - The overview panel uses `active_run` as a fallback when `/projects/{project_id}/status` still says `pending` after resume.
  - The overview panel now shows phase/pass from `active_run` while the worker status catches up.
  - Activity Logs now include the active in-progress run above completed runs, using the live `active_run` status while polling.
  - Activity Logs also keep the old overview behavior of showing a generic "Running: Phase..., Run..., Project..." row while the pipeline is running.
  - Activity Logs keep old per-run multiplier chips for completed runs when `group_multipliers` are available.
  - Live Stage Status now matches the old detail page behavior:
    - shows message text
    - shows training step progress from `current_step/currentStep` and `max_steps/maxSteps`
    - shows loss and PSNR when present
    - shows non-training stage progress bars from `stage_progress`
  - Project status is explicitly set to `processing` with `current_run_id` at pipeline worker handoff.
  - Workflow pipeline detail page now polls every 3 seconds while the pipeline is running/paused/stopping so resumed run state appears without a manual reload.
- Cooldown and stale terminal-state fix:
  - Start/resume/restart now clear stale `completed_at`, `last_error`, `active_run`, `cooldown_active`, `next_run_scheduled_at`, and active test model state before switching back to running.
  - Restart clears the same fields both in the reset payload and in the automatic running payload.
  - Start/resume/restart skip only stale initial cooldown waiting by clearing cooldown state before the orchestrator starts.
  - Normal cooldown still applies after the upcoming run completes, before the following run.
  - Do not skip the next run itself and do not skip normal between-run cooldown.
  - Cooldown ownership is now tied to `orchestrator_session_id`/`cooldown_session_id`.
  - Old paused/stopped cooldown threads cannot keep or clear cooldown state after a new resume/restart session starts.
  - Cooldown sleep is interruptible and checks that the sleeping orchestrator still owns the active pipeline session.
- Log tab refresh behavior:
  - Worker log accordions are now controlled by stable open state.
  - Auto-refresh preserves which accordion was open.
  - Auto-refresh is silent, so log content stays visible while new log text is fetched.
- Frontend TypeScript no-emit check passed after these changes:
  - command: `npx tsc --noEmit`
- Backend focused test note:
  - `python -m pytest ...` with system Python failed because pytest is not installed.
  - `.venv\Scripts\python.exe -m pytest bimba3d_backend/tests/test_workflow_pipelines_api.py -q` could not collect because the venv is missing `httpx`, which FastAPI/Starlette TestClient requires.
  - Syntax check passed with `.venv\Scripts\python.exe -m py_compile bimba3d_backend/app/services/training_pipeline_orchestrator.py bimba3d_backend/app/services/workflow_pipeline_service.py`.

Remaining frontend calls to old endpoints are intentionally not deleted yet:

- `ModelTrainingBuilderPage.tsx`
  - old separate builder route; new Train Models page uses an in-page modal instead.
  - Mark as legacy candidate after user confirms the new modal works.
- `PipelinesListPage.tsx` and `PipelineDetailsPage.tsx`
  - old pipeline list/detail pages still use `/training-pipeline/...`.
  - Mark as legacy candidates after user confirms the new workflow pipeline detail covers the needed UI.
- `ModelRegistryTab.tsx` and parts of `ProcessTab.tsx`
  - old project-level model/config surfaces still call `/projects/models` and `/training-pipeline/list`.
  - Keep for now unless they appear in the new required project flow.
  - They should be reviewed during the old-code deletion pass so no useful project details are lost.

Legacy separation checkpoint:

- Old code is now marked clearly rather than silently mixed with new UI:
  - `bimba3d_frontend/src/pages/ModelTrainingBuilderPage.tsx`
  - `bimba3d_frontend/src/pages/PipelinesListPage.tsx`
  - `bimba3d_frontend/src/pages/PipelineDetailsPage.tsx`
  - `bimba3d_frontend/src/pages/ProjectDetail.tsx`
  - `bimba3d_frontend/src/components/tabs/ModelRegistryTab.tsx`
- Active routes now avoid old pages:
  - `/model-training/new` redirects to `/model-training`.
  - `/pipelines/:id` renders the new workflow pipeline detail page.
  - `/project/:id` redirects to `/project-v2/:id`.
  - Create Project now navigates to `/project-v2/{project_id}`.
- Backend `bimba3d_backend/app/api/training_pipeline.py` is marked as legacy bridge, not unused:
  - New workflow APIs are the active API surface.
  - This legacy module remains temporarily because restart/retry/export/EXIF behavior still bridges through proven functions there.
- Deletion still waits for user confirmation after testing.

Legacy backup and active-code cleanup checkpoint:

- Moved inactive frontend legacy files out of `bimba3d_frontend/src` into `legacy_reference/frontend`:
  - `bimba3d_frontend/src/pages/ModelTrainingBuilderPage.tsx`
  - `bimba3d_frontend/src/pages/PipelinesListPage.tsx`
  - `bimba3d_frontend/src/pages/PipelineDetailsPage.tsx`
  - `bimba3d_frontend/src/pages/ProjectDetail.tsx`
  - `bimba3d_frontend/src/pages/Projects.tsx`
  - `bimba3d_frontend/src/pages/Upload.tsx`
  - `bimba3d_frontend/src/components/tabs/ModelRegistryTab.tsx`
- Active pipeline builder was renamed:
  - from `TrainingPipelinePage.tsx`
  - to `WorkflowPipelineBuilderPage.tsx`
  - route is now `/workflow/pipeline-builder`
- Active frontend no longer references:
  - `/training-pipeline`
  - `/training-pipeline/list`
  - `/projects/models`
  - legacy pipeline detail/list/builder pages
  - legacy project detail page
  - legacy project model registry tab
- `TrainingDataRowsTable` now reads learning rows from:
  - `/api/workflow/pipelines/{pipeline_id}/learning-rows`
- `ProcessTab` model list now reads from:
  - `/api/models`
- `ProcessTab` Training Data target list now reads from:
  - `/api/workflow/pipelines?stage=offline_data_preparation&limit=100`
- Backend public legacy route unmounted:
  - `main.py` no longer includes `/training-pipeline`
  - `main.py` no longer includes legacy `/training-pipeline` streaming route
  - EXIF streaming remains mounted only under `/api/workflow/pipelines`
- Backend legacy backup created:
  - `legacy_reference/backend/bimba3d_backend/app/api/training_pipeline.py`
- Active backend still keeps `bimba3d_backend/app/api/training_pipeline.py` as an internal bridge only.
  - It is not mounted as public API.
  - It is still imported by `workflow_pipeline_service.py` for restart/retry/predict/export until those functions are extracted.
  - Do not move this active bridge file until restart/retry/predict/export are implemented in new service modules.

Active workflow edit-config checkpoint:

- New Prepare Offline Data detail page header has `Edit Config`.
- New Run Tests detail page header has `Edit Config`.
- Both link to the existing shared pipeline editor:
  - `/workflow/pipeline-builder?edit={pipeline_id}`
- The editor already loads through `/api/workflow/pipelines/{pipeline_id}` and saves through `/api/workflow/pipelines/{pipeline_id}/config`.
- After saving, `/pipelines/{pipeline_id}` now resolves to the new workflow detail page, so the old detail page is bypassed.

Backend/frontend follow-up before deletion:

- User should manually test:
  - open workflow
  - open Prepare Offline Data
  - open Training Data
  - build final Training Data from an offline data pipeline
  - open Train Models
  - select a valid Training Data artifact
  - train Ridge and MLP
  - confirm latest model appears at the top
  - open model detail without "Model artifact was not found"
  - open Run Tests and export current test
- After user confirms, remove old routes/components in one planned pass.

Active code split checkpoint:

- Frontend `ProcessTab.tsx` has started being broken down without changing UI behavior:
  - New `bimba3d_frontend/src/components/projectProcess/processTypes.ts` holds project processing types.
  - New `bimba3d_frontend/src/components/projectProcess/processUtils.ts` holds pure formatting/default/config helpers.
  - New `bimba3d_frontend/src/components/projectProcess/InfoIcon.tsx` holds the shared small info icon wrapper.
  - New `bimba3d_frontend/src/components/projectProcess/ProcessToasts.tsx` holds process config/info toast rendering.
  - `ProcessTab.tsx` reduced from 6423 lines to 5953 lines.
  - `npx tsc --noEmit` passes after the split.
- Backend `projects.py` has started being broken down without changing endpoint behavior:
  - New `bimba3d_backend/app/services/project_paths.py` resolves regular and workflow-owned project directories.
  - New `bimba3d_backend/app/services/project_json.py` provides shared JSON read/write helpers.
  - New `bimba3d_backend/app/services/project_shared_config.py` owns project shared image/COLMAP config helpers.
  - `projects.py` now imports those helpers and no longer owns duplicate definitions for them.
  - `projects.py` reduced to 8362 lines after the first backend splits.
  - `python -m py_compile bimba3d_backend/app/api/projects.py bimba3d_backend/app/services/project_shared_config.py bimba3d_backend/app/services/project_paths.py bimba3d_backend/app/services/project_json.py` passes.
- Remaining active split targets:
  - Continue extracting `ProcessTab.tsx` into a small number of project-process components, preserving the current UI exactly.
  - Extract `projects.py` route groups and analytics/log helpers into focused modules.
  - Extract active bridge functions from `training_pipeline.py` into workflow/testing services before moving that bridge file out of active backend code.
  - Later split worker `entrypoint.py` and `gsplat_engine.py` carefully, keeping gsplat baseline behavior unchanged.

Cooldown behavior checkpoint:

- A fresh pipeline orchestrator session now skips exactly one initial cooldown.
- This applies to first start, resume, and restart because each creates/starts a new orchestrator session.
- The skip does not remove thermal management globally:
  - skipped/completed historical runs do not consume the skip;
  - the first actually executed run consumes the skip;
  - later non-skipped runs in the same session use the configured cooldown normally.
- Resume/restart/start also clear stale `cooldown_active`, `next_run_scheduled_at`, and `cooldown_session_id` before the worker loop begins.

Pipeline stop/pass terminology checkpoint:

- Overview activity log must display terminal status by `pipeline.status`, not by the presence of `completed_at`.
  - `completed_at` is an end timestamp and is also set for stopped/failed runs.
  - Frontend now shows `Pipeline stopped`, `Pipeline failed`, `Pipeline completed with failures`, or `Pipeline completed` as appropriate.
- UI should no longer explain the schedule as `phase + pass`.
  - Thesis/report language should be `phase` and `runs/exploration runs`.
  - Decision: remove `pass` as a frontend/backend concept in new code.
  - Implemented fresh-start cleanup:
    - new pipeline configs use `exploration_runs_per_project`;
    - pipeline progress uses `current_run`;
    - run history uses `run`;
    - old `passes`, `current_pass`, `pass_number`, and run `pass` fallbacks were removed from active workflow storage/service/orchestrator and new frontend workflow pages.
  - Existing old pipeline JSON is intentionally not supported now because the user is starting fresh.

Fresh-start cleanup checkpoint:

- Restart and retry-failed workflow actions no longer import the old `training_pipeline.py` API module.
  - New service:
    - `bimba3d_backend/app/services/workflow_pipeline_actions.py`
  - Restart now speaks only the active workflow schema:
    - `exploration_runs_per_project`
    - `current_run`
    - no `current_pass`
    - retry slot keys are `project|phase|run`
  - Restart still keeps the operational Windows-safe cleanup behavior:
    - stop active orchestrator
    - stop active local workers
    - retry locked-file deletion
    - preserve images, resized images, and sparse/COLMAP outputs
  - Retry-failed now stores fixed retry params without any pass-level key.
- Test prediction preview no longer imports the old `training_pipeline.py` API module.
  - New service:
    - `bimba3d_backend/app/services/workflow_prediction_preview.py`
  - Prediction preview is now workflow-owned and still writes:
    - `prediction_previews`
    - `prediction_preview_artifacts`
    - latest preview key
    - per-model and per-project intermediate check files
- Frontend project route cleanup:
  - Removed the `/project/:id` compatibility redirect.
  - Removed the `/project-v2/:id` route.
  - Active project detail route is now `/projects/:id`.
  - Updated create-project, dashboard, and workflow project links to use `/projects/:id`.
- Workflow EXIF route naming cleanup:
  - Renamed `bimba3d_backend/app/api/training_pipeline_streaming.py` to `bimba3d_backend/app/api/workflow_exif.py`.
  - Routes remain mounted under `/api/workflow/pipelines/...`, so the frontend EXIF tab keeps working.
  - This was a naming cleanup only; EXIF behavior was not changed.
- LiteGS active-code removal is done for frontend/backend active code.
  - Current targeted scans show no active LiteGS references in the workflow/frontend/service files checked.
- Remaining old API bridge:
  - `workflow_pipeline_service.export_current_test()` still imports `export_current_test_bundle` from `bimba3d_backend/app/api/training_pipeline.py`.
  - This is now the last direct active import from the old training-pipeline API bridge in the workflow service.
  - Next cleanup step: extract export-current-test into a workflow export service, then move/remove active `bimba3d_backend/app/api/training_pipeline.py`.
- Verification after this checkpoint:
  - `python -m py_compile bimba3d_backend/app/services/workflow_pipeline_actions.py bimba3d_backend/app/services/workflow_prediction_preview.py bimba3d_backend/app/services/workflow_pipeline_service.py`
  - `npx tsc --noEmit`

Fresh-start cleanup checkpoint 2:

- The old workflow export bridge was removed from active backend code.
  - New workflow-owned service:
    - `bimba3d_backend/app/services/workflow_test_export.py`
  - `workflow_pipeline_service.export_current_test()` now calls the workflow export service directly.
  - Active `bimba3d_backend/app/api/training_pipeline.py` was moved to `legacy_reference/backend_removed_api/...`.
- Project-level model elevation is removed from active project UI/API.
  - Removed the project Process tab "Elevate" button and modal.
  - Removed `POST /projects/{project_id}/runs/{run_id}/elevate-model`.
  - Removed the unused `ElevateModelRequest` schema.
  - Removed the unused `model_registry.elevate_learner_model()` helper.
  - Projects still list/select workflow-trained reusable models through the active model registry.
- Report-aligned model naming cleanup:
  - Active featurewise model files now use:
    - `featurewise_ridge_regression_runtime.py`
    - `featurewise_mlp_runtime.py`
    - `relative_quality_score_runtime.py`
  - Old single-file Ridge model fallbacks (`quality_model.json`, `convergence_model.json`) were removed from active Ridge loading.
  - Featurewise Ridge and MLP active selectors now use report-family strategy names:
    - `featurewise_ridge_regression`
    - `featurewise_mlp`
- Phase/run terminology cleanup:
  - AI input-mode snapshots now write `run` instead of `pass`.
  - Active scans show no old workflow `passes`, `current_pass`, `pass_num`, or `pass_number` terms in active source.
- Root and inactive-file cleanup moved old generated/debug/context artifacts to `legacy_reference`, including:
  - root debug scripts and generated reports,
  - inactive frontend pages/tabs,
  - old contextual-continuous tests,
  - old docs/generated notes,
  - root helper scripts.
- Remaining cleanup candidates before final legacy deletion:
  - `projects.py` still contains analytics backfill paths for old/incomplete runs. Since the project is now fresh-start, these can be removed in a focused comparison/test-results pass after confirming the new analytics writer always emits canonical summaries.
  - `ProcessTab.tsx` still contains top-level output handling for project files returned outside per-engine bundles. This is active behavior, not old route compatibility, but can be simplified once the backend response contract is finalized.
  - Operational fallbacks that should not be removed without a replacement:
    - EXIF/XMP extraction fallbacks,
    - COLMAP option/path handling,
    - Windows process cleanup,
    - React Suspense fallback UI,
    - CUDA/Torch compatibility documentation.
- Verification after this checkpoint:
  - `python -m py_compile` passes for edited backend modules checked in this pass.
  - `npx tsc --noEmit` passes after the frontend project-elevation removal.

Fresh-start cleanup checkpoint 3:

- Removed project analytics backfill behavior from active `projects.py`.
  - `run_analytics_v1.json` is now read as the canonical source.
  - The API no longer synthesizes missing analytics from `processing.log`, `stats/`, `eval_history.json`, or `comparison/experiment_summary.json`.
  - Project test-result rows skip runs that do not have canonical analytics.
  - Run comparison summaries return a clear missing-artifact error if canonical analytics is absent.
- Removed the old online adaptive controller path.
  - `bimba3d_backend/worker/ai_adaptive_light.py` is no longer imported by `gsplat_engine.py`.
  - Removed controller setup and runtime decision application from active gsplat execution.
  - Moved the old controller file, its tests, training/evaluation scripts, and generated `_adaptive_ai_global` dataset to `legacy_reference/backend_ai_adaptive/...`.
  - Project run listing now reports `ai_event_count` from current logs instead of `adaptive_event_count`.
  - Removed old `ai_*` adaptive-controller fields from the frontend telemetry export.
- Current active model logic is report-aligned through Featurewise Ridge/MLP modules; the old adaptive controller is not part of the thesis workflow.

Fresh-start cleanup checkpoint 4:

- Active model/training choices are now limited to the two thesis quality model families:
  - `featurewise_ridge_regression`
  - `featurewise_mlp`
- Removed remaining active convergence-model and old objective-mode plumbing.
  - Active scans show no `convergence_model`, `ridge_score_optimizer`, `featurewise_score_mlp`, `neural_featurewise`, `offline_objective`, or `ai_prediction_objective` references in active backend/frontend source.
  - Test/model prediction now uses one feature source, `exif_compact_featurewise`, and chooses Ridge vs MLP through `ai_selector_strategy`.
  - The model-training script now writes only `quality_model.json` and no longer uses an objective loop.
  - MLP training now exposes no objective argument; it trains against `score_quality`.
- Old convergence model artifacts and old offline-model datasets were moved under `legacy_reference/convergence_and_old_offline_models/...`.
- `score_convergence`, `r_convergence`, and `convergence_speed` remain only as scoring/summary metrics for charts and comparison context. They are not separate trainable model families or selectable training/testing modes.
- Verification after this checkpoint:
  - `python -m py_compile` passes for edited backend modules and scripts.
  - `npx tsc --noEmit` passes for the frontend.

Fresh-start cleanup checkpoint 5:

- Replaced active old objective terminology with score terminology.
  - Training Data and model training now use `relative_quality_score`.
  - Convergence display term is `convergence_score`.
  - Combined run/model score term is `relative_score` or `score`, depending on payload scope.
  - Candidate preview surfaces now use `candidate_score_checks` and `predicted_score`.
  - Pipeline aggregates now use `mean_relative_score` and `best_relative_score`.
- Removed remaining duplicate sidecar persistence for pipeline scoring data.
  - `input_mode_learning_results.json` is no longer written by gsplat.
  - Pipeline learning rows, project learning table, model checkpoint selection, and orchestrator score collection now read from canonical `analytics/run_analytics_v1.json`.
- Removed duplicate prediction preview artifacts.
  - Test prediction previews are stored in the pipeline record used by UI/export.
  - The extra `_test_prediction_checks` artifact files are no longer written.
- Added row-level duplicate protection.
  - Pipeline learning rows are de-duplicated by project/run.
  - Reusable Training Data build rows are de-duplicated by project/run/source pipeline before writing `rows.json`.
- Intentional remaining separate artifacts:
  - Per-run canonical analytics under each run folder.
  - Reusable Training Data `rows.json`, which is a normalized artifact generated from selected pipeline rows for model training.
  - Model artifacts/manifests generated from Training Data.
- Verification after this checkpoint:
  - Active-source scan has no old objective terminology, `input_mode_learning_results.json`, or `_test_prediction_checks` hits.
  - `python -m py_compile` passes for edited backend modules and scripts.
  - `npx tsc --noEmit` passes for the frontend.

Fresh-start cleanup checkpoint 6:

- Score reference step is now config-driven.
  - Relative quality score calculations no longer assume a fixed 5000-step comparison.
  - `compute_relative_quality_summary()` and the runtime score helper accept `score_reference_step`.
  - If no explicit `score_reference_step`/`comparison_step` is supplied, the run uses the configured `max_steps` as the score reference step.
  - Baseline comparison payloads now write dynamic loss keys such as `loss_at_7000_run` plus `score_reference_step`; normalized API/table rows expose `loss_at_reference_step_run` and `loss_at_reference_step_base`.
- New workflow pipelines default `max_steps` to 7000 in the builder, but the value remains frontend-configurable and saved in pipeline shared config.
- Test/offline pipeline execution no longer forces baseline phase 1 to 5000 steps.
  - `training_pipeline_orchestrator.py` now requires `max_steps` in the run config and passes that exact value to gsplat.
  - Missing `max_steps` raises a clear configuration error instead of silently falling back to 5000.
- gsplat save/eval fallback steps no longer use old fixed-step defaults.
  - When explicit save/eval intervals are absent, the fallback step is the configured `max_steps`.
- Canonical Training Data and export rows use normalized reference-step fields.
  - `pipeline_learning_rows.py`, `training_data_builder.py`, project test-result rows, and test export all read/write `score_reference_step` and `loss_at_reference_step_*`.
- Verification after this checkpoint:
  - Active-source scan has no old fixed-reference-step field names or fixed-step default hits.
  - Active-source scan has no old objective terminology hits.
  - `python -m py_compile` passes for edited backend modules and scripts.
  - Focused pytest passes for relative quality score, pipeline learning rows, and training pipeline orchestrator tests.
  - `npx tsc --noEmit` passes for the frontend.

Project testing / Training Data update checkpoint:

- Project Process config now treats project-level AI Guided Run as testing only.
  - Project page does not train Ridge/MLP models directly.
  - The model dropdown selects a workflow-trained model and uses that model's AI profile.
  - The project can mark selected results for Training Data update; model training happens later from the workflow Training Data page.
- Do not mix base-session selection with the run-type dropdown.
  - The dropdown label remains `Baseline Run` for the internal `mode="baseline"` path.
  - The first project run should continue to become base automatically through existing backend behavior.
  - Later base changes should remain handled through the existing project/session base workflow, not through the Training config modal.
- Backend follow-up for `update_training_data`:
  - When project testing has `update_training_data=true`, save only the selected/configured model evaluation point into the target Training Data artifact.
  - The saved row should include model evaluation metadata, including:
    - source workflow model id and name,
    - model family / AI profile,
    - run id and project id,
    - model evaluation point / checkpoint step,
    - score reference step,
    - selected multiplier groups and actual parameter mapping,
    - relative quality score and supporting metrics available at that evaluation point.
  - Do not export all checkpoints or all eval rows into Training Data from a project test run.

Project testing / Training Data backend implementation checkpoint:

- Done:
  - Workflow Ridge/MLP models can now be seeded into project-local selector folders for project AI Guided Run tests.
  - Workflow test pipelines use the same workflow model seeding/profile path, so selected Ridge/MLP models control the test selector strategy.
  - Project AI Guided Run starts gsplat from scratch for geometry training; the workflow model is used only for multiplier prediction.
  - `update_training_data=true` appends one validated completed project test row into the selected reusable Training Data artifact.
  - The appended row stores project/run source, workflow model id/name/family, source Training Data id, AI profile, model evaluation step, score reference step, features, selected multipliers, selected log multipliers, and relative quality score.
  - Appending refuses missing/empty/invalid Training Data targets and refuses runs without usable features, multipliers, or relative quality score.
  - `tune_end_step` follows configured `max_steps` when no explicit tuning end is supplied.
- Verification:
  - Backend compile check passed for the edited schema, services, project API, pipeline orchestrator, and gsplat engine.
  - Active frontend/backend source scan has no old objective naming hits.
  - Score calculation uses `score_reference_step` or `comparison_step` when supplied; otherwise it uses configured `max_steps`.

Model evaluation step checkpoint:

- Done:
  - Workflow model manifests now store `model_evaluation_step`.
  - Model training derives `model_evaluation_step` from Training Data row `score_reference_step`.
  - Model training refuses mixed or missing Training Data reference steps, so a single model cannot silently mix 7000-step and 30000-step targets.
  - Project AI Guided Run tests set `score_reference_step` from the selected workflow model when `update_training_data=true`, not from the project-level `max_steps`.
- Intended behavior:
  - Offline data pipelines and workflow test pipelines keep using their own configured max/last evaluation step for scoring.
  - If a model was trained from 7000-step Training Data and a project test is configured for 30000 max steps, the run may still continue to 30000.
  - Only when the project page has `Update training data` enabled, the saved Training Data contribution uses the selected model's evaluation point.
  - Project tests without Training Data update and workflow test pipelines should not be forced to the model evaluation point.

Pipeline timing and cooldown checkpoint:

- Pipeline overview Time cards now show estimated remaining time under elapsed time.
  - Estimate is based on elapsed pipeline time and completed run count.
  - Before the first completed run, remaining time shows as calculating.
- Cooldown behavior:
  - Start/resume/restart clears stale cooldown state before work begins.
  - Cooldown is no longer skipped after the first completed run.
  - Normal thermal cooldown applies after completed runs when thermal management is enabled.

Project Training Data update visibility checkpoint:

- After a project test successfully appends a row to Training Data, the backend writes `.training_data_update_latest.json` in the project folder.
- Project Test Results reads this marker through `/projects/{project_id}/ai-learning-table`.
- The Test Results tab shows a persistent banner with:
  - saved Training Data name/id,
  - model evaluation / score reference step,
  - source model name when available,
  - run id,
  - update timestamp.

Fixed multiplier application checkpoint:

- Found issue in completed run `chapel_of_the_holy_trinity_phase2_run1_20260627_141725`:
  - pipeline metadata recorded non-default group multipliers,
  - run analytics `initial_params` and `learning_param_rows.actual` showed default gsplat values,
  - therefore that run did not actually apply the fixed schedule values to gsplat.
- Root cause:
  - fixed schedule multipliers could be recorded for run-jitter-only exploration without being applied robustly to the gsplat parameter dict before `Config`/`DefaultStrategy` creation.
- Fix:
  - fixed group multiplier application now supports geometry, appearance, and densification values directly,
  - `lambda_dssim` is included in the appearance group,
  - `densification_multiplier` and `scale_lr_multiplier` are both recognized for densification,
  - fixed-schedule branch applies values if they have not already been applied,
  - `Actual Value` remains the exact value captured from gsplat `Config`/`DefaultStrategy`; it is not reconstructed for display.
- Existing affected runs should be rerun/restarted for valid offline data because stored analytics show default parameters were used.
- Multiplier group consistency correction:
  - Report grouping is the source of truth:
    - Geometry: `position_lr_init`, `position_lr_final`, `scaling_lr`, `rotation_lr`.
    - Appearance: `feature_lr`, `opacity_lr`, `lambda_dssim`.
    - Densification: `densify_grad_threshold`, `opacity_threshold`.
  - Runtime fixed/random multiplier application, backend learning-row builders, and frontend prepared-dataset grouping were corrected so `rotation_lr` uses the geometry multiplier, not the densification multiplier.
  - Existing runs created before this correction should be rerun before being used as trusted offline training data.

Relative quality score correction checkpoint:

- Pipeline learning-row scoring now follows the report formula with pipeline-level normalization:
  - completed pipeline rows are normalized together when the Training Data Rows tab/API is loaded,
  - `Q_quality = (PSNR_norm + SSIM_norm + (1 - LPIPS_norm)) / 3`,
  - `R_quality = Q_quality_run - Q_quality_baseline`.
- Removed the accidental pairwise min/max scoring behavior that could collapse small run-vs-baseline differences into hard `-1` or `+1`.
- Removed the temporary baseline-as-`1.0` ratio scoring idea; baseline quality is now its own `Q_quality` value.
- The shared helper for this is `bimba3d_backend/app/services/relative_quality_scoring.py`.
- The live Training Data Rows view recalculates scores from currently completed pipeline rows on fetch.
- Build Dataset freezes the currently recalculated row values into the Training Data artifact.
- Normalization is applied across all completed rows in the same pipeline together, including baseline and exploration runs. Baselines and exploration runs must not be normalized separately because `R_quality = Q_run - Q_baseline` requires both values to be on the same scale.
- Live `s_best`, `s_end`, and `s_run` values now use the normalized `Q_quality` scale (`0..1`) instead of raw runtime audit scores.
- The Chatteau run `chatteau_phase2_run1_20260627_145214` currently recomputes from source analytics as:
  - `Q_run = 0.3542417991037887`
  - `Q_baseline = 0.43360316382930497`
  - `R_quality = -0.07936136472551625`
- Existing saved Training Data artifacts that already contain the old `-1.0` value should be rebuilt from pipeline rows to refresh stored values.

Fixed log-space schedule checkpoint:

- Fixed schedule generation now uses the non-baseline exploration phase run count, not the Phase 1 baseline count.
- Offline data preparation Phase 2 reuses one fixed multiplier slot across all projects for the same exploration run number:
  - Phase 2 run 1 uses schedule index 0.
  - Phase 2 run 2 uses schedule index 1.
  - The schedule cursor must not advance per project.
- Stop/resume preserves generated schedule values. Restart regenerates the schedule and records seed/version/timestamp metadata.
- Existing `pipeline_748726953004` was repaired after being created with the old one-slot schedule. The current saved schedule now has random Phase 2 values from the configured group bounds, including index 0.
- Follow-up correction:
  - Phase 2 index 0 should also be a random log-space multiplier; `1.0` is only the baseline/default reference and should not be forced into exploration.
  - Frontend workflow builder now exposes editable group bounds:
    - geometry min/max,
    - appearance min/max,
    - densification min/max.
  - Default uniform bounds are:
    - geometry `[0.5, 2.0]`,
    - appearance `[0.5, 2.0]`,
    - densification `[0.7, 1.42]`.
  - Backend schedule generation reads these saved bounds from `shared_config`.
  - The mode dropdown is a preset helper; the saved numeric bounds are the source of truth.
  - Test pipeline model-scoring candidates are generated as deterministic balanced log-space values:
    - saved in `test_candidate_log_multipliers`,
    - generated from the same group-specific bounds,
    - default `test_candidate_count` is 30 points per group,
    - Ridge and MLP score/select over those saved candidates instead of an ad hoc evenly spaced grid when the schedule exists.
  - Offline data preparation does not use 30 candidate points. Its fixed random log-space schedule length is the configured number of Phase 2 exploration runs.
  - Pipeline detail UI displays the exploration seed, candidate seed, candidate count, and generation timestamp.
  - Model training now stores source log-multiplier bounds in the workflow model config.
  - Test pipeline candidate generation prefers bounds from the selected workflow model. If a model has no stored bounds metadata, the pipeline records `fixed_log_space_bounds_source=default_bounds_fallback`, and the UI shows a fallback warning.

Fixed schedule preview/save workflow:

- Fixed log-space multiplier schedules now support a temporary frontend preview flow:
  - `Regenerate Preview` calls the backend generator and stores the returned values only in React state.
  - Refreshing the page discards preview values.
  - `Save Preview for Processing` is the only action that writes the preview into the pipeline config.
- Backend save validation is strict:
  - saving is rejected while the pipeline is running,
  - saving is rejected if any non-baseline/phase-2+ run folder already exists,
  - baseline-only pipelines may still update the schedule because baseline does not consume exploration multipliers.
- This protects thesis reproducibility: once exploration/test multipliers may have been consumed, the fixed schedule is locked.

Pipeline retry/logging/storage correction:

- Root cause for mixed step logs after retry:
  - retry/start could create a new orchestrator while an old in-memory orchestrator or local worker was still alive,
  - local worker file handlers were attached to the root logger and not detached per run, so later run output could be written into previous project/run log files.
- Fix:
  - orchestrator start now stops an existing live orchestrator for the same pipeline before starting another one,
  - registered local worker processes are killed when replacing an existing active orchestrator,
  - local run log handlers are scoped to one `run_full_pipeline()` call and detached/closed in `finally`,
  - worker-log API now returns stable `id`, `project_name`, `run_id`, and `log_path`, and only falls back to project root logs when no per-run logs exist.
- C-drive storage correction:
  - backend startup now defaults `BIMBA3D_TEMP_DIR`, `TEMP`, `TMP`, `TMPDIR`, `XDG_CACHE_HOME`, `TORCH_HOME`, `MPLCONFIGDIR`, and `HF_HOME` to `E:/Thesis/Temp` unless explicitly overridden,
  - worker entrypoint mirrors the ML cache env vars so future worker caches do not grow under `C:/Users/ROG`.
- Existing old logs may already contain duplicated/mixed lines from the previous root-logger handler leak; new runs should produce isolated per-run logs after backend restart.
- Follow-up display fix:
  - `get_worker_logs()` now sorts the active pipeline run first and then newest run logs first.
  - `_iter_processing_logs()` reads `runs/:run_id/processing.log` by newest modification time instead of alphabetical run id.
  - This prevents a retry such as `Gaurisankar` from showing an older failed/static run accordion before the currently active retry log.
  - Worker-log display now filters run folders through current pipeline state: active run plus run ids still recorded in `pipeline.runs`.
  - Retry removes failed run records from the pipeline JSON, so superseded failed retry folders are hidden from the active log UI even if their old folders remain on disk for reference.
  - Retry cleanup now removes the failed run folder from disk after capturing retry parameters, so repeated retry attempts do not accumulate duplicate run folders.
  - Existing orphan retry folders in `E:\Thesis\PipelineProjects\Final_offline_data_June_27` were removed for `pipeline_55776ebd244e`.

Pipeline state storage checkpoint:
- Heavy pipeline outputs already live under the user-selected pipeline folder.
- New/updated pipeline state now writes the full state to `{pipeline_folder}/pipeline_state.json`.
- The backend registry under `bimba3d_backend/data/projects/training_pipelines` is now only a small discoverability pointer for new/updated pipelines.
- Existing running pipelines are not force-migrated mid-run; after backend restart, the next pipeline update writes the state beside the selected pipeline folder.
- Existing pipeline state files were migrated:
  - `pipeline_55776ebd244e` -> `E:\Thesis\PipelineProjects\Final_offline_data_June_27\pipeline_state.json`
  - `pipeline_5df0ad97ef5a` -> `E:\Thesis\PipelineTests\Final_test_June_27\pipeline_state.json`
  - `pipeline_748726953004` -> `E:\Thesis\PipelineProjects\offline-data-pipeline-delete\pipeline_state.json`
- The default `bimba3d_backend/data/projects/training_pipelines/*.json` files now remain as small registry pointers with `registry_only=true` and `pipeline_state_path`.
- The registry pointer files are expected to remain visible in the default location; they should stay tiny and contain no heavy run data.

Frontend chart checkpoint:
- Fixed Log-Space Multiplier Schedule charts now show multiple log-spaced x-axis ticks plus `1.0` when it falls inside the configured bounds.
- Relative Score Distribution charts now use the same richer real/log multiplier tick labels and have per-group fullscreen controls.
- Metric Distribution By Run adapts crowded project labels by skipping/rotating labels and uses larger charts in fullscreen mode.
- During a running baseline phase, Fixed Log-Space Multiplier Schedule shows `Baseline` / `1.000000` and does not highlight schedule index `0`; exploration indices only apply to phase 2+ runs.

Hard-cap / failed-run checkpoint:
- Gaurisankar baseline retry was confirmed to be baseline/default: phase 1, no selected multipliers, and retry-fixed params equal gsplat default hyperparameters.
- Gaussian hard-cap at step ~4400 is caused by `num_gaussians` exceeding the configured cap, not by phase-2 log-space multipliers.
- Gaurisankar overgrowth investigation:
  - new run used `max_steps=7000` and `densify_until_iter=5000`,
  - old baseline used `max_steps=5000` and `densify_until_iter=4000`,
  - upstream gsplat uses `cfg.max_steps` as the position/means LR scheduler horizon (`gamma=0.01 ** (1.0 / max_steps)`).
- Decision after checking upstream:
  - do not decouple the position LR scheduler from `max_steps`,
  - do not expose `position_lr_max_steps` in the workflow UI,
  - keep local `gsplat_upstream/simple_trainer.py` behavior aligned with upstream gsplat,
  - current `Final_offline_data_June_27` pipeline state was restored to `max_steps=7000`, `densify_until_iter=5000`, and no `position_lr_max_steps`.
- Training Data Rows now filters to successful pipeline run ids only, so failed hard-cap runs do not become training rows.
- Older stale duplicate Gaurisankar run folder `gaurisankar_phase1_run1_20260629_071824` was removed from the pipeline folder.
- Active hard-cap payload naming was updated from old reward terminology to relative-quality-score terminology where new results are written.

Temporary legacy COLMAP source hook:

- Workflow pipeline setup now asks `/projects?include_legacy_colmap_sources=true` only for the "Copy COLMAP from existing project" dropdown.
- This intentionally does not restore old legacy pipelines to the main workflow pipeline list.
- Backend project lookup temporarily scans legacy pipeline roots so selected old source projects can be resolved when copying `outputs/sparse`.
- Specific old pipeline folders needed for current data reuse:
  - `E:/Thesis/PipelineProjects/Training_June_2`
  - `E:/Thesis/PipelineTests/Test_2026-05-31`
- Remove this query flag scanner and the extra legacy-root resolver after the old sparse outputs have been copied into fresh pipelines and verified.

Multiplier application checkpoint:

- Upcoming offline-data runs should no longer clamp actual gsplat hyperparameters to the old absolute safety bounds after applying the generated multiplier schedule.
- Source of truth for exploration bounds is the data-preparation pipeline config/saved fixed log-space schedule:
  - geometry -> position, scale, and rotation learning rates,
  - appearance -> feature learning rate, opacity learning rate, and DSSIM lambda,
  - densification -> gradient and opacity thresholds.
- `learning_param_rows.final_multiplier` is the multiplier actually applied to gsplat and is now the training-data target for offline model training.
- `learning_param_rows.selected_multiplier` remains optional/empty for offline data preparation; it is primarily for model-predicted test/project runs.
- Training Data build now derives group-level `selected_multipliers` from `learning_param_rows.final_multiplier` when top-level selected multipliers are absent.
- Ridge and MLP model training now receive the source Training Data pipeline bounds, save those bounds in the model artifact/checkpoint, and scan predictions within those saved bounds.
- Existing completed rows that were previously clamped are not rewritten; rerun affected projects if those rows are needed for final training data.

Resume vs retry checkpoint:

- Plain Resume must continue normal pipeline progress and skip slots already marked failed.
- Failed slots are retried only through the explicit Retry Failed action.
- Retry Failed sets `config.retry_mode_active=true` only when it auto-starts the retry execution.
- Plain Resume clears `config.retry_fixed_params` and sets `config.retry_mode_active=false`.
- The orchestrator now skips previously failed slots unless the exact slot has retry-fixed params and retry mode is active.
- When all prepared retry slots are consumed successfully, retry mode is turned off.

Planned clamped-run retry marking:

- Identify old offline-data phase 2+ runs where `learning_param_rows.final_multiplier` differs from `learning_param_rows.log_multiplier` for any parameter.
- These are old double-bounded/clamped runs and should be rerun with the unclamped multiplier code before final Training Data is built.
- Do not delete old run folders or analytics during detection.
- Update pipeline run records only:
  - mark affected run records as `failed`,
  - add a clear reason such as `final_multiplier differs from generated log_multiplier; rerun required after clamp removal`.
- Retry Failed should then pick those runs up together with ordinary failed runs.
- If duplicate failed records exist for the same project/phase/run slot, retry cleanup should keep only the retry target slot semantics and avoid accumulating duplicate run folders.
- 2026-07-08 checkpoint for `pipeline_55776ebd244e` / `Final_offline_data_June_27`:
  - scanned current pipeline run analytics and training-data JSON artifacts for active run ids,
  - initial raw analytics scan found `0` current successful/partial run slots where top-level `final_multiplier != log_multiplier`,
  - follow-up scan of the exact Training Data Rows table source (`learning_param_rows.log_multiplier` vs `learning_param_rows.final_multiplier`) found `160` non-baseline old clamped rows,
  - those `160` run records were marked `failed` with reason `applied_multiplier_differed_from_log_space_multiplier`,
  - backup: `E:\Thesis\PipelineProjects\Final_offline_data_June_27\pipeline_state.before_double_clamp_failed_20260708_165256.json`,
  - resulting slot counters: `completed_runs=546`, `failed_runs=165`, `hard_cap_runs=25`, `pending_runs=0`.

Hard-cap retry classification checkpoint:

- Hard-cap runs are not normal retry failures.
- Use `status="hard_cap_reached"` with `reason="gaussian_hard_cap_reached"` for hard-cap attempts so Retry Failed ignores them.
- Retry Failed should target `status="failed"` records by default and should deduplicate by project/phase/run slot.
- The Retry Failed UI now exposes an explicit checkbox to include hard-cap runs; backend uses `include_hard_cap=true` only when selected.
- Gaussian hard cap is configurable from the workflow pipeline builder shared run configuration as `shared_config.gaussian_hard_cap`; default is `6,000,000`.
- Retry runs use the current saved pipeline config, so increasing `gaussian_hard_cap` before retrying hard-cap slots raises the cap for those retried runs.
- Retry mode now tracks active retry queue slots separately from original pipeline pending slots:
  - `config.retry_target_slots` records explicit retry slots for new retry attempts,
  - `pending_runs` / UI `Left` displays remaining retry queue slots while retry mode is active,
  - older in-flight retry sessions fall back to `config.retry_fixed_params` count for display.
- The orchestrator now skips `hard_cap_reached` slots unless those slots are explicitly present in the retry queue.
- Frontend counters:
  - `Failed` = retryable failed slots,
  - `Hard Cap` = non-retryable hard-cap slots,
  - `Left` / `pending_runs` = slots with no completed, failed, or hard-cap outcome yet.
- 2026-07-08 one-time state update for `pipeline_55776ebd244e`:
  - backup: `E:\Thesis\PipelineProjects\Final_offline_data_June_27\pipeline_state.before_hardcap_reclass_20260708_164151.json`,
  - reclassified `32` hard-cap failed attempt records,
  - resulting slot counters: `completed_runs=706`, `failed_runs=5`, `hard_cap_runs=25`, `pending_runs=0`.
- After marking double-clamped rows failed, current `pipeline_55776ebd244e` retry selection is:
  - `165` slots when hard caps are excluded,
  - `190` slots when hard caps are explicitly included.

Live overview telemetry checkpoint:

- Pipeline overview live progress now publishes and displays compact run metrics while gsplat is active:
  - log-step loss,
  - best loss with step,
  - latest eval PSNR, SSIM, and LPIPS with eval step.
- The overview shows two compact rows when data is available:
  - Current Run from the active worker status,
  - Baseline from the same project's stored baseline analytics.
- This is display telemetry only; it must not change relative-quality-score calculation, normalization, Training Data Rows eligibility, or dataset build behavior.

Compact featurewise model checkpoint:

- Added parallel compact model families beside the existing non-compact featurewise Ridge/MLP paths:
  - `compact_featurewise_ridge_regression`
  - `compact_featurewise_mlp`
- Compact models train one shared model over the selected compact descriptors instead of separate geometry, appearance, and densification model heads.
- Existing non-compact featurewise code remains active and is marked with `NON_COMPACT_FEATUREWISE` where practical so it can be found later if compact models become the main path.
- Train Models frontend selection includes all four model families: non-compact Ridge, non-compact MLP, compact Ridge, and compact MLP.
- Test pipeline model selection, overview filters, predictions, training-data rows, and resume-after-config-change behavior are intended to derive from configured model ids rather than hard-coded model counts.
- When a completed test pipeline config gains new selected models, saving config should move the pipeline back to resumable/stopped state while keeping old completed model runs intact.
- New notebook replicas exist for all four model families:
  - `notebooks/three_model_featurewise_ridge_quality_scoring.ipynb`
  - `notebooks/three_model_featurewise_mlp_quality_scoring.ipynb`
  - `notebooks/featurewise_ridge_quality_scoring.ipynb`
  - `notebooks/featurewise_mlp_quality_scoring.ipynb`
- Notebook training cells now include an explicit `%pip install tqdm` setup cell and use tqdm progress bars:
  - Ridge notebooks show lambda-search progress and per-lambda validation MSE details.
  - MLP notebooks show epoch progress with train loss, validation loss, best loss, and patience state.
- Platform Train Models modal currently shows synchronous request lifecycle logs plus backend-derived final training details. True live backend logs in the web app still require async training jobs with polling, SSE, or WebSocket.
