export interface SnapshotEntry {
  name: string;
  url: string;
  step: number | null;
  size?: number;
  format?: string;
}

export interface PreviewFile {
  name: string;
  url: string;
}

export interface EngineOutputBundle {
  name: string;
  label: string;
  hasModel: boolean;
  finalModelUrl: string | null;
  bestModelUrl: string | null;
  previews: PreviewFile[];
  snapshots: SnapshotEntry[];
}

export interface TelemetryTrainingRow {
  timestamp?: string | null;
  step?: number | null;
  max_steps?: number | null;
  loss?: number | null;
  elapsed_seconds?: number | null;
  eta?: string | null;
  speed?: string | null;
  source?: string | null;
}

export interface TelemetryEvalRow {
  step?: number | null;
  psnr?: number | null;
  lpips?: number | null;
  ssim?: number | null;
  num_gaussians?: number | null;
}

export interface TelemetryEventRow {
  timestamp?: string | null;
  type?: string | null;
  step?: number | null;
  summary?: string | null;
}

export interface TelemetryPayload {
  project_id: string;
  project_name?: string | null;
  run_id?: string | null;
  generated_at?: string;
  training_rows?: TelemetryTrainingRow[];
  event_rows?: TelemetryEventRow[];
  eval_rows?: TelemetryEvalRow[];
  latest_eval?: TelemetryEvalRow | null;
  training_summary?: {
    first_step?: number | null;
    last_step?: number | null;
    start_timestamp?: string | null;
    end_timestamp?: string | null;
    total_elapsed_seconds?: number | null;
    row_count?: number | null;
    best_loss?: number | null;
    best_loss_step?: number | null;
  };
  run_config?: {
    requested_params?: Record<string, unknown>;
    resolved_params?: Record<string, unknown>;
    shared_config_version?: number | null;
    active_sparse_shared_version?: number | null;
    run_shared_config_version?: number | null;
    shared_outdated?: boolean | null;
    base_session_id?: string | null;
    effective_shared_config?: Record<string, unknown> | null;
  } | null;
  ai_insights?: {
    ai_input_mode?: string | null;
    baseline_session_id?: string | null;
    selected_preset?: string | null;
    heuristic_preset?: string | null;
    cache_used?: boolean | null;
    score?: number | null;
    score_positive?: boolean | null;
    score_label?: string | null;
    score_mode?: string | null;
    score_preset?: string | null;
    feature_source?: string | null;
    initial_params?: Record<string, unknown>;
    feature_details?: Record<string, unknown>;
    feature_sources?: Record<string, unknown>;
    missing_flags?: Record<string, unknown>;
    learn_snapshot?: Record<string, unknown>;
  } | null;
  status?: {
    stage?: string | null;
    message?: string | null;
    currentStep?: number | null;
    maxSteps?: number | null;
    current_loss?: number | null;
  };
}

export interface MergeReportSourceDetail {
  relative_path?: string;
  used?: boolean;
  reason?: string;
  points?: number;
  aligned?: boolean;
  overlap_images?: number;
  scale?: number;
}

export interface SparseMergeReport {
  anchor_relative_path?: string;
  selected_relative_paths?: string[];
  merged_points?: number;
  created_at?: number;
  alignment?: string;
  source_details?: MergeReportSourceDetail[];
}

export interface ProjectRunInfo {
  run_id: string;
  run_name?: string | null;
  saved_at?: string | null;
  mode?: string | null;
  stage?: string | null;
  engine?: string | null;
  session_status?: "completed" | "pending" | string;
  max_steps?: number | null;
  tune_scope?: string | null;
  trend_scope?: string | null;
  ai_event_count?: number;
  has_run_config?: boolean;
  has_run_log?: boolean;
  is_base?: boolean;
  shared_config_version?: number | null;
  active_sparse_shared_version?: number | null;
  shared_outdated?: boolean;
  batch_plan_id?: string | null;
  batch_index?: number | null;
  batch_total?: number | null;
}

export type NewSessionConfigSource = "current" | "defaults";
export type TrainingEngine = "gsplat";
export type TuneScope = "core_individual" | "core_only" | "core_ai_optimization" | "core_individual_plus_strategy";
export type TrendScope = "run" | "phase";
export type AiInputMode = "" | "exif_compact_featurewise";
export type AiSelectorStrategy =
  | "featurewise_ridge_regression"
  | "featurewise_mlp"
  | "compact_featurewise_ridge_regression"
  | "compact_featurewise_mlp";
export type RunJitterMode = "fixed" | "random";
export type TuneScopeDropdownValue =
  | TuneScope
  | "core_ai_optimization__exif_compact_featurewise";
export type StartModelMode = "scratch" | "reuse";

export interface ReusableModelEntry {
  model_id: string;
  model_name?: string | null;
  source_project_id?: string | null;
  source_run_id?: string | null;
  created_at?: string | null;
  ai_profile?: {
    pipeline_kind?: "controller" | "input_mode" | null;
    ai_input_mode?: AiInputMode | null;
    ai_selector_strategy?: AiSelectorStrategy | null;
  } | null;
}

export interface TrainingDataTargetOption {
  id: string;
  label: string;
  detail?: string;
}

export type FeatureStatus = "present" | "defaulted" | "unknown";

export interface TelemetryFeatureRow {
  key: string;
  value: unknown;
  status: FeatureStatus;
  source: string;
}

