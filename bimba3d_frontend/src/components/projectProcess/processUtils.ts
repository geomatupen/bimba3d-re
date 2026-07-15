import type {
  AiInputMode,
  AiSelectorStrategy,
  FeatureStatus,
  ProjectRunInfo,
  RunJitterMode,
  StartModelMode,
  TelemetryFeatureRow,
  TrainingEngine,
  TrendScope,
  TuneScope,
} from "./processTypes";

export const extractSnapshotStep = (name?: string): number | null => {
  if (!name) return null;
  const match = name.match(/(\d+)(?!.*\d)/);
  return match ? parseInt(match[1], 10) : null;
};

export const formatEngineLabel = (name: string) =>
  name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());

export const sanitizeRunToken = (value: string): string =>
  value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^[-_]+|[-_]+$/g, "")
    .slice(0, 80);

export const sanitizeFilenameToken = (value: string): string =>
  value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^[-_]+|[-_]+$/g, "")
    .slice(0, 80);

export const formatDurationCompact = (seconds?: number | null): string => {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return "-";
  const total = Math.floor(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
};

export const formatTelemetryScalar = (value: unknown): string => {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return value.toLocaleString();
    return value.toFixed(6).replace(/\.?0+$/, "");
  }
  if (typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
};

export const formatTelemetryFieldLabel = (key: string): string =>
  key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

export const LEARNABLE_AI_PARAM_KEYS = new Set([
  "feature_lr",
  "opacity_lr",
  "scaling_lr",
  "rotation_lr",
  "position_lr_init",
  "densify_grad_threshold",
  "opacity_threshold",
  "lambda_dssim",
]);

export const formatFeatureSourceLabel = (source: unknown): string => {
  const token = String(source || "").trim().toLowerCase();
  if (token === "original_metadata") return "Original metadata";
  if (token === "processed_dimensions") return "Processed dimensions";
  if (token === "processed_pixels") return "Processed pixels";
  return "Unknown";
};

export const AI_TRAINING_FEATURE_KEYS = new Set([
  "gsd_median",
  "overlap_proxy",
  "coverage_spread",
  "camera_angle_bucket",
  "heading_consistency",
  "vegetation_cover_percentage",
  "vegetation_complexity_score",
  "terrain_roughness_proxy",
  "texture_density",
  "blur_motion_risk",
]);

export const buildTelemetryFeatureRows = (
  features: Record<string, unknown> | null | undefined,
  missingFlags?: Record<string, unknown> | null,
  featureSources?: Record<string, unknown> | null,
): TelemetryFeatureRow[] => {
  if (!features || typeof features !== "object") {
    return [];
  }

  return Object.entries(features)
    .filter(([key]) => AI_TRAINING_FEATURE_KEYS.has(key))
    .map(([key, value]) => {
      void missingFlags;
      const status: FeatureStatus = "present";
      const source = formatFeatureSourceLabel(featureSources?.[key]);
      return { key, value, status, source };
    });
};

export const buildDefaultRunName = (
  projectLabel: string | null | undefined,
  projectId: string,
  runs: ProjectRunInfo[] = []
): string => {
  const base = sanitizeRunToken(projectLabel || projectId || "project") || "project";
  const matcher = new RegExp(`^${base}_session(\\d+)$`);
  let nextIdx = 1;

  runs.forEach((run) => {
    const candidate = String(run.run_id || run.run_name || "").trim().toLowerCase();
    const match = candidate.match(matcher);
    if (!match) return;
    nextIdx = Math.max(nextIdx, Number.parseInt(match[1], 10) + 1);
  });

  return `${base}_session${nextIdx}`;
};

export const COLMAP_CAMERA_MODELS = [
  "SIMPLE_PINHOLE",
  "PINHOLE",
  "SIMPLE_RADIAL",
  "RADIAL",
  "OPENCV",
  "OPENCV_FISHEYE",
  "FULL_OPENCV",
  "FOV",
  "SIMPLE_RADIAL_FISHEYE",
  "RADIAL_FISHEYE",
  "THIN_PRISM_FISHEYE",
] as const;

export const getDefaultProcessConfig = () => ({
  mode: "baseline" as "baseline" | "modified",
  tune_start_step: 100,
  tune_min_improvement: 0.005,
  tune_end_step: 15000,
  tune_interval: 100,
  tune_scope: "core_ai_optimization" as TuneScope,
  trend_scope: "run" as TrendScope,
  ai_input_mode: "exif_compact_featurewise" as AiInputMode,
  ai_selector_strategy: "featurewise_ridge_regression" as AiSelectorStrategy,
  baseline_session_id: "",
  warmup_at_start: false,
  run_count: 1,
  run_jitter_mode: "fixed" as RunJitterMode,
  run_jitter_factor: 1,
  run_jitter_min: 0.5,
  run_jitter_max: 1.5,
  continue_on_failure: true,
  start_model_mode: "scratch" as StartModelMode,
  project_model_name: "",
  source_model_id: "",
  engine: "gsplat" as TrainingEngine,
  maxSteps: 15000,
  logInterval: 100,
  splatInterval: 31000,
  bestSplatInterval: 100,
  bestSplatStartStep: 2000,
  saveBestSplat: false,
  pngInterval: 50,
  evalInterval: 1000,
  saveInterval: 31000,
  sparse_preference: "best",
  sparse_merge_selection: [] as string[],
  update_training_data: false,
  training_data_target_id: "",
  colmap: {
    max_image_size: 1600,
    peak_threshold: 0.02,
    guided_matching: true,
    camera_model: "OPENCV",
    single_camera: true,
    camera_params: "",
    matching_type: "sequential",
    mapper_num_threads: 4,
    mapper_min_num_matches: 12,
    mapper_abs_pose_min_num_inliers: 15,
    mapper_init_min_num_inliers: 60,
    sift_matching_min_num_inliers: 12,
    run_image_registrator: true,
  },
  images_max_size: 1600,
  images_resize_enabled: true,
  densifyFromIter: 500,
  densifyUntilIter: 10000,
  densificationInterval: 100,
  densifyGradThreshold: 0.0002,
  opacityThreshold: 0.005,
  lambdaDssim: 0.2,
});
