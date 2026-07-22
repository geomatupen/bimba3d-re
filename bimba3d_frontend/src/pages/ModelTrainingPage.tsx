import { useCallback, useEffect, useMemo, useState } from "react";
import { Brain, ChartNoAxesCombined, FileJson, Plus, RefreshCw, Search, Settings2, Upload, X } from "lucide-react";
import { api } from "../api/client";
import ModelSummaryList, { type WorkflowModel } from "../components/workflow/ModelSummaryList";
import WorkflowActionPanel from "../components/workflow/WorkflowActionPanel";
import WorkflowShell from "../components/workflow/WorkflowShell";

type ModelMode =
  | "featurewise_ridge_regression"
  | "featurewise_mlp"
  | "compact_featurewise_ridge_regression"
  | "compact_featurewise_mlp"
  | "compact_descriptor_mlp";
type SortBy = "time_desc" | "time_asc" | "name";
type TrainingLogLevel = "info" | "success" | "warning" | "error";
type UploadModelMode =
  | "compact_featurewise_ridge_regression"
  | "compact_featurewise_mlp"
  | "compact_descriptor_mlp";

interface TrainingLogEntry {
  timestamp: string;
  level: TrainingLogLevel;
  message: string;
}

const MODEL_MODE_OPTIONS: Array<{ value: ModelMode; label: string; description: string }> = [
  {
    value: "compact_featurewise_ridge_regression",
    label: "Compact Featurewise Ridge Regression",
    description: "One shared Ridge model for all multiplier groups.",
  },
  {
    value: "compact_featurewise_mlp",
    label: "Compact Featurewise MLP",
    description: "One shared MLP model for all multiplier groups.",
  },
  {
    value: "compact_descriptor_mlp",
    label: "Compact Descriptor MLP",
    description: "One shared MLP using only 10 descriptors + 3 log multipliers.",
  },
  {
    value: "featurewise_ridge_regression",
    label: "Featurewise Ridge Regression",
    description: "Legacy group-wise Ridge models for geometry, appearance, and densification.",
  },
  {
    value: "featurewise_mlp",
    label: "Featurewise MLP",
    description: "Legacy group-wise MLP heads for multiplier groups.",
  },
];

const isMlpModelFamily = (family?: string | null) =>
  family === "featurewise_mlp" || family === "compact_featurewise_mlp" || family === "compact_descriptor_mlp";

const isRidgeModelFamily = (family?: string | null) =>
  family === "featurewise_ridge_regression" ||
  family === "compact_featurewise_ridge_regression";

const isUploadMlpMode = (mode: UploadModelMode) => mode === "compact_featurewise_mlp" || mode === "compact_descriptor_mlp";

interface TrainingDataSource {
  training_data_id: string;
  name: string;
  status: string;
  source_pipeline_id: string;
  row_count: number;
  schema_valid: boolean;
  created_at: string;
  last_built_at?: string | null;
  errors?: string[];
}

const mapWorkflowModel = (model: any): WorkflowModel => ({
  ...model,
  artifact_format: model.artifact_format || model.model_family || "model",
  created_at: model.created_at || model.trained_at || null,
  ai_profile: model.ai_profile || {
    ai_input_mode: model.model_family || null,
  },
  provenance_summary: model.provenance_summary || {
    unique_project_count: model.training_samples,
  },
});

const apiErrorMessage = (err: any, fallback: string) => {
  const detail = err?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (detail?.message && detail?.details) return `${detail.message} ${detail.details}`;
  if (detail?.message) return detail.message;
  if (detail?.details) return detail.details;
  return fallback;
};

const formatElapsed = (seconds: number) => {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return mins > 0 ? `${mins}m ${secs.toString().padStart(2, "0")}s` : `${secs}s`;
};

const normalizeTrainingLogLevel = (level: unknown): TrainingLogLevel => {
  return level === "success" || level === "warning" || level === "error" ? level : "info";
};

const trainingWaitMessages = (mode: ModelMode, autoLambda: boolean) => {
  const ridge =
    mode === "featurewise_ridge_regression" ||
    mode === "compact_featurewise_ridge_regression";
  return [
    { delayMs: 900, message: "Backend step: loading and filtering Training Data rows." },
    { delayMs: 2200, message: "Backend step: validating score reference step and multiplier bounds." },
    ...(ridge
      ? [
          {
            delayMs: 3800,
            message: autoLambda
              ? "Backend step: fitting Ridge models across lambda candidates."
              : "Backend step: fitting Ridge model with the selected lambda.",
          },
          { delayMs: 6500, message: "Backend step: selecting validation winner and preparing JSON artifact." },
        ]
      : [
          { delayMs: 3800, message: "Backend step: building tensors and train/validation split." },
          { delayMs: 6500, message: "Backend step: optimizing MLP epochs with early stopping." },
          { delayMs: 10500, message: "Backend step: selecting best epoch and preparing checkpoint metadata." },
        ]),
  ];
};

export default function ModelTrainingPage() {
  const [models, setModels] = useState<WorkflowModel[]>([]);
  const [trainingDataTargets, setTrainingDataTargets] = useState<TrainingDataSource[]>([]);
  const [loading, setLoading] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [selectedTrainingDataId, setSelectedTrainingDataId] = useState("");
  const [uploadTrainingDataId, setUploadTrainingDataId] = useState("");
  const [modelMode, setModelMode] = useState<ModelMode>("compact_featurewise_ridge_regression");
  const [uploadModelMode, setUploadModelMode] = useState<UploadModelMode>("compact_featurewise_ridge_regression");
  const [modelName, setModelName] = useState("");
  const [uploadModelName, setUploadModelName] = useState("");
  const [uploadArtifactFile, setUploadArtifactFile] = useState<File | null>(null);
  const [uploadMetadataFile, setUploadMetadataFile] = useState<File | null>(null);
  const [uploadingModel, setUploadingModel] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<any | null>(null);
  const [autoSelectLambda, setAutoSelectLambda] = useState(true);
  const [lambdaInput, setLambdaInput] = useState("2.0");
  const [training, setTraining] = useState(false);
  const [trainingStartedAt, setTrainingStartedAt] = useState<number | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [trainingLogs, setTrainingLogs] = useState<TrainingLogEntry[]>([]);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [trainResult, setTrainResult] = useState<any | null>(null);
  const [modelActionError, setModelActionError] = useState<string | null>(null);
  const [modelToDelete, setModelToDelete] = useState<WorkflowModel | null>(null);
  const [modelToRename, setModelToRename] = useState<WorkflowModel | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [modelActionBusy, setModelActionBusy] = useState(false);
  const [datasetValidation, setDatasetValidation] = useState<{
    checkedId: string;
    loading: boolean;
    rowCount: number;
    valid: boolean;
    error: string | null;
  }>({
    checkedId: "",
    loading: false,
    rowCount: 0,
    valid: false,
    error: null,
  });
  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState<SortBy>("time_desc");

  const appendTrainingLog = useCallback((message: string, level: TrainingLogLevel = "info") => {
    setTrainingLogs((current) => [
      ...current.slice(-59),
      {
        timestamp: new Date().toLocaleTimeString(),
        level,
        message,
      },
    ]);
  }, []);

  useEffect(() => {
    if (!trainingStartedAt) return;
    const updateElapsed = () => setElapsedSeconds(Math.max(0, Math.floor((Date.now() - trainingStartedAt) / 1000)));
    updateElapsed();
    if (!training) return;
    const timer = window.setInterval(updateElapsed, 1000);
    return () => window.clearInterval(timer);
  }, [training, trainingStartedAt]);

  const loadModels = useCallback(async () => {
    setLoading(true);
    try {
      const [registryRes, sourcesRes] = await Promise.all([
        api.get("/api/models"),
        api.get("/api/workflow/model-training/training-data-sources"),
      ]);
      const registryModels = Array.isArray(registryRes.data?.items) ? registryRes.data.items.map(mapWorkflowModel) : [];
      const sources = Array.isArray(sourcesRes.data?.items) ? sourcesRes.data.items : [];
      setTrainingDataTargets(sources);
      setSelectedTrainingDataId((current) => current || sources[0]?.training_data_id || "");
      setUploadTrainingDataId((current) => current || sources[0]?.training_data_id || "");
      setModels(registryModels);
    } catch (err) {
      console.error("Failed to load trained models", err);
      setModels([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadModels();
  }, [loadModels]);

  const reportModels = useMemo(
    () => {
      const query = searchQuery.trim().toLowerCase();
      const filtered = models.filter((model) => {
        const format = String(model.artifact_format || "").toLowerCase();
        const mode = String(model.ai_profile?.ai_input_mode || "").toLowerCase();
        const isResearchModel = mode.includes("featurewise") || format.includes("ridge") || format.includes("mlp") || format.includes("featurewise");
        if (!isResearchModel) return false;
        if (!query) return true;
        return (
          String(model.model_name || "").toLowerCase().includes(query) ||
          String(model.model_id || "").toLowerCase().includes(query) ||
          format.includes(query) ||
          mode.includes(query)
        );
      });

      filtered.sort((a, b) => {
        if (sortBy === "name") {
          return String(a.model_name || a.model_id).localeCompare(String(b.model_name || b.model_id), undefined, {
            numeric: true,
            sensitivity: "base",
          });
        }
        const aTime = a.created_at ? Date.parse(a.created_at) || 0 : 0;
        const bTime = b.created_at ? Date.parse(b.created_at) || 0 : 0;
        return sortBy === "time_asc" ? aTime - bTime : bTime - aTime;
      });
      return filtered;
    },
    [models, searchQuery, sortBy],
  );

  const selectedTrainingData = useMemo(
    () => trainingDataTargets.find((source) => source.training_data_id === selectedTrainingDataId),
    [selectedTrainingDataId, trainingDataTargets],
  );

  const defaultModelName = useCallback(
    (mode: ModelMode) => {
      const base = selectedTrainingData?.name || "Training Data";
      const selectedOption = MODEL_MODE_OPTIONS.find((option) => option.value === mode);
      return `${base} - ${selectedOption?.label || mode}`;
    },
    [selectedTrainingData],
  );

  const openCreateModal = () => {
    const mode: ModelMode = "compact_featurewise_ridge_regression";
    setModelMode(mode);
    setModelName(defaultModelName(mode));
    setAutoSelectLambda(true);
    setLambdaInput("2.0");
    setTrainError(null);
    setTrainResult(null);
    setTrainingStartedAt(null);
    setElapsedSeconds(0);
    setTrainingLogs([
      {
        timestamp: new Date().toLocaleTimeString(),
        level: "info",
        message: "Ready. Select Training Data and model type, then start training.",
      },
    ]);
    setDatasetValidation({
      checkedId: "",
      loading: false,
      rowCount: 0,
      valid: false,
      error: null,
    });
    setModalOpen(true);
  };

  const openUploadModal = () => {
    const mode: UploadModelMode = "compact_featurewise_ridge_regression";
    setUploadModelMode(mode);
    setUploadModelName("Uploaded Featurewise Ridge Quality Model");
    setUploadTrainingDataId(selectedTrainingDataId || trainingDataTargets[0]?.training_data_id || "");
    setUploadArtifactFile(null);
    setUploadMetadataFile(null);
    setUploadError(null);
    setUploadResult(null);
    setUploadModalOpen(true);
  };

  const validateTrainingData = useCallback(async (trainingDataId: string) => {
    if (!trainingDataId) {
      appendTrainingLog("Training Data validation skipped: no dataset selected.", "warning");
      setDatasetValidation({
        checkedId: "",
        loading: false,
        rowCount: 0,
        valid: false,
        error: "Select Training Data first.",
      });
      return;
    }

    appendTrainingLog(`Checking Training Data: ${trainingDataId}`);
    setDatasetValidation({
      checkedId: trainingDataId,
      loading: true,
      rowCount: 0,
      valid: false,
      error: null,
    });
    try {
      const res = await api.get(`/api/workflow/training-data/${encodeURIComponent(trainingDataId)}/validity`);
      const rowCount = Number(res.data?.row_count || 0);
      const valid = Boolean(res.data?.usable_for_model_training);
      setDatasetValidation({
        checkedId: trainingDataId,
        loading: false,
        rowCount,
        valid,
        error: valid ? null : "Selected Training Data has no valid rows. Build Training Data first.",
      });
      appendTrainingLog(
        valid
          ? `Training Data is usable (${rowCount} row${rowCount === 1 ? "" : "s"}).`
          : "Training Data is not usable for model training yet.",
        valid ? "success" : "warning",
      );
    } catch (err: any) {
      appendTrainingLog(apiErrorMessage(err, "Training Data validation failed."), "error");
      setDatasetValidation({
        checkedId: trainingDataId,
        loading: false,
        rowCount: 0,
        valid: false,
        error: apiErrorMessage(err, "Selected Training Data is not built yet. Build it from the Training Data page first."),
      });
    }
  }, [appendTrainingLog]);

  useEffect(() => {
    if (!modalOpen) return;
    void validateTrainingData(selectedTrainingDataId);
  }, [modalOpen, selectedTrainingDataId, validateTrainingData]);

  const trainModel = async () => {
    if (!selectedTrainingDataId) {
      setTrainError("Select Training Data before training a model.");
      appendTrainingLog("Training blocked: no Training Data selected.", "error");
      return;
    }

    const payload: Record<string, unknown> = {
      model_family: modelMode,
      source_training_data_id: selectedTrainingDataId,
      source_pipeline_id: selectedTrainingData?.source_pipeline_id || undefined,
      // Stored with the model as the fallback prediction grid when testing does not provide one.
      candidate_points: 30,
    };

    if (modelName.trim()) {
      payload.model_name = modelName.trim();
    }

    if (isRidgeModelFamily(modelMode)) {
      if (autoSelectLambda) {
        payload.lambda_ridge = null;
      } else {
        const parsed = Number(lambdaInput);
        if (!Number.isFinite(parsed) || parsed <= 0) {
          setTrainError("Please enter a valid ridge lambda value greater than 0.");
          appendTrainingLog("Training blocked: invalid Ridge lambda value.", "error");
          return;
        }
        payload.lambda_ridge = parsed;
      }
    }

    setTraining(true);
    const startedAt = Date.now();
    const selectedOption = MODEL_MODE_OPTIONS.find((option) => option.value === modelMode);
    setTrainingStartedAt(startedAt);
    setElapsedSeconds(0);
    setTrainingLogs([
      {
        timestamp: new Date().toLocaleTimeString(),
        level: "info",
        message: `Starting ${selectedOption?.label || modelMode}.`,
      },
      {
        timestamp: new Date().toLocaleTimeString(),
        level: "info",
        message: `Training Data: ${selectedTrainingData?.name || selectedTrainingDataId}.`,
      },
      {
        timestamp: new Date().toLocaleTimeString(),
        level: "info",
        message:
          isRidgeModelFamily(modelMode)
            ? autoSelectLambda
              ? "Ridge lambda: auto-selecting best candidate."
              : `Ridge lambda: ${lambdaInput}.`
            : "MLP training: running neural optimizer with early stopping.",
      },
    ]);
    setTrainError(null);
    setTrainResult(null);
    try {
      const waitTimers = trainingWaitMessages(modelMode, autoSelectLambda).map((entry) =>
        window.setTimeout(() => appendTrainingLog(entry.message), entry.delayMs),
      );
      appendTrainingLog("Submitted training request. Waiting for backend training to finish...");
      try {
        const res = await api.post("/api/workflow/model-training/train", payload);
        waitTimers.forEach((timer) => window.clearTimeout(timer));
        setTrainResult(res.data);
        const metrics = res.data?.metrics || {};
        const backendLogs = Array.isArray(metrics.training_log) ? metrics.training_log : [];
        backendLogs.forEach((entry: any) => {
          if (entry?.message) {
            appendTrainingLog(String(entry.message), normalizeTrainingLogLevel(entry.level));
          }
        });
        if (isRidgeModelFamily(res.data?.model_family)) {
          appendTrainingLog(`Training completed. Selected lambda: ${metrics.lambda_selected ?? metrics.selected_lambda ?? "n/a"}.`, "success");
        } else {
          appendTrainingLog(
            `Training completed. Epochs: ${metrics.epochs_trained ?? "-"}; best val loss: ${
              metrics.best_val_loss !== undefined && metrics.best_val_loss !== null ? Number(metrics.best_val_loss).toFixed(6) : "-"
            }.`,
            "success",
          );
        }
      } finally {
        waitTimers.forEach((timer) => window.clearTimeout(timer));
      }
      await loadModels();
    } catch (err: any) {
      const message = apiErrorMessage(err, "Model training failed. Build Training Data first, then retry.");
      setTrainError(message);
      appendTrainingLog(message, "error");
    } finally {
      setTraining(false);
    }
  };

  const openRenameModel = (model: WorkflowModel) => {
    setModelActionError(null);
    setModelToRename(model);
    setRenameValue(model.model_name || model.model_id);
  };

  const saveModelName = async () => {
    if (!modelToRename) return;
    const name = renameValue.trim();
    if (!name) {
      setModelActionError("Model name is required.");
      return;
    }

    setModelActionBusy(true);
    setModelActionError(null);
    try {
      await api.patch(`/api/models/${encodeURIComponent(modelToRename.model_id)}`, { model_name: name });
      setModelToRename(null);
      setRenameValue("");
      await loadModels();
    } catch (err: any) {
      setModelActionError(apiErrorMessage(err, "Failed to rename model."));
    } finally {
      setModelActionBusy(false);
    }
  };

  const deleteModel = async () => {
    if (!modelToDelete) return;

    setModelActionBusy(true);
    setModelActionError(null);
    try {
      await api.delete(`/api/models/${encodeURIComponent(modelToDelete.model_id)}`);
      setModelToDelete(null);
      await loadModels();
    } catch (err: any) {
      setModelActionError(apiErrorMessage(err, "Failed to delete model."));
    } finally {
      setModelActionBusy(false);
    }
  };

  const uploadModel = async () => {
    if (!uploadTrainingDataId) {
      setUploadError("Select the Training Data source used to train this model.");
      return;
    }
    if (!uploadModelName.trim()) {
      setUploadError("Model name is required.");
      return;
    }
    if (!uploadArtifactFile) {
      setUploadError(isUploadMlpMode(uploadModelMode) ? "Select the .pt checkpoint file." : "Select the Ridge model JSON file.");
      return;
    }

    const source = trainingDataTargets.find((item) => item.training_data_id === uploadTrainingDataId);
    const form = new FormData();
    form.append("model_family", uploadModelMode);
    form.append("model_name", uploadModelName.trim());
    form.append("source_training_data_id", uploadTrainingDataId);
    if (source?.source_pipeline_id) {
      form.append("source_pipeline_id", source.source_pipeline_id);
    }
    form.append("artifact_file", uploadArtifactFile);
    if (uploadMetadataFile) {
      form.append("metadata_file", uploadMetadataFile);
    }

    setUploadingModel(true);
    setUploadError(null);
    setUploadResult(null);
    try {
      const res = await api.post("/api/workflow/model-training/upload", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setUploadResult(res.data);
      await loadModels();
    } catch (err: any) {
      setUploadError(apiErrorMessage(err, "Model upload failed."));
    } finally {
      setUploadingModel(false);
    }
  };

  return (
    <WorkflowShell
      eyebrow="Stage 2"
      title="Train Models"
      backTo="/workflow"
      breadcrumbs={[
        { label: "Research Workflow", to: "/workflow" },
        { label: "Train Models" },
      ]}
    >
      <div className="space-y-4">
        <WorkflowActionPanel
          title="Train Research Models"
          subtitle="Select a Training Data dataset, then train featurewise or compact Ridge/MLP models."
          icon={Brain}
          tone="emerald"
          actionIcon={Plus}
          actionLabel="Create New"
          actionOnClick={openCreateModal}
          secondaryActionIcon={Upload}
          secondaryActionLabel="Upload Model"
          secondaryActionOnClick={openUploadModal}
        >
          <div className="grid gap-3 md:grid-cols-3">
            <div className="rounded-lg border border-slate-200 p-3">
              <FileJson className="mb-2 h-4 w-4 text-emerald-600" />
              <div className="text-sm font-semibold text-slate-900">Training Data</div>
              <div className="text-xs text-slate-500">Training consumes the selected final Training Data dataset.</div>
            </div>
            <div className="rounded-lg border border-slate-200 p-3">
              <Settings2 className="mb-2 h-4 w-4 text-emerald-600" />
              <div className="text-sm font-semibold text-slate-900">Configuration</div>
              <div className="text-xs text-slate-500">Choose legacy group-wise or compact shared Ridge/MLP training.</div>
            </div>
            <div className="rounded-lg border border-slate-200 p-3">
              <ChartNoAxesCombined className="mb-2 h-4 w-4 text-emerald-600" />
              <div className="text-sm font-semibold text-slate-900">Reports</div>
              <div className="text-xs text-slate-500">Metrics and model metadata are retained for report tables.</div>
            </div>
          </div>
        </WorkflowActionPanel>
        <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-start justify-between gap-3">
            <div className="flex items-start gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-600 to-emerald-700 text-white shadow-sm">
                <Brain className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-slate-950">Trained Models</h2>
                <p className="mt-1 text-sm text-slate-600">Registered featurewise and compact model artifacts available for testing.</p>
              </div>
            </div>
            <button
              onClick={() => void loadModels()}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
            >
              <RefreshCw className="h-4 w-4" />
              Refresh
            </button>
          </div>
          <div className="mb-4 grid gap-3 md:grid-cols-[1fr_180px]">
            <label className="relative block">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <input
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                placeholder="Search models by name, id, type, or mode"
                className="w-full rounded-lg border border-slate-200 py-2 pl-9 pr-3 text-sm outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100"
              />
            </label>
            <select
              value={sortBy}
              onChange={(event) => setSortBy(event.target.value as SortBy)}
              className="rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100"
            >
              <option value="time_desc">Sort: Newest</option>
              <option value="time_asc">Sort: Oldest</option>
              <option value="name">Sort: Name</option>
            </select>
          </div>
          <ModelSummaryList
            emptyMessage="No trained research models found yet."
            loading={loading}
            models={reportModels}
            onDeleteModel={(model) => {
              setModelActionError(null);
              setModelToDelete(model);
            }}
            onRenameModel={openRenameModel}
          />
        </section>
      </div>
      {(modelToRename || modelToDelete) && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-md rounded-xl bg-white p-5 shadow-xl">
            <div className="mb-4 flex items-start justify-between gap-3">
              <div>
                <h3 className="text-lg font-semibold text-slate-950">
                  {modelToRename ? "Edit Model Name" : "Delete Model"}
                </h3>
                <p className="mt-1 text-sm text-slate-600">
                  {modelToRename
                    ? "Rename the registered workflow model. The model id and artifact files stay unchanged."
                    : "This removes the registered model artifact from Train Models and testing selection."}
                </p>
              </div>
              <button
                onClick={() => {
                  if (modelActionBusy) return;
                  setModelToRename(null);
                  setModelToDelete(null);
                  setModelActionError(null);
                }}
                className="rounded p-1 text-slate-500 hover:bg-slate-100 hover:text-slate-800"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {modelToRename && (
              <div>
                <label className="mb-1 block text-sm font-medium text-slate-700">Model Name</label>
                <input
                  value={renameValue}
                  onChange={(event) => setRenameValue(event.target.value)}
                  className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100"
                  disabled={modelActionBusy}
                />
              </div>
            )}

            {modelToDelete && (
              <div className="rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800">
                Delete <strong>{modelToDelete.model_name || modelToDelete.model_id}</strong>? This cannot be undone.
              </div>
            )}

            {modelActionError && (
              <div className="mt-3 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">{modelActionError}</div>
            )}

            <div className="mt-5 flex justify-end gap-2 border-t border-slate-200 pt-4">
              <button
                onClick={() => {
                  if (modelActionBusy) return;
                  setModelToRename(null);
                  setModelToDelete(null);
                  setModelActionError(null);
                }}
                className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
                disabled={modelActionBusy}
              >
                Cancel
              </button>
              {modelToRename ? (
                <button
                  onClick={() => void saveModelName()}
                  disabled={modelActionBusy || !renameValue.trim()}
                  className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-700 disabled:opacity-60"
                >
                  {modelActionBusy ? "Saving..." : "Save"}
                </button>
              ) : (
                <button
                  onClick={() => void deleteModel()}
                  disabled={modelActionBusy}
                  className="rounded-lg bg-rose-600 px-4 py-2 text-sm font-semibold text-white hover:bg-rose-700 disabled:opacity-60"
                >
                  {modelActionBusy ? "Deleting..." : "Delete"}
                </button>
              )}
            </div>
          </div>
        </div>
      )}
      {uploadModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="flex max-h-[calc(100vh-2rem)] w-full max-w-lg flex-col overflow-hidden rounded-xl bg-white shadow-xl">
            <div className="shrink-0 p-5 pb-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h3 className="text-lg font-semibold text-slate-950">Upload Model</h3>
                  <p className="mt-1 text-sm text-slate-600">
                    Register a notebook-trained compact model so it can be selected in testing configuration.
                  </p>
                </div>
                <button
                  onClick={() => {
                    if (!uploadingModel) setUploadModalOpen(false);
                  }}
                  className="rounded p-1 text-slate-500 hover:bg-slate-100 hover:text-slate-800"
                  disabled={uploadingModel}
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>

            <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-5 pb-5">
              <div className="rounded-lg border border-blue-200 bg-blue-50 p-3 text-sm text-blue-800">
                Ridge uploads use the notebook JSON model. MLP uploads use the notebook `.pt` checkpoint; add the metadata JSON when available so metrics and seed details are shown in the model record.
              </div>

              <div>
                <label className="mb-1 block text-sm font-medium text-slate-700">Training Data Source</label>
                <select
                  value={uploadTrainingDataId}
                  onChange={(event) => setUploadTrainingDataId(event.target.value)}
                  disabled={uploadingModel || trainingDataTargets.length === 0}
                  className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 disabled:opacity-60"
                >
                  {trainingDataTargets.length === 0 ? (
                    <option value="">No Training Data available</option>
                  ) : (
                    trainingDataTargets.map((source) => (
                      <option key={source.training_data_id} value={source.training_data_id}>
                        {source.name || source.training_data_id} ({source.row_count} rows)
                      </option>
                    ))
                  )}
                </select>
                <p className="mt-1 text-xs text-slate-500">
                  Select the prepared Training Data folder that was used to train the uploaded notebook model.
                </p>
              </div>

              <div>
                <label className="mb-1 block text-sm font-medium text-slate-700">Model Type</label>
                <select
                  value={uploadModelMode}
                  onChange={(event) => {
                    const mode = event.target.value as UploadModelMode;
                    setUploadModelMode(mode);
                    setUploadArtifactFile(null);
                    setUploadMetadataFile(null);
                    setUploadModelName(
                      isUploadMlpMode(mode)
                        ? "Uploaded Featurewise MLP Quality Model"
                        : "Uploaded Featurewise Ridge Quality Model",
                    );
                  }}
                  disabled={uploadingModel}
                  className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 disabled:opacity-60"
                >
                  <option value="compact_featurewise_ridge_regression">Featurewise Ridge Quality Model</option>
                  <option value="compact_featurewise_mlp">Featurewise MLP Quality Model</option>
                  <option value="compact_descriptor_mlp">Compact Descriptor MLP Quality Model</option>
                </select>
              </div>

              <div>
                <label className="mb-1 block text-sm font-medium text-slate-700">Model Name</label>
                <input
                  value={uploadModelName}
                  onChange={(event) => setUploadModelName(event.target.value)}
                  disabled={uploadingModel}
                  className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 disabled:opacity-60"
                  placeholder="Enter model name"
                />
              </div>

              <div>
                <label className="mb-1 block text-sm font-medium text-slate-700">
                  {isUploadMlpMode(uploadModelMode) ? "MLP Checkpoint (.pt)" : "Ridge Model JSON"}
                </label>
                <input
                  key={`artifact-${uploadModelMode}`}
                  type="file"
                  accept={isUploadMlpMode(uploadModelMode) ? ".pt,.pth" : ".json,application/json"}
                  onChange={(event) => setUploadArtifactFile(event.target.files?.[0] || null)}
                  disabled={uploadingModel}
                  className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm file:mr-3 file:rounded-md file:border-0 file:bg-slate-100 file:px-3 file:py-1.5 file:text-sm file:font-semibold file:text-slate-700 hover:file:bg-slate-200 disabled:opacity-60"
                />
                {uploadArtifactFile && (
                  <div className="mt-1 truncate text-xs text-slate-500" title={uploadArtifactFile.name}>
                    Selected: {uploadArtifactFile.name}
                  </div>
                )}
              </div>

              <div>
                <label className="mb-1 block text-sm font-medium text-slate-700">Metadata JSON</label>
                <input
                  key={`metadata-${uploadModelMode}`}
                  type="file"
                  accept=".json,application/json"
                  onChange={(event) => setUploadMetadataFile(event.target.files?.[0] || null)}
                  disabled={uploadingModel}
                  className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm file:mr-3 file:rounded-md file:border-0 file:bg-slate-100 file:px-3 file:py-1.5 file:text-sm file:font-semibold file:text-slate-700 hover:file:bg-slate-200 disabled:opacity-60"
                />
                <p className="mt-1 text-xs text-slate-500">
                  Optional for Ridge. Recommended for MLP because it carries readable training metrics.
                </p>
                {uploadMetadataFile && (
                  <div className="mt-1 truncate text-xs text-slate-500" title={uploadMetadataFile.name}>
                    Selected: {uploadMetadataFile.name}
                  </div>
                )}
              </div>

              {uploadError && <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">{uploadError}</div>}
              {uploadResult && (
                <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-800">
                  <div className="font-semibold">Model uploaded successfully.</div>
                  <div className="mt-1 break-all text-xs">{uploadResult.model_id}</div>
                  <div className="mt-1 text-xs">It is now available in testing model selection.</div>
                </div>
              )}
            </div>

            <div className="sticky bottom-0 flex shrink-0 justify-end gap-2 border-t border-slate-200 bg-white p-5 shadow-[0_-8px_16px_rgba(15,23,42,0.06)]">
              <button
                onClick={() => setUploadModalOpen(false)}
                className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
                disabled={uploadingModel}
              >
                {uploadResult ? "Close" : "Cancel"}
              </button>
              {!uploadResult && (
                <button
                  onClick={() => void uploadModel()}
                  disabled={uploadingModel || !uploadTrainingDataId || !uploadModelName.trim() || !uploadArtifactFile}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-700 disabled:opacity-60"
                >
                  <Upload className="h-4 w-4" />
                  {uploadingModel ? "Uploading..." : "Upload Model"}
                </button>
              )}
            </div>
          </div>
        </div>
      )}
      {modalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="flex max-h-[calc(100vh-2rem)] w-full max-w-lg flex-col overflow-hidden rounded-xl bg-white shadow-xl">
            <div className="shrink-0 p-5 pb-4">
              <div className="flex items-start justify-between gap-3">
              <div>
                <h3 className="text-lg font-semibold text-slate-950">Create Model Training Pipeline</h3>
                <p className="mt-1 text-sm text-slate-600">Select Training Data first, then configure and train the model.</p>
              </div>
              <button
                onClick={() => setModalOpen(false)}
                className="rounded p-1 text-slate-500 hover:bg-slate-100 hover:text-slate-800"
              >
                <X className="h-4 w-4" />
              </button>
              </div>
            </div>

            <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-5 pb-5">
              <div>
                <label className="mb-1 block text-sm font-medium text-slate-700">Training Data</label>
                <select
                  value={selectedTrainingDataId}
                  onChange={(event) => {
                    setSelectedTrainingDataId(event.target.value);
                    setTrainResult(null);
                    setTrainError(null);
                  }}
                  disabled={training || trainingDataTargets.length === 0}
                  className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 disabled:opacity-60"
                >
                  {trainingDataTargets.length === 0 ? (
                    <option value="">No Training Data available</option>
                  ) : (
                    trainingDataTargets.map((source) => (
                      <option key={source.training_data_id} value={source.training_data_id}>
                        {source.name || source.training_data_id} ({source.row_count} rows)
                      </option>
                    ))
                  )}
                </select>
              </div>

              {datasetValidation.loading && (
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-600">
                  Checking selected Training Data...
                </div>
              )}

              {!datasetValidation.loading && selectedTrainingDataId && !datasetValidation.valid && (
                <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
                  {datasetValidation.error || "Selected Training Data is not ready for model training."}
                </div>
              )}

              {!datasetValidation.loading && datasetValidation.valid && (
                <>
                  <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-800">
                    Training Data ready: {datasetValidation.rowCount} row{datasetValidation.rowCount === 1 ? "" : "s"} available.
                  </div>
                  <div>
                    <label className="mb-1 block text-sm font-medium text-slate-700">Model</label>
                    <select
                      value={modelMode}
                      onChange={(event) => {
                        const nextMode = event.target.value as ModelMode;
                        setModelMode(nextMode);
                        setModelName(defaultModelName(nextMode));
                        setTrainResult(null);
                      }}
                      disabled={training}
                      className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 disabled:opacity-60"
                    >
                      {MODEL_MODE_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                    <p className="mt-1 text-xs text-slate-500">
                      {MODEL_MODE_OPTIONS.find((option) => option.value === modelMode)?.description}
                    </p>
                  </div>

                  <div>
                    <label className="mb-1 block text-sm font-medium text-slate-700">Model Name</label>
                    <input
                      value={modelName}
                      onChange={(event) => setModelName(event.target.value)}
                      disabled={training}
                      className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 disabled:opacity-60"
                      placeholder="Enter model name"
                    />
                  </div>

                  {isRidgeModelFamily(modelMode) && (
                    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                      <label className="flex items-center gap-2 text-sm font-medium text-slate-700">
                        <input
                          type="checkbox"
                          checked={autoSelectLambda}
                          onChange={(event) => setAutoSelectLambda(event.target.checked)}
                          disabled={training}
                          className="h-4 w-4 rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
                        />
                        Auto-select best lambda
                      </label>
                      {!autoSelectLambda && (
                        <div className="mt-3">
                          <label className="mb-1 block text-xs font-medium text-slate-600">Lambda (ridge)</label>
                          <input
                            type="number"
                            min="0"
                            step="0.1"
                            value={lambdaInput}
                            onChange={(event) => setLambdaInput(event.target.value)}
                            disabled={training}
                            className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100 disabled:opacity-60"
                          />
                        </div>
                      )}
                    </div>
                  )}
                </>
              )}

              {trainError && <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">{trainError}</div>}
              {trainResult && (
                <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-800">
                  <div className="font-semibold">Model trained successfully.</div>
                  <div className="mt-1 text-xs">
                    Samples used: {trainResult.training_samples ?? "-"}
                    {trainResult.metrics?.mean_relative_quality_score !== undefined && trainResult.metrics?.mean_relative_quality_score !== null
                      ? ` | Mean relative score: ${Number(trainResult.metrics.mean_relative_quality_score).toFixed(4)}`
                      : ""}
                  </div>
                  {isRidgeModelFamily(trainResult.model_family) &&
                    (trainResult.metrics?.lambda_selected || trainResult.metrics?.selected_lambda || trainResult.config?.lambda_ridge) && (
                    <div className="mt-1 text-xs">
                      Lambda: {trainResult.metrics?.lambda_selected || trainResult.metrics?.selected_lambda || trainResult.config.lambda_ridge}
                      {trainResult.config?.regularization ? ` | Regularization: ${trainResult.config.regularization}` : ""}
                    </div>
                  )}
                  {isMlpModelFamily(trainResult.model_family) && (
                    <div className="mt-1 text-xs">
                      Epochs: {trainResult.metrics?.epochs_trained ?? "-"} / {trainResult.metrics?.max_epochs ?? trainResult.config?.max_epochs ?? "-"}
                      {trainResult.metrics?.best_val_loss !== undefined && trainResult.metrics?.best_val_loss !== null
                        ? ` | Best val loss: ${Number(trainResult.metrics.best_val_loss).toFixed(6)}`
                        : ""}
                    </div>
                  )}
                </div>
              )}

              {(trainingLogs.length > 0 || training) && (
                <div className="rounded-lg border border-slate-800 bg-slate-950 p-3 text-xs text-slate-100">
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <div className="font-semibold">Training Activity</div>
                    {trainingStartedAt && (
                      <div className="shrink-0 text-[11px] text-slate-400">
                        {training ? "Running" : "Elapsed"} {formatElapsed(elapsedSeconds)}
                      </div>
                    )}
                  </div>
                  <div className="max-h-44 space-y-1 overflow-auto font-mono leading-5">
                    {trainingLogs.map((entry, index) => {
                      const levelClass =
                        entry.level === "error"
                          ? "text-red-300"
                          : entry.level === "warning"
                            ? "text-amber-300"
                            : entry.level === "success"
                              ? "text-emerald-300"
                              : "text-slate-200";
                      return (
                        <div key={`${entry.timestamp}-${index}`} className={levelClass}>
                          <span className="text-slate-500">[{entry.timestamp}]</span> {entry.message}
                        </div>
                      );
                    })}
                  </div>
                  {training && (
                    <div className="mt-2 border-t border-slate-800 pt-2 text-[11px] text-slate-400">
                      Backend training is running. Detailed Ridge/MLP metrics appear here as soon as the request finishes.
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="sticky bottom-0 flex shrink-0 justify-end gap-2 border-t border-slate-200 bg-white p-5 shadow-[0_-8px_16px_rgba(15,23,42,0.06)]">
              <button
                onClick={() => setModalOpen(false)}
                className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
              >
                {trainResult ? "Close" : "Cancel"}
              </button>
              {!trainResult && (
                <button
                  onClick={() => void trainModel()}
                  disabled={training || !selectedTrainingDataId || !datasetValidation.valid}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-700 disabled:opacity-60"
                >
                  <Brain className="h-4 w-4" />
                  {training ? "Training..." : "Train Model"}
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </WorkflowShell>
  );
}
