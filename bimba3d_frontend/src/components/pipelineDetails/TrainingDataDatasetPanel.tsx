import { Database, Download, RefreshCw, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api } from "../../api/client";
import { getPipelineStage, type WorkflowPipeline } from "../workflow/PipelineSummaryList";

interface OfflineDataset {
  has_dataset?: boolean;
  training_data_id?: string;
  path?: string;
  rows?: any[];
  summary?: Record<string, any>;
  dataset?: {
    row_count?: number;
    rows?: any[];
    path?: string;
    summary?: Record<string, any>;
  };
}

interface TrainingDataDatasetPanelProps {
  allowPipelineSelection?: boolean;
  description?: string;
  pipelineId?: string;
  title?: string;
}

type SortKey = "project" | "hard_cap" | "relative_quality_score" | "convergence_score";
type SortDirection = "asc" | "desc";

const getRowCount = (dataset: OfflineDataset | null) => {
  if (!dataset) return 0;
  if (typeof dataset.dataset?.row_count === "number") return dataset.dataset.row_count;
  if (Array.isArray(dataset.dataset?.rows)) return dataset.dataset.rows.length;
  if (Array.isArray(dataset.rows)) return dataset.rows.length;
  return 0;
};

const getPreviewRows = (dataset: OfflineDataset | null) => {
  if (!dataset) return [];
  if (Array.isArray(dataset.dataset?.rows)) return dataset.dataset.rows;
  if (Array.isArray(dataset.rows)) return dataset.rows;
  return [];
};

const ROWS_PER_PAGE = 25;

const isObjectValue = (value: any) => value !== null && typeof value === "object";

const MULTIPLIER_GROUPS: Record<string, { label: string; params: string[] }> = {
  geometry_lr_mult: {
    label: "Geometry",
    params: ["position_lr_init", "position_lr_final", "scaling_lr", "rotation_lr"],
  },
  appearance_lr_mult: {
    label: "Appearance",
    params: ["feature_lr", "opacity_lr", "lambda_dssim"],
  },
  densification_mult: {
    label: "Densification",
    params: ["densify_grad_threshold", "opacity_threshold"],
  },
};

const COMPACT_FEATURE_KEYS = new Set([
  "gsd_median",
  "overlap_proxy",
  "coverage_spread",
  "camera_angle_bucket",
  "heading_consistency",
  "texture_density",
  "blur_motion_risk",
  "terrain_roughness_proxy",
  "terrain_roughness",
  "vegetation_complexity_score",
  "vegetation_complexity",
  "vegetation_cover_percentage",
  "vegetation_cover",
]);

const formatModalValue = (value: any) => {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(6);
  if (value === null || value === undefined) return "-";
  if (isObjectValue(value)) return JSON.stringify(value);
  return String(value);
};

const formatMetric = (value: any, digits = 4) => {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return value.toFixed(digits);
};

const formatSummaryNumber = (value: any, digits = 6) => {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(digits);
};

const countObjectValues = (value: any) => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return 0;
  return Object.values(value).filter((entry) => entry !== null && entry !== undefined).length;
};

const cleanXFeatures = (value: any) => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return Object.fromEntries(
    Object.entries(value).filter(([key]) => !String(key).toLowerCase().endsWith("_missing")),
  );
};

const isHardCapPenaltyRow = (row: any) => Boolean(row?.metadata?.is_hard_cap_penalty_row);

const csvCell = (value: any) => {
  if (value === null || value === undefined) return "";
  const text = typeof value === "string" ? value : isObjectValue(value) ? JSON.stringify(value) : String(value);
  return `"${text.replace(/"/g, '""')}"`;
};

const downloadTextFile = (filename: string, content: string) => {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

const safeFileToken = (value: string) =>
  value.replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/-+/g, "-").replace(/^-|-$/g, "") || "training-data";

const sortValue = (row: any, key: SortKey) => {
  if (key === "project") return String(row.project_name || row.project_id || "").toLowerCase();
  if (key === "hard_cap") return isHardCapPenaltyRow(row) ? 1 : 0;
  const value = row[key];
  return typeof value === "number" && Number.isFinite(value) ? value : Number.NEGATIVE_INFINITY;
};

const uniqueNumericValues = (values: any[]) => {
  const seen = new Set<number>();
  values.forEach((value) => {
    if (typeof value === "number" && Number.isFinite(value)) seen.add(value);
  });
  return [...seen].sort((a, b) => a - b);
};

const pickLatestTrainingData = (items: any[]) => {
  if (!Array.isArray(items) || items.length === 0) return null;
  return [...items].sort((a, b) => {
    const aTime = Date.parse(a.last_built_at || a.created_at || "") || 0;
    const bTime = Date.parse(b.last_built_at || b.created_at || "") || 0;
    return bTime - aTime;
  })[0];
};

const toDataset = (manifest: any, rows: any[] = []): OfflineDataset => ({
  has_dataset: rows.length > 0,
  training_data_id: manifest?.training_data_id,
  path: manifest?.rows_path,
  rows,
  summary: {
    status: manifest?.status,
    schema_valid: manifest?.schema_valid,
    feature_schema: manifest?.feature_schema,
    multiplier_schema: manifest?.multiplier_schema,
    last_built_at: manifest?.last_built_at,
    build_options: manifest?.build_options || {},
    build_summary: manifest?.build_summary || {},
    errors: manifest?.errors || [],
  },
  dataset: {
    row_count: typeof manifest?.row_count === "number" ? manifest.row_count : rows.length,
    rows,
    path: manifest?.rows_path,
    summary: {
      status: manifest?.status,
      schema_valid: manifest?.schema_valid,
      feature_schema: manifest?.feature_schema,
      multiplier_schema: manifest?.multiplier_schema,
      last_built_at: manifest?.last_built_at,
      build_options: manifest?.build_options || {},
      build_summary: manifest?.build_summary || {},
      errors: manifest?.errors || [],
    },
  },
});

export default function TrainingDataDatasetPanel({
  allowPipelineSelection = false,
  description = "Offline training rows produced by this data preparation pipeline.",
  pipelineId,
  title = "Prepared Dataset",
}: TrainingDataDatasetPanelProps) {
  const [dataset, setDataset] = useState<OfflineDataset | null>(null);
  const [loading, setLoading] = useState(false);
  const [building, setBuilding] = useState(false);
  const [pipelines, setPipelines] = useState<WorkflowPipeline[]>([]);
  const [pipelinesLoading, setPipelinesLoading] = useState(false);
  const [selectedPipelineId, setSelectedPipelineId] = useState(pipelineId || "");
  const [buildError, setBuildError] = useState<string | null>(null);
  const [buildMessage, setBuildMessage] = useState<string | null>(null);
  const [buildModalOpen, setBuildModalOpen] = useState(false);
  const [includeHardCapPenaltyRows, setIncludeHardCapPenaltyRows] = useState(false);
  const [page, setPage] = useState(1);
  const [sortConfig, setSortConfig] = useState<{ key: SortKey; direction: SortDirection } | null>(null);
  const [detailModal, setDetailModal] = useState<{
    column: string;
    data: any;
    project?: string;
    rowData?: any;
    rowIndex: number;
    run?: string;
  } | null>(null);

  useEffect(() => {
    setSelectedPipelineId(pipelineId || "");
  }, [pipelineId]);

  const loadPipelines = useCallback(async () => {
    if (!allowPipelineSelection) return;
    setPipelinesLoading(true);
    try {
      const res = await api.get("/api/workflow/pipelines?stage=offline_data_preparation&limit=100");
      const dataPipelines = (res.data?.items || []).filter(
        (pipeline: WorkflowPipeline) => getPipelineStage(pipeline) === "training_data",
      );
      setPipelines(dataPipelines);
      setSelectedPipelineId((current) => current || dataPipelines[0]?.id || "");
    } catch (err) {
      console.error("Failed to load training data pipelines", err);
      setPipelines([]);
    } finally {
      setPipelinesLoading(false);
    }
  }, [allowPipelineSelection]);

  const activePipelineId = allowPipelineSelection ? selectedPipelineId : pipelineId || "";

  const loadDataset = useCallback(async () => {
    if (!activePipelineId) {
      setDataset(null);
      return;
    }
    setLoading(true);
    try {
      setBuildError(null);
      const manifestRes = await api.get(`/api/workflow/training-data/by-source-pipeline/${encodeURIComponent(activePipelineId)}`);
      const manifest = pickLatestTrainingData(manifestRes.data?.items || []);
      if (!manifest) {
        setDataset(null);
        setPage(1);
        return;
      }
      const rowsRes = await api.get(`/api/workflow/training-data/${encodeURIComponent(manifest.training_data_id)}/rows`);
      setDataset(toDataset(manifest, rowsRes.data?.rows || []));
      setPage(1);
    } catch (err) {
      console.error("Failed to load prepared training data", err);
      setBuildError("Failed to load prepared training data.");
      setDataset(null);
    } finally {
      setLoading(false);
    }
  }, [activePipelineId]);

  const buildDataset = useCallback(async (includePenaltyRows: boolean) => {
    if (!activePipelineId) return;
    setBuilding(true);
    try {
      setBuildError(null);
      setBuildMessage(null);
      const sourcePipeline = pipelines.find((pipeline) => pipeline.id === activePipelineId);
      const manifestRes = await api.get(`/api/workflow/training-data/by-source-pipeline/${encodeURIComponent(activePipelineId)}`);
      let manifest = pickLatestTrainingData(manifestRes.data?.items || []);
      if (!manifest) {
        const createRes = await api.post("/api/workflow/training-data", {
          name: `${sourcePipeline?.name || activePipelineId} Training Data`,
          source_pipeline_id: activePipelineId,
          feature_schema: "mode3_exif_flight_scene_v1",
        });
        manifest = createRes.data;
      }
      const buildRes = await api.post(
        `/api/workflow/training-data/${encodeURIComponent(manifest.training_data_id)}/build-from-pipeline/${encodeURIComponent(activePipelineId)}`,
        { include_hard_cap_penalty_rows: includePenaltyRows },
      );
      const builtManifest = buildRes.data?.manifest || manifest;
      const rowsRes = await api.get(`/api/workflow/training-data/${encodeURIComponent(builtManifest.training_data_id)}/rows`);
      setDataset(toDataset(builtManifest, rowsRes.data?.rows || []));
      const importedRows = Number(buildRes.data?.imported_rows || 0);
      const errors = Array.isArray(buildRes.data?.errors) ? buildRes.data.errors : [];
      if (errors.length > 0) {
        setBuildError(errors.slice(0, 5).join(" | "));
      } else {
        const hardCapRows = Number(buildRes.data?.hard_cap_penalty_rows || 0);
        const hardCapPenalty = buildRes.data?.hard_cap_penalty;
        const hardCapSuffix = hardCapRows > 0 && typeof hardCapPenalty === "number"
          ? ` Included ${hardCapRows} hard-cap penalty row${hardCapRows === 1 ? "" : "s"} at ${hardCapPenalty.toFixed(6)}.`
          : "";
        setBuildMessage(`Built ${importedRows} training data rows.${hardCapSuffix}`);
      }
      setPage(1);
      setBuildModalOpen(false);
    } catch (err) {
      console.error("Failed to build prepared training data", err);
      const maybeResponse = (err as any)?.response?.data?.detail;
      const details = typeof maybeResponse?.details === "string" ? maybeResponse.details : "";
      const message = typeof maybeResponse?.message === "string" ? maybeResponse.message : "Failed to build prepared training data.";
      setBuildError(details ? `${message} ${details}` : message);
    } finally {
      setBuilding(false);
    }
  }, [activePipelineId, pipelines]);

  useEffect(() => {
    void loadPipelines();
  }, [loadPipelines]);

  useEffect(() => {
    if (allowPipelineSelection && !activePipelineId) return;
    void loadDataset();
  }, [activePipelineId, allowPipelineSelection, loadDataset]);

  const rowCount = getRowCount(dataset);
  const previewRows = getPreviewRows(dataset);
  const sortedRows = sortConfig
    ? [...previewRows].sort((a, b) => {
        const aValue = sortValue(a, sortConfig.key);
        const bValue = sortValue(b, sortConfig.key);
        let comparison = 0;
        if (typeof aValue === "string" || typeof bValue === "string") {
          comparison = String(aValue).localeCompare(String(bValue));
        } else {
          comparison = Number(aValue) - Number(bValue);
        }
        return sortConfig.direction === "asc" ? comparison : -comparison;
      })
    : previewRows;
  const summary = dataset?.summary || dataset?.dataset?.summary || {};
  const filePath = dataset?.path || dataset?.dataset?.path || "";
  const totalPages = Math.max(1, Math.ceil(sortedRows.length / ROWS_PER_PAGE));
  const visibleRows = sortedRows.slice((page - 1) * ROWS_PER_PAGE, page * ROWS_PER_PAGE);
  const showingFrom = sortedRows.length === 0 ? 0 : (page - 1) * ROWS_PER_PAGE + 1;
  const showingTo = Math.min(page * ROWS_PER_PAGE, sortedRows.length);
  const buildSummary = summary.build_summary || {};
  const hardCapPenaltyRows = Number(buildSummary.hard_cap_penalty_rows || 0);
  const hardCapPenalty = buildSummary.hard_cap_penalty;
  const hardCapLimitValues = uniqueNumericValues(
    Array.isArray(buildSummary.gaussian_hard_cap_values) && buildSummary.gaussian_hard_cap_values.length > 0
      ? buildSummary.gaussian_hard_cap_values
      : buildSummary.gaussian_hard_cap !== undefined
        ? [buildSummary.gaussian_hard_cap]
        : [],
  );
  const hardCapLimitLabel = hardCapLimitValues.length === 1
    ? formatSummaryNumber(hardCapLimitValues[0], 0)
    : hardCapLimitValues.length > 1
      ? hardCapLimitValues.map((value) => formatSummaryNumber(value, 0)).join(", ")
      : "";

  const toggleSort = (key: SortKey) => {
    setSortConfig((current) => {
      const direction: SortDirection = current?.key === key && current.direction === "asc" ? "desc" : "asc";
      return { key, direction };
    });
    setPage(1);
  };

  const sortLabel = (key: SortKey) => {
    if (sortConfig?.key !== key) return "Sort";
    return sortConfig.direction === "asc" ? "Sorted ascending" : "Sorted descending";
  };

  const sortGlyph = (key: SortKey) => {
    if (sortConfig?.key !== key) return "↕";
    return sortConfig.direction === "asc" ? "↑" : "↓";
  };

  const downloadDatasetCsv = useCallback(() => {
    if (sortedRows.length === 0) return;
    const headers = [
      "row",
      "phase",
      "project",
      "project_id",
      "run_id",
      "hard_cap",
      "relative_quality_score",
      "convergence_score",
      "score_reference_step",
      "loss_at_reference_step_run",
      "loss_at_reference_step_base",
      "selected_multipliers",
      "selected_log_multipliers",
      "x_features",
      "metadata",
      "source",
    ];
    const lines = [
      headers.map(csvCell).join(","),
      ...sortedRows.map((row, index) => {
        const fields = [
          index + 1,
          row.is_baseline_run ? "Ph1" : "Ph2+",
          row.project_name || row.project_id || "",
          row.project_id || "",
          row.run_id || row.run_name || "",
          isHardCapPenaltyRow(row) ? "Yes" : "",
          row.relative_quality_score ?? "",
          row.convergence_score ?? "",
          row.score_reference_step ?? "",
          row.loss_at_reference_step_run ?? "",
          row.loss_at_reference_step_base ?? "",
          row.selected_multipliers || {},
          row.selected_log_multipliers || {},
          cleanXFeatures(row.x_features),
          index === 0 ? row.metadata || {} : "",
          index === 0 ? row.source || {} : "",
        ];
        return fields.map(csvCell).join(",");
      }),
    ];
    const filename = `${safeFileToken(dataset?.training_data_id || activePipelineId || "training-data")}_dataset_rows.csv`;
    downloadTextFile(filename, lines.join("\n"));
  }, [activePipelineId, dataset?.training_data_id, sortedRows]);

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-950">{title}</h2>
          <p className="mt-1 text-sm text-slate-600">{description}</p>
        </div>
      </div>

      {allowPipelineSelection && (
        <div className="mb-4 rounded-lg border border-slate-200 bg-slate-50 p-3">
          <div className="grid gap-3 lg:grid-cols-[1fr_auto] lg:items-end">
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wide text-slate-500">
                Training Data Target
              </label>
              <select
                value={selectedPipelineId}
                onChange={(event) => setSelectedPipelineId(event.target.value)}
                disabled={pipelinesLoading || building || loading}
                className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100 disabled:opacity-60"
              >
                {!selectedPipelineId && <option value="">Select a data preparation pipeline</option>}
                {pipelines.map((pipeline) => (
                  <option key={pipeline.id} value={pipeline.id}>
                    {pipeline.name || pipeline.id} ({pipeline.id})
                  </option>
                ))}
              </select>
              <p className="mt-1 text-xs text-slate-500">
                Select the target first, then build or refresh the final training dataset.
              </p>
            </div>
            <div className="flex flex-wrap items-center justify-end gap-2">
              <button
                onClick={() => setBuildModalOpen(true)}
                disabled={building || loading || !activePipelineId}
                className="inline-flex items-center gap-1.5 rounded-lg bg-blue-600 px-3 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-60"
              >
                <RefreshCw className={`h-4 w-4 ${building ? "animate-spin" : ""}`} />
                {rowCount > 0 ? "Rebuild Dataset" : "Build Dataset"}
              </button>
              <button
                onClick={() => void loadDataset()}
                disabled={loading || !activePipelineId}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-60"
              >
                <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
                Refresh
              </button>
              <button
                onClick={downloadDatasetCsv}
                disabled={previewRows.length === 0}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-60"
              >
                <Download className="h-4 w-4" />
                Download CSV
              </button>
            </div>
          </div>
        </div>
      )}

      {(buildError || buildMessage || (Array.isArray(summary.errors) && summary.errors.length > 0)) && (
        <div
          className={`mb-4 rounded-lg border px-3 py-2 text-sm ${
            buildError || (Array.isArray(summary.errors) && summary.errors.length > 0)
              ? "border-red-200 bg-red-50 text-red-700"
              : "border-emerald-200 bg-emerald-50 text-emerald-700"
          }`}
        >
          {buildError || buildMessage || summary.errors.slice(0, 5).join(" | ")}
        </div>
      )}

      {!allowPipelineSelection && (
        <div className="mb-4 flex flex-wrap items-center justify-end gap-2">
          <button
            onClick={() => setBuildModalOpen(true)}
            disabled={building || loading || !activePipelineId}
            className="inline-flex items-center gap-1.5 rounded-lg bg-blue-600 px-3 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-60"
          >
            <RefreshCw className={`h-4 w-4 ${building ? "animate-spin" : ""}`} />
            {rowCount > 0 ? "Rebuild Dataset" : "Build Dataset"}
          </button>
          <button
            onClick={() => void loadDataset()}
            disabled={loading || !activePipelineId}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-60"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </button>
          <button
            onClick={downloadDatasetCsv}
            disabled={previewRows.length === 0}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-60"
          >
            <Download className="h-4 w-4" />
            Download CSV
          </button>
        </div>
      )}

      {allowPipelineSelection && !activePipelineId ? (
        <div className="rounded-lg border border-dashed border-slate-300 p-5 text-sm text-slate-600">
          Select a data preparation pipeline before building training data.
        </div>
      ) : loading ? (
        <div className="rounded-lg border border-slate-200 p-4 text-sm text-slate-500">Loading prepared dataset...</div>
      ) : !dataset || rowCount === 0 ? (
        <div className="rounded-lg border border-dashed border-slate-300 p-5">
          <div className="mb-2 flex h-10 w-10 items-center justify-center rounded-lg bg-blue-50 text-blue-700">
            <Database className="h-5 w-5" />
          </div>
          <h3 className="text-sm font-semibold text-slate-900">No prepared dataset available yet</h3>
          <p className="mt-1 text-sm text-slate-600">
            This page is reserved for the finalized training_data output. The builder flow will create or refresh this dataset.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
            <span className="inline-flex items-center gap-1">
              <span className="font-semibold uppercase tracking-wide text-slate-500">Rows</span>
              <span className="text-sm font-bold text-slate-950">{rowCount}</span>
            </span>
            {hardCapPenaltyRows > 0 && (
              <span className="inline-flex items-center gap-1 rounded border border-orange-200 bg-orange-50 px-2 py-1 text-orange-800">
                <span className="font-semibold uppercase tracking-wide">Hard Cap</span>
                <span className="text-sm font-bold">{hardCapPenaltyRows}</span>
                <span className="font-mono text-[10px]">
                  Penalty {formatSummaryNumber(hardCapPenalty)}
                </span>
                {hardCapLimitLabel && (
                  <span className="font-mono text-[10px]" title="Gaussian hard cap limit captured from the rows used in this dataset build">
                    Gaussian {hardCapLimitLabel}
                  </span>
                )}
              </span>
            )}
            <span className="inline-flex items-center gap-1">
              <span className="font-semibold uppercase tracking-wide text-slate-500">Summary</span>
              <span className="text-sm font-bold text-slate-950">{Object.keys(summary).length}</span>
            </span>
            <span className="min-w-0 flex-1 truncate font-mono text-[10px] text-slate-500" title={filePath}>
              {filePath || "-"}
            </span>
          </div>

          {previewRows.length > 0 && (
            <div className="overflow-hidden rounded-lg border border-slate-200">
              <div className="border-b border-slate-200 bg-slate-50 px-2 py-1.5 text-sm font-semibold text-slate-800">
                Dataset Rows
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full text-left text-[11px]">
                  <thead className="bg-white text-slate-500">
                    <tr>
                      <th className="border-b border-slate-200 px-2 py-1.5 text-center font-semibold">Phase</th>
                      <th className="border-b border-slate-200 px-2 py-1.5 font-semibold">
                        <button
                          type="button"
                          onClick={() => toggleSort("project")}
                          title={sortLabel("project")}
                          className="inline-flex items-center gap-1 font-semibold text-slate-600 hover:text-slate-950"
                        >
                          Project <span className="text-[10px]">{sortGlyph("project")}</span>
                        </button>
                      </th>
                      <th className="w-[190px] border-b border-slate-200 px-2 py-1.5 font-semibold">Run ID</th>
                      <th className="w-[68px] border-b border-slate-200 px-1 py-1.5 text-center font-semibold">
                        <button
                          type="button"
                          onClick={() => toggleSort("hard_cap")}
                          title={sortLabel("hard_cap")}
                          className="inline-flex items-center gap-0.5 whitespace-nowrap font-semibold text-slate-600 hover:text-slate-950"
                        >
                          Hard Cap <span className="text-[10px]">{sortGlyph("hard_cap")}</span>
                        </button>
                      </th>
                      <th className="border-b border-slate-200 px-2 py-1.5 text-right font-semibold">
                        <button
                          type="button"
                          onClick={() => toggleSort("relative_quality_score")}
                          title={sortLabel("relative_quality_score")}
                          className="inline-flex items-center gap-1 font-semibold text-slate-600 hover:text-slate-950"
                        >
                          R_quality <span className="text-[10px]">{sortGlyph("relative_quality_score")}</span>
                        </button>
                      </th>
                      <th className="border-b border-slate-200 px-2 py-1.5 text-right font-semibold">
                        <button
                          type="button"
                          onClick={() => toggleSort("convergence_score")}
                          title={sortLabel("convergence_score")}
                          className="inline-flex items-center gap-1 font-semibold text-slate-600 hover:text-slate-950"
                        >
                          R_conv <span className="text-[10px]">{sortGlyph("convergence_score")}</span>
                        </button>
                      </th>
                      <th className="border-b border-slate-200 px-2 py-1.5 font-semibold">Multipliers</th>
                      <th className="border-b border-slate-200 px-2 py-1.5 font-semibold">Features</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visibleRows.map((row, index) => {
                      const absoluteRowIndex = (page - 1) * ROWS_PER_PAGE + index + 1;
                      const projectName = row.project_name || row.project_id || "";
                      const runName = row.run_id || row.run_name || "";
                      const isHardCapPenaltyRow = Boolean(row.metadata?.is_hard_cap_penalty_row);
                      return (
                        <tr key={`${runName}-${absoluteRowIndex}`} className="border-b border-slate-100 last:border-0 hover:bg-slate-50">
                          <td className="px-2 py-1.5 text-center">
                            <span className={`inline-flex rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                              row.is_baseline_run ? "bg-green-100 text-green-800" : "bg-orange-100 text-orange-800"
                            }`}>
                              {row.is_baseline_run ? "Ph1" : "Ph2+"}
                            </span>
                          </td>
                          <td className="max-w-[150px] truncate px-2 py-1.5 text-slate-700" title={projectName}>
                            {projectName || "-"}
                          </td>
                          <td className="w-[190px] max-w-[190px] truncate px-2 py-1.5 font-mono text-[10px] text-slate-600" title={runName}>
                            {runName || "-"}
                          </td>
                          <td className="w-[68px] px-1 py-1.5 text-center">
                            {isHardCapPenaltyRow ? (
                              <span className="inline-flex rounded bg-orange-100 px-1.5 py-0.5 text-[10px] font-semibold text-orange-800">
                                Yes
                              </span>
                            ) : (
                              <span className="text-slate-300">-</span>
                            )}
                          </td>
                          <td className={`px-2 py-1.5 text-right font-semibold ${
                            row.relative_quality_score > 0 ? "text-green-700" : row.relative_quality_score < 0 ? "text-red-700" : "text-slate-700"
                          }`}>
                            <div>{formatMetric(row.relative_quality_score)}</div>
                            {isHardCapPenaltyRow && (
                              <div className="mt-0.5 text-[9px] font-semibold uppercase tracking-wide text-orange-700">Hard cap penalty</div>
                            )}
                          </td>
                          <td className={`px-2 py-1.5 text-right font-semibold ${
                            row.convergence_score > 0 ? "text-green-700" : row.convergence_score < 0 ? "text-red-700" : "text-slate-700"
                          }`}>
                            {formatMetric(row.convergence_score)}
                          </td>
                          <td className="px-2 py-1.5">
                            <button
                              onClick={() => setDetailModal({
                                column: "selected_multipliers",
                                data: row.selected_multipliers || {},
                                project: projectName,
                                rowData: row,
                                rowIndex: absoluteRowIndex,
                                run: runName,
                              })}
                              className="rounded border border-blue-200 bg-blue-50 px-2 py-0.5 text-[10px] font-semibold text-blue-700 hover:bg-blue-100"
                            >
                              {countObjectValues(row.selected_multipliers)} values
                            </button>
                          </td>
                          <td className="px-2 py-1.5">
                            <button
                              onClick={() => setDetailModal({
                                column: "x_features",
                                data: cleanXFeatures(row.x_features),
                                project: projectName,
                                rowData: row,
                                rowIndex: absoluteRowIndex,
                                run: runName,
                              })}
                              className="rounded border border-blue-200 bg-blue-50 px-2 py-0.5 text-[10px] font-semibold text-blue-700 hover:bg-blue-100"
                            >
                              {countObjectValues(cleanXFeatures(row.x_features))} features
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div className="flex items-center justify-between gap-3 border-t border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
                <span>
                  Showing {showingFrom} to {showingTo} of {sortedRows.length} rows
                </span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setPage((current) => Math.max(1, current - 1))}
                    disabled={page === 1}
                    className="rounded border border-slate-200 bg-white px-2 py-1 font-semibold text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                  >
                    Previous
                  </button>
                  <span className="rounded border border-slate-200 bg-white px-2 py-1 font-semibold text-slate-700">
                    Page {page} of {totalPages}
                  </span>
                  <button
                    onClick={() => setPage((current) => Math.min(totalPages, current + 1))}
                    disabled={page >= totalPages}
                    className="rounded border border-slate-200 bg-white px-2 py-1 font-semibold text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                  >
                    Next
                  </button>
                </div>
              </div>
            </div>
          )}

          {Object.keys(summary).length > 0 && (
            <details className="rounded-lg border border-slate-200 bg-slate-50">
              <summary className="cursor-pointer px-3 py-2 text-sm font-semibold text-slate-800">Dataset Summary</summary>
              <pre className="overflow-auto border-t border-slate-200 bg-slate-950 p-3 text-xs text-slate-100">
                {JSON.stringify(summary, null, 2)}
              </pre>
            </details>
          )}
        </div>
      )}
      {buildModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={() => !building && setBuildModalOpen(false)}>
          <div
            className="flex max-h-[88vh] w-full max-w-xl flex-col overflow-hidden rounded-lg bg-white shadow-xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-center justify-between gap-3 border-b border-slate-200 px-4 py-3">
              <div className="min-w-0">
                <h3 className="text-sm font-semibold text-slate-950">{rowCount > 0 ? "Rebuild Dataset" : "Build Dataset"}</h3>
                <p className="mt-0.5 truncate text-xs text-slate-500">{activePipelineId}</p>
              </div>
              <button
                onClick={() => setBuildModalOpen(false)}
                disabled={building}
                className="rounded p-1 text-slate-500 hover:bg-slate-100 hover:text-slate-800 disabled:opacity-50"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
              <label className="flex items-start gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
                <input
                  type="checkbox"
                  checked={includeHardCapPenaltyRows}
                  onChange={(event) => setIncludeHardCapPenaltyRows(event.target.checked)}
                  disabled={building}
                  className="mt-0.5 h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500 disabled:opacity-60"
                />
                <span className="min-w-0">
                  <span className="block text-sm font-semibold text-slate-800">Include hard-cap penalty rows</span>
                  <span className="mt-1 block text-xs leading-5 text-slate-500">
                    Hard-capped runs did not finish normally, but they are useful negative examples. During this build only, each one gets the lowest normal relative quality score found in the dataset. Offline scoring and normalization are not changed.
                  </span>
                </span>
              </label>
              {hardCapPenaltyRows > 0 && (
                <div className="mt-3 rounded-lg border border-orange-200 bg-orange-50 px-3 py-2 text-xs text-orange-800">
                  Current dataset includes {hardCapPenaltyRows} hard-cap penalty row{hardCapPenaltyRows === 1 ? "" : "s"}
                  {typeof hardCapPenalty === "number" ? ` at ${hardCapPenalty.toFixed(6)}` : ""}.
                </div>
              )}
            </div>
            <div className="sticky bottom-0 flex items-center justify-end gap-2 border-t border-slate-200 bg-white px-4 py-3">
              <button
                type="button"
                onClick={() => setBuildModalOpen(false)}
                disabled={building}
                className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-60"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => void buildDataset(includeHardCapPenaltyRows)}
                disabled={building || loading || !activePipelineId}
                className="inline-flex items-center gap-1.5 rounded-lg bg-blue-600 px-3 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-60"
              >
                <RefreshCw className={`h-4 w-4 ${building ? "animate-spin" : ""}`} />
                {building ? "Building" : rowCount > 0 ? "Rebuild Dataset" : "Build Dataset"}
              </button>
            </div>
          </div>
        </div>
      )}
      {detailModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={() => setDetailModal(null)}>
          <div className="flex max-h-[88vh] w-full max-w-4xl flex-col overflow-hidden rounded-lg bg-white shadow-xl" onClick={(event) => event.stopPropagation()}>
            <div className="flex items-center justify-between gap-3 border-b border-slate-200 px-4 py-3">
              <div className="min-w-0">
                <h3 className="text-sm font-semibold text-slate-950">{detailModal.column}</h3>
                <p className="mt-0.5 truncate text-xs text-slate-500">
                  Row {detailModal.rowIndex}
                  {detailModal.project ? ` - ${detailModal.project}` : ""}
                  {detailModal.run ? ` - ${detailModal.run}` : ""}
                </p>
              </div>
              <button onClick={() => setDetailModal(null)} className="rounded p-1 text-slate-500 hover:bg-slate-100 hover:text-slate-800">
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto p-4">
            {isObjectValue(detailModal.data) && !Array.isArray(detailModal.data) ? (
              <>
                <div className="mb-2 text-xs text-slate-500">
                  {Object.keys(detailModal.data).length} fields,{" "}
                  {new Set(Object.values(detailModal.data).map((value) => isObjectValue(value) ? JSON.stringify(value) : String(value))).size} unique values
                </div>
                {detailModal.column === "selected_multipliers" ? (
                  <div className="max-h-[70vh] space-y-3 overflow-auto">
                    {Object.entries(MULTIPLIER_GROUPS).map(([groupKey, group]) => {
                      const groupValue = detailModal.data?.[groupKey];
                      const logValues = detailModal.rowData?.selected_log_multipliers || {};
                      const logValue = logValues[groupKey];
                      return (
                        <div key={groupKey} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                          <div className="mb-2 flex items-center justify-between gap-3">
                            <div>
                              <h4 className="text-sm font-semibold text-slate-950">{group.label}</h4>
                              <p className="font-mono text-[10px] text-slate-500">{groupKey}</p>
                            </div>
                            <div className="text-right">
                              <div className="rounded bg-blue-50 px-2 py-1 font-mono text-xs font-bold text-blue-700">
                                {formatModalValue(groupValue)}
                              </div>
                              {logValue !== undefined && (
                                <div className="mt-1 font-mono text-[10px] text-slate-500">
                                  log: {formatModalValue(logValue)}
                                </div>
                              )}
                            </div>
                          </div>
                          <div className="rounded border border-slate-200 bg-white px-2 py-1.5">
                            <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-slate-500">
                              Applies To
                            </div>
                            <div className="flex flex-wrap gap-1">
                              {group.params.map((paramKey) => (
                                <span key={paramKey} className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[10px] text-slate-700">
                                  {paramKey}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : detailModal.column === "x_features" ? (
                  <div className="max-h-[70vh] overflow-auto">
                    <div className="grid gap-1 md:grid-cols-3 lg:grid-cols-4">
                      {Object.entries(detailModal.data)
                        .sort(([aKey], [bKey]) => {
                          const aUsed = COMPACT_FEATURE_KEYS.has(aKey) ? 0 : 1;
                          const bUsed = COMPACT_FEATURE_KEYS.has(bKey) ? 0 : 1;
                          return aUsed - bUsed || aKey.localeCompare(bKey);
                        })
                        .map(([key, value]) => {
                          const usedByCompact = COMPACT_FEATURE_KEYS.has(key);
                          return (
                            <div
                              key={key}
                              className={`flex min-w-0 items-center justify-between gap-2 rounded border px-2 py-1 ${
                                usedByCompact
                                  ? "border-emerald-200 bg-emerald-50"
                                  : "border-slate-200 bg-slate-50"
                              }`}
                            >
                              <span
                                className={`truncate font-mono text-[10px] ${usedByCompact ? "text-emerald-800" : "text-slate-500"}`}
                                title={key}
                              >
                                {key}
                              </span>
                              <span
                                className={`shrink-0 font-mono text-[10px] font-semibold ${
                                  usedByCompact ? "text-emerald-950" : "text-slate-900"
                                }`}
                                title={formatModalValue(value)}
                              >
                                {formatModalValue(value)}
                              </span>
                            </div>
                          );
                        })}
                    </div>
                    <div className="mt-3 flex flex-wrap gap-2 border-t border-slate-200 pt-3 text-[10px] font-semibold text-slate-600">
                      <span className="rounded border border-emerald-200 bg-emerald-50 px-2 py-1 text-emerald-800">
                        Green: compact featurewise descriptor
                      </span>
                      <span className="rounded border border-slate-200 bg-slate-50 px-2 py-1 text-slate-600">
                        Gray: extra stored feature
                      </span>
                    </div>
                  </div>
                ) : (
                  <div className="grid max-h-[70vh] gap-1 overflow-auto md:grid-cols-3 lg:grid-cols-4">
                    {Object.entries(detailModal.data).map(([key, value]) => (
                      <div key={key} className="flex min-w-0 items-center justify-between gap-2 rounded border border-slate-200 bg-slate-50 px-2 py-1">
                        <span className="truncate font-mono text-[10px] text-slate-500" title={key}>{key}</span>
                        <span className="shrink-0 font-mono text-[10px] font-semibold text-slate-900" title={formatModalValue(value)}>
                          {formatModalValue(value)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <pre className="max-h-[70vh] overflow-auto rounded bg-slate-950 p-3 text-xs text-slate-100">
                {JSON.stringify(detailModal.data, null, 2)}
              </pre>
            )}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

