import { BarChart3, RefreshCw, X } from "lucide-react";
import { useMemo, useState } from "react";
import { api } from "../../api/client";
import type { PipelineDetail } from "./types";

interface CandidateCheckRow {
  candidate_log_multiplier?: number | null;
  candidate_multiplier?: number | null;
  predicted_score?: number | null;
  predicted_reward?: number | null;
  selected?: boolean;
}

interface PredictionRow {
  project_id?: string | null;
  project_name?: string | null;
  model_id?: string | null;
  model_name?: string | null;
  mode?: string | null;
  status?: string | null;
  error?: string | null;
  selected_preset?: string | null;
  selected_multipliers?: Record<string, unknown> | null;
  selected_log_multipliers?: Record<string, unknown> | null;
  features?: Record<string, unknown> | null;
  effective_params?: Record<string, unknown> | null;
  score_spreads?: Record<string, unknown> | null;
  candidate_points?: number | null;
  candidate_score_checks?: Record<string, CandidateCheckRow[]> | null;
  n_runs?: number | null;
  has_signal?: boolean | null;
}

interface TestingPredictionsPanelProps {
  pipeline: PipelineDetail;
  selectedModelId?: string | null;
}

const GROUPS = [
  { key: "geometry_lr_mult", label: "Geometry LR", members: ["position_lr_init_mult", "scaling_lr_mult", "rotation_lr_mult"] },
  { key: "appearance_lr_mult", label: "Appearance LR", members: ["feature_lr_mult", "opacity_lr_mult", "lambda_dssim_mult"] },
  { key: "densification_mult", label: "Densification", members: ["densify_grad_threshold_mult", "opacity_threshold_mult"] },
];

const EFFECTIVE_PARAM_KEYS = [
  "position_lr_init",
  "position_lr_final",
  "scaling_lr",
  "rotation_lr",
  "feature_lr",
  "opacity_lr",
  "lambda_dssim",
  "densify_grad_threshold",
  "opacity_threshold",
];

const DESCRIPTOR_KEYS = [
  "gsd_median",
  "overlap_proxy",
  "coverage_spread",
  "camera_angle_bucket",
  "heading_consistency",
  "texture_density",
  "blur_motion_risk",
  "terrain_roughness_proxy",
  "vegetation_cover_percentage",
  "vegetation_complexity_score",
];

function formatNumber(value: unknown, digits = 4) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(3);
  return value.toFixed(digits);
}

function compactId(value: string | null | undefined) {
  if (!value) return "-";
  return value.length > 42 ? `${value.slice(0, 18)}...${value.slice(-16)}` : value;
}

function getConfiguredModelIds(pipeline: PipelineDetail): string[] {
  const ids = Array.isArray(pipeline.config?.source_model_ids)
    ? pipeline.config.source_model_ids
    : pipeline.config?.source_model_id
      ? [pipeline.config.source_model_id]
      : [];
  return ids.map((id: unknown) => String(id || "").trim()).filter(Boolean);
}

function addModelLabel(labels: Map<string, string>, rawId: unknown, rawName: unknown) {
  const id = String(rawId || "").trim();
  const name = String(rawName || "").trim();
  if (id && name && name !== id) labels.set(id, name);
}

function collectModelLabels(pipeline: PipelineDetail, rows: PredictionRow[]): Map<string, string> {
  const labels = new Map<string, string>();
  rows.forEach((row) => addModelLabel(labels, row.model_id, row.model_name));

  const config = pipeline.config || {};
  const possibleLists = [
    config.source_models,
    config.models,
    config.model_records,
    config.workflow_models,
    config.selected_models,
  ];
  possibleLists.forEach((list) => {
    if (!Array.isArray(list)) return;
    list.forEach((item) => {
      if (!item || typeof item !== "object") return;
      addModelLabel(
        labels,
        item.model_id || item.id || item.source_model_id,
        item.model_name || item.name || item.label,
      );
    });
  });

  const possibleMaps = [
    config.source_model_names,
    config.model_names,
    config.model_name_by_id,
  ];
  possibleMaps.forEach((map) => {
    if (!map || typeof map !== "object" || Array.isArray(map)) return;
    Object.entries(map).forEach(([id, name]) => addModelLabel(labels, id, name));
  });

  return labels;
}

function modelLabel(modelId: string | null | undefined, labels: Map<string, string>) {
  if (!modelId) return "-";
  return labels.get(modelId) || compactId(modelId);
}

function latestPreviewRows(pipeline: PipelineDetail): PredictionRow[] {
  const latestPreviewKey = String(pipeline.latest_prediction_preview_key || "").trim();
  const previews = pipeline.prediction_previews || {};
  const latest = latestPreviewKey ? previews?.[latestPreviewKey] : null;
  const rows = Array.isArray(latest?.results) ? latest.results : Array.isArray(latest?.rows) ? latest.rows : [];
  return rows.filter((row: unknown): row is PredictionRow => !!row && typeof row === "object");
}

function valueForGroup(source: Record<string, unknown> | null | undefined, groupKey: string) {
  if (!source) return null;
  const direct = source[groupKey];
  if (typeof direct === "number") return direct;
  const group = GROUPS.find((entry) => entry.key === groupKey);
  if (!group) return null;
  for (const member of group.members) {
    const value = source[member];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function keyValueList(source: Record<string, unknown> | null | undefined, keys: string[], digits = 4) {
  if (!source) return <span className="text-slate-400">-</span>;
  const pairs = keys
    .map((key) => [key, source[key]] as const)
    .filter(([, value]) => value !== undefined && value !== null);
  if (pairs.length === 0) return <span className="text-slate-400">-</span>;
  return (
    <div className="space-y-0.5 font-mono text-[10px] leading-tight text-slate-700">
      {pairs.map(([key, value]) => (
        <div key={key} className="grid grid-cols-[135px_minmax(64px,1fr)] gap-2">
          <span className="truncate text-slate-500" title={key}>{key}</span>
          <span>{typeof value === "number" ? formatNumber(value, digits) : String(value)}</span>
        </div>
      ))}
    </div>
  );
}

function CandidateMiniChart({ checks }: { checks: CandidateCheckRow[] }) {
  const points = checks
    .map((entry, index) => {
      const multiplier = typeof entry.candidate_multiplier === "number"
        ? entry.candidate_multiplier
        : typeof entry.candidate_log_multiplier === "number"
          ? Math.exp(entry.candidate_log_multiplier)
          : null;
      const score = typeof entry.predicted_score === "number"
        ? entry.predicted_score
        : typeof entry.predicted_reward === "number"
          ? entry.predicted_reward
          : null;
      return multiplier && score !== null ? { index, multiplier, score, selected: !!entry.selected } : null;
    })
    .filter((point): point is { index: number; multiplier: number; score: number; selected: boolean } => !!point);

  if (points.length === 0) {
    return <div className="rounded-md border border-dashed border-slate-300 p-4 text-xs text-slate-500">No candidate scores recorded.</div>;
  }

  const width = 720;
  const height = 180;
  const pad = { left: 44, right: 14, top: 14, bottom: 34 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const scores = points.map((point) => point.score);
  const logs = points.map((point) => Math.log(Math.max(point.multiplier, 1e-12)));
  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);
  const minLog = Math.min(...logs);
  const maxLog = Math.max(...logs);
  const scaleX = (multiplier: number) =>
    pad.left + ((Math.log(Math.max(multiplier, 1e-12)) - minLog) / (maxLog - minLog || 1)) * plotW;
  const scaleY = (score: number) =>
    pad.top + plotH - ((score - minScore) / (maxScore - minScore || 1)) * plotH;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-48 w-full rounded-md border border-slate-200 bg-white">
      <line x1={pad.left} y1={pad.top} x2={pad.left} y2={pad.top + plotH} stroke="#cbd5e1" />
      <line x1={pad.left} y1={pad.top + plotH} x2={pad.left + plotW} y2={pad.top + plotH} stroke="#cbd5e1" />
      <polyline
        points={points.map((point) => `${scaleX(point.multiplier)},${scaleY(point.score)}`).join(" ")}
        fill="none"
        stroke="#2563eb"
        strokeWidth="2"
      />
      {points.map((point) => (
        <circle
          key={point.index}
          cx={scaleX(point.multiplier)}
          cy={scaleY(point.score)}
          r={point.selected ? 5 : 3}
          fill={point.selected ? "#f59e0b" : "#2563eb"}
          stroke="#fff"
          strokeWidth="1.5"
        >
          <title>{`multiplier=${formatNumber(point.multiplier, 6)} score=${formatNumber(point.score, 6)}`}</title>
        </circle>
      ))}
      <text x={pad.left} y={height - 10} fill="#64748b" fontSize="11">multiplier</text>
      <text x="8" y={pad.top + 10} fill="#64748b" fontSize="11">score</text>
    </svg>
  );
}

function CandidateModal({ row, modelLabels, onClose }: { row: PredictionRow; modelLabels: Map<string, string>; onClose: () => void }) {
  const [group, setGroup] = useState(GROUPS[0].key);
  const checks = Array.isArray(row.candidate_score_checks?.[group]) ? row.candidate_score_checks[group] : [];
  const selected = valueForGroup(row.selected_multipliers, group);
  const selectedLog = valueForGroup(row.selected_log_multipliers, group);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/55 p-4">
      <div className="flex max-h-[90vh] w-full max-w-5xl flex-col overflow-hidden rounded-lg bg-white shadow-xl">
        <div className="flex items-start justify-between gap-4 border-b border-slate-200 px-5 py-4">
          <div className="min-w-0">
            <h3 className="text-base font-semibold text-slate-950">Multiplier Selection</h3>
            <p className="mt-1 truncate text-xs text-slate-600">
              {row.project_name || row.project_id || "Project"} - {row.model_name || modelLabel(row.model_id, modelLabels)}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-slate-200 text-slate-600 hover:bg-slate-50"
            title="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="overflow-auto p-5">
          <div className="mb-4 grid gap-3 md:grid-cols-[260px_minmax(0,1fr)]">
            <label className="text-xs font-semibold text-slate-700">
              Group
              <select
                value={group}
                onChange={(event) => setGroup(event.target.value)}
                className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1.5 text-xs font-normal"
              >
                {GROUPS.map((entry) => (
                  <option key={entry.key} value={entry.key}>{entry.label}</option>
                ))}
              </select>
            </label>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div className="rounded-md border border-slate-200 bg-slate-50 p-2">
                <div className="text-slate-500">Selected multiplier</div>
                <div className="font-mono font-semibold text-slate-900">{formatNumber(selected, 6)}</div>
              </div>
              <div className="rounded-md border border-slate-200 bg-slate-50 p-2">
                <div className="text-slate-500">Log multiplier</div>
                <div className="font-mono font-semibold text-slate-900">{formatNumber(selectedLog, 6)}</div>
              </div>
              <div className="rounded-md border border-slate-200 bg-slate-50 p-2">
                <div className="text-slate-500">Score spread</div>
                <div className="font-mono font-semibold text-slate-900">{formatNumber(row.score_spreads?.[group], 6)}</div>
              </div>
            </div>
          </div>
          <CandidateMiniChart checks={checks} />
          <div className="mt-4 max-h-72 overflow-auto rounded-md border border-slate-200">
            <table className="w-full min-w-[680px] text-xs">
              <thead className="sticky top-0 bg-slate-100 text-slate-700">
                <tr>
                  <th className="px-2 py-1.5 text-left font-semibold">Candidate</th>
                  <th className="px-2 py-1.5 text-left font-semibold">Log Multiplier</th>
                  <th className="px-2 py-1.5 text-left font-semibold">Multiplier</th>
                  <th className="px-2 py-1.5 text-left font-semibold">Predicted Score</th>
                  <th className="px-2 py-1.5 text-left font-semibold">Selected</th>
                </tr>
              </thead>
              <tbody>
                {checks.map((entry, index) => (
                  <tr key={`${group}-${index}`} className={`border-t border-slate-100 ${entry.selected ? "bg-amber-50" : ""}`}>
                    <td className="px-2 py-1.5 text-slate-700">{index + 1}</td>
                    <td className="px-2 py-1.5 font-mono text-slate-700">{formatNumber(entry.candidate_log_multiplier, 6)}</td>
                    <td className="px-2 py-1.5 font-mono text-slate-700">{formatNumber(entry.candidate_multiplier, 6)}</td>
                    <td className="px-2 py-1.5 font-mono text-slate-700">{formatNumber(entry.predicted_score ?? entry.predicted_reward, 6)}</td>
                    <td className="px-2 py-1.5 text-slate-700">{entry.selected ? "yes" : "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function TestingPredictionsPanel({ pipeline, selectedModelId }: TestingPredictionsPanelProps) {
  const hardCapRuns = Number(pipeline.hard_cap_runs || 0);
  const configuredModels = getConfiguredModelIds(pipeline);
  const initialModelId = selectedModelId && configuredModels.includes(selectedModelId)
    ? selectedModelId
    : configuredModels[0] || "";
  const [modelId, setModelId] = useState(initialModelId);
  const [refreshing, setRefreshing] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [modalRow, setModalRow] = useState<PredictionRow | null>(null);

  const rows = useMemo(() => latestPreviewRows(pipeline), [pipeline]);
  const modelLabels = useMemo(() => collectModelLabels(pipeline, rows), [pipeline, rows]);
  const displayRows = modelId ? rows.filter((row) => row.model_id === modelId) : rows;
  const latestPreviewKey = String(pipeline.latest_prediction_preview_key || "").trim();
  const latestPreview = latestPreviewKey ? pipeline.prediction_previews?.[latestPreviewKey] : null;
  const generatedAt = typeof latestPreview?.generated_at === "string" ? latestPreview.generated_at : null;

  const predictForModel = async () => {
    if (!modelId || refreshing) return;
    setRefreshing(true);
    setMessage(null);
    try {
      await api.post(`/api/workflow/pipelines/${pipeline.id}/predict-multipliers`, { model_id: modelId });
      setMessage("Prediction preview updated. Refreshing pipeline details...");
      window.setTimeout(() => window.location.reload(), 450);
    } catch (err: any) {
      setMessage(err.response?.data?.detail?.message || err.response?.data?.detail || "Failed to update prediction preview.");
    } finally {
      setRefreshing(false);
    }
  };

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-950">Predictions</h2>
          <p className="mt-1 text-sm text-slate-600">
            Project-level multiplier predictions, descriptors, candidate score checks, and final parameter values.
          </p>
          {generatedAt && <p className="mt-1 text-xs text-slate-500">Preview generated: {new Date(generatedAt).toLocaleString()}</p>}
          {hardCapRuns > 0 && (
            <p className="mt-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
              {hardCapRuns} run(s) reached Gaussian hard cap.
            </p>
          )}
        </div>
        <div className="flex flex-wrap items-end gap-2">
          <label className="text-xs font-semibold text-slate-700">
            Model
            <select
              value={modelId}
              onChange={(event) => setModelId(event.target.value)}
              className="mt-1 w-80 max-w-full rounded-md border border-slate-300 px-2 py-1.5 text-xs font-normal"
            >
              {configuredModels.length === 0 && <option value="">No models configured</option>}
              {configuredModels.map((id) => (
                <option key={id} value={id}>{modelLabel(id, modelLabels)}</option>
              ))}
            </select>
          </label>
          <button
            type="button"
            onClick={() => void predictForModel()}
            disabled={!modelId || refreshing}
            className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
            title="Run prediction preview for the selected model"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${refreshing ? "animate-spin" : ""}`} />
            {refreshing ? "Predicting" : "Predict"}
          </button>
        </div>
      </div>

      {message && <div className="mb-3 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-700">{message}</div>}

      {rows.length === 0 ? (
        <div className="rounded-lg border border-dashed border-slate-300 p-6 text-sm text-slate-500">
          No prediction preview rows available yet. Select a model and run Predict.
        </div>
      ) : displayRows.length === 0 ? (
        <div className="rounded-lg border border-dashed border-slate-300 p-6 text-sm text-slate-500">
          No prediction rows for model <span className="font-mono text-xs">{modelId}</span>. Run Predict for this model.
        </div>
      ) : (
        <div className="overflow-auto rounded-lg border border-slate-200">
          <table className="w-full min-w-[1900px] text-[11px] leading-tight">
            <thead className="sticky top-0 z-10 bg-slate-100 text-slate-700">
              <tr>
                <th className="px-2 py-1.5 text-left font-semibold">Project</th>
                <th className="px-2 py-1.5 text-left font-semibold">Model</th>
                <th className="px-2 py-1.5 text-left font-semibold">Mode</th>
                <th className="px-2 py-1.5 text-left font-semibold">Selected Multipliers</th>
                <th className="px-2 py-1.5 text-left font-semibold">Log Multipliers</th>
                <th className="px-2 py-1.5 text-left font-semibold">Descriptors Used</th>
                <th className="px-2 py-1.5 text-left font-semibold">Final Values Used</th>
                <th className="px-2 py-1.5 text-left font-semibold">Evidence</th>
                <th className="px-2 py-1.5 text-left font-semibold">Status</th>
              </tr>
            </thead>
            <tbody>
              {displayRows.map((row, index) => (
                <tr key={`${row.project_name}-${row.model_id}-${index}`} className="border-t border-slate-100 align-top">
                  <td className="px-2 py-1.5 align-top font-semibold text-slate-800">
                    <div>{row.project_name || row.project_id || "-"}</div>
                    {row.project_id && <div className="font-mono text-[10px] font-normal text-slate-500">{row.project_id}</div>}
                  </td>
                  <td className="px-2 py-1.5 align-top text-slate-700" title={row.model_id || ""}>
                    <div className="font-semibold">{row.model_name || modelLabel(row.model_id, modelLabels)}</div>
                    {row.model_id && <div className="font-mono text-[10px] text-slate-500">{compactId(row.model_id)}</div>}
                  </td>
                  <td className="px-2 py-1.5 align-top text-slate-700">
                    <div>{row.selected_preset || "-"}</div>
                    <div className="font-mono text-[10px] text-slate-500">{row.mode || "-"}</div>
                  </td>
                  <td className="px-2 py-1.5 align-top">
                    {keyValueList(
                      row.selected_multipliers,
                      GROUPS.flatMap((group) => group.members),
                      6,
                    )}
                  </td>
                  <td className="px-2 py-1.5 align-top">
                    {keyValueList(
                      row.selected_log_multipliers,
                      GROUPS.flatMap((group) => group.members),
                      6,
                    )}
                  </td>
                  <td className="px-2 py-1.5 align-top">
                    {keyValueList(row.features, DESCRIPTOR_KEYS, 4)}
                  </td>
                  <td className="px-2 py-1.5 align-top">
                    {keyValueList(row.effective_params, EFFECTIVE_PARAM_KEYS, 8)}
                  </td>
                  <td className="px-2 py-1.5 align-top">
                    <button
                      type="button"
                      onClick={() => setModalRow(row)}
                      disabled={!row.candidate_score_checks}
                      className="inline-flex items-center gap-1.5 rounded-md border border-blue-200 bg-blue-50 px-2 py-1 text-xs font-semibold text-blue-700 hover:bg-blue-100 disabled:border-slate-200 disabled:bg-slate-50 disabled:text-slate-400"
                    >
                      <BarChart3 className="h-3.5 w-3.5" />
                      Chart
                    </button>
                    <div className="mt-1 text-[10px] text-slate-500">
                      {row.candidate_points || 0} candidates, n={row.n_runs ?? "-"}, signal={row.has_signal === false ? "no" : "yes"}
                    </div>
                  </td>
                  <td className="max-w-[180px] px-2 py-1.5 align-top text-slate-700">
                    {row.status === "ok" ? (
                      <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-[10px] font-semibold text-emerald-700">OK</span>
                    ) : (
                      <span className="break-words text-red-700">{row.error || row.status || "error"}</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {modalRow && <CandidateModal row={modalRow} modelLabels={modelLabels} onClose={() => setModalRow(null)} />}
    </section>
  );
}
