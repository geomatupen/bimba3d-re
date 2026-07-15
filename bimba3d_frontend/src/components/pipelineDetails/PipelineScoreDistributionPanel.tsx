import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../../api/client";
interface PipelineScoreDistributionPanelProps {
  pipelineId: string;
  refreshKey?: string | number | null;
  selectedModelId?: string | null;
  title?: string;
}

interface ScoreRow {
  candidate_score_checks?: Record<string, unknown> | null;
  final_loss?: number | null;
  final_lpips?: number | null;
  final_psnr?: number | null;
  final_ssim?: number | null;
  is_baseline_row?: boolean;
  model_id?: string | null;
  project_id?: string | null;
  project_name?: string | null;
  score?: number | null;
  source_model_id?: string | null;
  relative_quality_score?: number | null;
  run_id?: string | null;
  selected_log_multipliers?: Record<string, unknown> | null;
  selected_multipliers?: Record<string, unknown> | null;
  test_model_id?: string | null;
}

interface PipelineDetail {
  config?: {
    pipeline_type?: string;
    shared_config?: Record<string, unknown>;
    test_candidate_log_multipliers?: Record<string, unknown>;
  };
}

interface PlotPoint {
  isBaseline?: boolean;
  label?: string;
  logX?: number | null;
  project: string;
  run: string;
  x: number;
  y: number;
}

const GROUPS = [
  { key: "geometry_lr_mult", label: "Geometry", minKey: "geometry_log_multiplier_min", maxKey: "geometry_log_multiplier_max", defaultBounds: [0.2, 5] },
  { key: "appearance_lr_mult", label: "Appearance", minKey: "appearance_log_multiplier_min", maxKey: "appearance_log_multiplier_max", defaultBounds: [0.2, 5] },
  { key: "densification_mult", label: "Densification", minKey: "densification_log_multiplier_min", maxKey: "densification_log_multiplier_max", defaultBounds: [0.44, 2.29] },
];

const GROUP_MEMBER_KEYS: Record<string, string[]> = {
  appearance_lr_mult: ["feature_lr_mult", "opacity_lr_mult", "lambda_dssim_mult"],
  densification_mult: ["densify_grad_threshold_mult", "opacity_threshold_mult"],
  geometry_lr_mult: ["position_lr_init_mult", "scaling_lr_mult", "rotation_lr_mult"],
};

const numeric = (value: unknown): number | null => (typeof value === "number" && Number.isFinite(value) ? value : null);

const scoreOf = (row: ScoreRow): number | null => numeric(row.relative_quality_score);

const projectKey = (row: ScoreRow): string => String(row.project_name || row.project_id || "");

const modelKey = (row: ScoreRow): string => String(row.model_id || row.test_model_id || row.source_model_id || "");

function FullscreenIcon({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="#334155" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M8 3H3v5" />
      <path d="M16 3h5v5" />
      <path d="M21 16v5h-5" />
      <path d="M3 16v5h5" />
    </svg>
  );
}

const objectOf = (value: unknown): Record<string, unknown> => {
  if (!value) return {};
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
    } catch {
      return {};
    }
  }
  return typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
};

const numericFromGroup = (source: Record<string, unknown>, groupKey: string): number | null => {
  const direct = numeric(source[groupKey]);
  if (direct !== null) return direct;
  for (const key of GROUP_MEMBER_KEYS[groupKey] || []) {
    const value = numeric(source[key]);
    if (value !== null) return value;
  }
  return null;
};

function pointsForGroup(rows: ScoreRow[], groupKey: string): PlotPoint[] {
  return rows
    .map<PlotPoint | null>((row) => {
      const selectedLogMultipliers = objectOf(row.selected_log_multipliers);
      const selectedMultipliers = objectOf(row.selected_multipliers);
      const rawMultiplier = numericFromGroup(selectedMultipliers, groupKey);
      const logValue = numericFromGroup(selectedLogMultipliers, groupKey) ?? (rawMultiplier !== null ? Math.log(Math.max(rawMultiplier, 1e-12)) : null);
      const score = scoreOf(row);
      if (rawMultiplier === null || score === null) return null;
      return {
        logX: logValue,
        project: row.project_name || row.project_id || "unknown",
        run: row.run_id || "",
        x: rawMultiplier,
        y: score,
      };
    })
    .filter((item): item is PlotPoint => item !== null);
}

const formatTick = (value: number): string => {
  if (Math.abs(value) >= 10) return value.toFixed(0);
  if (Math.abs(value) >= 1) return value.toFixed(2);
  return value.toFixed(3);
};

const formatRealLogTick = (value: number): string[] => [formatTick(value), `ln ${formatTick(Math.log(Math.max(value, 1e-12)))}`];

const logSpaceTicks = ([min, max]: [number, number]): number[] => {
  const safeMin = Math.max(min, 1e-12);
  const safeMax = Math.max(max, safeMin + 1e-12);
  const logMin = Math.log(safeMin);
  const logMax = Math.log(safeMax);
  const candidates = [
    safeMin,
    Math.exp(logMin + (logMax - logMin) * 0.25),
    Math.exp(logMin + (logMax - logMin) * 0.5),
    Math.exp(logMin + (logMax - logMin) * 0.75),
    safeMax,
  ];
  if (safeMin < 1 && safeMax > 1) {
    candidates.push(1);
  }
  return candidates
    .sort((a, b) => a - b)
    .filter((value, index, arr) => index === 0 || Math.abs(value - arr[index - 1]) > Math.max(value * 0.015, 1e-6));
};

const paddedRange = (min: number, max: number): [number, number] => {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (Math.abs(max - min) < 1e-12) {
    const pad = Math.max(Math.abs(min) * 0.1, 0.01);
    return [min - pad, max + pad];
  }
  const pad = (max - min) * 0.08;
  return [min - pad, max + pad];
};

function MiniScatter({ bounds, fullscreen, points }: { bounds?: [number, number]; fullscreen?: boolean; points: PlotPoint[] }) {
  if (points.length === 0) {
    return <div className="rounded border border-dashed border-slate-300 p-3 text-xs text-slate-500">No score points available.</div>;
  }

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const [minX, maxX] = bounds || paddedRange(Math.min(...xs), Math.max(...xs));
  // Always include y=0 so the "equal to baseline" reference line is always visible.
  const rawMinY = Math.min(0, ...ys);
  const rawMaxY = Math.max(0, ...ys);
  const [minY, maxY] = paddedRange(rawMinY, rawMaxY);
  const best = points.reduce((acc, point) => (point.y > acc.y ? point : acc), points[0]);
  const width = fullscreen ? 1280 : 340;
  const height = fullscreen ? 520 : 220;
  const plot = { left: 54, right: 18, top: 18, bottom: 54 };
  const plotWidth = width - plot.left - plot.right;
  const plotHeight = height - plot.top - plot.bottom;
  const xTicks = logSpaceTicks([minX, maxX]);
  const yTicks = [minY, (minY + maxY) / 2, maxY];
  const logMinX = Math.log(Math.max(minX, 1e-12));
  const logMaxX = Math.log(Math.max(maxX, minX + 1e-12));
  const scaleX = (value: number) => {
    const logValue = Math.log(Math.max(value, 1e-12));
    return plot.left + ((logValue - logMinX) / (logMaxX - logMinX || 1)) * plotWidth;
  };
  const scaleY = (value: number) => plot.top + plotHeight - ((value - minY) / (maxY - minY || 1)) * plotHeight;
  // Zero line is always shown as the "equal to baseline" reference.
  const zeroY = scaleY(0);

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} className={`${fullscreen ? "h-[calc(100vh-13rem)] min-h-[28rem]" : "h-56"} w-full rounded border border-slate-200 bg-white`}>
        <rect x={plot.left} y={plot.top} width={plotWidth} height={plotHeight} fill="#f8fafc" />
        {yTicks.map((tick) => {
          const y = scaleY(tick);
          return (
            <g key={`y-${tick}`}>
              <line x1={plot.left} x2={plot.left + plotWidth} y1={y} y2={y} stroke="#e2e8f0" strokeWidth="1" />
              <text x={plot.left - 8} y={y + 3} textAnchor="end" className="fill-slate-500 text-[10px]">
                {formatTick(tick)}
              </text>
            </g>
          );
        })}
        {xTicks.map((tick, index) => {
          const x = scaleX(tick);
          const labelTick = index === 0 || index === Math.floor(xTicks.length / 2) || index === xTicks.length - 1;
          return (
            <g key={`x-${tick}`}>
              <line x1={x} x2={x} y1={plot.top} y2={plot.top + plotHeight} stroke="#e2e8f0" strokeWidth="1" />
              {labelTick && (
                <>
                  <text x={x} y={plot.top + plotHeight + 15} textAnchor="middle" className="fill-slate-500 text-[10px]">
                    {formatRealLogTick(tick)[0]}
                  </text>
                  <text x={x} y={plot.top + plotHeight + 28} textAnchor="middle" className="fill-slate-400 text-[9px]">
                    {formatRealLogTick(tick)[1]}
                  </text>
                </>
              )}
            </g>
          );
        })}
        <line x1={plot.left} x2={plot.left + plotWidth} y1={plot.top + plotHeight} y2={plot.top + plotHeight} stroke="#94a3b8" strokeWidth="1" />
        <line x1={plot.left} x2={plot.left} y1={plot.top} y2={plot.top + plotHeight} stroke="#94a3b8" strokeWidth="1" />
        {/* Baseline quality reference — always shown at score=0 */}
        {zeroY >= plot.top && zeroY <= plot.top + plotHeight && (
          <g>
            <line
              x1={plot.left}
              x2={plot.left + plotWidth}
              y1={zeroY}
              y2={zeroY}
              stroke="#f97316"
              strokeDasharray="4 3"
              strokeWidth="1.5"
            />
            <text x={plot.left + plotWidth - 4} y={zeroY - 4} textAnchor="end" className="fill-orange-600 text-[10px] font-medium">
              0 Rquality (baseline)
            </text>
          </g>
        )}
        <text x={plot.left + plotWidth / 2} y={height - 12} textAnchor="middle" className="fill-slate-700 text-[11px] font-medium">
          Real multiplier value, with ln(multiplier) shown below
        </text>
        <text x="13" y={plot.top + plotHeight / 2} textAnchor="middle" transform={`rotate(-90 13 ${plot.top + plotHeight / 2})`} className="fill-slate-700 text-[11px] font-medium">
          Rquality
        </text>
        {points.map((point, index) => (
          <circle
            key={`${point.run}-${index}`}
            cx={scaleX(point.x)}
            cy={scaleY(point.y)}
            r={point === best ? 4 : 3}
            fill={point === best ? "#16a34a" : "#2563eb"}
            opacity={point === best ? 1 : 0.72}
          >
            <title>{`${point.project}\n${point.run}\nMultiplier: ${point.x.toFixed(6)}\nln(multiplier): ${(point.logX ?? Math.log(Math.max(point.x, 1e-12))).toFixed(6)}\nRquality: ${point.y.toFixed(6)}`}</title>
          </circle>
        ))}
      </svg>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-600">
        <div className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-blue-600 opacity-75" />
          Phase/test run
        </div>
        <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-green-600" />
            Best Rquality
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-0 w-5 border-t-2 border-dashed border-orange-500" />
            Baseline (score=0)
          </div>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] text-slate-600">
        <div>
          Best Rquality: <span className="font-semibold text-green-700">{best.y.toFixed(6)}</span>
        </div>
        <div title="Real multiplier value sampled from the configured log-space range.">
          Best multiplier: <span className="font-mono font-semibold text-slate-900">{best.x.toFixed(6)}</span>
        </div>
      </div>
    </div>
  );
}

const METRICS = [
  { key: "final_loss", label: "Loss", direction: "lower is better" },
  { key: "final_psnr", label: "PSNR", direction: "higher is better" },
  { key: "final_ssim", label: "SSIM", direction: "higher is better" },
  { key: "final_lpips", label: "LPIPS", direction: "lower is better" },
] as const;

function metricPoints(rows: ScoreRow[], metricKey: (typeof METRICS)[number]["key"]): PlotPoint[] {
  return rows
    .map<PlotPoint | null>((row) => {
      const value = numeric(row[metricKey]);
      if (value === null) return null;
      return {
        isBaseline: !!row.is_baseline_row,
        label: row.is_baseline_row ? "Baseline" : "Run",
        project: row.project_name || row.project_id || "unknown",
        run: row.run_id || "",
        x: 0,
        y: value,
      };
    })
    .filter((item): item is PlotPoint => item !== null);
}

function MetricStripChart({
  fullscreen,
  metric,
  onFullscreen,
  points,
}: {
  fullscreen?: boolean;
  metric: (typeof METRICS)[number];
  onFullscreen?: () => void;
  points: PlotPoint[];
}) {
  if (points.length === 0) {
    return <div className="rounded border border-dashed border-slate-300 p-3 text-xs text-slate-500">No {metric.label} values available.</div>;
  }

  const projects = Array.from(new Set(points.map((point) => point.project)));
  const ys = points.map((point) => point.y);
  const [minY, maxY] = paddedRange(Math.min(...ys), Math.max(...ys));
  const projectLaneWidth = fullscreen ? 108 : 92;
  const width = Math.max(fullscreen ? 720 : 360, projects.length * projectLaneWidth + 54);
  const height = fullscreen ? 590 : 275;
  const plot = { left: 46, right: 8, top: 18, bottom: fullscreen ? 154 : 112 };
  const plotWidth = width - plot.left - plot.right;
  const plotHeight = height - plot.top - plot.bottom;
  const yTicks = [minY, (minY + maxY) / 2, maxY];
  const scaleY = (value: number) => plot.top + plotHeight - ((value - minY) / (maxY - minY || 1)) * plotHeight;
  const scaleX = (project: string, isBaseline?: boolean, runIndex = 0) => {
    const groupIndex = Math.max(projects.indexOf(project), 0);
    const groupWidth = plotWidth / Math.max(projects.length, 1);
    const groupStart = plot.left + groupIndex * groupWidth;
    const center = groupStart + groupWidth / 2;
    if (isBaseline) return center - Math.min(groupWidth * 0.18, 18);
    const jitter = ((runIndex % 5) - 2) * Math.min(groupWidth * 0.035, 4);
    return center + Math.min(groupWidth * 0.18, 18) + jitter;
  };
  const runCounters = new Map<string, number>();
  const rotateLabels = projects.length > 4;

  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <div>
          <div className="text-sm font-semibold text-slate-950">{metric.label}</div>
          <div className="text-[11px] text-slate-500">{metric.direction}</div>
        </div>
        {onFullscreen && (
          <button
            type="button"
            onClick={onFullscreen}
            title="Open chart fullscreen"
            aria-label={`Open ${metric.label} chart fullscreen`}
            className="inline-flex h-7 w-7 items-center justify-center rounded border border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
          >
            <FullscreenIcon />
          </button>
        )}
      </div>
      <div className="overflow-x-auto pb-1">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{ minWidth: `${width}px`, width: "100%" }}
        className="h-auto max-w-none rounded border border-slate-200 bg-white"
      >
        <rect x={plot.left} y={plot.top} width={plotWidth} height={plotHeight} fill="#f8fafc" />
        {yTicks.map((tick) => {
          const y = scaleY(tick);
          return (
            <g key={`metric-y-${metric.key}-${tick}`}>
              <line x1={plot.left} x2={plot.left + plotWidth} y1={y} y2={y} stroke="#e2e8f0" />
              <text x={plot.left - 8} y={y + 3} textAnchor="end" className="fill-slate-500 text-[10px]">
                {formatTick(tick)}
              </text>
            </g>
          );
        })}
        <line x1={plot.left} x2={plot.left + plotWidth} y1={plot.top + plotHeight} y2={plot.top + plotHeight} stroke="#94a3b8" />
        <line x1={plot.left} x2={plot.left} y1={plot.top} y2={plot.top + plotHeight} stroke="#94a3b8" />
        {projects.map((project, index) => {
          const groupWidth = plotWidth / Math.max(projects.length, 1);
          const separatorX = plot.left + index * groupWidth;
          const x = scaleX(project);
          return (
            <g key={project}>
              {index > 0 && (
                <line
                  x1={separatorX}
                  x2={separatorX}
                  y1={plot.top}
                  y2={plot.top + plotHeight}
                  stroke="#cbd5e1"
                  strokeDasharray="3 3"
                />
              )}
              <text
                x={x}
                y={plot.top + plotHeight + 18}
                textAnchor={rotateLabels ? "end" : "middle"}
                transform={rotateLabels ? `rotate(-35 ${x} ${plot.top + plotHeight + 18})` : undefined}
                className="fill-slate-500 text-[9px]"
              >
                {project.length > 16 ? `${project.slice(0, 15)}...` : project}
              </text>
              <title>{project}</title>
            </g>
          );
        })}
        <text x={plot.left + plotWidth / 2} y={height - 18} textAnchor="middle" className="fill-slate-500 text-[10px]">
          Scroll horizontally to inspect all project groups. Hover points for full project and run names.
        </text>
        <text x="14" y={plot.top + plotHeight / 2} textAnchor="middle" transform={`rotate(-90 14 ${plot.top + plotHeight / 2})`} className="fill-slate-700 text-[11px] font-medium">
          {metric.label}
        </text>
        {points.map((point, index) => {
          const runIndex = runCounters.get(point.project) || 0;
          if (!point.isBaseline) runCounters.set(point.project, runIndex + 1);
          const x = scaleX(point.project, point.isBaseline, runIndex);
          return (
            <circle
              key={`${metric.key}-${point.project}-${point.run}-${index}`}
              cx={x}
              cy={scaleY(point.y)}
              r={point.isBaseline ? 4 : 3}
              fill={point.isBaseline ? "#f59e0b" : "#2563eb"}
              opacity={point.isBaseline ? 1 : 0.76}
            >
              <title>{`${point.project}\n${point.run}\n${metric.label}: ${point.y.toFixed(6)}\n${point.label}`}</title>
            </circle>
          );
        })}
      </svg>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-600">
        <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-amber-500" />Baseline</span>
        <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-blue-600 opacity-75" />Non-baseline run</span>
      </div>
    </div>
  );
}

export default function PipelineScoreDistributionPanel({
  pipelineId,
  refreshKey,
  selectedModelId,
  title = "Relative Score Distribution",
}: PipelineScoreDistributionPanelProps) {
  const [rows, setRows] = useState<ScoreRow[]>([]);
  const [pipelineDetail, setPipelineDetail] = useState<PipelineDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedProject, setSelectedProject] = useState("all");
  const [metricSelectedProject, setMetricSelectedProject] = useState("all");
  const [metricsFullscreen, setMetricsFullscreen] = useState(false);
  const [scoreFullscreenGroupKey, setScoreFullscreenGroupKey] = useState<string | null>(null);
  const [fullscreenMetricKey, setFullscreenMetricKey] = useState<(typeof METRICS)[number]["key"] | null>(null);

  const loadRows = useCallback(async () => {
    setLoading(true);
    try {
      const [rowsRes, pipelineRes] = await Promise.all([
        api.get(`/api/workflow/pipelines/${pipelineId}/learning-rows`),
        api.get(`/api/workflow/pipelines/${pipelineId}`),
      ]);
      setRows(Array.isArray(rowsRes.data?.rows) ? rowsRes.data.rows : []);
      setPipelineDetail(pipelineRes.data || null);
    } catch (err) {
      console.error("Failed to load score rows", err);
      setRows([]);
      setPipelineDetail(null);
    } finally {
      setLoading(false);
    }
  }, [pipelineId]);

  // Reload when the pipeline is refreshed (updated_at changes) or on initial mount.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { void loadRows(); }, [loadRows, refreshKey]);

  const displayRows = useMemo(() => {
    if (!selectedModelId) return rows;
    const projectsWithSelectedModel = new Set(
      rows
        .filter((row) => !row.is_baseline_row && modelKey(row) === selectedModelId)
        .map(projectKey)
        .filter(Boolean),
    );
    return rows.filter((row) => {
      if (row.is_baseline_row) return projectsWithSelectedModel.has(projectKey(row));
      return modelKey(row) === selectedModelId;
    });
  }, [rows, selectedModelId]);

  const projects = useMemo(
    () => Array.from(new Set(displayRows.map((row) => row.project_name || row.project_id).filter(Boolean) as string[])).sort(),
    [displayRows],
  );

  useEffect(() => {
    if (selectedProject !== "all" && !projects.includes(selectedProject)) {
      setSelectedProject("all");
    }
    if (metricSelectedProject !== "all" && !projects.includes(metricSelectedProject)) {
      setMetricSelectedProject("all");
    }
  }, [metricSelectedProject, projects, selectedProject]);

  const filteredRows = useMemo(
    () =>
      displayRows.filter((row) => {
        if (row.is_baseline_row) return false;
        if (selectedProject === "all") return true;
        return (row.project_name || row.project_id) === selectedProject;
      }),
    [displayRows, selectedProject],
  );
  const metricRows = useMemo(
    () =>
      displayRows.filter((row) => {
        if (metricSelectedProject === "all") return true;
        return (row.project_name || row.project_id) === metricSelectedProject;
      }),
    [displayRows, metricSelectedProject],
  );
  const groupBounds = useMemo(() => {
    const pipelineType = String(pipelineDetail?.config?.pipeline_type || "").toLowerCase();
    const candidateLogMultipliers = pipelineDetail?.config?.test_candidate_log_multipliers || {};
    const sharedConfig = pipelineDetail?.config?.shared_config || {};
    return GROUPS.reduce<Record<string, [number, number]>>((acc, group) => {
      const candidateLogs = candidateLogMultipliers[group.key];
      if (pipelineType === "test" && Array.isArray(candidateLogs)) {
        const candidateValues = candidateLogs
          .filter((value): value is number => typeof value === "number" && Number.isFinite(value))
          .map((value) => Math.exp(value))
          .filter((value) => Number.isFinite(value) && value > 0);
        if (candidateValues.length > 0) {
          acc[group.key] = [Math.min(...candidateValues), Math.max(...candidateValues)];
          return acc;
        }
      }
      const min = numeric(sharedConfig[group.minKey]);
      const max = numeric(sharedConfig[group.maxKey]);
      const fallback = group.defaultBounds as [number, number];
      acc[group.key] = min !== null && max !== null && max > min ? [min, max] : fallback;
      return acc;
    }, {});
  }, [pipelineDetail]);

  const renderMetricChartGrid = (fullscreen = false) => (
    <div className={`grid gap-3 ${fullscreen ? "grid-cols-1" : "xl:grid-cols-2"}`}>
      {METRICS.map((metric) => (
        <MetricStripChart
          key={metric.key}
          fullscreen={fullscreen}
          metric={metric}
          onFullscreen={() => setFullscreenMetricKey(metric.key)}
          points={metricPoints(metricRows, metric.key)}
        />
      ))}
    </div>
  );
  const fullscreenMetric = METRICS.find((metric) => metric.key === fullscreenMetricKey) || null;
  const fullscreenScoreGroup = GROUPS.find((group) => group.key === scoreFullscreenGroupKey) || null;

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-950">{title}</h2>
          <p className="mt-1 text-sm text-slate-600">
            Quality by selected multiplier group. Baseline rows are excluded; each point is a non-baseline run compared with its own project baseline.
          </p>
          {selectedModelId && (
            <p className="mt-1 text-xs text-amber-700">
              Filtered: <span className="font-mono">{selectedModelId}</span>
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <select
            value={selectedProject}
            onChange={(event) => setSelectedProject(event.target.value)}
            className="rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
          >
            <option value="all">All projects</option>
            {projects.map((project) => (
              <option key={project} value={project}>
                {project}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => void loadRows()}
            disabled={loading}
            title="Reload score data"
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-60"
          >
            <svg className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" /><path d="M3 3v5h5" /></svg>
            Refresh
          </button>
        </div>
      </div>

      {loading ? (
        <div className="rounded border border-slate-200 p-3 text-sm text-slate-500">Loading score rows...</div>
      ) : (
        <div className="space-y-5">
          <div className="grid gap-3 lg:grid-cols-3">
            {GROUPS.map((group) => (
              <div key={group.key} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div className="mb-2 flex items-start justify-between gap-2">
                  <div>
                    <div className="text-sm font-semibold text-slate-950">{group.label}</div>
                    <div className="font-mono text-[10px] text-slate-500">{group.key}</div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setScoreFullscreenGroupKey(group.key)}
                    title="Open chart fullscreen"
                    aria-label={`Open ${group.label} score chart fullscreen`}
                    className="inline-flex h-7 w-7 items-center justify-center rounded border border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                  >
                    <FullscreenIcon />
                  </button>
                </div>
                <MiniScatter bounds={groupBounds[group.key]} points={pointsForGroup(filteredRows, group.key)} />
              </div>
            ))}
          </div>

          <div className="border-t border-slate-200 pt-4">
            <div className="mb-3 space-y-3">
              <div>
                <h3 className="text-base font-semibold text-slate-950">Metric Distribution By Run</h3>
                <p className="text-sm text-slate-600">
                  Grouped dot charts compare each project baseline with its non-baseline runs.
                </p>
              </div>
              <div className="flex w-full items-center gap-2">
                <select
                  value={metricSelectedProject}
                  onChange={(event) => setMetricSelectedProject(event.target.value)}
                  className="min-w-0 flex-1 rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
                >
                  <option value="all">All projects</option>
                  {projects.map((project) => (
                    <option key={project} value={project}>
                      {project}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={() => setMetricsFullscreen(true)}
                  title="Open all metric charts fullscreen"
                  aria-label="Open all metric charts fullscreen"
                  className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                >
                  <FullscreenIcon className="h-5 w-5" />
                </button>
              </div>
            </div>
            <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-600">
              <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-amber-500" />Baseline</span>
              <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-blue-600 opacity-75" />Non-baseline run</span>
            </div>
            {renderMetricChartGrid()}
          </div>
        </div>
      )}

      {metricsFullscreen && (
        <div className="fixed inset-0 z-50 bg-slate-950/70 p-4">
          <div className="flex h-full flex-col rounded-lg bg-white shadow-2xl">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 px-4 py-3">
              <div>
                <h3 className="text-base font-semibold text-slate-950">Metric Distribution By Run</h3>
                <p className="text-xs text-slate-600">
                  {metricSelectedProject === "all" ? "All projects" : metricSelectedProject}
                </p>
              </div>
              <div className="flex w-full flex-wrap items-center gap-2">
                <select
                  value={metricSelectedProject}
                  onChange={(event) => setMetricSelectedProject(event.target.value)}
                  className="min-w-0 flex-1 rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
                >
                  <option value="all">All projects</option>
                  {projects.map((project) => (
                    <option key={project} value={project}>
                      {project}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={() => setMetricsFullscreen(false)}
                  className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
                >
                  Close
                </button>
              </div>
            </div>
            <div className="overflow-auto p-4">
              <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-600">
                <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-amber-500" />Baseline</span>
                <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-blue-600 opacity-75" />Non-baseline run</span>
              </div>
              {renderMetricChartGrid(true)}
            </div>
          </div>
        </div>
      )}

      {fullscreenScoreGroup && (
        <div className="fixed inset-0 z-50 bg-slate-950/70 p-4">
          <div className="flex h-full flex-col rounded-lg bg-white shadow-2xl">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 px-4 py-3">
              <div>
                <h3 className="text-base font-semibold text-slate-950">{fullscreenScoreGroup.label}</h3>
                <p className="text-xs text-slate-600">
                  {selectedProject === "all" ? "All projects" : selectedProject} · Rquality by selected multiplier
                </p>
              </div>
              <div className="flex w-full flex-wrap items-center gap-2">
                <select
                  value={selectedProject}
                  onChange={(event) => setSelectedProject(event.target.value)}
                  className="min-w-0 flex-1 rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
                >
                  <option value="all">All projects</option>
                  {projects.map((project) => (
                    <option key={project} value={project}>
                      {project}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={() => setScoreFullscreenGroupKey(null)}
                  className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
                >
                  Close
                </button>
              </div>
            </div>
            <div className="min-h-0 flex-1 overflow-auto p-4">
              <MiniScatter
                bounds={groupBounds[fullscreenScoreGroup.key]}
                fullscreen
                points={pointsForGroup(filteredRows, fullscreenScoreGroup.key)}
              />
            </div>
          </div>
        </div>
      )}

      {fullscreenMetric && (
        <div className="fixed inset-0 z-50 bg-slate-950/70 p-4">
          <div className="flex h-full flex-col rounded-lg bg-white shadow-2xl">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 px-4 py-3">
              <div>
                <h3 className="text-base font-semibold text-slate-950">{fullscreenMetric.label}</h3>
                <p className="text-xs text-slate-600">
                  {fullscreenMetric.direction} · {metricSelectedProject === "all" ? "All projects" : metricSelectedProject}
                </p>
              </div>
              <div className="flex w-full flex-wrap items-center gap-2">
                <select
                  value={metricSelectedProject}
                  onChange={(event) => setMetricSelectedProject(event.target.value)}
                  className="min-w-0 flex-1 rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
                >
                  <option value="all">All projects</option>
                  {projects.map((project) => (
                    <option key={project} value={project}>
                      {project}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={() => setFullscreenMetricKey(null)}
                  className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
                >
                  Close
                </button>
              </div>
            </div>
            <div className="min-h-0 flex-1 overflow-auto p-4">
              <MetricStripChart fullscreen metric={fullscreenMetric} points={metricPoints(metricRows, fullscreenMetric.key)} />
            </div>
          </div>
        </div>
      )}
    </section>
  );
}




