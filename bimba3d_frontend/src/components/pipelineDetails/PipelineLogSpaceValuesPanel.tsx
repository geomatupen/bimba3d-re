import { Hash, Shuffle } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api } from "../../api/client";
import SvgChartExportButton from "../common/SvgChartExportButton";
import type { PipelineDetail } from "./types";

interface PipelineLogSpaceValuesPanelProps {
  pipeline: PipelineDetail;
  predictionRows?: any[];
  selectedModelId?: string | null;
}

type SchedulePatch = Record<string, unknown>;

const GROUP_LABELS: Record<string, string> = {
  appearance_lr: "Appearance",
  appearance_lr_mult: "Appearance",
  scale_lr: "Densification",
  densification_mult: "Densification",
  geometry_lr: "Geometry",
  geometry_lr_mult: "Geometry",
};

const GROUP_BOUNDS: Record<string, { minKey: string; maxKey: string; fallback: [number, number] }> = {
  appearance_lr: { minKey: "appearance_log_multiplier_min", maxKey: "appearance_log_multiplier_max", fallback: [0.2, 5] },
  appearance_lr_mult: { minKey: "appearance_log_multiplier_min", maxKey: "appearance_log_multiplier_max", fallback: [0.2, 5] },
  scale_lr: { minKey: "densification_log_multiplier_min", maxKey: "densification_log_multiplier_max", fallback: [0.44, 2.29] },
  densification_mult: { minKey: "densification_log_multiplier_min", maxKey: "densification_log_multiplier_max", fallback: [0.44, 2.29] },
  geometry_lr: { minKey: "geometry_log_multiplier_min", maxKey: "geometry_log_multiplier_max", fallback: [0.2, 5] },
  geometry_lr_mult: { minKey: "geometry_log_multiplier_min", maxKey: "geometry_log_multiplier_max", fallback: [0.2, 5] },
};

const formatValue = (value: unknown) => {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return value.toFixed(6);
};

const numeric = (value: unknown): number | null => (typeof value === "number" && Number.isFinite(value) ? value : null);

const formatBounds = ([min, max]: [number, number]) => `${formatTick(min)} to ${formatTick(max)}`;

const formatTick = (value: number): string => {
  if (Math.abs(value) >= 10) return value.toFixed(0);
  if (Math.abs(value) >= 1) return value.toFixed(2);
  return value.toFixed(3);
};

function FullscreenChartIcon() {
  return (
    <span aria-hidden="true" style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", fontSize: 16, lineHeight: "16px" }}>
      ⛶
    </span>
  );
}

const logSpaceTicks = ([min, max]: [number, number], divisions = 4): number[] => {
  const safeMin = Math.max(min, 1e-12);
  const safeMax = Math.max(max, safeMin + 1e-12);
  const logMin = Math.log(safeMin);
  const logMax = Math.log(safeMax);
  const candidates = Array.from({ length: divisions + 1 }, (_, index) =>
    Math.exp(logMin + (logMax - logMin) * (index / divisions)),
  );
  if (safeMin < 1 && safeMax > 1) {
    candidates.push(1);
  }
  return candidates
    .sort((a, b) => a - b)
    .filter((value, index, arr) => index === 0 || Math.abs(value - arr[index - 1]) > Math.max(value * 0.015, 1e-6));
};

const groupBounds = (pipeline: PipelineDetail, group: string): [number, number] => {
  const info = GROUP_BOUNDS[group];
  if (!info) return [0.2, 5];
  const config = pipeline.config || {};
  const savedBounds = (config as Record<string, unknown>).fixed_log_space_bounds;
  if (savedBounds && typeof savedBounds === "object") {
    const raw = (savedBounds as Record<string, unknown>)[group];
    if (Array.isArray(raw) && raw.length >= 2) {
      const min = numeric(raw[0]);
      const max = numeric(raw[1]);
      if (min !== null && max !== null && max > min) return [min, max];
    }
  }
  const sharedConfig = pipeline.config?.shared_config || {};
  const min = numeric((sharedConfig as Record<string, unknown>)[info.minKey]);
  const max = numeric((sharedConfig as Record<string, unknown>)[info.maxKey]);
  return min !== null && max !== null && max > min ? [min, max] : info.fallback;
};

const getGroupValues = (source: unknown): Record<string, number[]> => {
  if (!source || typeof source !== "object") return {};
  const result: Record<string, number[]> = {};
  Object.entries(source as Record<string, unknown>).forEach(([key, value]) => {
    if (Array.isArray(value)) {
      result[key] = value.filter((item): item is number => typeof item === "number" && Number.isFinite(item));
    }
  });
  return result;
};

const getCandidateMultiplierValues = (source: unknown): Record<string, number[]> => {
  if (!source || typeof source !== "object") return {};
  const result: Record<string, number[]> = {};
  Object.entries(source as Record<string, unknown>).forEach(([key, value]) => {
    if (Array.isArray(value)) {
      result[key] = value
        .filter((item): item is number => typeof item === "number" && Number.isFinite(item))
        .map((logValue) => Math.exp(logValue));
    }
  });
  return result;
};

const getCandidateLogValues = (source: unknown): Record<string, number[]> => {
  if (!source || typeof source !== "object") return {};
  const result: Record<string, number[]> = {};
  Object.entries(source as Record<string, unknown>).forEach(([key, value]) => {
    if (Array.isArray(value)) {
      result[key] = value.filter((item): item is number => typeof item === "number" && Number.isFinite(item));
    }
  });
  return result;
};

const shuffleList = (values: number[]) => {
  const shuffled = [...values];
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }
  return shuffled;
};

const groupMultiplierValue = (source: unknown, group: string): number | null => {
  if (!source || typeof source !== "object") return null;
  const record = source as Record<string, unknown>;
  const aliases: Record<string, string[]> = {
    appearance_lr_mult: ["appearance_lr_mult", "appearance_lr", "appearance"],
    densification_mult: ["densification_mult", "scale_lr", "densification"],
    geometry_lr_mult: ["geometry_lr_mult", "geometry_lr", "geometry"],
  };
  for (const key of aliases[group] || [group]) {
    const raw = record[key];
    const direct = numeric(raw);
    if (direct !== null) return direct;
    if (raw && typeof raw === "object") {
      const nested = numeric((raw as Record<string, unknown>).multiplier);
      if (nested !== null) return nested;
    }
  }
  return null;
};

const selectedCandidateIndex = (checks: CandidateScoreCheck[], selectedGroupMultiplier: number | null = null) => {
  const selectedByFlag = checks.findIndex((entry) => Boolean(entry?.selected));
  if (selectedByFlag >= 0 || selectedGroupMultiplier === null || selectedGroupMultiplier <= 0) return selectedByFlag;
  const targetLog = Math.log(selectedGroupMultiplier);
  return checks.reduce(
    (best, entry, index) => {
      const multiplier = numeric(entry?.candidate_multiplier);
      const logMultiplier = numeric(entry?.candidate_log_multiplier) ?? (multiplier && multiplier > 0 ? Math.log(multiplier) : null);
      if (logMultiplier === null) return best;
      const distance = Math.abs(logMultiplier - targetLog);
      return distance < best.distance ? { index, distance } : best;
    },
    { index: -1, distance: Number.POSITIVE_INFINITY },
  ).index;
};

type CandidateScoreCheck = {
  candidate_log_multiplier?: number;
  candidate_multiplier?: number;
  predicted_score?: number;
  selected?: boolean;
};

function CandidateScoreChart({
  bounds,
  checks,
  fullscreen,
  group,
  onFullscreen,
  selectedGroupMultiplier,
}: {
  bounds: [number, number];
  checks: CandidateScoreCheck[];
  fullscreen?: boolean;
  group: string;
  onFullscreen?: () => void;
  selectedGroupMultiplier?: number | null;
}) {
  const [svgElement, setSvgElement] = useState<SVGSVGElement | null>(null);
  const captureSvg = useCallback((node: SVGSVGElement | null) => setSvgElement(node), []);

  const points = checks
    .map((entry, index) => ({
      index,
      multiplier: numeric(entry.candidate_multiplier) ?? Math.exp(numeric(entry.candidate_log_multiplier) ?? 0),
      logMultiplier: numeric(entry.candidate_log_multiplier),
      score: numeric(entry.predicted_score),
      selected: Boolean(entry.selected),
    }))
    .filter((point) => point.multiplier > 0 && point.score !== null);

  if (points.length === 0) {
    return <div className="rounded border border-dashed border-slate-300 p-3 text-xs text-slate-500">No model scores available for this project yet.</div>;
  }

  const [boundMinX, boundMaxX] = bounds;
  const minX = Math.min(boundMinX, ...points.map((point) => point.multiplier));
  const maxX = Math.max(boundMaxX, ...points.map((point) => point.multiplier));
  const scores = points.map((point) => point.score as number);
  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);
  const scorePad = Math.max((maxScore - minScore) * 0.12, 1e-6);
  // Always include 0 so baseline reference line is visible.
  const minY = Math.min(0, minScore - scorePad);
  const maxY = Math.max(0, maxScore + scorePad);
  const width = fullscreen ? 1280 : 340;
  const height = fullscreen ? 560 : 244;
  const plot = { left: fullscreen ? 76 : 58, right: fullscreen ? 28 : 16, top: 20, bottom: 78 };
  const plotWidth = width - plot.left - plot.right;
  const plotHeight = height - plot.top - plot.bottom;
  const logMinX = Math.log(Math.max(minX, 1e-12));
  const logMaxX = Math.log(Math.max(maxX, minX + 1e-12));
  const scaleX = (value: number) => {
    const logValue = Math.log(Math.max(value, 1e-12));
    return plot.left + ((logValue - logMinX) / (logMaxX - logMinX || 1)) * plotWidth;
  };
  const scaleY = (value: number) => plot.top + (1 - ((value - minY) / (maxY - minY || 1))) * plotHeight;
  const xTicks = logSpaceTicks([minX, maxX], 8);
  const yTicks = [minScore, (minScore + maxScore) / 2, maxScore];
  const sortedPoints = [...points].sort((a, b) => a.multiplier - b.multiplier);
  const axisY = plot.top + plotHeight;
  const selectedIndex = selectedCandidateIndex(checks, selectedGroupMultiplier ?? null);
  const isPointSelected = (point: { index: number; selected: boolean }) =>
    point.selected || point.index === selectedIndex;
  const selectedPoint = points.find((point) => isPointSelected(point));
  const exportName = `candidate_score_${GROUP_LABELS[group] || group}`;

  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div>
          <div className="text-sm font-semibold text-slate-950">{GROUP_LABELS[group] || group}</div>
          <div className="font-mono text-[10px] text-slate-500">predicted score by candidate</div>
        </div>
        <div className="flex shrink-0 flex-nowrap items-center gap-1.5">
          {onFullscreen && (
            <button
              type="button"
              onClick={onFullscreen}
              title="Open chart fullscreen"
              aria-label={`Open ${GROUP_LABELS[group] || group} candidate score chart fullscreen`}
              className="inline-flex h-7 w-7 items-center justify-center rounded border border-slate-300 bg-slate-50 text-slate-900 shadow-sm hover:bg-white"
            >
              <FullscreenChartIcon />
            </button>
          )}
          <SvgChartExportButton filename={exportName} svgElement={svgElement} />
        </div>
      </div>
      {selectedPoint && (
        <div className="mb-2 rounded border border-emerald-200 bg-emerald-50 px-2 py-1 text-[11px] text-emerald-900">
          <span className="font-semibold">Selected score: </span>
          <span className="font-mono">{formatTick(selectedPoint.score as number)}</span>
          <span className="mx-1 text-emerald-700">|</span>
          <span className="font-semibold">multiplier: </span>
          <span className="font-mono">{formatTick(selectedPoint.multiplier)}</span>
        </div>
      )}
      <svg ref={captureSvg} viewBox={`0 0 ${width} ${height}`} className={`${fullscreen ? "h-[calc(100vh-13rem)] min-h-[28rem]" : "h-64"} w-full rounded border border-slate-200 bg-white`}>
        <rect x={plot.left} y={plot.top} width={plotWidth} height={plotHeight} fill="#f8fafc" />
        {xTicks.map((tick, index) => {
          const x = scaleX(tick);
          const labelTick = index === 0 || index % 2 === 0 || index === xTicks.length - 1;
          return (
            <g key={`${group}-score-x-${tick}`}>
              <line x1={x} x2={x} y1={plot.top} y2={plot.top + plotHeight} stroke="#e2e8f0" />
              {labelTick && (
                <text x={x} y={plot.top + plotHeight + 17} textAnchor="middle" className="fill-slate-500 text-[11px]">
                  {formatTick(tick)}
                </text>
              )}
            </g>
          );
        })}
        {yTicks.map((tick) => {
          const y = scaleY(tick);
          return (
            <g key={`${group}-score-y-${tick}`}>
              <line x1={plot.left} x2={plot.left + plotWidth} y1={y} y2={y} stroke="#e2e8f0" />
              <text x={plot.left - 9} y={y + 4} textAnchor="end" className="fill-slate-500 text-[10px]">
                {formatTick(tick)}
              </text>
            </g>
          );
        })}
        <line x1={plot.left} x2={plot.left} y1={plot.top} y2={axisY} stroke="#94a3b8" />
        <line x1={plot.left} x2={plot.left + plotWidth} y1={axisY} y2={axisY} stroke="#94a3b8" />
        {/* y=0 baseline reference */}
        {(() => {
          const zeroY = scaleY(0);
          if (zeroY < plot.top || zeroY > plot.top + plotHeight) return null;
          return (
            <g>
              <line x1={plot.left} x2={plot.left + plotWidth} y1={zeroY} y2={zeroY}
                stroke="#f97316" strokeDasharray="4 3" strokeWidth="1.5" />
            </g>
          );
        })()}
        <polyline
          fill="none"
          stroke="#2563eb"
          strokeWidth="1.6"
          points={sortedPoints.map((point) => `${scaleX(point.multiplier)},${scaleY(point.score as number)}`).join(" ")}
        />
        {points.map((point) => {
          const x = scaleX(point.multiplier);
          const scoreY = scaleY(point.score as number);
          return isPointSelected(point) ? (
            <line
              key={`${group}-selected-guide-${point.index}`}
              x1={x}
              x2={x}
              y1={scoreY}
              y2={axisY}
              stroke="#16a34a"
              strokeDasharray="3 3"
              strokeWidth="1.2"
              opacity="0.75"
            />
          ) : null;
        })}
        <text x={plot.left + plotWidth / 2} y={height - 38} textAnchor="middle" className="fill-slate-700 text-[12px] font-medium">
          Candidate multiplier (log-scaled)
        </text>
        <text
          x={14}
          y={plot.top + plotHeight / 2}
          textAnchor="middle"
          transform={`rotate(-90 14 ${plot.top + plotHeight / 2})`}
          className="fill-slate-600 text-[12px] font-medium"
        >
          Predicted score
        </text>
        {points.map((point) => {
          const title = [
            `Candidate ${point.index}`,
            `Multiplier: ${point.multiplier.toFixed(6)}`,
            `ln(multiplier): ${(point.logMultiplier ?? Math.log(point.multiplier)).toFixed(6)}`,
            `Predicted score: ${(point.score as number).toFixed(6)}`,
            isPointSelected(point) ? "Selected best" : null,
          ].filter(Boolean).join("\n");
          return (
            <g key={`${group}-score-point-${point.index}`}>
              <circle
                cx={scaleX(point.multiplier)}
                cy={scaleY(point.score as number)}
                r={isPointSelected(point) ? 3.4 : 2.5}
                fill={isPointSelected(point) ? "#16a34a" : "#2563eb"}
                opacity={isPointSelected(point) ? 1 : 0.72}
              >
                <title>{title}</title>
              </circle>
            </g>
          );
        })}
        <g transform={`translate(${plot.left - 36} ${height - 14})`} className="fill-slate-600 text-[10px]">
          <circle cx="0" cy="0" r="3" fill="#2563eb" opacity="0.75" />
          <text x="8" y="3">Predicted score</text>
          <circle cx="110" cy="0" r="3.4" fill="#16a34a" />
          <text x="120" y="3">Selected</text>
          <line x1="178" x2="198" y1="0" y2="0" stroke="#f97316" strokeDasharray="4 3" strokeWidth="1.4" />
          <text x="204" y="3">Baseline score=0</text>
        </g>
      </svg>
    </div>
  );
}

function ScheduleDistributionChart({
  bounds,
  fullscreen,
  group,
  onFullscreen,
  selectedIndex,
  selectedIndices,
  values,
}: {
  bounds: [number, number];
  fullscreen?: boolean;
  group: string;
  onFullscreen?: () => void;
  selectedIndex: number;
  selectedIndices?: Set<number>;
  values: number[];
}) {
  const [svgElement, setSvgElement] = useState<SVGSVGElement | null>(null);
  const captureSvg = useCallback((node: SVGSVGElement | null) => setSvgElement(node), []);

  if (values.length === 0) {
    return <div className="rounded border border-dashed border-slate-300 p-3 text-xs text-slate-500">No schedule values available.</div>;
  }

  const valueMin = Math.min(...values.filter((value) => Number.isFinite(value) && value > 0));
  const valueMax = Math.max(...values.filter((value) => Number.isFinite(value) && value > 0));
  const [boundMinX, boundMaxX] = bounds;
  const minX = Math.min(boundMinX, Number.isFinite(valueMin) ? valueMin : boundMinX);
  const maxX = Math.max(boundMaxX, Number.isFinite(valueMax) ? valueMax : boundMaxX);
  const width = fullscreen ? 1280 : 340;
  const height = fullscreen ? 520 : 220;
  const plot = { left: fullscreen ? 48 : 28, right: fullscreen ? 24 : 8, top: 18, bottom: 54 };
  const plotWidth = width - plot.left - plot.right;
  const plotHeight = height - plot.top - plot.bottom;
  const logMinX = Math.log(Math.max(minX, 1e-12));
  const logMaxX = Math.log(Math.max(maxX, minX + 1e-12));
  const scaleX = (value: number) => {
    const logValue = Math.log(Math.max(value, 1e-12));
    return plot.left + ((logValue - logMinX) / (logMaxX - logMinX || 1)) * plotWidth;
  };
  const centerY = plot.top + plotHeight / 2;
  const xTicks = logSpaceTicks([minX, maxX]);
  const pointGroups = values.reduce<Array<{ value: number; indices: number[]; selected: boolean }>>((acc, value, index) => {
    const key = value.toFixed(6);
    const existing = acc.find((item) => item.value.toFixed(6) === key);
    const indexSelected = index === selectedIndex || Boolean(selectedIndices?.has(index));
    if (existing) {
      existing.indices.push(index);
      existing.selected = existing.selected || indexSelected;
    } else {
      acc.push({ value, indices: [index], selected: indexSelected });
    }
    return acc;
  }, []);
  const labelStride = Math.max(1, Math.ceil(pointGroups.length / 10));
  const exportName = `log_space_schedule_${GROUP_LABELS[group] || group}`;

  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div>
          <div className="text-sm font-semibold text-slate-950">{GROUP_LABELS[group] || group}</div>
          <div className="font-mono text-[10px] text-slate-500">{group}</div>
        </div>
        <div className="flex shrink-0 flex-nowrap items-center gap-1.5">
          {onFullscreen && (
            <button
              type="button"
              onClick={onFullscreen}
              title="Open chart fullscreen"
              aria-label={`Open ${GROUP_LABELS[group] || group} schedule chart fullscreen`}
              className="inline-flex h-7 w-7 items-center justify-center rounded border border-slate-300 bg-slate-50 text-slate-900 shadow-sm hover:bg-white"
            >
              <FullscreenChartIcon />
            </button>
          )}
          <SvgChartExportButton filename={exportName} svgElement={svgElement} />
        </div>
      </div>
      <svg ref={captureSvg} viewBox={`0 0 ${width} ${height}`} className={`${fullscreen ? "h-[calc(100vh-13rem)] min-h-[28rem]" : "h-56"} w-full rounded border border-slate-200 bg-white`}>
        <rect x={plot.left} y={plot.top} width={plotWidth} height={plotHeight} fill="#f8fafc" />
        {xTicks.map((tick, index) => {
          const x = scaleX(tick);
          const labelTick = index === 0 || index === Math.floor(xTicks.length / 2) || index === xTicks.length - 1;
          return (
            <g key={`${group}-tick-${tick}`}>
              <line x1={x} x2={x} y1={plot.top} y2={plot.top + plotHeight} stroke="#e2e8f0" />
              {labelTick && (
                <>
                  <text x={x} y={plot.top + plotHeight + 15} textAnchor="middle" className="fill-slate-500 text-[10px]">
                    {formatTick(tick)}
                  </text>
                  <text x={x} y={plot.top + plotHeight + 28} textAnchor="middle" className="fill-slate-400 text-[9px]">
                    ln {formatTick(Math.log(Math.max(tick, 1e-12)))}
                  </text>
                </>
              )}
            </g>
          );
        })}
        <line x1={plot.left} x2={plot.left + plotWidth} y1={centerY} y2={centerY} stroke="#94a3b8" />
        <line x1={plot.left} x2={plot.left + plotWidth} y1={plot.top + plotHeight} y2={plot.top + plotHeight} stroke="#94a3b8" />
        <text x={plot.left + plotWidth / 2} y={height - 12} textAnchor="middle" className="fill-slate-700 text-[11px] font-medium">
          Real multiplier value, with ln(multiplier) shown below
        </text>
        {pointGroups.map((point, pointIndex) => {
          const x = Math.max(plot.left + 4, Math.min(plot.left + plotWidth - 4, scaleX(point.value)));
          const label = point.indices.length === 1 ? `${point.indices[0]}` : point.indices.map((index) => `${index}`).join(", ");
          const showLabel = point.selected || pointIndex % labelStride === 0 || pointIndex === pointGroups.length - 1;
          const title = [
            label,
            `Multiplier: ${point.value.toFixed(6)}`,
            `ln(multiplier): ${Math.log(Math.max(point.value, 1e-12)).toFixed(6)}`,
            point.selected ? "Selected/current" : null,
          ].filter(Boolean).join("\n");
          return (
            <g key={`${group}-point-${point.value.toFixed(6)}-${point.indices.join("-")}`}>
              {showLabel && (
                <text
                  x={x}
                  y={centerY - 12}
                  textAnchor="middle"
                  className={`text-[9px] font-semibold ${point.selected ? "fill-green-700" : "fill-slate-500"}`}
                >
                  {point.indices.length > 2 ? `${point.indices.length} values` : label}
                </text>
              )}
              <circle
                cx={x}
                cy={centerY}
                r={point.selected ? 5 : point.indices.length > 1 ? 4.5 : 3.5}
                fill={point.selected ? "#16a34a" : "#2563eb"}
                opacity={point.selected ? 1 : 0.68}
              >
                <title>{title}</title>
              </circle>
            </g>
          );
        })}
      </svg>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-600">
        <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-blue-600 opacity-75" />Saved schedule value</span>
        {(selectedIndex >= 0 || (selectedIndices && selectedIndices.size > 0)) && (
          <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-green-600" />Selected by model</span>
        )}
      </div>
    </div>
  );
}

export default function PipelineLogSpaceValuesPanel({ pipeline, predictionRows = [], selectedModelId }: PipelineLogSpaceValuesPanelProps) {
  const [fullscreenGroup, setFullscreenGroup] = useState<string | null>(null);
  const [previewSchedule, setPreviewSchedule] = useState<SchedulePatch | null>(null);
  const [savedSchedule, setSavedSchedule] = useState<SchedulePatch | null>(null);
  const [selectedPredictionKey, setSelectedPredictionKey] = useState("");
  const [scheduleBusy, setScheduleBusy] = useState(false);
  const [scheduleMessage, setScheduleMessage] = useState<{ type: "success" | "error" | "info"; text: string } | null>(null);
  useEffect(() => {
    setSelectedPredictionKey("");
  }, [pipeline.id, selectedModelId]);
  const displayConfig = {
    ...(pipeline.config || {}),
    ...(savedSchedule || {}),
    ...(previewSchedule || {}),
  };
  const pipelineType = String(displayConfig.pipeline_type || pipeline.pipeline_type || "").toLowerCase();
  const isTestPipeline = pipelineType === "test";
  const isPreviewActive = !!previewSchedule;
  const configuredValues = isTestPipeline
    ? getCandidateMultiplierValues(displayConfig.test_candidate_log_multipliers)
    : getGroupValues(displayConfig.pre_generated_log_multipliers);
  const activeRun = pipeline.active_run as { phase?: number; status?: string } | undefined;
  const currentPhase = typeof pipeline.current_phase === "number" ? pipeline.current_phase : null;
  const pipelineStatus = String(pipeline.status || "").toLowerCase();
  const cooldownActive = Boolean(pipeline.cooldown_active);
  const activePhase = Number(activeRun?.phase || currentPhase || 1);
  const baselineActive = activePhase === 1 && (activeRun?.status === "running" || cooldownActive || pipelineStatus === "running");
  const nonBaselineActive = activePhase > 1 && (activeRun?.status === "running" || cooldownActive || pipelineStatus === "running");
  const hasNonBaselineRuns = Array.isArray(pipeline.runs)
    && pipeline.runs.some((run: any) => Number(run?.phase || 0) > 1);
  const scheduleEditBlocked = isTestPipeline && (nonBaselineActive || hasNonBaselineRuns);
  const currentIndex = baselineActive ? -1 : typeof displayConfig.multiplier_current_index === "number" ? displayConfig.multiplier_current_index : -1;
  const scheduleSeed = displayConfig.fixed_log_space_seed;
  const groupSeeds = displayConfig.fixed_log_space_group_seeds;
  const candidateSeed = displayConfig.test_candidate_seed;
  const candidateCount = displayConfig.test_candidate_count;
  const candidateGeneration = displayConfig.test_candidate_generation;
  const generatedAt = displayConfig.fixed_log_space_generated_at;
  const boundsSource = displayConfig.fixed_log_space_bounds_source;
  const usingFallbackBounds = boundsSource === "default_bounds_fallback";
  const activeCandidateChecks = pipeline.active_run?.candidate_score_checks;
  const activeHasCandidateChecks = activeCandidateChecks
    && typeof activeCandidateChecks === "object"
    && Object.values(activeCandidateChecks).some((checks) => Array.isArray(checks) && checks.length > 0);
  const completedRunPredictionRows = Array.isArray(pipeline.runs)
    ? pipeline.runs
        .filter((run: any) => Number(run?.phase || run?.phase_number || 0) > 1)
        .filter((run: any) => {
          const checks = run?.candidate_score_checks;
          return checks && typeof checks === "object" && Object.values(checks).some((items) => Array.isArray(items) && items.length > 0);
        })
        .map((run: any) => ({
          project_name: run.project_name || run.project || run.project_id || "Project",
          model_id: run.model_id || run.source_model_id || run.test_model_id || run.current_test_model_id || run.selected_model_id || "",
          status: "ok",
          run_id: run.run_id,
          selected_preset: run.selected_preset,
          selected_multipliers: run.selected_multipliers || {},
          selected_log_multipliers: run.selected_log_multipliers || {},
          group_multipliers: run.group_multipliers || {},
          group_log_multipliers: run.group_log_multipliers || {},
          candidate_score_checks: run.candidate_score_checks,
          completed_run: true,
        }))
    : [];
  const livePredictionRows = activeHasCandidateChecks
    ? [
        {
          project_name: pipeline.active_run?.project_name || "Active project",
          model_id: pipeline.active_run?.test_model_id || pipeline.current_test_model_id || "",
          status: "ok",
          selected_multipliers: pipeline.active_run?.selected_multipliers || {},
          selected_log_multipliers: pipeline.active_run?.selected_log_multipliers || {},
          group_multipliers: (pipeline.active_run as any)?.group_multipliers || {},
          group_log_multipliers: (pipeline.active_run as any)?.group_log_multipliers || {},
          candidate_score_checks: activeCandidateChecks,
          live: true,
        },
        ...completedRunPredictionRows,
        ...predictionRows,
      ]
    : [...completedRunPredictionRows, ...predictionRows];
  const predictionOptions = isTestPipeline
    ? livePredictionRows
        .map((row: any, index: number) => {
          const rowModelId = String(row?.model_id || row?.test_model_id || row?.source_model_id || "");
          const checksByGroup = row?.candidate_score_checks && typeof row.candidate_score_checks === "object" ? row.candidate_score_checks : {};
          const hasChecks = Object.values(checksByGroup).some((checks) => Array.isArray(checks) && checks.length > 0);
          if (selectedModelId && rowModelId !== selectedModelId) return null;
          if (!hasChecks) return null;
          return {
            key: `${row?.project_name || row?.project_id || "project"}::${rowModelId || "model"}::${index}`,
            label: `${row?.live ? "Live: " : ""}${row?.project_name || row?.project_id || "Project"}${rowModelId ? ` - ${rowModelId}` : ""}`,
            row,
          };
        })
        .filter((option): option is { key: string; label: string; row: any } => Boolean(option))
    : [];
  const effectivePredictionKey = predictionOptions.some((option) => option.key === selectedPredictionKey)
    ? selectedPredictionKey
    : predictionOptions[0]?.key || "";
  const selectedPredictionRow = predictionOptions.find((option) => option.key === effectivePredictionKey)?.row || null;
  const selectedPredictionChecksByGroup = (
    selectedPredictionRow?.candidate_score_checks && typeof selectedPredictionRow.candidate_score_checks === "object"
      ? selectedPredictionRow.candidate_score_checks
      : {}
  ) as Record<string, CandidateScoreCheck[]>;
  const selectedGroupMultipliers = (
    selectedPredictionRow?.group_multipliers && typeof selectedPredictionRow.group_multipliers === "object"
      ? selectedPredictionRow.group_multipliers
      : {}
  );
  const predictionCandidateValues = isTestPipeline && selectedPredictionRow
    ? Object.entries(selectedPredictionChecksByGroup).reduce<Record<string, number[]>>((acc, [group, checks]) => {
        if (!Array.isArray(checks)) return acc;
        const values = checks
          .map((entry) => numeric(entry?.candidate_multiplier))
          .filter((value): value is number => value !== null && value > 0);
        if (values.length > 0) acc[group] = values;
        return acc;
      }, {})
    : {};
  const values = isTestPipeline && Object.keys(predictionCandidateValues).length > 0
    ? predictionCandidateValues
    : configuredValues;
  const groups = Object.keys(values);
  const totalSlots = Math.max(0, ...groups.map((group) => values[group]?.length || 0));
  const selectedIndex = !isTestPipeline && currentIndex >= 0 ? Math.min(currentIndex, Math.max(totalSlots - 1, 0)) : -1;
  const selectedIndicesByGroup = isTestPipeline && selectedPredictionRow
    ? Object.entries(selectedPredictionChecksByGroup).reduce<Record<string, Set<number>>>((acc, [group, checks]) => {
        if (!Array.isArray(checks)) return acc;

        const index = selectedCandidateIndex(checks, groupMultiplierValue(selectedGroupMultipliers, group));
        if (index >= 0) {
          if (!acc[group]) acc[group] = new Set<number>();
          acc[group].add(index);
        }
        return acc;
      }, {})
    : {};
  const scheduleMethod = displayConfig.fixed_log_space_method;
  const intervalCount = displayConfig.fixed_log_space_interval_count;
  const isGridCandidateSet = isTestPipeline && (
    candidateGeneration === "grid_log_space"
    || candidateGeneration === "balanced_log_space"
    || Object.keys(displayConfig.test_candidate_log_multipliers || {}).length > 0
  );
  const candidateLabel = isGridCandidateSet
    ? `grid log-space${candidateCount || totalSlots ? ` (${candidateCount || totalSlots} points/group)` : ""}`
    : `offline schedule (${totalSlots} values/group)`;
  const displayPipeline = { ...pipeline, config: displayConfig } as PipelineDetail;
  const boundsForGroup = (group: string): [number, number] => {
    const groupValues = values[group] || [];
    const positiveValues = groupValues.filter((value) => Number.isFinite(value) && value > 0);
    if (isTestPipeline && positiveValues.length > 0) {
      return [Math.min(...positiveValues), Math.max(...positiveValues)];
    }
    return groupBounds(displayPipeline, group);
  };

  if (groups.length === 0) {
    return null;
  }
  const fullscreenValues = fullscreenGroup ? values[fullscreenGroup] || [] : [];

  const regeneratePreview = async () => {
    setScheduleBusy(true);
    setScheduleMessage(null);
    try {
      const response = await api.post(`/api/workflow/pipelines/${pipeline.id}/fixed-log-space-schedule/preview`);
      setPreviewSchedule(response.data?.schedule || null);
      setScheduleMessage({ type: "info", text: "Temporary interval-sampled values generated. Refreshing the page will discard them unless you save them." });
    } catch (error: any) {
      setScheduleMessage({ type: "error", text: error?.response?.data?.detail?.message || error?.message || "Failed to regenerate preview." });
    } finally {
      setScheduleBusy(false);
    }
  };

  const savePreviewForProcessing = async () => {
    if (!previewSchedule) return;
    setScheduleBusy(true);
    setScheduleMessage(null);
    try {
      const response = await api.post(`/api/workflow/pipelines/${pipeline.id}/fixed-log-space-schedule/save-preview`, {
        schedule: previewSchedule,
      });
      const saved = response.data?.schedule || previewSchedule;
      setSavedSchedule(saved);
      setPreviewSchedule(null);
      setScheduleMessage({ type: "success", text: response.data?.message || "Preview values saved for processing." });
    } catch (error: any) {
      setScheduleMessage({ type: "error", text: error?.response?.data?.detail?.message || error?.message || "Failed to save preview values." });
    } finally {
      setScheduleBusy(false);
    }
  };

  const shufflePreviewValues = () => {
    const baseSchedule = previewSchedule || displayConfig;
    const baseMultipliers = isTestPipeline
      ? getCandidateLogValues((baseSchedule as Record<string, unknown>).test_candidate_log_multipliers)
      : getGroupValues((baseSchedule as Record<string, unknown>).pre_generated_log_multipliers);
    const shuffledMultipliers = Object.fromEntries(
      Object.entries(baseMultipliers).map(([group, groupValues]) => [group, shuffleList(groupValues)])
    );
    setPreviewSchedule({
      ...baseSchedule,
      ...(isTestPipeline
        ? {
            test_candidate_log_multipliers: shuffledMultipliers,
            test_candidate_generation: "grid_log_space",
            test_candidate_count: totalSlots,
          }
        : {
            pre_generated_log_multipliers: shuffledMultipliers,
            fixed_log_space_method: (baseSchedule as Record<string, unknown>).fixed_log_space_method || "bounded_log_space_interval_sampling",
            fixed_log_space_interval_count: (baseSchedule as Record<string, unknown>).fixed_log_space_interval_count || totalSlots,
          }),
      fixed_log_space_generated_at: new Date().toISOString(),
    });
    setScheduleMessage({
      type: "info",
      text: isTestPipeline
        ? "Temporary candidate order shuffled. The same grid values are reordered only; save to use this order for model input."
        : "Temporary schedule shuffled. The same values are reordered only; save the schedule to use this order.",
    });
  };

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-950">
            {isTestPipeline ? "Model Candidate Grid" : "Log-Space Multiplier Schedule"}
          </h2>
          <p className="mt-1 text-sm text-slate-600">
            {isTestPipeline
              ? "Candidate multiplier values and model-predicted scores used to choose the best value before Gaussian Splatting training runs."
              : "One value is sampled inside each configured run interval, then group values can be shuffled before saving."}
          </p>
          {isPreviewActive && (
            <p className="mt-2 rounded border border-amber-200 bg-amber-50 px-3 py-2 text-xs font-medium text-amber-800">
              Viewing temporary preview values. They are not used for processing until saved.
            </p>
          )}
        </div>
        <div className="flex flex-col items-end gap-2">
          {!isTestPipeline && <div className="rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-right">
            <div className="text-xs font-semibold uppercase tracking-wide text-blue-700">{baselineActive ? "Current Run" : "Current Index"}</div>
            <div className="text-xl font-bold text-blue-950">{baselineActive ? "Baseline" : currentIndex >= 0 ? currentIndex : "N/A"}</div>
          </div>}
          <div className="flex flex-wrap justify-end gap-2">
            {!isTestPipeline && (
              <button
                type="button"
                onClick={regeneratePreview}
                disabled={scheduleBusy}
                className="rounded-lg border-2 border-blue-300 bg-white px-3 py-2 text-xs font-semibold text-blue-800 shadow-sm hover:border-blue-400 hover:bg-blue-50 disabled:cursor-not-allowed disabled:opacity-60"
              >
                Generate New Values
              </button>
            )}
            <button
              type="button"
              onClick={shufflePreviewValues}
              disabled={scheduleBusy || scheduleEditBlocked}
              className="inline-flex items-center gap-1.5 rounded-lg border-2 border-blue-300 bg-white px-3 py-2 text-xs font-semibold text-blue-800 shadow-sm hover:border-blue-400 hover:bg-blue-50 disabled:cursor-not-allowed disabled:opacity-60"
              title={scheduleEditBlocked ? "Candidate order cannot be changed after prediction/test runs have started." : undefined}
            >
              <Shuffle className="h-3.5 w-3.5" />
              {isTestPipeline ? "Shuffle Candidate Order" : "Shuffle Values"}
            </button>
            <button
              type="button"
              onClick={savePreviewForProcessing}
              disabled={scheduleBusy || !previewSchedule || scheduleEditBlocked}
              className="rounded-lg border border-emerald-200 bg-emerald-600 px-3 py-2 text-xs font-semibold text-white hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50"
              title={scheduleEditBlocked ? "Candidate order cannot be saved after prediction/test runs have started." : undefined}
            >
              {isTestPipeline ? "Save Order for Model Input" : "Save Schedule for Processing"}
            </button>
            {previewSchedule && (
              <button
                type="button"
                onClick={() => {
                  setPreviewSchedule(null);
                  setScheduleMessage({ type: "info", text: "Temporary preview discarded." });
                }}
                disabled={scheduleBusy}
                className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
              >
                Discard Preview
              </button>
            )}
          </div>
        </div>
      </div>

      {scheduleMessage && (
        <div
          className={`mb-4 rounded border px-3 py-2 text-xs ${
            scheduleMessage.type === "success"
              ? "border-emerald-200 bg-emerald-50 text-emerald-800"
              : scheduleMessage.type === "error"
                ? "border-red-200 bg-red-50 text-red-800"
                : "border-blue-200 bg-blue-50 text-blue-800"
          }`}
        >
          {scheduleMessage.text}
        </div>
      )}

      {scheduleEditBlocked && (
        <div className="mb-4 rounded border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          Candidate order is locked after prediction/test runs start. Baseline-only runs can still shuffle and save before that point.
        </div>
      )}

      <div className="mb-4 grid gap-2 text-xs text-slate-700 md:grid-cols-4">
        <div className="rounded border border-slate-200 bg-slate-50 px-3 py-2">
          <div className="font-semibold text-slate-900">Exploration Seed</div>
          <div className="font-mono text-[11px]">
            {scheduleSeed ?? (groupSeeds && typeof groupSeeds === "object" ? "mixed per-group seeds" : "N/A")}
          </div>
        </div>
        <div className="rounded border border-slate-200 bg-slate-50 px-3 py-2">
          <div className="font-semibold text-slate-900">Schedule Method</div>
          <div className="font-mono text-[11px]">
            {scheduleMethod === "bounded_log_space_interval_sampling"
              ? `interval sampled${intervalCount || totalSlots ? ` (${intervalCount || totalSlots} intervals)` : ""}`
              : scheduleMethod
                ? String(scheduleMethod)
                : "method not recorded"}
          </div>
        </div>
        <div className="rounded border border-slate-200 bg-slate-50 px-3 py-2">
          <div className="font-semibold text-slate-900">Candidate Set</div>
          <div className="font-mono text-[11px]">
            {isGridCandidateSet
              ? candidateLabel
              : candidateSeed
                ? `${candidateSeed} (${totalSlots} values/group)`
                : candidateLabel}
          </div>
        </div>
        <div className="rounded border border-slate-200 bg-slate-50 px-3 py-2">
          <div className="font-semibold text-slate-900">Generated</div>
          <div className="font-mono text-[11px]">{generatedAt ? new Date(generatedAt).toLocaleString() : "N/A"}</div>
        </div>
      </div>

      <div className={`mb-4 rounded border px-3 py-2 text-xs ${
        usingFallbackBounds ? "border-amber-200 bg-amber-50 text-amber-800" : "border-slate-200 bg-slate-50 text-slate-700"
      }`}>
        {isTestPipeline ? (
          <div className="flex flex-wrap gap-x-5 gap-y-1">
            {groups.map((group) => (
              <span key={`bounds-${group}`}>
                <span className="font-semibold">{GROUP_LABELS[group] || group}: </span>
                <span className="font-mono">{formatBounds(boundsForGroup(group))}</span>
              </span>
            ))}
          </div>
        ) : (
          <>
            <span className="font-semibold">Bounds source: </span>
            <span className="font-mono">{boundsSource || "N/A"}</span>
          </>
        )}
        {usingFallbackBounds && (
          <span className="ml-2">Model/source bounds were not available, so default group bounds are being used.</span>
        )}
      </div>

      {isTestPipeline && predictionOptions.length > 0 && (
        <div className="mb-4 rounded-lg border border-slate-200 bg-slate-50 p-3">
          <label className="mb-1 block text-xs font-semibold text-slate-800">Model candidate score curves</label>
          <p className="mb-2 text-[11px] text-slate-600">
            These are model-predicted candidate scores, not the final Gaussian Splatting test result.
          </p>
          <select
            value={effectivePredictionKey}
            onChange={(event) => setSelectedPredictionKey(event.target.value)}
            className="w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-800"
          >
            {predictionOptions.map((option) => (
              <option key={option.key} value={option.key}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      )}

      <div className="mb-4 grid gap-3 lg:grid-cols-3">
        {groups.map((group) => {
          const checks = selectedPredictionChecksByGroup[group];
          return isTestPipeline && Array.isArray(checks) && checks.length > 0 ? (
            <CandidateScoreChart
              key={`score-chart-${group}`}
              bounds={boundsForGroup(group)}
              checks={checks}
              group={group}
              onFullscreen={() => setFullscreenGroup(group)}
              selectedGroupMultiplier={groupMultiplierValue(selectedGroupMultipliers, group)}
            />
          ) : (
            <ScheduleDistributionChart
              key={`schedule-chart-${group}`}
              bounds={boundsForGroup(group)}
              group={group}
              onFullscreen={() => setFullscreenGroup(group)}
              selectedIndex={selectedIndex}
              selectedIndices={selectedIndicesByGroup[group]}
              values={values[group] || []}
            />
          );
        })}
      </div>

      <div className="space-y-3">
        {groups.map((group) => {
          const groupValues = values[group] || [];
          const selectedValue = selectedIndex >= 0 ? groupValues[selectedIndex] : null;
          const selectedSet = selectedIndicesByGroup[group] || new Set<number>();
          return (
            <div key={group} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-50 text-blue-700 ring-1 ring-blue-100">
                  <Hash className="h-4 w-4" />
                </div>
                <div>
                  <div className="text-sm font-semibold text-slate-950">{GROUP_LABELS[group] || group}</div>
                  <div className="font-mono text-[10px] text-slate-500">{group}</div>
                </div>
              </div>
                <div className="rounded border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-right">
                  <div className="text-[10px] font-semibold uppercase tracking-wide text-emerald-700">
                    {isTestPipeline ? "Candidates" : selectedIndex >= 0 ? `Selected ${selectedIndex}` : "Baseline"}
                  </div>
                  <div className="font-mono text-xs font-bold text-emerald-950">
                    {isTestPipeline ? `${groupValues.length} values` : selectedIndex >= 0 ? formatValue(selectedValue) : "1.000000"}
                  </div>
                </div>
              </div>
              <div className="overflow-x-auto rounded border border-slate-200 bg-white px-2 py-2">
                <div className="flex min-w-max items-center gap-1.5">
                  {groupValues.map((value, index) => {
                    const selected = index === selectedIndex || selectedSet.has(index);
                    return (
                      <div
                    key={`${group}-${index}`}
                        title={`Index ${index}: ${formatValue(value)}${selected ? " selected" : ""}`}
                        className={`flex min-w-[72px] flex-col items-center rounded border px-2 transition ${
                      selected
                            ? "scale-105 border-emerald-300 bg-emerald-100 py-2 text-emerald-950 shadow-sm"
                            : "border-slate-200 bg-slate-50 py-1.5 text-slate-600"
                    }`}
                  >
                        <span className="text-[9px] font-semibold">{index}</span>
                        <span className="font-mono text-[10px]">{formatValue(value)}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {fullscreenGroup && (
        <div className="fixed inset-0 z-50 bg-slate-950/70 p-4">
          <div className="flex h-full flex-col rounded-lg bg-white shadow-2xl">
            <div className="flex items-center justify-between gap-3 border-b border-slate-200 px-4 py-3">
              <div>
                <h3 className="text-base font-semibold text-slate-950">{GROUP_LABELS[fullscreenGroup] || fullscreenGroup}</h3>
                <p className="font-mono text-xs text-slate-600">{fullscreenGroup}</p>
              </div>
              <button
                type="button"
                onClick={() => setFullscreenGroup(null)}
                className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
              >
                Close
              </button>
            </div>
            <div className="overflow-auto p-4">
              {isTestPipeline && Array.isArray(selectedPredictionChecksByGroup[fullscreenGroup]) ? (
                <CandidateScoreChart
                  bounds={boundsForGroup(fullscreenGroup)}
                  checks={selectedPredictionChecksByGroup[fullscreenGroup]}
                  fullscreen
                  group={fullscreenGroup}
                  selectedGroupMultiplier={groupMultiplierValue(selectedGroupMultipliers, fullscreenGroup)}
                />
              ) : (
                <ScheduleDistributionChart
                  bounds={boundsForGroup(fullscreenGroup)}
                  fullscreen
                  group={fullscreenGroup}
                  selectedIndex={selectedIndex}
                  selectedIndices={selectedIndicesByGroup[fullscreenGroup]}
                  values={fullscreenValues}
                />
              )}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
