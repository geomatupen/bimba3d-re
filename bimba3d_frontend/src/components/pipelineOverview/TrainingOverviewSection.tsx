import { useEffect, useMemo, useState } from "react";

interface LearningParamRow {
  key: string;
  actual: number | null;
  selected_multiplier: number | null;
  selected_multiplier_raw?: number | null;
  log_multiplier?: number | null;
  jitter?: number | null;
  final_multiplier: number | null;
}

interface TrainingOverviewRow {
  project_name: string;
  run_id: string;
  run_name?: string | null;
  best_loss?: number | null;
  final_loss?: number | null;
  best_psnr?: number | null;
  final_psnr?: number | null;
  best_ssim?: number | null;
  final_ssim?: number | null;
  best_lpips?: number | null;
  final_lpips?: number | null;
  score?: number | null;
  relative_quality_score?: number | null;
  convergence_score?: number | null;
  learning_param_rows?: LearningParamRow[] | null;
}

interface CoverageRow {
  key: string;
  samples: number;
  minLog: number;
  maxLog: number;
  coveragePct: number;
}

interface TrainingOverviewSectionProps {
  rows: TrainingOverviewRow[];
  loading: boolean;
  restartVersion?: number;
  restartToken?: string | null;
  lastRestartAt?: string | null;
  preGeneratedLogMultipliers?: {
    geometry_lr?: number[];
    appearance_lr?: number[];
    scale_lr?: number[];
  };
  multiplierCurrentIndex?: number;
}

const COVERAGE_BINS = 12;

function toFinite(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function average(values: Array<number | null>): number | null {
  const valid = values.filter((value): value is number => value !== null);
  if (valid.length === 0) return null;
  return valid.reduce((sum, value) => sum + value, 0) / valid.length;
}

function formatNumber(value: number | null, digits = 4): string {
  if (value === null) return "-";
  return value.toFixed(digits);
}

function summarizeCoverage(rows: TrainingOverviewRow[]): CoverageRow[] {
  const map = new Map<string, number[]>();

  for (const row of rows) {
    for (const param of row.learning_param_rows ?? []) {
      const logValue = toFinite(param.log_multiplier ?? param.jitter ?? null);
      if (logValue === null) continue;
      const list = map.get(param.key) ?? [];
      list.push(logValue);
      map.set(param.key, list);
    }
  }

  const result: CoverageRow[] = [];
  for (const [key, values] of map.entries()) {
    if (values.length === 0) continue;
    const minLog = Math.min(...values);
    const maxLog = Math.max(...values);
    const span = Math.max(maxLog - minLog, 1e-9);
    const occupied = new Set<number>();

    for (const value of values) {
      const normalized = (value - minLog) / span;
      const bin = Math.min(COVERAGE_BINS - 1, Math.max(0, Math.floor(normalized * COVERAGE_BINS)));
      occupied.add(bin);
    }

    result.push({
      key,
      samples: values.length,
      minLog,
      maxLog,
      coveragePct: (occupied.size / COVERAGE_BINS) * 100,
    });
  }

  return result.sort((a, b) => b.coveragePct - a.coveragePct || a.key.localeCompare(b.key));
}

export default function TrainingOverviewSection({
  rows,
  loading,
  restartVersion,
  restartToken,
  lastRestartAt,
  preGeneratedLogMultipliers,
  multiplierCurrentIndex,
}: TrainingOverviewSectionProps) {
  const [selectedRunKey, setSelectedRunKey] = useState("");

  const runOptions = useMemo(
    () =>
      rows.map((row) => ({
        key: `${row.project_name}::${row.run_id}`,
        label: `${row.project_name} Â· ${row.run_name || row.run_id}`,
      })),
    [rows],
  );

  useEffect(() => {
    if (runOptions.length === 0) {
      setSelectedRunKey("");
      return;
    }
    if (!runOptions.some((option) => option.key === selectedRunKey)) {
      setSelectedRunKey(runOptions[0].key);
    }
  }, [runOptions, selectedRunKey]);

  const selectedRow = useMemo(
    () => rows.find((row) => `${row.project_name}::${row.run_id}` === selectedRunKey) ?? null,
    [rows, selectedRunKey],
  );

  const summary = useMemo(() => {
    const uniqueProjects = new Set(rows.map((row) => row.project_name)).size;
    const avgScore = average(rows.map((row) => toFinite(row.score)));
    const avgScoreQuality = average(rows.map((row) => toFinite(row.relative_quality_score)));
    const avgScoreConvergence = average(rows.map((row) => toFinite(row.convergence_score)));
    const avgFinalLoss = average(rows.map((row) => toFinite(row.final_loss)));

    return {
      totalRuns: rows.length,
      uniqueProjects,
      avgScore,
      avgScoreQuality,
      avgScoreConvergence,
      avgFinalLoss,
    };
  }, [rows]);

  const coverageRows = useMemo(() => summarizeCoverage(rows), [rows]);

  const fitSnapshot = useMemo(() => {
    if (!selectedRow) return [] as Array<{ label: string; best: number | null; final: number | null; invert?: boolean }>;
    return [
      { label: "Loss", best: toFinite(selectedRow.best_loss), final: toFinite(selectedRow.final_loss), invert: true },
      { label: "PSNR", best: toFinite(selectedRow.best_psnr), final: toFinite(selectedRow.final_psnr) },
      { label: "SSIM", best: toFinite(selectedRow.best_ssim), final: toFinite(selectedRow.final_ssim) },
      { label: "LPIPS", best: toFinite(selectedRow.best_lpips), final: toFinite(selectedRow.final_lpips), invert: true },
    ];
  }, [selectedRow]);

  if (loading) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-7 w-7 border-b-2 border-indigo-600" />
        </div>
      </div>
    );
  }

  if (rows.length === 0) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h2 className="text-sm font-semibold text-gray-900 mb-2">Training Overview</h2>
        <p className="text-xs text-gray-600">No training learning rows available yet.</p>
      </div>
    );
  }

  return (
    <section className="bg-white border border-gray-200 rounded-lg p-4 space-y-4">
      <div>
        <h2 className="text-sm font-semibold text-gray-900">Training Overview</h2>
        <p className="text-xs text-gray-600 mt-1">
          Restart-scoped exploration coverage, scores, and fit quality from pipeline learning rows.
        </p>
        <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]">
          <span className="px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200">
            Restart v{Number.isFinite(Number(restartVersion)) ? Number(restartVersion) : 0}
          </span>
          {restartToken ? (
            <span className="px-2 py-0.5 rounded bg-gray-50 text-gray-700 border border-gray-200 font-mono">
              token {restartToken.slice(0, 8)}
            </span>
          ) : null}
          {lastRestartAt ? (
            <span className="text-gray-500">last restart {new Date(lastRestartAt).toLocaleString()}</span>
          ) : (
            <span className="text-gray-500">no restart recorded yet</span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2 text-xs">
        <div className="border border-gray-200 rounded p-2 bg-gray-50">
          <div className="text-gray-500">Runs</div>
          <div className="font-semibold text-gray-900">{summary.totalRuns}</div>
        </div>
        <div className="border border-gray-200 rounded p-2 bg-gray-50">
          <div className="text-gray-500">Projects</div>
          <div className="font-semibold text-gray-900">{summary.uniqueProjects}</div>
        </div>
        <div className="border border-gray-200 rounded p-2 bg-gray-50">
          <div className="text-gray-500">Avg Score</div>
          <div className="font-semibold text-gray-900">{formatNumber(summary.avgScore, 4)}</div>
        </div>
        <div className="border border-gray-200 rounded p-2 bg-gray-50">
          <div className="text-gray-500">Avg R Quality</div>
          <div className="font-semibold text-gray-900">{formatNumber(summary.avgScoreQuality, 4)}</div>
        </div>
        <div className="border border-gray-200 rounded p-2 bg-gray-50">
          <div className="text-gray-500">Avg R Conv</div>
          <div className="font-semibold text-gray-900">{formatNumber(summary.avgScoreConvergence, 4)}</div>
        </div>
        <div className="border border-gray-200 rounded p-2 bg-gray-50">
          <div className="text-gray-500">Avg Final Loss</div>
          <div className="font-semibold text-gray-900">{formatNumber(summary.avgFinalLoss, 6)}</div>
        </div>
      </div>

      {preGeneratedLogMultipliers && Object.keys(preGeneratedLogMultipliers).length > 0 && (
        <div className="border border-gray-200 rounded-lg p-3">
          <h3 className="text-xs font-semibold text-gray-800 mb-2">Pre-Generated Log Space Multipliers</h3>
          <div className="text-[10px] text-gray-600 mb-2">Current Index: {multiplierCurrentIndex ?? 0} / {(preGeneratedLogMultipliers.geometry_lr?.length ?? 0)}</div>
          <div className="space-y-2">
            {Object.entries(preGeneratedLogMultipliers).map(([group, values]) => {
              const currentIdx = multiplierCurrentIndex ?? 0;
              return (
                <div key={group} className="bg-gray-50 border border-gray-200 rounded p-2">
                  <div className="text-[11px] font-medium text-gray-700 mb-1">{group}</div>
                  <div className="flex flex-wrap gap-1">
                    {(values ?? []).map((value, idx) => (
                      <div
                        key={idx}
                        className={`px-2 py-0.5 rounded text-[10px] font-mono ${
                          idx < currentIdx
                            ? "bg-emerald-100 text-emerald-700 border border-emerald-200"
                            : idx === currentIdx
                              ? "bg-amber-100 text-amber-700 border border-amber-300 font-semibold"
                              : "bg-gray-100 text-gray-600 border border-gray-200"
                        }`}
                        title={`Index ${idx}: ${value.toFixed(4)}`}
                      >
                        {value.toFixed(3)}
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="border border-gray-200 rounded-lg p-3">
        <h3 className="text-xs font-semibold text-gray-800 mb-2">Exploration Coverage (Log Multipliers)</h3>
        {coverageRows.length === 0 ? (
          <p className="text-xs text-gray-500">No log-multiplier values found.</p>
        ) : (
          <div className="space-y-2">
            {coverageRows.map((row) => (
              <div key={row.key}>
                <div className="flex justify-between text-[11px] text-gray-700 mb-0.5">
                  <span>{row.key}</span>
                  <span>
                    {row.coveragePct.toFixed(1)}% Â· n={row.samples} Â· [{row.minLog.toFixed(3)}, {row.maxLog.toFixed(3)}]
                  </span>
                </div>
                <div className="w-full h-2 rounded bg-gray-200 overflow-hidden">
                  <div className="h-2 bg-indigo-500" style={{ width: `${Math.max(2, row.coveragePct)}%` }} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="border border-gray-200 rounded-lg p-3 space-y-2">
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium text-gray-700">Run</label>
          <select
            value={selectedRunKey}
            onChange={(event) => setSelectedRunKey(event.target.value)}
            className="px-2 py-1 text-xs border border-gray-300 rounded max-w-[460px]"
          >
            {runOptions.map((option) => (
              <option key={option.key} value={option.key}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        {selectedRow ? (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
              <div className="bg-gray-50 border border-gray-200 rounded p-2">
                <div className="text-gray-500">Best Loss</div>
                <div className="font-semibold text-gray-900">{formatNumber(toFinite(selectedRow.best_loss), 6)}</div>
              </div>
              <div className="bg-gray-50 border border-gray-200 rounded p-2">
                <div className="text-gray-500">Final Loss</div>
                <div className="font-semibold text-gray-900">{formatNumber(toFinite(selectedRow.final_loss), 6)}</div>
              </div>
              <div className="bg-gray-50 border border-gray-200 rounded p-2">
                <div className="text-gray-500">Final PSNR</div>
                <div className="font-semibold text-gray-900">{formatNumber(toFinite(selectedRow.final_psnr), 4)}</div>
              </div>
              <div className="bg-gray-50 border border-gray-200 rounded p-2">
                <div className="text-gray-500">Final LPIPS</div>
                <div className="font-semibold text-gray-900">{formatNumber(toFinite(selectedRow.final_lpips), 4)}</div>
              </div>
            </div>

            <div className="border border-gray-200 rounded p-2 bg-white">
              <div className="text-[11px] font-semibold text-gray-700 mb-2">Fit Snapshot (Best vs Final)</div>
              <div className="space-y-2">
                {fitSnapshot.map((item) => {
                  const best = item.best;
                  const final = item.final;
                  const min = Math.min(...[best, final].filter((v): v is number => v !== null));
                  const max = Math.max(...[best, final].filter((v): v is number => v !== null));
                  const span = max - min;
                  const bestPct = best === null ? null : (span < 1e-12 ? 50 : ((best - min) / span) * 100);
                  const finalPct = final === null ? null : (span < 1e-12 ? 50 : ((final - min) / span) * 100);
                  return (
                    <div key={item.label} className="grid grid-cols-[80px_1fr] items-center gap-2">
                      <div className="text-[11px] text-gray-600">{item.label}</div>
                      <div className="relative h-5 rounded bg-gray-100 border border-gray-200">
                        {bestPct !== null && <div className="absolute top-0 bottom-0 w-[2px] bg-indigo-600" style={{ left: `${bestPct}%` }} title={`best ${formatNumber(best, 4)}`} />}
                        {finalPct !== null && <div className="absolute top-0 bottom-0 w-[2px] bg-emerald-600" style={{ left: `${finalPct}%` }} title={`final ${formatNumber(final, 4)}`} />}
                        <div className="absolute inset-0 flex items-center justify-between px-2 text-[10px] text-gray-600">
                          <span>B {formatNumber(best, 4)}</span>
                          <span>F {formatNumber(final, 4)}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="overflow-x-auto border border-gray-200 rounded">
              <table className="min-w-[760px] w-full text-xs">
                <thead className="bg-gray-50 border-b border-gray-200">
                  <tr>
                    <th className="px-2 py-1 text-left font-semibold text-gray-700">Key</th>
                    <th className="px-2 py-1 text-left font-semibold text-gray-700">Actual</th>
                    <th className="px-2 py-1 text-left font-semibold text-gray-700">Selected Multiplier</th>
                    <th className="px-2 py-1 text-left font-semibold text-gray-700">Log Multiplier</th>
                    <th className="px-2 py-1 text-left font-semibold text-gray-700">Final Multiplier</th>
                  </tr>
                </thead>
                <tbody>
                  {(selectedRow.learning_param_rows ?? []).map((param) => {
                    const selected = toFinite(param.selected_multiplier_raw ?? param.selected_multiplier ?? null);
                    const logMultiplier = toFinite(param.log_multiplier ?? param.jitter ?? null);
                    return (
                      <tr key={param.key} className="border-b border-gray-100">
                        <td className="px-2 py-1 text-gray-800">{param.key}</td>
                        <td className="px-2 py-1 text-gray-700">{formatNumber(toFinite(param.actual), 6)}</td>
                        <td className="px-2 py-1 text-gray-700">{formatNumber(selected, 6)}</td>
                        <td className="px-2 py-1 text-gray-700">{formatNumber(logMultiplier, 6)}</td>
                        <td className="px-2 py-1 text-gray-700">{formatNumber(toFinite(param.final_multiplier), 6)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <p className="text-xs text-gray-500">No run selected.</p>
        )}
      </div>
    </section>
  );
}

