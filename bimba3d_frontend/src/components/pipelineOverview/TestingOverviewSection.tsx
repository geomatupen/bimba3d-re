import { useEffect, useMemo, useState } from "react";

interface CandidateCheckRow {
  candidate_log_multiplier: number;
  candidate_multiplier: number;
  predicted_score: number;
  selected: boolean;
}

interface PredictionRow {
  project_name: string;
  model_id?: string | null;
  status: string;
  has_signal?: boolean;
  n_runs?: number;
  score_spreads?: Record<string, number>;
  candidate_points?: number;
  selected_log_multipliers?: Record<string, number>;
  selected_multipliers?: Record<string, number>;
  candidate_score_checks?: Record<string, CandidateCheckRow[]>;
}

interface GroupCoverage {
  group: string;
  rows: number;
  selectedCandidates: number;
  candidatePoints: number;
  coveragePct: number;
}

const GROUP_LABELS: Record<string, string> = {
  appearance_lr_mult: "Appearance",
  densification_mult: "Densification",
  geometry_lr_mult: "Geometry",
};

interface TestingOverviewSectionProps {
  rows: PredictionRow[];
  loading: boolean;
  restartVersion?: number;
  restartToken?: string | null;
  lastRestartAt?: string | null;
  previewGeneratedAt?: string | null;
}

function formatNumber(value: number | null, digits = 4): string {
  if (value === null || !Number.isFinite(value)) return "-";
  return value.toFixed(digits);
}

function average(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function buildCoverage(rows: PredictionRow[]): GroupCoverage[] {
  const groups = new Map<string, { selectedBins: Set<number>; candidatePoints: number; rows: number }>();

  for (const row of rows) {
    if (row.status !== "ok") continue;
    const checksByGroup = row.candidate_score_checks ?? {};

    for (const [group, checks] of Object.entries(checksByGroup)) {
      if (!Array.isArray(checks) || checks.length === 0) continue;
      const selectedIndex = checks.findIndex((entry) => entry.selected);
      const existing = groups.get(group) ?? { selectedBins: new Set<number>(), candidatePoints: 0, rows: 0 };

      if (selectedIndex >= 0) {
        existing.selectedBins.add(selectedIndex);
      }
      existing.candidatePoints = Math.max(existing.candidatePoints, checks.length, row.candidate_points ?? 0);
      existing.rows += 1;
      groups.set(group, existing);
    }
  }

  const result: GroupCoverage[] = [];
  for (const [group, value] of groups.entries()) {
    const candidatePoints = Math.max(value.candidatePoints, 1);
    result.push({
      group,
      rows: value.rows,
      selectedCandidates: value.selectedBins.size,
      candidatePoints,
      coveragePct: (value.selectedBins.size / candidatePoints) * 100,
    });
  }

  return result.sort((a, b) => b.coveragePct - a.coveragePct || a.group.localeCompare(b.group));
}

export default function TestingOverviewSection({
  rows,
  loading,
  lastRestartAt,
  previewGeneratedAt,
}: TestingOverviewSectionProps) {
  const [selectedTestKey, setSelectedTestKey] = useState("");
  const [selectedGroup, setSelectedGroup] = useState("geometry_lr_mult");

  const okRows = useMemo(() => rows.filter((row) => row.status === "ok"), [rows]);

  const testOptions = useMemo(
    () =>
      okRows.map((row, index) => ({
        key: `${row.project_name}::${row.model_id || "no_model"}::${index}`,
        label: `${row.project_name} - ${row.model_id || "model"}`,
        row,
      })),
    [okRows],
  );

  useEffect(() => {
    if (testOptions.length === 0) {
      setSelectedTestKey("");
      return;
    }
    if (!testOptions.some((option) => option.key === selectedTestKey)) {
      setSelectedTestKey(testOptions[0].key);
    }
  }, [testOptions, selectedTestKey]);

  const selectedTest = useMemo(
    () => testOptions.find((option) => option.key === selectedTestKey)?.row ?? null,
    [selectedTestKey, testOptions],
  );

  const groupOptions = useMemo(() => {
    if (!selectedTest) return ["geometry_lr_mult", "appearance_lr_mult", "densification_mult"];
    const keys = Object.keys(selectedTest.candidate_score_checks ?? {});
    return keys.length > 0 ? keys : ["geometry_lr_mult", "appearance_lr_mult", "densification_mult"];
  }, [selectedTest]);

  useEffect(() => {
    if (!groupOptions.includes(selectedGroup)) {
      setSelectedGroup(groupOptions[0]);
    }
  }, [groupOptions, selectedGroup]);

  const selectedChecks = useMemo(() => {
    if (!selectedTest) return [] as CandidateCheckRow[];
    const checks = selectedTest.candidate_score_checks?.[selectedGroup];
    return Array.isArray(checks) ? checks : [];
  }, [selectedGroup, selectedTest]);

  const summary = useMemo(() => {
    const total = rows.length;
    const ok = rows.filter((row) => row.status === "ok").length;
    const signal = rows.filter((row) => row.status === "ok" && row.has_signal !== false).length;

    const spreadValues: number[] = [];
    const nRunValues: number[] = [];

    for (const row of rows) {
      if (row.status !== "ok") continue;
      for (const spread of Object.values(row.score_spreads ?? {})) {
        if (typeof spread === "number" && Number.isFinite(spread)) {
          spreadValues.push(spread);
        }
      }
      if (typeof row.n_runs === "number" && Number.isFinite(row.n_runs)) {
        nRunValues.push(row.n_runs);
      }
    }

    return {
      total,
      ok,
      failed: total - ok,
      signalRate: ok > 0 ? (signal / ok) * 100 : 0,
      avgSpread: average(spreadValues),
      avgNRuns: average(nRunValues),
    };
  }, [rows]);

  const coverageRows = useMemo(() => buildCoverage(rows), [rows]);

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
        <h2 className="text-sm font-semibold text-gray-900 mb-2">Testing Overview</h2>
        <p className="text-xs text-gray-600">No test prediction preview rows available yet.</p>
      </div>
    );
  }

  return (
    <section className="bg-white border border-gray-200 rounded-lg p-4 space-y-4">
      <div>
        <h2 className="text-sm font-semibold text-gray-900">Testing Overview</h2>
        <p className="text-xs text-gray-600 mt-1">
          Exploration coverage and candidate score surfaces by test row.
        </p>
        <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-gray-500">
          {lastRestartAt && (
            <span>Last restart: {new Date(lastRestartAt).toLocaleString()}</span>
          )}
          {previewGeneratedAt && (
            <span>Preview generated: {new Date(previewGeneratedAt).toLocaleString()}</span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2 text-xs" style={{display:'none'}}>
        <div className="border border-gray-200 rounded p-2 bg-gray-50">
          <div className="text-gray-500">Rows</div>
          <div className="font-semibold text-gray-900">{summary.total}</div>
        </div>
        <div className="border border-gray-200 rounded p-2 bg-gray-50">
          <div className="text-gray-500">OK</div>
          <div className="font-semibold text-green-700">{summary.ok}</div>
        </div>
        <div className="border border-gray-200 rounded p-2 bg-gray-50">
          <div className="text-gray-500">Failed</div>
          <div className="font-semibold text-red-700">{summary.failed}</div>
        </div>
        <div className="border border-gray-200 rounded p-2 bg-gray-50">
          <div className="text-gray-500">Signal Rate</div>
          <div className="font-semibold text-gray-900">{summary.signalRate.toFixed(1)}%</div>
        </div>
        <div className="border border-gray-200 rounded p-2 bg-gray-50">
          <div className="text-gray-500">Avg Spread</div>
          <div className="font-semibold text-gray-900">{formatNumber(summary.avgSpread, 6)}</div>
        </div>
        <div className="border border-gray-200 rounded p-2 bg-gray-50">
          <div className="text-gray-500">Avg n_runs</div>
          <div className="font-semibold text-gray-900">{formatNumber(summary.avgNRuns, 2)}</div>
        </div>
      </div>

      <div className="border border-gray-200 rounded-lg p-3">
        <div className="mb-3">
          <h3 className="text-xs font-semibold text-gray-800">Selected Candidate Coverage</h3>
          <p className="mt-1 text-[11px] leading-5 text-gray-600">
            Shows how widely the model-selected multipliers are spread across the candidate grid. A higher bar means the model is choosing more different candidate positions across projects, not only repeating the same grid value.
          </p>
        </div>
        {coverageRows.length === 0 ? (
          <p className="text-xs text-gray-500">No candidate score checks available.</p>
        ) : (
          <div className="space-y-3">
            {coverageRows.map((row) => (
              <div key={row.group}>
                <div className="flex justify-between text-[11px] text-gray-700 mb-0.5">
                  <span className="font-medium">{GROUP_LABELS[row.group] || row.group}</span>
                  <span>
                    {row.selectedCandidates} of {row.candidatePoints} candidates selected - {row.rows} row{row.rows === 1 ? "" : "s"}
                  </span>
                </div>
                <div className="w-full h-2 rounded bg-gray-200 overflow-hidden">
                  <div className="h-2 bg-emerald-500" style={{ width: `${Math.max(2, row.coveragePct)}%` }} />
                </div>
                <div className="mt-1 text-[10px] text-gray-500">
                  Coverage: {row.coveragePct.toFixed(1)}% of candidate positions were selected at least once.
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="border border-gray-200 rounded-lg p-3 space-y-2">
        <div className="grid gap-2 md:grid-cols-[minmax(0,1fr)_220px]">
          <label className="min-w-0 text-xs font-medium text-gray-700">
            <span className="mb-1 block">Test Row</span>
          <select
            value={selectedTestKey}
            onChange={(event) => setSelectedTestKey(event.target.value)}
              className="w-full min-w-0 rounded border border-gray-300 px-2 py-1.5 text-xs"
          >
            {testOptions.map((option) => (
              <option key={option.key} value={option.key}>
                {option.label}
              </option>
            ))}
          </select>
          </label>

          <label className="min-w-0 text-xs font-medium text-gray-700">
            <span className="mb-1 block">Group</span>
          <select
            value={selectedGroup}
            onChange={(event) => setSelectedGroup(event.target.value)}
              className="w-full min-w-0 rounded border border-gray-300 px-2 py-1.5 text-xs"
          >
            {groupOptions.map((group) => (
              <option key={group} value={group}>
                {group}
              </option>
            ))}
          </select>
          </label>
        </div>

        {selectedTest ? (
          <>
            <div className="overflow-hidden rounded border border-gray-200">
              <div className="max-h-72 overflow-auto">
              <table className="min-w-[760px] w-full text-xs">
                <thead className="sticky top-0 z-10 bg-gray-50 border-b border-gray-200">
                  <tr>
                    <th className="px-2 py-1 text-left font-semibold text-gray-700">Candidate #</th>
                    <th className="px-2 py-1 text-left font-semibold text-gray-700">Log Multiplier</th>
                    <th className="px-2 py-1 text-left font-semibold text-gray-700">Multiplier</th>
                    <th className="px-2 py-1 text-left font-semibold text-gray-700">Predicted Score</th>
                    <th className="px-2 py-1 text-left font-semibold text-gray-700">Selected</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedChecks.map((entry, index) => (
                    <tr key={`${selectedGroup}-${index}`} className="border-b border-gray-100">
                      <td className="px-2 py-1 text-gray-700">{index + 1}</td>
                      <td className="px-2 py-1 text-gray-700 font-mono">{formatNumber(entry.candidate_log_multiplier, 6)}</td>
                      <td className="px-2 py-1 text-gray-700 font-mono">{formatNumber(entry.candidate_multiplier, 6)}</td>
                      <td className="px-2 py-1 text-gray-700">{formatNumber(entry.predicted_score, 6)}</td>
                      <td className="px-2 py-1 text-gray-700">{entry.selected ? "yes" : "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              </div>
            </div>
          </>
        ) : (
          <p className="text-xs text-gray-500">No test row selected.</p>
        )}
      </div>
    </section>
  );
}

