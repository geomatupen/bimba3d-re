import { useCallback, useEffect, useState } from "react";
import { CheckCircle2, Download, RefreshCw, Table2 } from "lucide-react";
import { api } from "../../api/client";
import LearningParamRows from "../learning/LearningParamRows";

interface ProjectTestResultsTabProps {
  projectId: string;
}

interface ProjectTestResultRow {
  run_id: string;
  run_name?: string | null;
  is_baseline_row?: boolean;
  model_id?: string | null;
  best_loss_step?: number | null;
  best_loss?: number | null;
  final_loss_step?: number | null;
  final_loss?: number | null;
  best_psnr_step?: number | null;
  best_psnr?: number | null;
  final_psnr_step?: number | null;
  final_psnr?: number | null;
  best_ssim_step?: number | null;
  best_ssim?: number | null;
  final_ssim_step?: number | null;
  final_ssim?: number | null;
  best_lpips_step?: number | null;
  best_lpips?: number | null;
  final_lpips_step?: number | null;
  final_lpips?: number | null;
  score?: number | null;
  run_best_l?: number | null;
  run_best_q?: number | null;
  run_best_t?: number | null;
  run_best_s?: number | null;
  run_end_l?: number | null;
  run_end_q?: number | null;
  run_end_t?: number | null;
  run_end_s?: number | null;
  run_end_elapsed?: number | null;
  base_best_l?: number | null;
  base_best_q?: number | null;
  base_best_t?: number | null;
  base_end_l?: number | null;
  base_end_q?: number | null;
  base_end_t?: number | null;
  base_end_elapsed?: number | null;
  s_best?: number | null;
  s_end?: number | null;
  s_run?: number | null;
  s_base_best?: number | null;
  s_base_end?: number | null;
  s_base?: number | null;
  remarks?: string | null;
  final_multiplier_formula?: string | null;
  learned_input_params_source?: string | null;
  learned_input_params_status?: string | null;
  learning_param_rows?: Array<{
    actual: number | null;
    final_multiplier: number | null;
    jitter: number | null;
    key: string;
    selected_multiplier: number | null;
  }> | null;
}

interface TrainingDataUpdateInfo {
  updated_at?: string | null;
  run_id?: string | null;
  training_data_id?: string | null;
  training_data_name?: string | null;
  row_count?: number | null;
  source_model_name?: string | null;
  source_model_family?: string | null;
  model_evaluation_step?: number | null;
  score_reference_step?: number | null;
}

const fmt = (value: number | null | undefined, digits = 6) =>
  typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "-";

const fmtStep = (value: number | null | undefined) =>
  typeof value === "number" && Number.isFinite(value) ? value.toLocaleString() : "-";

const fmtLearnedParamSource = (source: string | null | undefined, status: string | null | undefined) => {
  const normalized = String(source || "").trim().toLowerCase();
  if (normalized === "run_start_log") return "source: run-start log";
  if (normalized === "missing_run_start_log") return "source: missing run-start marker";
  if (normalized === "baseline_config") return "source: baseline config";
  if (normalized === "run_config") return "source: run config";
  if (status) return `status: ${status}`;
  return "source: not captured";
};

const renderLearningParamRows = (
  runId: string,
  rows: ProjectTestResultRow["learning_param_rows"],
  field: "actual" | "final_multiplier",
) => <LearningParamRows runId={runId} rows={rows} field={field} />;

export default function ProjectTestResultsTab({ projectId }: ProjectTestResultsTabProps) {
  const [rows, setRows] = useState<ProjectTestResultRow[]>([]);
  const [trainingDataUpdate, setTrainingDataUpdate] = useState<TrainingDataUpdateInfo | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const loadRows = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.get(`/projects/${projectId}/ai-learning-table`);
      setRows(Array.isArray(res.data?.rows) ? res.data.rows : []);
      setTrainingDataUpdate(res.data?.training_data_update && typeof res.data.training_data_update === "object" ? res.data.training_data_update : null);
      setMessage(typeof res.data?.message === "string" ? res.data.message : null);
    } catch (err) {
      console.error("Failed to load project test results:", err);
      setRows([]);
      setTrainingDataUpdate(null);
      setMessage("Failed to load project test results.");
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    void loadRows();
  }, [loadRows]);

  const downloadResults = () => {
    const blob = new Blob([JSON.stringify({ project_id: projectId, rows }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${projectId}_test_results.json`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-6">
      <div className="overflow-hidden rounded-xl border border-gray-200 bg-white shadow-sm">
        <div className="flex items-center justify-between border-b border-gray-200 px-6 py-4">
          <div className="flex items-center gap-3">
            <Table2 className="h-5 w-5 text-gray-600" />
            <div>
              <h2 className="text-lg font-semibold text-gray-900">Test Results</h2>
              <p className="mt-0.5 text-xs text-gray-500">Per-project model test results and learned parameter outcomes</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => void loadRows()}
              disabled={loading}
              className="inline-flex items-center gap-1.5 rounded-md bg-gray-100 px-2.5 py-1.5 text-xs font-medium text-gray-700 transition-colors hover:bg-gray-200 disabled:opacity-60"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </button>
            <button
              onClick={downloadResults}
              className="inline-flex items-center gap-1.5 rounded-md bg-blue-600 px-2.5 py-1.5 text-xs font-medium text-white transition-colors hover:bg-blue-700"
            >
              <Download className="h-3.5 w-3.5" />
              Download Results
            </button>
          </div>
        </div>

        <div className="h-[600px] overflow-y-auto rounded-b-xl bg-slate-50">
          <div className="border-b border-gray-200 bg-white px-6 py-4 text-xs text-slate-600">
            <strong>Project Test Results:</strong> Shows all test runs within this project, including quality metrics,
            score breakdowns, timing values, and learning parameter multipliers used by each run.
          </div>
          {trainingDataUpdate && (
            <div className="border-b border-emerald-200 bg-emerald-50 px-6 py-3 text-xs text-emerald-900">
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
                <span className="inline-flex items-center gap-1.5 font-semibold">
                  <CheckCircle2 className="h-4 w-4" />
                  Training Data updated
                </span>
                <span>
                  Saved to: <strong>{trainingDataUpdate.training_data_name || trainingDataUpdate.training_data_id || "Training Data"}</strong>
                </span>
                <span>
                  Evaluation step: <strong>{trainingDataUpdate.model_evaluation_step || trainingDataUpdate.score_reference_step || "-"}</strong>
                </span>
                {trainingDataUpdate.source_model_name && (
                  <span>
                    Model: <strong>{trainingDataUpdate.source_model_name}</strong>
                  </span>
                )}
                {trainingDataUpdate.run_id && (
                  <span>
                    Run: <span className="font-mono">{trainingDataUpdate.run_id}</span>
                  </span>
                )}
                {trainingDataUpdate.updated_at && (
                  <span className="text-emerald-700">
                    {new Date(trainingDataUpdate.updated_at).toLocaleString()}
                  </span>
                )}
              </div>
            </div>
          )}
          <div className="p-4">
            <div className="max-h-[520px] overflow-auto rounded-lg border border-gray-200 bg-white">
              <table className="min-w-[2700px] w-full text-xs">
                <thead className="bg-slate-100 text-slate-700">
                  <tr>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Run</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Model</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Actual Value</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Final Multiplier</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Final Formula</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Best Loss</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Final Loss (- better)</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Best PSNR</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Final PSNR (+ better)</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Best SSIM</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Final SSIM (+ better)</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Best LPIPS</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Final LPIPS (- better)</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Run End Time (s)</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Base End Time (s)</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Run Best (l,q,t,s)</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Run End (l,q,t,s)</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Base Best (l,q,t,s)</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Base End (l,q,t,s)</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">S Best</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">S End</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">S Run</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">S Base Best</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">S Base End</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">S Base</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Score</th>
                    <th className="sticky top-0 z-10 bg-slate-100 px-2 py-2 text-left font-semibold">Remarks</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.length === 0 ? (
                    <tr>
                      <td className="px-3 py-6 text-slate-500" colSpan={27}>
                        {loading ? "Loading project test results..." : message || "No project test results available yet."}
                      </td>
                    </tr>
                  ) : (
                    rows.map((row) => (
                      <tr key={row.run_id} className={`border-t border-gray-100 align-top ${row.is_baseline_row ? "bg-amber-50" : ""}`}>
                        <td className="px-2 py-2 text-slate-800">
                          <div className="font-semibold">{row.run_name || row.run_id}</div>
                          <div className="text-[10px] text-slate-500">{row.run_id}</div>
                        </td>
                        <td className="px-2 py-2 text-slate-700">{row.model_id || "-"}</td>
                        <td className="max-w-[360px] whitespace-pre-wrap break-words px-2 py-2 font-mono text-[10px] text-slate-700">
                          <div className="space-y-0.5">
                            {renderLearningParamRows(row.run_id, row.learning_param_rows, "actual")}
                            <div className="mt-1 font-sans text-[10px] text-slate-500">
                              {fmtLearnedParamSource(row.learned_input_params_source, row.learned_input_params_status)}
                            </div>
                          </div>
                        </td>
                        <td className="max-w-[360px] whitespace-pre-wrap break-words px-2 py-2 font-mono text-[10px] text-slate-700">
                          {renderLearningParamRows(row.run_id, row.learning_param_rows, "final_multiplier")}
                        </td>
                        <td className="max-w-[360px] whitespace-pre-wrap break-words px-2 py-2 font-mono text-[10px] text-slate-700">
                          {(row.final_multiplier_formula || "").replaceAll("jitter", "log_multiplier")}
                        </td>
                        <td className="px-2 py-2 text-slate-700">{fmt(row.best_loss)} @ {fmtStep(row.best_loss_step)}</td>
                        <td className="px-2 py-2 text-slate-700">{fmt(row.final_loss)} @ {fmtStep(row.final_loss_step)}</td>
                        <td className="px-2 py-2 text-slate-700">{fmt(row.best_psnr, 4)} @ {fmtStep(row.best_psnr_step)}</td>
                        <td className="px-2 py-2 text-slate-700">{fmt(row.final_psnr, 4)} @ {fmtStep(row.final_psnr_step)}</td>
                        <td className="px-2 py-2 text-slate-700">{fmt(row.best_ssim, 4)} @ {fmtStep(row.best_ssim_step)}</td>
                        <td className="px-2 py-2 text-slate-700">{fmt(row.final_ssim, 4)} @ {fmtStep(row.final_ssim_step)}</td>
                        <td className="px-2 py-2 text-slate-700">{fmt(row.best_lpips, 4)} @ {fmtStep(row.best_lpips_step)}</td>
                        <td className="px-2 py-2 text-slate-700">{fmt(row.final_lpips, 4)} @ {fmtStep(row.final_lpips_step)}</td>
                        <td className="px-2 py-2 text-slate-700">{fmt(row.run_end_elapsed, 2)}</td>
                        <td className="px-2 py-2 text-slate-700">{fmt(row.base_end_elapsed, 2)}</td>
                        <td className="px-2 py-2 text-slate-700">{fmt(row.run_best_l, 4)}, {fmt(row.run_best_q, 4)}, {fmt(row.run_best_t, 4)}, {fmt(row.run_best_s, 4)}</td>
                        <td className="px-2 py-2 text-slate-700">{fmt(row.run_end_l, 4)}, {fmt(row.run_end_q, 4)}, {fmt(row.run_end_t, 4)}, {fmt(row.run_end_s, 4)}</td>
                        <td className="px-2 py-2 text-slate-700">{fmt(row.base_best_l, 4)}, {fmt(row.base_best_q, 4)}, {fmt(row.base_best_t, 4)}, {fmt(row.s_base_best, 4)}</td>
                        <td className="px-2 py-2 text-slate-700">{fmt(row.base_end_l, 4)}, {fmt(row.base_end_q, 4)}, {fmt(row.base_end_t, 4)}, {fmt(row.s_base_end, 4)}</td>
                        <td className="px-2 py-2 text-slate-700">{fmt(row.s_best, 6)}</td>
                        <td className="px-2 py-2 text-slate-700">{fmt(row.s_end, 6)}</td>
                        <td className="px-2 py-2 text-slate-700">{fmt(row.s_run, 6)}</td>
                        <td className="px-2 py-2 text-slate-700">{fmt(row.s_base_best, 6)}</td>
                        <td className="px-2 py-2 text-slate-700">{fmt(row.s_base_end, 6)}</td>
                        <td className="px-2 py-2 text-slate-700">{fmt(row.s_base, 6)}</td>
                        <td className="px-2 py-2 font-semibold text-slate-700">{fmt(row.score, 6)}</td>
                        <td className="px-2 py-2 text-slate-700">{row.remarks || "-"}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

