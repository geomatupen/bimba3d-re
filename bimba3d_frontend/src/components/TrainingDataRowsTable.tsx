import { RefreshCw } from "lucide-react";
import { useEffect, useState, useCallback } from "react";
import { api } from "../api/client";
import LearningParamRows from "./learning/LearningParamRows";

interface TrainingDataRowsTableRow {
 project_name: string;
 run_id: string;
 run_name?: string | null;
 is_baseline_row?: boolean;
 selected_preset?: string | null;
 learned_input_params?: Record<string, unknown> | null;
 selected_multipliers?: Record<string, unknown> | null;
 final_multiplier_formula?: string | null;
 learning_param_rows?: Array<{
 key: string;
 actual: number | null;
 selected_multiplier: number | null;
 selected_multiplier_raw?: number | null;
 log_multiplier?: number | null;
 jitter?: number | null;
 final_multiplier: number | null;
 }> | null;
 learned_input_params_source?: string | null;
 learned_input_params_status?: string | null;
 best_loss?: number | null;
 best_loss_step?: number | null;
 final_loss?: number | null;
 final_loss_step?: number | null;
 best_psnr?: number | null;
 best_psnr_step?: number | null;
 final_psnr?: number | null;
 final_psnr_step?: number | null;
 best_ssim?: number | null;
 best_ssim_step?: number | null;
 final_ssim?: number | null;
 final_ssim_step?: number | null;
 best_lpips?: number | null;
 best_lpips_step?: number | null;
 final_lpips?: number | null;
 final_lpips_step?: number | null;
 baseline_final_psnr?: number | null;
 baseline_final_ssim?: number | null;
 baseline_final_lpips?: number | null;
 delta_psnr?: number | null;
 delta_ssim?: number | null;
 delta_lpips?: number | null;
 time_seconds?: number | null;
 time_diff_seconds?: number | null;
 run_best_l?: number | null;
 run_best_q?: number | null;
 run_best_t?: number | null;
 run_best_s?: number | null;
 run_end_l?: number | null;
 run_end_q?: number | null;
 run_end_t?: number | null;
 run_end_s?: number | null;
 s_best?: number | null;
 s_end?: number | null;
 s_run?: number | null;
 s_base_best?: number | null;
 s_base_end?: number | null;
 s_base?: number | null;
 // Score decomposition (document 4.X.8)
 relative_quality_score?: number | null;
 convergence_score?: number | null;
 auc_loss_run?: number | null;
 auc_loss_base?: number | null;
 // Loss @ reference step used to compute R_conv
 score_reference_step?: number | null;
 loss_at_reference_step_run?: number | null;
 loss_at_reference_step_base?: number | null;
 exploration_mode?: string | null;
 remarks?: string | null;
 model_id?: string | null;
 test_model_id?: string | null;
 source_model_id?: string | null;
 source_workflow_model_id?: string | null;
 source_model_name?: string | null;
 source_model_family?: string | null;
}

function renderLearningParamRows(
 runId: string,
 rows: TrainingDataRowsTableRow["learning_param_rows"],
 field: "actual" | "selected_multiplier" | "jitter" | "log_multiplier" | "final_multiplier",
) {
 return <LearningParamRows runId={runId} rows={rows} field={field === "jitter" ? "log_multiplier" : field} />;
}

function scoreCell(value: number | null | undefined, decimals = 6) {
 if (typeof value !== "number") return <span className="text-slate-400">-</span>;
 const cls = value > 0 ? "text-green-700" : value < 0 ? "text-red-700" : "text-slate-700";
 return <span className={`font-semibold ${cls}`}>{value.toFixed(decimals)}</span>;
}

function metricAtStep(value: number | null | undefined, step: number | null | undefined, decimals = 6) {
 if (typeof value !== "number") return <span className="text-slate-400">-</span>;
 return (
 <span>
 {value.toFixed(decimals)} <span className="text-slate-400">@ {typeof step === "number" ? step.toLocaleString() : "-"}</span>
 </span>
 );
}

function lossAtReference(value: number | null | undefined, step: number | null | undefined) {
 return metricAtStep(value, step, 6);
}

function finalMetricDelta(value: number | null | undefined) {
 return scoreCell(value, 4);
}

function formatSeconds(value: number | null | undefined) {
 if (typeof value !== "number") return <span className="text-slate-400">-</span>;
 return <span>{value.toFixed(1)}s</span>;
}

export default function TrainingDataRowsTable({
 pipelineId,
 selectedModelId,
 showFinalMetricDeltas = false,
}: {
 pipelineId: string;
 selectedModelId?: string | null;
 showFinalMetricDeltas?: boolean;
}) {
 const [rows, setRows] = useState<TrainingDataRowsTableRow[]>([]);
 const [loading, setLoading] = useState(false);
 const [message, setMessage] = useState<string | null>(null);

 const fetchTable = useCallback(async () => {
 if (!pipelineId) return;
 setLoading(true);
 try {
 const res = await api.get(`/api/workflow/pipelines/${pipelineId}/learning-rows`);
 const data = Array.isArray(res.data?.rows) ? res.data.rows : [];
 setRows(data);
 setMessage(typeof res.data?.message === "string" ? res.data.message : null);
 } catch (err) {
 console.error("Failed to fetch training data rows", err);
 setRows([]);
 setMessage(null);
 } finally {
 setLoading(false);
 }
 }, [pipelineId]);

 useEffect(() => {
 void fetchTable();
 }, [fetchTable]);

 if (loading && rows.length === 0) {
 return (
 <div className="flex items-center justify-center py-12">
 <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
 </div>
 );
 }

 const rowModelKey = (row: TrainingDataRowsTableRow) =>
 String((row as any).model_id || (row as any).test_model_id || (row as any).source_workflow_model_id || (row as any).source_model_id || "");

 const displayRows = selectedModelId
 ? rows.filter((row) => {
 if (row.is_baseline_row) return true; // always show baseline rows for context
 return rowModelKey(row) === selectedModelId;
 })
 : rows;
 const visibleModelIds = Array.from(
 new Set(displayRows.map(rowModelKey).filter((modelId) => modelId.trim().length > 0)),
 );
 const hasFinalMetricDeltas = showFinalMetricDeltas || displayRows.some(
 (row) =>
 typeof row.delta_psnr === "number" ||
 typeof row.delta_ssim === "number" ||
 typeof row.delta_lpips === "number",
 );

 if (rows.length === 0) {
 return (
 <div className="text-center py-12 text-gray-500">
 {message || "No training data rows available yet. Complete preparation runs to see results."}
 </div>
 );
 }

 if (displayRows.length === 0 && selectedModelId) {
 return (
 <div className="text-center py-12 text-gray-500">
 No rows for model <span className="font-mono text-xs">{selectedModelId}</span> yet. Rows appear once runs complete successfully.
 </div>
 );
 }

 const TH = ({ children }: { children: React.ReactNode }) => (
 <th className="sticky top-0 z-10 whitespace-nowrap bg-slate-100 px-1.5 py-1 text-left font-semibold">
 {children}
 </th>
 );

 return (
 <div className="space-y-2">
 <div className="flex flex-wrap items-center justify-between gap-2">
 {selectedModelId && (
 <span className="rounded-full border border-amber-200 bg-amber-50 px-2.5 py-0.5 text-[11px] font-semibold text-amber-800">
 Model: {selectedModelId}
 </span>
 )}
 {!selectedModelId && visibleModelIds.length > 0 && (
 <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-0.5 text-[11px] font-semibold text-slate-700">
 Models shown: {visibleModelIds.length}
 </span>
 )}
 <div className="ml-auto">
 <button
 type="button"
 onClick={() => void fetchTable()}
 disabled={loading}
 className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
 >
 <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
 {loading ? "Refreshing" : "Refresh"}
 </button>
 </div>
 </div>
 <div className="overflow-auto max-h-[600px] bg-white border border-gray-200 rounded-lg">
 <table className={`${hasFinalMetricDeltas ? "min-w-[3300px]" : "min-w-[3000px]"} w-full text-[11px] leading-tight`}>
 <thead className="bg-slate-100 text-slate-700">
 <tr>
 <TH>Project</TH>
 <TH>Run</TH>
 <TH>Model</TH>
 <TH>Baseline</TH>
 <TH>Actual Value</TH>
 <TH>Selected Multiplier</TH>
 <TH>Log-Space Multiplier</TH>
 <TH>Applied Multiplier</TH>
 <TH>Final Formula</TH>
 <TH>Best Loss</TH>
            <TH>Final Loss (lower is better)</TH>
 <TH>Best PSNR</TH>
            <TH>Final PSNR (higher is better)</TH>
 <TH>Best SSIM</TH>
            <TH>Final SSIM (higher is better)</TH>
 <TH>Best LPIPS</TH>
            <TH>Final LPIPS (lower is better)</TH>
 {hasFinalMetricDeltas && (
 <>
 <TH>Δ PSNR</TH>
 <TH>Δ SSIM</TH>
 <TH>Δ LPIPS</TH>
 </>
 )}
 <TH>Run Best (l,q,t,s)</TH>
 <TH>Run End (l,q,t,s)</TH>
 <TH>S Best</TH>
 <TH>S End</TH>
 <TH>S Run</TH>
 <TH>S Base Best</TH>
 <TH>S Base End</TH>
 <TH>S Base</TH>
 <TH>R Quality</TH>
 <TH>Time</TH>
 <TH>Time Diff</TH>
 <TH>R Conv</TH>
 <TH>Loss @ Ref Run</TH>
 <TH>Loss @ Ref Base</TH>
 <TH>AUC Run</TH>
 <TH>AUC Base</TH>
 <TH>Explore</TH>
 <TH>Remarks</TH>
 </tr>
 </thead>
 <tbody>
 {displayRows.map((row) => (
 <tr
 key={`${row.project_name}-${row.run_id}`}
 className={`border-t border-gray-100 align-top ${row.is_baseline_row ? "bg-amber-50" : ""}`}
 >
 <td className="px-1.5 py-1 align-top font-medium text-slate-800">{row.project_name}</td>
 <td className="px-1.5 py-1 align-top text-slate-800">
 <div className="font-semibold">{row.run_name || row.run_id}</div>
 <div className="text-[10px] text-slate-500">{row.run_id}</div>
 </td>
 <td className="max-w-[260px] px-1.5 py-1 align-top text-slate-700">
 {row.is_baseline_row && !rowModelKey(row) ? (
 <span className="rounded bg-amber-100 px-1.5 py-0.5 text-[10px] font-semibold text-amber-800">Baseline</span>
 ) : (
 <>
 <div className="truncate font-semibold" title={row.source_model_name || rowModelKey(row)}>
 {row.source_model_name || rowModelKey(row) || "-"}
 </div>
 {rowModelKey(row) && row.source_model_name !== rowModelKey(row) && (
 <div className="truncate font-mono text-[10px] text-slate-500" title={rowModelKey(row)}>
 {rowModelKey(row)}
 </div>
 )}
 {row.source_model_family && (
 <div className="truncate text-[10px] text-slate-500" title={row.source_model_family}>
 {row.source_model_family}
 </div>
 )}
 </>
 )}
 </td>
 <td className="px-1.5 py-1 text-center align-top">
 {row.is_baseline_row ? (
 <span className="inline-block rounded bg-amber-200 px-1.5 py-0.5 text-[10px] font-medium text-amber-900">
 Baseline
 </span>
 ) : (
 <span className="text-slate-400">-</span>
 )}
 </td>
 <td className="px-1.5 py-1 align-top font-mono text-[10px] text-slate-700">
 <div className="space-y-0.5">
 {renderLearningParamRows(row.run_id, row.learning_param_rows, "actual")}
 {row.learned_input_params_source && (
 <div className="mt-1 text-[10px] font-sans text-slate-500 italic">
 ({row.learned_input_params_source})
 </div>
 )}
 </div>
 </td>
 <td className="px-1.5 py-1 align-top font-mono text-[10px] text-slate-700">
 {renderLearningParamRows(row.run_id, row.learning_param_rows, "selected_multiplier")}
 </td>
 <td className="px-1.5 py-1 align-top font-mono text-[10px] text-slate-700">
 {renderLearningParamRows(row.run_id, row.learning_param_rows, "log_multiplier")}
 </td>
 <td className="px-1.5 py-1 align-top font-mono text-[10px] text-slate-700">
 {renderLearningParamRows(row.run_id, row.learning_param_rows, "final_multiplier")}
 </td>
 <td className="whitespace-pre-wrap break-words px-1.5 py-1 align-top font-mono text-[10px] text-slate-700">
 {(row.final_multiplier_formula || "").replaceAll("jitter", "log_multiplier")}
 </td>
 <td className="px-1.5 py-1 align-top text-slate-700">
 {metricAtStep(row.best_loss, row.best_loss_step)}
 </td>
 <td className="px-1.5 py-1 align-top text-slate-700">
 {metricAtStep(row.final_loss, row.final_loss_step)}
 </td>
 <td className="px-1.5 py-1 align-top text-slate-700">
 {metricAtStep(row.best_psnr, row.best_psnr_step, 4)}
 </td>
 <td className="px-1.5 py-1 align-top text-slate-700">
 {metricAtStep(row.final_psnr, row.final_psnr_step, 4)}
 </td>
 <td className="px-1.5 py-1 align-top text-slate-700">
 {metricAtStep(row.best_ssim, row.best_ssim_step, 4)}
 </td>
 <td className="px-1.5 py-1 align-top text-slate-700">
 {metricAtStep(row.final_ssim, row.final_ssim_step, 4)}
 </td>
 <td className="px-1.5 py-1 align-top text-slate-700">
 {metricAtStep(row.best_lpips, row.best_lpips_step, 4)}
 </td>
 <td className="px-1.5 py-1 align-top text-slate-700">
 {metricAtStep(row.final_lpips, row.final_lpips_step, 4)}
 </td>
 {hasFinalMetricDeltas && (
 <>
 <td className="px-1.5 py-1 align-top" title={`Baseline final PSNR: ${typeof row.baseline_final_psnr === "number" ? row.baseline_final_psnr.toFixed(4) : "-"}`}>
 {finalMetricDelta(row.delta_psnr)}
 </td>
 <td className="px-1.5 py-1 align-top" title={`Baseline final SSIM: ${typeof row.baseline_final_ssim === "number" ? row.baseline_final_ssim.toFixed(4) : "-"}`}>
 {finalMetricDelta(row.delta_ssim)}
 </td>
 <td className="px-1.5 py-1 align-top" title={`Baseline final LPIPS: ${typeof row.baseline_final_lpips === "number" ? row.baseline_final_lpips.toFixed(4) : "-"}`}>
 {finalMetricDelta(row.delta_lpips)}
 </td>
 </>
 )}
 <td className="px-1.5 py-1 align-top text-slate-700">
 {[row.run_best_l, row.run_best_q, row.run_best_t, row.run_best_s]
 .map((v) => (typeof v === "number" ? v.toFixed(4) : "-"))
 .join(", ")}
 </td>
 <td className="px-1.5 py-1 align-top text-slate-700">
 {[row.run_end_l, row.run_end_q, row.run_end_t, row.run_end_s]
 .map((v) => (typeof v === "number" ? v.toFixed(4) : "-"))
 .join(", ")}
 </td>
 <td className="px-1.5 py-1 align-top text-slate-700">
 {typeof row.s_best === "number" ? row.s_best.toFixed(6) : "-"}
 </td>
 <td className="px-1.5 py-1 align-top text-slate-700">
 {typeof row.s_end === "number" ? row.s_end.toFixed(6) : "-"}
 </td>
 <td className="px-1.5 py-1 align-top text-slate-700">
 {typeof row.s_run === "number" ? row.s_run.toFixed(6) : "-"}
 </td>
 <td className="px-1.5 py-1 align-top text-slate-700">
 {typeof row.s_base_best === "number" ? row.s_base_best.toFixed(6) : "-"}
 </td>
 <td className="px-1.5 py-1 align-top text-slate-700">
 {typeof row.s_base_end === "number" ? row.s_base_end.toFixed(6) : "-"}
 </td>
 <td className="px-2 py-2 text-slate-700">
 {typeof row.s_base === "number" ? row.s_base.toFixed(6) : "-"}
 </td>
 {/* R_quality blank for baseline runs */}
 <td className="px-1.5 py-1 align-top">
 {row.is_baseline_row ? <span className="text-slate-400 text-[10px]"></span> : scoreCell(row.relative_quality_score)}
 </td>
 <td className="px-1.5 py-1 align-top text-slate-700">
 {formatSeconds(row.time_seconds)}
 </td>
 <td className="px-1.5 py-1 align-top">
 {row.is_baseline_row ? <span className="text-slate-400 text-[10px]"></span> : scoreCell(row.time_diff_seconds, 1)}
 </td>
 {/* R_conv = Loss_reference_step_baseline Loss_reference_step_run */}
 <td className="px-1.5 py-1 align-top">
 {row.is_baseline_row ? <span className="text-slate-400 text-[10px]"></span> : scoreCell(row.convergence_score, 6)}
 </td>
 {/* Loss @ reference step for this run */}
 <td className="px-1.5 py-1 align-top font-mono text-[10px] text-slate-700">
 {lossAtReference(row.loss_at_reference_step_run, row.score_reference_step)}
 </td>
 {/* Loss @ reference step for baseline */}
 <td className="px-1.5 py-1 align-top font-mono text-[10px] text-slate-700">
 {lossAtReference(row.loss_at_reference_step_base, row.score_reference_step)}
 </td>
 {/* AUC informational */}
 <td className="px-1.5 py-1 align-top font-mono text-[10px] text-slate-700">
 {typeof row.auc_loss_run === "number" ? row.auc_loss_run.toFixed(2) : "-"}
 </td>
 <td className="px-1.5 py-1 align-top font-mono text-[10px] text-slate-700">
 {typeof row.auc_loss_base === "number" ? row.auc_loss_base.toFixed(2) : "-"}
 </td>
 {/* Thompson Sampling mode used at prediction time */}
 <td className="px-1.5 py-1 align-top text-[10px] text-slate-700">
 {row.exploration_mode || "-"}
 </td>
 <td className="max-w-[200px] break-words px-1.5 py-1 align-top text-[10px] text-slate-600">
 {row.remarks || "-"}
 </td>
 </tr>
 ))}
 </tbody>
 </table>
 </div>
 </div>
 );
}
