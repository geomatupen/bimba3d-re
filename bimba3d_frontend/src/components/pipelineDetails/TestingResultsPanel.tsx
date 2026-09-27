import type { PipelineDetail } from "./types";
import TrainingDataRowsTable from "../TrainingDataRowsTable";
import { Info } from "lucide-react";

interface TestingResultsPanelProps {
  pipeline: PipelineDetail;
}

export default function TestingResultsPanel({ pipeline }: TestingResultsPanelProps) {
  const controls = ((pipeline.runs || []) as any[]).filter((run) => run.controlled_experiment);
  const metricDifference = (run: any, key: "psnr" | "ssim" | "lpips", referenceKind: "model" | "baseline") => {
    const reference = referenceKind === "model" ? run.reference_model_metrics?.[key] : run.reference_baseline_metrics?.[key];
    const actual = run[key];
    if ((referenceKind === "model" && run.reference_model_hard_cap) || typeof reference !== "number" || typeof actual !== "number") return "-";
    const delta = key === "lpips" ? reference - actual : actual - reference;
    return `${delta >= 0 ? "+" : ""}${delta.toFixed(key === "psnr" ? 3 : 4)}`;
  };
  const metricCell = (run: any, key: "psnr" | "ssim" | "lpips") => {
    const value = run[key];
    if (typeof value !== "number") return "-";
    const step = typeof run.actual_step === "number" ? ` @${run.actual_step.toLocaleString()}` : "";
    const modelDifference = metricDifference(run, key, "model");
    const baselineDifference = metricDifference(run, key, "baseline");
    return <><div className="font-medium">{value.toFixed(key === "psnr" ? 3 : 4)}{step}</div><div className="whitespace-nowrap text-slate-500">{modelDifference} vs M · {baselineDifference} vs B</div></>;
  };
  const configuredModelIds = Array.isArray(pipeline.config?.source_model_ids)
    ? pipeline.config.source_model_ids.filter(Boolean)
    : pipeline.config?.source_model_id
      ? [pipeline.config.source_model_id]
      : [];

  return (
    <div className="space-y-4">
    {controls.length > 0 && <section className="overflow-x-auto rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="mb-3 flex items-center gap-1 text-base font-semibold text-slate-950">Additional Experiments <span className="text-slate-500" title="M is the selected model run. B is the project's original baseline run. Positive deltas mean the controlled run performed better; for LPIPS, lower values are better." aria-label="Metric comparison information"><Info size={14} aria-hidden="true" /></span></h2>
      <table className="w-full min-w-[900px] text-left text-xs">
        <thead><tr className="border-b border-slate-200 text-slate-600"><th className="py-2">Project</th><th>Control</th><th>Reference run</th><th>Target / achieved</th><th>PSNR @ step</th><th>SSIM @ step</th><th>LPIPS @ step</th><th>Status</th></tr></thead>
        <tbody>{controls.map((run) => {
          const isTime = run.controlled_experiment === "time_constrained_test";
          const target = isTime ? run.target_time_seconds : run.target_gaussians;
          const actual = isTime && run.time_budget_basis === "training_loop_pre_eval" ? run.actual_training_loop_seconds : isTime ? run.actual_time_seconds : run.actual_gaussians;
          const matched = typeof target === "number" && target > 0 && typeof actual === "number" && Math.abs(actual - target) / target <= 0.05;
          const formatted = (value: unknown) => typeof value === "number" ? isTime ? `${value.toFixed(1)} s` : value.toLocaleString() : "-";
          const atLimit = typeof run.actual_step === "number" && typeof run.max_steps_ceiling === "number" && run.actual_step >= run.max_steps_ceiling;
          const status = run.reference_model_hard_cap ? "Reference hard cap"
            : run.status !== "success" ? run.status
            : !isTime && !run.gaussian_cap_freeze_applied
              ? atLimit ? `Target not reached: ${formatted(actual)} at step ${run.actual_step.toLocaleString()} (limit)` : `Target not reached: ${formatted(actual)} at step ${run.actual_step?.toLocaleString() ?? "-"}`
            : !isTime && typeof run.gaussian_cap_step === "number" && typeof run.actual_step === "number" && run.actual_step - run.gaussian_cap_step < 1000 ? "Ceiling before +1,000 steps"
            : matched ? "Within 5%" : atLimit ? "Step ceiling; unmatched" : "Budget not matched";
          return <tr key={run.run_id || `${run.project_name}-${run.controlled_experiment}`} className="border-b border-slate-100">
            <td className="py-2 pr-3">{run.project_name}</td>
            <td className="pr-3">{isTime ? "Time" : "Gaussian"}</td>
            <td className="pr-3 font-mono" title={run.reference_model_run_id}>{run.reference_model_run_id || "-"}</td>
            <td className="pr-3" title={!isTime ? "Final Gaussian count may differ from the count when growth was frozen" : undefined}>
              {formatted(target)} / {formatted(actual)}
              {isTime && <span className="ml-1 inline-flex items-center gap-1 text-slate-500">
                {run.time_budget_basis === "training_loop_pre_eval" ? "(loop)" : "(legacy total)"}
                <span title={run.time_budget_basis === "training_loop_pre_eval" ? "This is time spent in the training loop before final evaluation. The report total also includes evaluation, export, and other post-training work." : "This older control uses total training time, including final evaluation."} aria-label="Time measurement information"><Info size={13} aria-hidden="true" /></span>
              </span>}
              {isTime && typeof run.reference_model_metrics?.reported_total_seconds === "number" && <div className="text-slate-500">Model report total: {run.reference_model_metrics.reported_total_seconds.toFixed(1)} s</div>}
            </td>
            <td className="pr-3">{metricCell(run, "psnr")}</td>
            <td className="pr-3">{metricCell(run, "ssim")}</td>
            <td className="pr-3">{metricCell(run, "lpips")}</td>
            <td>{status}</td>
          </tr>;
        })}</tbody>
      </table>
    </section>}
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-slate-950">Training Data Rows</h2>
        <p className="mt-1 text-sm text-slate-600">
          Recorded test rows with model ids, selected multipliers, scores, score terms, and report values.
        </p>
        {configuredModelIds.length > 0 && (
          <p className="mt-1 text-xs text-slate-500">
            Showing rows for all {configuredModelIds.length} configured model{configuredModelIds.length === 1 ? "" : "s"} in this test pipeline.
          </p>
        )}
      </div>
      <TrainingDataRowsTable pipelineId={pipeline.id} showFinalMetricDeltas excludeControlledExperiments />
    </section>
    </div>
  );
}
