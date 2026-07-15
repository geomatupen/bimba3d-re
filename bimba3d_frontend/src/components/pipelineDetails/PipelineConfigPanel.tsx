import type { PipelineDetail } from "./types";

interface PipelineConfigPanelProps {
  pipeline: PipelineDetail;
}

export default function PipelineConfigPanel({ pipeline }: PipelineConfigPanelProps) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <h2 className="text-lg font-semibold text-slate-950">Configuration</h2>
      <div className="mt-4 grid gap-3 md:grid-cols-3">
        <div className="rounded-lg border border-slate-200 p-3">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Pipeline Type</div>
          <div className="mt-1 text-sm font-semibold text-slate-950">{pipeline.pipeline_type || pipeline.config?.pipeline_type || "training_data"}</div>
        </div>
        <div className="rounded-lg border border-slate-200 p-3">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Max Steps</div>
          <div className="mt-1 text-sm font-semibold text-slate-950">{pipeline.config?.shared_config?.max_steps ?? "-"}</div>
        </div>
        <div className="rounded-lg border border-slate-200 p-3">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Eval Interval</div>
          <div className="mt-1 text-sm font-semibold text-slate-950">{pipeline.config?.shared_config?.eval_interval ?? "-"}</div>
        </div>
      </div>
      <pre className="mt-4 max-h-[520px] overflow-auto rounded-lg bg-slate-950 p-4 text-xs text-slate-100">
        {JSON.stringify(pipeline.config || {}, null, 2)}
      </pre>
    </section>
  );
}
