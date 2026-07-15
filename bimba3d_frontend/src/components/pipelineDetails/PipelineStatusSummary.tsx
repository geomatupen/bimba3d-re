import { AlertTriangle, CheckCircle2, Clock, Database, Gauge } from "lucide-react";
import type { PipelineDetail } from "./types";

interface PipelineStatusSummaryProps {
  pipeline: PipelineDetail;
}

const formatDate = (value?: string | null) => {
  if (!value) return "-";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
};

export default function PipelineStatusSummary({ pipeline }: PipelineStatusSummaryProps) {
  const progress = pipeline.total_runs > 0 ? Math.round((pipeline.completed_runs / pipeline.total_runs) * 100) : 0;
  const hardCapRuns = Number(pipeline.hard_cap_runs || 0);

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="grid gap-4 lg:grid-cols-5">
        <div>
          <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg bg-blue-50 text-blue-700">
            <Database className="h-4 w-4" />
          </div>
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Runs</div>
          <div className="mt-1 text-2xl font-bold text-slate-950">
            {pipeline.completed_runs}/{pipeline.total_runs}
          </div>
        </div>
        <div>
          <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg bg-emerald-50 text-emerald-700">
            <CheckCircle2 className="h-4 w-4" />
          </div>
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Success</div>
          <div className="mt-1 text-2xl font-bold text-slate-950">
            {pipeline.success_rate === null ? "-" : `${pipeline.success_rate.toFixed(1)}%`}
          </div>
        </div>
        <div>
          <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg bg-amber-50 text-amber-700">
            <Clock className="h-4 w-4" />
          </div>
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Current</div>
          <div className="mt-1 text-sm font-semibold text-slate-950">
            Phase {pipeline.current_phase || 0}, Run {pipeline.current_run || 0}
          </div>
          <div className="mt-1 text-xs text-slate-500">Created {formatDate(pipeline.created_at)}</div>
        </div>
        <div>
          <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg bg-rose-50 text-rose-700">
            <AlertTriangle className="h-4 w-4" />
          </div>
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Failed</div>
          <div className="mt-1 text-2xl font-bold text-slate-950">{pipeline.failed_runs}</div>
        </div>
        <div>
          <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg bg-amber-50 text-amber-700">
            <Gauge className="h-4 w-4" />
          </div>
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Hard Cap</div>
          <div className="mt-1 text-2xl font-bold text-slate-950">{hardCapRuns}</div>
        </div>
      </div>

      <div className="mt-5">
        <div className="mb-1 flex justify-between text-xs font-semibold text-slate-600">
          <span>Progress</span>
          <span>{progress}%</span>
        </div>
        <div className="h-2.5 overflow-hidden rounded-full bg-slate-100 shadow-inner">
          <div className="h-full rounded-full bg-gradient-to-r from-blue-500 to-indigo-600" style={{ width: `${progress}%` }} />
        </div>
      </div>

      {pipeline.last_error && (
        <div className="mt-4 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          <strong>Error:</strong> {pipeline.last_error}
        </div>
      )}
    </section>
  );
}
