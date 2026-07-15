import { ArrowLeft, RefreshCw } from "lucide-react";
import { useNavigate } from "react-router-dom";
import type { ReactNode } from "react";
import PipelineBreadcrumbs from "./PipelineBreadcrumbs";
import type { PipelineDetail } from "./types";

interface PipelineDetailShellProps {
  actions?: ReactNode;
  badge: string;
  children: ReactNode;
  breadcrumbs?: { label: string; to?: string }[];
  onRefresh: () => void;
  pipeline: PipelineDetail;
  refreshing?: boolean;
}

const statusClasses = (status: string) => {
  switch (status.toLowerCase()) {
    case "running":
      return "border-green-200 bg-green-50 text-green-700";
    case "paused":
      return "border-yellow-200 bg-yellow-50 text-yellow-700";
    case "completed":
      return "border-blue-200 bg-blue-50 text-blue-700";
    case "completed_with_hard_caps":
    case "hard_cap_reached":
      return "border-orange-200 bg-orange-50 text-orange-700";
    case "failed":
    case "completed_with_failures":
      return "border-red-200 bg-red-50 text-red-700";
    default:
      return "border-slate-200 bg-slate-50 text-slate-700";
  }
};

export default function PipelineDetailShell({
  actions,
  badge,
  breadcrumbs = [],
  children,
  onRefresh,
  pipeline,
  refreshing = false,
}: PipelineDetailShellProps) {
  const navigate = useNavigate();
  const goBack = () => {
    if (window.history.length > 1) {
      navigate(-1);
      return;
    }
    navigate("/all-pipelines");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50">
      <header className="bg-gradient-to-r from-blue-600 via-blue-700 to-indigo-700 shadow-xl">
        <div className="mx-auto max-w-7xl px-6 py-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={goBack}
                className="inline-flex items-center gap-2 rounded-xl border border-white/20 bg-white/10 px-3 py-2 text-sm font-medium text-white backdrop-blur-sm transition-all duration-200 hover:scale-105 hover:bg-white/20"
              >
                <ArrowLeft className="h-4 w-4 text-white" />
                Back
              </button>
              <div className="min-w-0">
                <div className="mb-2">
                  <PipelineBreadcrumbs items={breadcrumbs} />
                </div>
                <div className="mb-1 inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/10 px-2 py-0.5 text-xs font-semibold uppercase tracking-wide text-white">
                  {badge}
                </div>
                <h1 className="truncate text-2xl font-bold text-white">{pipeline.name}</h1>
                <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-blue-100">
                  <span>{pipeline.id}</span>
                  <span className={`rounded-full border px-2 py-0.5 font-semibold ${statusClasses(pipeline.status)}`}>
                    {pipeline.status}
                  </span>
                </div>
              </div>
            </div>
            <div className="flex flex-col items-start gap-2 sm:flex-row sm:items-center lg:justify-end">
              {actions}
              <button
                onClick={onRefresh}
                disabled={refreshing}
                className="inline-flex items-center justify-center gap-2 rounded-xl border border-white/20 bg-white/10 px-4 py-2 text-sm font-semibold text-white backdrop-blur-sm transition hover:bg-white/20 disabled:opacity-60"
              >
                <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
                Refresh
              </button>
            </div>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-7xl space-y-4 px-6 py-6">{children}</main>
    </div>
  );
}
