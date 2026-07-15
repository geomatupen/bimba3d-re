import { ArrowLeft } from "lucide-react";
import { useNavigate } from "react-router-dom";
import type { ReactNode } from "react";
import PipelineBreadcrumbs from "../pipelineDetails/PipelineBreadcrumbs";

interface WorkflowShellProps {
  title: string;
  eyebrow: string;
  children: ReactNode;
  backTo?: string;
  breadcrumbs?: { label: string; to?: string }[];
}

export default function WorkflowShell({ title, eyebrow, children, backTo = "/", breadcrumbs = [] }: WorkflowShellProps) {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50">
      <header className="bg-gradient-to-r from-blue-600 via-blue-700 to-indigo-700 shadow-xl">
        <div className="mx-auto max-w-7xl px-6 py-6">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate(backTo)}
              className="inline-flex items-center gap-2 rounded-xl border border-white/20 bg-white/10 px-3 py-2 text-sm font-medium text-white backdrop-blur-sm transition-all duration-200 hover:scale-105 hover:bg-white/20"
              aria-label="Back"
            >
              <ArrowLeft className="h-4 w-4 text-white" />
              Back
            </button>
            <div>
              {breadcrumbs.length > 0 && (
                <div className="mb-2">
                  <PipelineBreadcrumbs items={breadcrumbs} />
                </div>
              )}
              <div className="text-xs font-semibold uppercase tracking-wide text-blue-100">{eyebrow}</div>
              <h1 className="text-2xl font-bold text-white">{title}</h1>
            </div>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-6 py-6">{children}</main>
    </div>
  );
}
