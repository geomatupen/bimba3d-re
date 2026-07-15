import { ArrowLeft, Check, Clock } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import PipelineBreadcrumbs from "../pipelineDetails/PipelineBreadcrumbs";

interface ProjectStatus {
  name?: string | null;
  status: string;
}

interface ProjectPageHeaderProps {
  projectId: string;
  projectStatus: ProjectStatus | null;
  returnTo?: string | null;
  returnToPipeline?: string | null;
  source?: string | null;
}

export default function ProjectPageHeader({ projectId, projectStatus, returnTo, returnToPipeline, source }: ProjectPageHeaderProps) {
  const navigate = useNavigate();
  const workflowSource = source === "workflow" || source === "workflow-projects";
  const backTarget = returnTo
    ? returnTo
    : returnToPipeline
    ? `/workflow/pipelines/${returnToPipeline}`
    : workflowSource
      ? "/workflow"
      : "/";
  const breadcrumbItems = returnToPipeline
    ? [
        { label: "Research Workflow", to: "/workflow" },
        { label: "Pipeline", to: returnTo || `/workflow/pipelines/${returnToPipeline}` },
        { label: projectStatus?.name || `Project ${projectId.slice(0, 8)}` },
      ]
    : workflowSource
      ? [
          { label: "Research Workflow", to: "/workflow" },
          { label: "Projects", to: "/workflow" },
          { label: projectStatus?.name || `Project ${projectId.slice(0, 8)}` },
        ]
      : [
          { label: "Projects", to: "/" },
          { label: projectStatus?.name || `Project ${projectId.slice(0, 8)}` },
        ];

  return (
    <header className="bg-gradient-to-r from-blue-600 via-blue-700 to-indigo-700 shadow-xl">
      <div className="max-w-7xl mx-auto px-6 lg:px-8 py-7">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            {returnToPipeline ? (
              <button
                onClick={() => navigate(backTarget)}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-white/10 hover:bg-white/20 backdrop-blur-sm border border-white/20 text-white text-sm font-medium transition-all duration-200 hover:scale-105"
              >
                <ArrowLeft className="w-4 h-4" />
                Back to Pipeline
              </button>
            ) : (
              <Link
                to={backTarget}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-white/10 hover:bg-white/20 backdrop-blur-sm border border-white/20 text-white text-sm font-medium transition-all duration-200 hover:scale-105"
              >
                <ArrowLeft className="w-4 h-4" />
                Back
              </Link>
            )}
            <div>
              <div className="mb-2">
                <PipelineBreadcrumbs items={breadcrumbItems} />
              </div>
              <div className="inline-flex items-center gap-2 px-2 py-0.5 rounded-full bg-white/10 backdrop-blur-sm border border-white/20 mb-1">
                <span className="text-xs font-medium text-white uppercase tracking-wider">Project</span>
              </div>
              <h1 className="text-2xl font-bold text-white mb-1">
                {projectStatus?.name || `Project ${projectId.slice(0, 8)}`}
              </h1>
              <p className="text-xs text-blue-100 font-mono">ID: {projectId}</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {projectStatus && (
              <span className={`px-4 py-2 rounded-xl text-xs font-semibold shadow-lg backdrop-blur-sm border-2 ${
                projectStatus.status === "completed" || projectStatus.status === "done"
                  ? "bg-emerald-50/90 text-emerald-700 border-emerald-200"
                  : projectStatus.status === "processing"
                    ? "bg-blue-50/90 text-blue-700 border-blue-200"
                    : projectStatus.status === "failed"
                      ? "bg-rose-50/90 text-rose-700 border-rose-200"
                      : "bg-white/90 text-slate-700 border-slate-200"
              } inline-flex items-center gap-1.5`}>
                {(projectStatus.status === "processing" || projectStatus.status === "stopping") && (
                  <Clock className="w-3.5 h-3.5 text-blue-600 animate-pulse" />
                )}
                {(projectStatus.status === "completed" || projectStatus.status === "done") && (
                  <Check className="w-3.5 h-3.5 text-emerald-600" />
                )}
                {projectStatus.status}
              </span>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
