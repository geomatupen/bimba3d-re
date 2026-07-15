import { ExternalLink, Workflow } from "lucide-react";
import { useNavigate } from "react-router-dom";

export interface WorkflowPipeline {
  id: string;
  name: string;
  status: string;
  pipeline_type?: string;
  workflow_stage?: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  current_phase: number;
  current_run: number;
  total_runs: number;
  completed_runs: number;
  failed_runs: number;
  hard_cap_runs?: number;
  mean_relative_score?: number | null;
  success_rate?: number | null;
  last_error?: string | null;
}

export type WorkflowPipelineStage = "training_data" | "model_training" | "test";

interface PipelineSummaryListProps {
  detailBasePath?: string;
  emptyMessage: string;
  loading: boolean;
  pipelines: WorkflowPipeline[];
  tone: "blue" | "amber";
  showStageLabel?: boolean;
}

const statusClasses = (status: string) => {
  switch (status.toLowerCase()) {
    case "running":
      return "border-green-300 bg-green-50 text-green-700";
    case "paused":
      return "border-yellow-300 bg-yellow-50 text-yellow-700";
    case "completed":
      return "border-blue-300 bg-blue-50 text-blue-700";
    case "completed_with_hard_caps":
    case "hard_cap_reached":
      return "border-orange-300 bg-orange-50 text-orange-700";
    case "failed":
    case "completed_with_failures":
      return "border-red-300 bg-red-50 text-red-700";
    default:
      return "border-slate-300 bg-slate-50 text-slate-700";
  }
};

const toneClasses = {
  blue: {
    badge: "bg-blue-100 text-blue-700",
  },
  amber: {
    badge: "bg-amber-100 text-amber-700",
  },
};

export const getPipelineStage = (pipeline: Pick<WorkflowPipeline, "pipeline_type" | "workflow_stage">): WorkflowPipelineStage => {
  const stage = String(pipeline.workflow_stage || "").toLowerCase();
  if (stage === "testing" || stage === "testing_pipeline" || stage === "test") return "test";
  if (stage === "model_training") return "model_training";
  if (stage === "offline_data_preparation" || stage === "training_data") return "training_data";

  const type = String(pipeline.pipeline_type || "offline_data").toLowerCase();
  if (type === "test" || type.includes("test")) return "test";
  if (type === "model_training" || type === "model" || type.includes("model") || type.includes("train_model")) {
    return "model_training";
  }
  return "training_data";
};

const stageLabelClasses = {
  training_data: "bg-blue-100 text-blue-700",
  model_training: "bg-emerald-100 text-emerald-700",
  test: "bg-amber-100 text-amber-700",
};

const stageDisplayLabel = {
  training_data: "DATA",
  model_training: "TRAIN",
  test: "TEST",
};

const formatDuration = (start: string | null, end: string | null) => {
  if (!start) return "";
  const startTime = new Date(start).getTime();
  const endTime = end ? new Date(end).getTime() : Date.now();
  if (Number.isNaN(startTime) || Number.isNaN(endTime)) return "";
  const minutes = Math.max(0, Math.floor((endTime - startTime) / 60000));
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  return hours > 0 ? `${hours}h ${mins}m` : `${mins}m`;
};

export default function PipelineSummaryList({
  detailBasePath = "/workflow/pipelines",
  emptyMessage,
  loading,
  pipelines,
  tone,
  showStageLabel = false,
}: PipelineSummaryListProps) {
  const navigate = useNavigate();

  if (loading) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-5 text-sm text-slate-500 shadow-sm">
        Loading pipelines...
      </div>
    );
  }

  if (pipelines.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-slate-300 bg-white p-5 text-sm text-slate-500 shadow-sm">
        {emptyMessage}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {pipelines.map((pipeline) => {
        const completion =
          pipeline.total_runs > 0
            ? Math.round((pipeline.completed_runs / pipeline.total_runs) * 100)
            : 0;
        const progressWidth = pipeline.total_runs > 0
          ? (pipeline.completed_runs / pipeline.total_runs) * 100
          : 0;
        const stage = getPipelineStage(pipeline);
        const relativeScore = pipeline.mean_relative_score;
        const typeLabel = stageDisplayLabel[stage];
        const typeLabelClass = showStageLabel ? stageLabelClasses[stage] : toneClasses[tone].badge;
        const createdDate = new Date(pipeline.created_at);
        const timingText = pipeline.started_at
          ? pipeline.completed_at
            ? `Completed - ${formatDuration(pipeline.started_at, pipeline.completed_at)}`
            : `Running - ${formatDuration(pipeline.started_at, null)}`
          : `Created ${Number.isNaN(createdDate.getTime()) ? pipeline.created_at : createdDate.toLocaleDateString()}`;

        return (
          <div
            key={pipeline.id}
            onClick={() => navigate(`${detailBasePath}/${pipeline.id}`)}
            className="group relative block cursor-pointer overflow-hidden rounded-xl border border-slate-300 bg-white shadow-sm transition-all duration-300 hover:border-blue-400 hover:shadow-lg"
          >
            <div className="flex items-center gap-3 p-3">
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-indigo-600 text-white shadow-md transition-transform duration-300 group-hover:scale-105">
                <Workflow className="h-6 w-6" />
              </div>

              <div className="min-w-0 flex-1 space-y-1.5">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <h3 className="mb-0.5 truncate text-sm font-bold text-slate-900 transition-colors group-hover:text-blue-600">
                      {pipeline.name}
                      <span className={`ml-2 rounded px-1.5 py-0.5 text-[10px] font-medium ${typeLabelClass}`}>
                        {typeLabel}
                      </span>
                    </h3>
                    <p className="text-xs text-slate-500">{timingText}</p>
                  </div>
                  <div className="flex items-center gap-1">
                    <span className={`shrink-0 rounded-full border px-2 py-0.5 text-xs font-semibold ${statusClasses(pipeline.status)}`}>
                      {pipeline.status}
                    </span>
                    <ExternalLink className="h-4 w-4 text-slate-400 group-hover:text-blue-600" />
                  </div>
                </div>

                <div className="space-y-0.5">
                  <div className="flex justify-between text-xs font-medium text-slate-600">
                    <span>Progress: {pipeline.completed_runs}/{pipeline.total_runs}</span>
                    <span className="text-blue-600">{completion}%</span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100 shadow-inner">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-indigo-600 transition-all duration-300"
                      style={{ width: `${progressWidth}%` }}
                    />
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-4 text-xs">
                  {pipeline.status === "running" && (
                    <span className="text-slate-600">
                      Phase {pipeline.current_phase} - Run {pipeline.current_run}
                    </span>
                  )}
                  {typeof relativeScore === "number" && Number.isFinite(relativeScore) && (
                    <span className="text-slate-600">
                      <span className="text-slate-500">Relative Score:</span>{" "}
                      <span className="font-semibold">{relativeScore.toFixed(3)}</span>
                    </span>
                  )}
                  {typeof pipeline.success_rate === "number" && Number.isFinite(pipeline.success_rate) && (
                    <span className="text-slate-600">
                      <span className="text-slate-500">Success:</span>{" "}
                      <span className="font-semibold">{pipeline.success_rate.toFixed(1)}%</span>
                    </span>
                  )}
                  {pipeline.failed_runs > 0 && (
                    <span className="font-semibold text-red-600">{pipeline.failed_runs} failed</span>
                  )}
                  {(pipeline.hard_cap_runs || 0) > 0 && (
                    <span className="font-semibold text-orange-600">{pipeline.hard_cap_runs} hard cap</span>
                  )}
                </div>

                {pipeline.last_error && (
                  <div className="truncate rounded border border-red-200 bg-red-50 px-2 py-1 text-xs text-red-700" title={pipeline.last_error}>
                    <strong>Error:</strong> {pipeline.last_error}
                  </div>
                )}

              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

