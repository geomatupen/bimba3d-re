import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../../api/client";
import type { PipelineDetail } from "./types";

interface PipelineOverviewPanelProps {
  pipeline: PipelineDetail;
  variant: "training_data" | "model_training" | "test";
}

const formatDuration = (start: string | null, end: string | null) => {
  if (!start) return "N/A";
  const startTime = new Date(start).getTime();
  const endTime = end ? new Date(end).getTime() : Date.now();
  if (Number.isNaN(startTime) || Number.isNaN(endTime)) return "N/A";
  const minutes = Math.max(0, Math.floor((endTime - startTime) / 60000));
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  return hours > 0 ? `${hours}h ${mins}m` : `${mins}m`;
};

const formatSeconds = (seconds: number | null) => {
  if (seconds === null || !Number.isFinite(seconds) || seconds < 0) return "N/A";
  const rounded = Math.max(0, Math.round(seconds));
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const secs = rounded % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${secs}s`;
  return `${secs}s`;
};

const formatDate = (value: string | null | undefined) => {
  if (!value) return "N/A";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
};

const resolveProjectId = (project: any) => {
  if (!project || typeof project === "string") return null;
  return project?.project_id || project?.id || null;
};

const resolveProjectLabel = (project: any) => {
  if (typeof project === "string") return project;
  return project?.name || project?.project_id || project?.id || "N/A";
};

const percent = (done: number, total: number) => {
  if (!total || total <= 0) return 0;
  return Math.max(0, Math.min(100, Math.round((done / total) * 100)));
};

const numberFromStatus = (status: any, ...keys: string[]) => {
  for (const key of keys) {
    const value = status?.[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
};

const numberFromObject = (source: any, ...keys: string[]) => {
  for (const key of keys) {
    const value = source?.[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
};

const formatMetricValue = (value: number | null, decimals = 4) => {
  if (value === null || !Number.isFinite(value)) return "-";
  return value.toFixed(decimals);
};

const formatMetricStep = (step: number | null) => {
  if (step === null || !Number.isFinite(step)) return "";
  return `@${Math.round(step).toLocaleString()}`;
};

const formatCompactNumber = (value: unknown, decimals = 6) => {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return value.toFixed(decimals).replace(/\.?0+$/, "");
};

const normalizeName = (value: unknown) =>
  String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[\s_-]+/g, "");

const rowToLiveMetrics = (row: any) => ({
  run_id: row?.run_id || null,
  latest_loss: numberFromObject(row, "final_loss"),
  latest_loss_step: numberFromObject(row, "final_loss_step"),
  best_loss: numberFromObject(row, "best_loss"),
  best_loss_step: numberFromObject(row, "best_loss_step"),
  psnr: numberFromObject(row, "final_psnr"),
  psnr_step: numberFromObject(row, "final_psnr_step"),
  ssim: numberFromObject(row, "final_ssim"),
  ssim_step: numberFromObject(row, "final_ssim_step"),
  lpips: numberFromObject(row, "final_lpips"),
  lpips_step: numberFromObject(row, "final_lpips_step"),
});

const completionActivity = (
  status: string | null | undefined,
  lastError?: string | null,
  failedRuns = 0,
  hardCapRuns = 0,
) => {
  const normalized = String(status || "").toLowerCase();
  if (lastError || failedRuns > 0) {
    return {
      label: normalized === "completed_with_failures" ? "Pipeline completed with failures" : "Pipeline ended with error",
      rowClass: "border-red-200 bg-red-50",
      timeClass: "text-red-600",
      textClass: "text-red-800",
    };
  }
  if (hardCapRuns > 0 || normalized === "completed_with_hard_caps" || normalized === "hard_cap_reached") {
    return {
      label: "Pipeline completed with hard cap reached",
      rowClass: "border-orange-200 bg-orange-50",
      timeClass: "text-orange-600",
      textClass: "text-orange-800",
    };
  }
  if (normalized === "stopped") {
    return {
      label: "Pipeline stopped",
      rowClass: "border-slate-200 bg-slate-50",
      timeClass: "text-slate-600",
      textClass: "text-slate-800",
    };
  }
  if (normalized === "failed" || normalized === "completed_with_failures") {
    return {
      label: normalized === "failed" ? "Pipeline failed" : "Pipeline completed with failures",
      rowClass: "border-red-200 bg-red-50",
      timeClass: "text-red-600",
      textClass: "text-red-800",
    };
  }
  return {
    label: "Pipeline completed",
    rowClass: "border-green-200 bg-green-50",
    timeClass: "text-green-600",
    textClass: "text-green-800",
  };
};

const runStatusClass = (status?: string | null) => {
  const normalized = String(status || "").toLowerCase();
  if (normalized === "success" || normalized === "completed") return "bg-green-100 text-green-700";
  if (normalized === "running") return "bg-blue-100 text-blue-700";
  if (normalized === "hard_cap_reached" || normalized === "partial_completed") return "bg-orange-100 text-orange-700";
  if (normalized === "failed" || normalized === "error") return "bg-red-100 text-red-700";
  return "bg-slate-100 text-slate-700";
};

export default function PipelineOverviewPanel({ pipeline, variant }: PipelineOverviewPanelProps) {
  const navigate = useNavigate();
  const [currentProjectStatus, setCurrentProjectStatus] = useState<any>(null);
  const [baselineRowMetrics, setBaselineRowMetrics] = useState<any>(null);
  const projects = Array.isArray(pipeline.config?.projects) ? pipeline.config.projects : [];
  const runs = Array.isArray(pipeline.runs) ? pipeline.runs : [];
  const progress = percent(pipeline.completed_runs, pipeline.total_runs);
  const hardCapRuns = Number(pipeline.hard_cap_runs || 0);
  const pendingRuns = Number.isFinite(Number(pipeline.pending_runs))
    ? Number(pipeline.pending_runs || 0)
    : Math.max(0, pipeline.total_runs - pipeline.completed_runs - pipeline.failed_runs - hardCapRuns);
  const currentProject = projects[pipeline.current_project_index];
  const baselineRunId = typeof currentProject === "object" ? currentProject?.baseline_run_id : null;
  const activeRun = pipeline.status === "running" ? pipeline.active_run || null : null;
  const currentProjectId = resolveProjectId(currentProject);
  const hasCurrentProject = Boolean(currentProject);
  const activeProjectLabel = activeRun?.project_name || resolveProjectLabel(currentProject);
  const displayedRunId = currentProjectStatus?.current_run_id || activeRun?.run_id || "N/A";
  const rawProjectStatus = String(currentProjectStatus?.status || "").toLowerCase();
  const displayedStatus = activeRun?.status === "running" && (!rawProjectStatus || rawProjectStatus === "pending")
    ? "running"
    : currentProjectStatus?.status || activeRun?.status || "Checking...";
  const displayedPhase = activeRun?.phase ?? pipeline.current_phase;
  const displayedRun = activeRun?.run ?? pipeline.current_run;
  const currentStep = numberFromStatus(currentProjectStatus, "current_step", "currentStep");
  const maxSteps = numberFromStatus(currentProjectStatus, "max_steps", "maxSteps");
  const currentLoss = numberFromStatus(currentProjectStatus, "current_loss");
  const psnr = numberFromStatus(currentProjectStatus, "psnr");
  const liveMetrics = currentProjectStatus?.live_metrics || {};
  const baselineLiveMetrics = baselineRowMetrics || {};
  const runMetricRows = [
    {
      label: "Current Run",
      tone: "text-blue-950",
      metrics: liveMetrics,
      fallbackLoss: currentLoss,
      fallbackPsnr: psnr,
    },
    {
      label: "Baseline",
      tone: "text-slate-700",
      metrics: baselineLiveMetrics,
      fallbackLoss: null,
      fallbackPsnr: null,
    },
  ];
  const selectedMultipliers = activeRun?.selected_multipliers || {};
  const activeMultiplierRows = [
    {
      label: "Geometry",
      selected: numberFromObject(selectedMultipliers, "position_lr_init_mult", "scaling_lr_mult", "rotation_lr_mult"),
    },
    {
      label: "Appearance",
      selected: numberFromObject(selectedMultipliers, "feature_lr_mult", "opacity_lr_mult", "lambda_dssim_mult"),
    },
    {
      label: "Densification",
      selected: numberFromObject(selectedMultipliers, "densify_grad_threshold_mult", "opacity_threshold_mult"),
    },
  ];
  const hasActiveSelection = activeMultiplierRows.some((row) => row.selected !== null);
  const elapsedSeconds = (() => {
    if (!pipeline.started_at) return null;
    const startTime = new Date(pipeline.started_at).getTime();
    const endTime = pipeline.completed_at ? new Date(pipeline.completed_at).getTime() : Date.now();
    if (Number.isNaN(startTime) || Number.isNaN(endTime)) return null;
    return Math.max(0, (endTime - startTime) / 1000);
  })();
  const remainingSeconds = (() => {
    if (pipeline.completed_at || pipeline.status !== "running") return 0;
    const completed = Number(pipeline.completed_runs || 0);
    const total = Number(pipeline.total_runs || 0);
    if (!elapsedSeconds || completed <= 0 || total <= completed) return null;
    return (elapsedSeconds / completed) * Math.max(0, total - completed);
  })();
  const activityRuns = activeRun?.run_id
    ? [
        {
          project_name: activeRun.project_name,
          phase: activeRun.phase,
          run: activeRun.run,
          run_id: activeRun.run_id,
          run_name: `Active Run - Phase ${activeRun.phase ?? "-"}`,
          status: activeRun.status || "running",
          timestamp: activeRun.started_at,
          is_active: true,
          test_model_id: activeRun.test_model_id,
        },
        ...runs.filter((run: any) => run?.run_id !== activeRun.run_id),
    ]
    : runs;
  const activityItems = [
    {
      key: "created",
      timestamp: pipeline.created_at,
      rowClass: "border-gray-200 bg-gray-50",
      timeClass: "text-gray-500",
      textClass: "text-gray-700",
      content: <>Pipeline created: {pipeline.name}</>,
    },
    ...(pipeline.started_at
      ? [{
          key: "started",
          timestamp: pipeline.started_at,
          rowClass: "border-gray-200 bg-gray-50",
          timeClass: "text-gray-500",
          textClass: "text-gray-700",
          content: <>Pipeline started</>,
        }]
      : []),
    ...activityRuns.map((run: any, index: number) => ({
      key: `run-${run.run_id || run.id || index}`,
      timestamp: run.timestamp || run.created_at,
      rowClass: run.is_active ? "border-blue-200 bg-blue-50" : "border-gray-200 bg-gray-50",
      timeClass: "text-gray-500",
      textClass: "text-gray-700",
      content: (
        <>
          <span className="font-medium">{run.project_name || run.project || "Project"}</span> - {run.run_name || `Phase ${run.phase ?? "-"}, Run ${run.run ?? index + 1}`}
          {run.phase === 1 && <span className="ml-2 rounded bg-gray-100 px-2 py-0.5 text-xs text-gray-700">Baseline</span>}
          {run.status && (
            <span
              className={`ml-2 rounded px-2 py-0.5 text-xs ${runStatusClass(run.status)}`}
            >
              {run.status}
            </span>
          )}
          {run.run_id && <span className="ml-2 font-mono text-[10px] text-gray-500">{run.run_id}</span>}
          {run.test_model_id && <span className="ml-2 text-xs text-amber-700">Model: {run.test_model_id}</span>}
          {typeof run.score === "number" && <span className="ml-2 text-xs text-gray-600">Score: {run.score.toFixed(4)}</span>}
          {run.group_multipliers && Object.keys(run.group_multipliers).length > 0 && (
            <span className="ml-2 rounded bg-indigo-50 px-1.5 py-0.5 font-mono text-[10px] text-indigo-700" title="Per-group log-space multipliers applied">
              geo x{(run.group_multipliers.geometry_lr?.multiplier ?? 1.0).toFixed(3)}{" "}
              app x{(run.group_multipliers.appearance_lr?.multiplier ?? 1.0).toFixed(3)}{" "}
              den x{(run.group_multipliers.densification?.multiplier ?? 1.0).toFixed(3)}
            </span>
          )}
        </>
      ),
    })),
    ...(pipeline.status === "running" && !pipeline.cooldown_active
      ? [{
          key: "running",
          timestamp: new Date().toISOString(),
          rowClass: "border-blue-200 bg-blue-50",
          timeClass: "text-blue-600",
          textClass: "text-blue-800",
          content: <>Running: Phase {displayedPhase}, Run {pipeline.current_run || displayedRun}, Project {projects.length ? pipeline.current_project_index + 1 : 0}/{projects.length}</>,
        }]
      : []),
    ...(pipeline.cooldown_active && pipeline.next_run_scheduled_at
      ? [{
          key: "cooldown",
          timestamp: new Date().toISOString(),
          rowClass: "border-yellow-200 bg-yellow-50",
          timeClass: "text-yellow-600",
          textClass: "text-yellow-800 text-xs",
          content: <>Cooldown active - Next run scheduled at {new Date(pipeline.next_run_scheduled_at).toLocaleTimeString()}</>,
        }]
      : []),
    ...(pipeline.completed_at
      ? (() => {
          const activity = completionActivity(
            pipeline.status,
            pipeline.last_error,
            Number(pipeline.failed_runs || 0),
            Number(pipeline.hard_cap_runs || 0),
          );
          return [{
            key: "completed",
            timestamp: pipeline.completed_at,
            rowClass: activity.rowClass,
            timeClass: activity.timeClass,
            textClass: activity.textClass,
            content: <>{activity.label}</>,
          }];
        })()
      : []),
    ...(pipeline.last_error
      ? [{
          key: "error",
          timestamp: pipeline.completed_at || new Date().toISOString(),
          rowClass: "border-red-200 bg-red-50",
          timeClass: "text-red-600",
          textClass: "text-red-800",
          content: <>Error: {pipeline.last_error}</>,
        }]
      : []),
  ].sort((a, b) => {
    const aTime = Date.parse(a.timestamp || "") || 0;
    const bTime = Date.parse(b.timestamp || "") || 0;
    return aTime - bTime;
  });

  const loadCurrentProjectStatus = useCallback(async () => {
    if (pipeline.status !== "running" || !currentProjectId) {
      setCurrentProjectStatus(null);
      return;
    }
    try {
      const res = await api.get(`/projects/${currentProjectId}/status`);
      setCurrentProjectStatus(res.data);
    } catch (err) {
      console.error("Failed to load current project status", err);
      setCurrentProjectStatus(null);
    }
  }, [currentProjectId, pipeline.status]);

  useEffect(() => {
    void loadCurrentProjectStatus();
    if (pipeline.status !== "running" || !currentProjectId) return;

    const timer = window.setInterval(() => {
      void loadCurrentProjectStatus();
    }, 3000);

    return () => window.clearInterval(timer);
  }, [currentProjectId, loadCurrentProjectStatus, pipeline.status]);

  useEffect(() => {
    let cancelled = false;

    const loadBaselineRowMetrics = async () => {
      if (!pipeline.id || !activeProjectLabel) {
        setBaselineRowMetrics(null);
        return;
      }

      try {
        const res = await api.get(`/api/workflow/pipelines/${pipeline.id}/learning-rows`);
        if (cancelled) return;
        const rows = Array.isArray(res.data?.rows) ? res.data.rows : [];
        const projectKey = normalizeName(activeProjectLabel);
        const baselineRow = rows.find((row: any) => {
          if (!row?.is_baseline_row) return false;
          if (baselineRunId && row.run_id === baselineRunId) return true;
          return normalizeName(row.project_name || row.project_id) === projectKey;
        });
        setBaselineRowMetrics(baselineRow ? rowToLiveMetrics(baselineRow) : null);
      } catch (err) {
        if (!cancelled) {
          console.error("Failed to load baseline learning row metrics", err);
          setBaselineRowMetrics(null);
        }
      }
    };

    void loadBaselineRowMetrics();

    return () => {
      cancelled = true;
    };
  }, [activeProjectLabel, baselineRunId, pipeline.id]);

  const liveTitle = variant === "test" ? "Live Testing Progress" : variant === "model_training" ? "Live Model Training Progress" : "Live Training Data Progress";

  return (
    <div className="space-y-3">
      {variant === "test" && (pipeline.active_run?.test_model_id || pipeline.current_test_model_id || pipeline.config?.source_model_id) && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 shadow-sm">
          <div className="flex items-center gap-2">
            <span className="rounded bg-amber-200 px-2 py-0.5 text-xs font-bold text-amber-800">TEST PIPELINE</span>
            <span className="text-sm text-amber-900">
              Model: <strong>{pipeline.active_run?.test_model_id || pipeline.current_test_model_id || pipeline.config?.source_model_id}</strong>
            </span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-sm">
          <h2 className="mb-2 text-sm font-semibold text-gray-900">Pipeline Status</h2>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <p className="text-gray-500">Status</p>
              <p className="text-sm font-medium text-gray-900">{pipeline.status}</p>
            </div>
            <div>
              <p className="text-gray-500">Phase</p>
              <p className="text-sm font-medium text-gray-900">{pipeline.current_phase}/{pipeline.config?.phases?.length || 0}</p>
            </div>
            <div>
              <p className="text-gray-500">Run</p>
              <p className="text-sm font-medium text-gray-900">{pipeline.current_run || 0}</p>
            </div>
            <div>
              <p className="text-gray-500">Current Project</p>
              <p className="text-sm font-medium text-gray-900">
                {projects.length ? `${pipeline.current_project_index + 1}/${projects.length}` : "0/0"}
              </p>
            </div>
          </div>
        </div>

        <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-sm">
          <h2 className="mb-2 text-sm font-semibold text-gray-900">Progress</h2>
          <div className="space-y-2">
            <div>
              <div className="mb-1 flex justify-between text-xs text-gray-600">
                <span>Total: {pipeline.completed_runs}/{pipeline.total_runs}</span>
                <span>{progress}%</span>
              </div>
              <div className="h-2 w-full rounded-full bg-gray-200">
                <div className="h-2 rounded-full bg-indigo-600 transition-all" style={{ width: `${progress}%` }} />
              </div>
            </div>
            <div className="grid grid-cols-4 gap-1 text-xs">
              <div>
                <p className="text-gray-500">Done</p>
                <p className="text-sm font-semibold text-gray-900">{pipeline.completed_runs}</p>
              </div>
              <div>
                <p className="text-gray-500">Failed</p>
                <p className="text-sm font-semibold text-red-600">{pipeline.failed_runs}</p>
              </div>
              <div>
                <p className="text-gray-500">Hard Cap</p>
                <p className="text-sm font-semibold text-amber-600">{hardCapRuns}</p>
              </div>
              <div>
                <p className="text-gray-500">Left</p>
                <p className="text-sm font-semibold text-gray-900">{pendingRuns}</p>
              </div>
            </div>
          </div>
        </div>

        <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-sm">
          <h2 className="mb-2 text-sm font-semibold text-gray-900">Learning Stats</h2>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-500">Mean:</span>
              <span className="font-semibold text-gray-900">{typeof pipeline.mean_relative_score === "number" ? pipeline.mean_relative_score.toFixed(4) : "N/A"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Best:</span>
              <span className="font-semibold text-green-600">{typeof pipeline.best_relative_score === "number" ? pipeline.best_relative_score.toFixed(4) : "N/A"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Success:</span>
              <span className="font-semibold text-gray-900">{typeof pipeline.success_rate === "number" ? `${pipeline.success_rate.toFixed(1)}%` : "N/A"}</span>
            </div>
          </div>
        </div>

        <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-sm">
          <h2 className="mb-2 text-sm font-semibold text-gray-900">Time</h2>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between gap-2">
              <span className="text-gray-500">Started:</span>
              <span className="text-right text-[10px] font-medium text-gray-900">{formatDate(pipeline.started_at)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Elapsed:</span>
              <span className="font-medium text-gray-900">{formatDuration(pipeline.started_at, pipeline.completed_at)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Remaining:</span>
              <span className="font-medium text-gray-900">
                {pipeline.status === "running" && remainingSeconds === null ? "Calculating..." : formatSeconds(remainingSeconds)}
              </span>
            </div>
            {pipeline.completed_at && (
              <div className="flex justify-between gap-2">
                <span className="text-gray-500">Done:</span>
                <span className="text-right text-[10px] font-medium text-gray-900">{formatDate(pipeline.completed_at)}</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {pipeline.status === "running" && (currentProjectId || activeRun || hasCurrentProject) && (
        <div className="rounded-lg border-2 border-blue-300 bg-gradient-to-br from-blue-50 to-indigo-50 p-4 shadow-md">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 animate-pulse rounded-full bg-blue-600" />
              <h2 className="text-sm font-bold text-blue-900">{liveTitle}</h2>
            </div>
            {currentProjectId && (
              <button
                onClick={() => navigate(`/projects/${currentProjectId}?from=workflow-projects`)}
                className="rounded bg-blue-600 px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-blue-700"
              >
                Go to Project
              </button>
            )}
          </div>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <div className="rounded-lg border border-blue-200 bg-white/70 p-3 backdrop-blur-sm">
              <p className="mb-2 text-xs font-semibold text-gray-700">Current Execution</p>
              <div className="space-y-1.5 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-600">Project:</span>
                  <span className="font-semibold text-gray-900">{activeProjectLabel}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Run ID:</span>
                  <span className="font-mono text-[10px] text-gray-900">{displayedRunId}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Status:</span>
                  <span className="font-medium text-blue-700">{displayedStatus}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Phase / Run:</span>
                  <span className="font-medium text-gray-900">{displayedPhase} / {displayedRun}</span>
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-blue-200 bg-white/70 p-3 backdrop-blur-sm">
              <p className="mb-2 text-xs font-semibold text-gray-700">Training Progress</p>
              <div className="space-y-2">
                {currentProjectStatus?.progress !== undefined && (
                  <div>
                    <div className="mb-1 flex justify-between text-xs text-gray-600">
                      <span>Overall</span>
                      <span className="font-semibold">{currentProjectStatus.progress}%</span>
                    </div>
                    <div className="h-2 w-full rounded-full bg-gray-200">
                      <div className="h-2 rounded-full bg-gradient-to-r from-blue-500 to-indigo-600 transition-all" style={{ width: `${currentProjectStatus.progress}%` }} />
                    </div>
                  </div>
                )}
                {currentProjectStatus?.stage && (
                  <div className="text-xs">
                    <div className="mb-1 flex justify-between text-gray-600">
                      <span className="font-medium">Stage: {currentProjectStatus.stage}</span>
                      {currentProjectStatus.stage_progress !== undefined && <span className="font-semibold">{currentProjectStatus.stage_progress}%</span>}
                    </div>
                    {currentProjectStatus.stage_progress !== undefined && (
                      <div className="h-1.5 w-full rounded-full bg-gray-200">
                        <div className="h-1.5 rounded-full bg-blue-400 transition-all" style={{ width: `${currentProjectStatus.stage_progress}%` }} />
                      </div>
                    )}
                  </div>
                )}
                {!currentProjectStatus && (
                  <div className="rounded border border-blue-100 bg-blue-50 px-2 py-1.5 text-xs text-blue-800">
                    Waiting for live project status...
                  </div>
                )}
              </div>
            </div>
          </div>

          {variant === "test" && hasActiveSelection && (
            <div className="mt-3 rounded-lg border border-emerald-200 bg-emerald-50 p-3">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <p className="text-xs font-semibold text-emerald-800">Selected Multipliers For Current Test Run</p>
                {activeRun?.selected_preset && (
                  <span className="rounded border border-emerald-200 bg-white px-2 py-0.5 text-[10px] font-semibold text-emerald-700">
                    {activeRun.selected_preset}
                  </span>
                )}
              </div>
              <div className="grid gap-2 md:grid-cols-3">
                {activeMultiplierRows.map((row) => (
                  <div key={row.label} className="rounded border border-emerald-100 bg-white/80 px-2.5 py-2 text-xs">
                    <div className="font-semibold text-emerald-950">{row.label}</div>
                    <div className="mt-1 flex justify-between gap-2">
                      <span className="text-slate-500">Selected multiplier</span>
                      <span className="font-mono font-semibold text-slate-900">{formatCompactNumber(row.selected)}</span>
                    </div>
                    <div className="mt-0.5 flex justify-between gap-2">
                      <span className="text-slate-500">ln(selected)</span>
                      <span className="font-mono text-slate-700">
                        {row.selected && row.selected > 0 ? formatCompactNumber(Math.log(row.selected)) : "-"}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {currentProjectStatus?.stage && currentProjectStatus?.message && (
            <div className="mt-3 rounded-lg border border-blue-200 bg-blue-50 p-3">
              <p className="mb-1 text-xs font-semibold text-blue-700">Stage Status</p>
              <p className="mb-2 whitespace-pre-line text-xs text-blue-900">{currentProjectStatus.message}</p>

              {currentProjectStatus.stage === "training" && currentStep !== null && maxSteps !== null && maxSteps > 0 && (
                <div className="mt-2">
                  <div className="mb-1 flex items-center justify-between text-xs text-blue-700">
                    <span>Training Step {currentStep.toLocaleString()} / {maxSteps.toLocaleString()}</span>
                    <span className="font-semibold">{((currentStep / maxSteps) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-blue-100">
                    <div
                      className="h-2 rounded-full bg-blue-600 transition-all duration-300"
                      style={{ width: `${Math.min((currentStep / maxSteps) * 100, 100)}%` }}
                    />
                  </div>

                  <div className="mt-3 overflow-x-auto rounded-md border border-blue-100 bg-white/70">
                    <table className="min-w-full text-left text-[11px] text-slate-700">
                      <thead className="bg-blue-100/70 text-[10px] uppercase tracking-wide text-blue-800">
                        <tr>
                          <th className="px-2 py-1 font-semibold">Run</th>
                          <th className="px-2 py-1 font-semibold">Loss</th>
                          <th className="px-2 py-1 font-semibold">Best Loss</th>
                          <th className="px-2 py-1 font-semibold">PSNR</th>
                          <th className="px-2 py-1 font-semibold">SSIM</th>
                          <th className="px-2 py-1 font-semibold">LPIPS</th>
                        </tr>
                      </thead>
                      <tbody>
                        {runMetricRows.map((row) => {
                          const latestLoss = numberFromObject(row.metrics, "latest_loss") ?? row.fallbackLoss;
                          const latestLossStep = numberFromObject(row.metrics, "latest_loss_step");
                          const bestLoss = numberFromObject(row.metrics, "best_loss");
                          const bestLossStep = numberFromObject(row.metrics, "best_loss_step");
                          const rowPsnr = numberFromObject(row.metrics, "psnr") ?? row.fallbackPsnr;
                          const psnrStep = numberFromObject(row.metrics, "psnr_step");
                          const rowSsim = numberFromObject(row.metrics, "ssim");
                          const ssimStep = numberFromObject(row.metrics, "ssim_step");
                          const rowLpips = numberFromObject(row.metrics, "lpips");
                          const lpipsStep = numberFromObject(row.metrics, "lpips_step");

                          return (
                            <tr key={row.label} className="border-t border-blue-100">
                              <td className={`whitespace-nowrap px-2 py-1.5 font-semibold ${row.tone}`}>{row.label}</td>
                              <td className="whitespace-nowrap px-2 py-1.5">
                                <span className="font-semibold">{formatMetricValue(latestLoss, 6)}</span>
                                <span className="ml-1 text-slate-400">{formatMetricStep(latestLossStep)}</span>
                              </td>
                              <td className="whitespace-nowrap px-2 py-1.5">
                                <span className="font-semibold">{formatMetricValue(bestLoss, 6)}</span>
                                <span className="ml-1 text-slate-400">{formatMetricStep(bestLossStep)}</span>
                              </td>
                              <td className="whitespace-nowrap px-2 py-1.5">
                                <span className="font-semibold">{formatMetricValue(rowPsnr, 2)}</span>
                                <span className="ml-1 text-slate-400">{formatMetricStep(psnrStep)}</span>
                              </td>
                              <td className="whitespace-nowrap px-2 py-1.5">
                                <span className="font-semibold">{formatMetricValue(rowSsim, 4)}</span>
                                <span className="ml-1 text-slate-400">{formatMetricStep(ssimStep)}</span>
                              </td>
                              <td className="whitespace-nowrap px-2 py-1.5">
                                <span className="font-semibold">{formatMetricValue(rowLpips, 4)}</span>
                                <span className="ml-1 text-slate-400">{formatMetricStep(lpipsStep)}</span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {currentProjectStatus.stage !== "training" && typeof currentProjectStatus.stage_progress === "number" && (
                <div className="mt-2">
                  <div className="mb-1 flex items-center justify-between text-xs text-blue-700">
                    <span>{currentProjectStatus.stage || "Current Stage"} Progress</span>
                    <span className="font-semibold">{currentProjectStatus.stage_progress}%</span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-blue-100">
                    <div
                      className="h-2 rounded-full bg-blue-600 transition-all duration-300"
                      style={{ width: `${Math.max(0, Math.min(currentProjectStatus.stage_progress, 100))}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
        <h2 className="mb-3 text-sm font-semibold text-gray-900">Activity Logs</h2>
        <div className="max-h-96 space-y-2 overflow-y-auto text-sm">
          {activityItems.length > 0 ? (
            activityItems.map((item) => (
              <div
                key={item.key}
                className={`flex items-start gap-3 rounded border p-2 ${item.rowClass}`}
              >
                <span className={`whitespace-nowrap font-mono text-xs ${item.timeClass}`}>{formatDate(item.timestamp)}</span>
                <span className={`flex-1 ${item.textClass}`}>{item.content}</span>
              </div>
            ))
          ) : (
            <div className="py-4 text-center text-xs text-gray-500">No runs yet. Pipeline runs will appear here.</div>
          )}
        </div>
      </div>

      {(pipeline.last_error || pipeline.status === "failed" || pipeline.failed_runs > 0) && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-3 shadow-sm">
          <h2 className="mb-2 text-sm font-semibold text-red-900">Errors & Issues</h2>
          {pipeline.last_error && (
            <div className="rounded border border-red-200 bg-white p-2">
              <p className="mb-1 text-xs font-medium text-red-900">Last Error:</p>
              <p className="whitespace-pre-wrap font-mono text-xs text-red-700">{pipeline.last_error}</p>
            </div>
          )}
          {pipeline.failed_runs > 0 && <p className="mt-2 text-xs text-red-800">{pipeline.failed_runs} run(s) failed.</p>}
        </div>
      )}
    </div>
  );
}

