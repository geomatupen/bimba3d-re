import { Pause, Play, RotateCcw, Square } from "lucide-react";
import type { MouseEvent } from "react";
import { useState } from "react";
import { api } from "../../api/client";

interface PipelineActionControlsProps {
  compact?: boolean;
  onComplete?: () => void | Promise<void>;
  pipeline: {
    failed_runs?: number;
    hard_cap_runs?: number;
    id: string;
    name?: string;
    status: string;
  };
  stopPropagation?: boolean;
}

type PipelineAction = "start" | "pause" | "resume" | "stop";

const actionLabel = {
  start: "Start",
  pause: "Pause",
  resume: "Resume",
  stop: "Stop",
};

const buttonClasses = {
  start: "bg-green-600 text-white hover:bg-green-700",
  pause: "bg-yellow-600 text-white hover:bg-yellow-700",
  resume: "bg-green-600 text-white hover:bg-green-700",
  stop: "bg-red-600 text-white hover:bg-red-700",
  restart: "bg-orange-600 text-white hover:bg-orange-700",
  retry: "bg-indigo-600 text-white hover:bg-indigo-700",
};

const getActionsForStatus = (status: string): PipelineAction[] => {
  const normalized = status.toLowerCase();
  if (normalized === "pending") return ["start"];
  if (normalized === "running") return ["pause", "stop"];
  if (normalized === "paused") return ["resume", "stop"];
  if (["stopped", "failed", "completed_with_failures"].includes(normalized)) return ["resume"];
  return [];
};

const canRestart = (status: string) => status.toLowerCase() !== "running";

export default function PipelineActionControls({
  compact = false,
  onComplete,
  pipeline,
  stopPropagation = false,
}: PipelineActionControlsProps) {
  const [actioning, setActioning] = useState<PipelineAction | "restart" | null>(null);
  const [message, setMessage] = useState<{ text: string; type: "success" | "error" } | null>(null);
  const [restartOpen, setRestartOpen] = useState(false);
  const [retryOpen, setRetryOpen] = useState(false);
  const [includeHardCapRetries, setIncludeHardCapRetries] = useState(false);
  const [keepBaseline, setKeepBaseline] = useState(true);
  const [keepLogSpaceSchedule, setKeepLogSpaceSchedule] = useState(true);

  const showMessage = (text: string, type: "success" | "error") => {
    setMessage({ text, type });
    window.setTimeout(() => setMessage(null), 3500);
  };

  const runAction = async (action: PipelineAction) => {
    if (actioning) return;
    setActioning(action);
    try {
      await api.post(`/api/workflow/pipelines/${pipeline.id}/${action}`);
      showMessage(`Pipeline ${actionLabel[action].toLowerCase()} request sent.`, "success");
      await onComplete?.();
    } catch (err: any) {
      showMessage(err.response?.data?.detail || `Failed to ${action} pipeline.`, "error");
    } finally {
      setActioning(null);
    }
  };

  const runRestart = async () => {
    if (actioning) return;
    setActioning("restart");
    try {
      const query = new URLSearchParams({
        keep_baseline: String(keepBaseline),
        keep_log_space_schedule: String(keepLogSpaceSchedule),
      });
      await api.post(`/api/workflow/pipelines/${pipeline.id}/restart?${query.toString()}`);
      showMessage("Pipeline restart request sent.", "success");
      setRestartOpen(false);
      await onComplete?.();
    } catch (err: any) {
      showMessage(err.response?.data?.detail || "Failed to restart pipeline.", "error");
    } finally {
      setActioning(null);
    }
  };

  const runRetryFailed = async (includeHardCap = false) => {
    if (actioning) return;
    setActioning("restart");
    try {
      const query = new URLSearchParams({
        auto_start: "true",
        include_hard_cap: String(includeHardCap),
      });
      const res = await api.post(`/api/workflow/pipelines/${pipeline.id}/retry-failed?${query.toString()}`);
      const retried = Number(res.data?.failed_count || 0);
      showMessage(
        retried > 0
          ? `Retry started for ${retried} run(s).`
          : "No retryable runs found.",
        "success",
      );
      setRetryOpen(false);
      await onComplete?.();
    } catch (err: any) {
      showMessage(err.response?.data?.detail || "Failed to retry failed runs.", "error");
    } finally {
      setActioning(null);
    }
  };

  const actions = getActionsForStatus(pipeline.status);
  const sizeClass = compact ? "px-2 py-1 text-[11px]" : "px-3 py-2 text-sm";
  const iconClass = compact ? "h-3.5 w-3.5" : "h-4 w-4";
  const hardCapCount = Number(pipeline.hard_cap_runs || 0);

  const maybeStopPropagation = (event: MouseEvent) => {
    if (stopPropagation) {
      event.stopPropagation();
    }
  };

  return (
    <div className="relative" onClick={maybeStopPropagation}>
      <div className="flex flex-wrap items-center gap-1.5">
        {actions.map((action) => {
          const Icon = action === "pause" ? Pause : action === "stop" ? Square : Play;
          return (
            <button
              key={action}
              onClick={() => void runAction(action)}
              disabled={Boolean(actioning)}
              className={`inline-flex items-center gap-1.5 rounded-lg font-semibold disabled:opacity-60 ${sizeClass} ${buttonClasses[action]}`}
            >
              <Icon className={iconClass} />
              {actionLabel[action]}
            </button>
          );
        })}
        {canRestart(pipeline.status) && (
          <button
            onClick={() => setRestartOpen(true)}
            disabled={Boolean(actioning)}
            className={`inline-flex items-center gap-1.5 rounded-lg font-semibold disabled:opacity-60 ${sizeClass} ${buttonClasses.restart}`}
          >
            <RotateCcw className={iconClass} />
            Restart
          </button>
        )}
        {(Number(pipeline.failed_runs || 0) > 0 || hardCapCount > 0) && pipeline.status.toLowerCase() !== "running" && (
          <button
            onClick={() => {
              if (hardCapCount > 0) {
                setRetryOpen(true);
                return;
              }
              void runRetryFailed(false);
            }}
            disabled={Boolean(actioning)}
            className={`inline-flex items-center gap-1.5 rounded-lg font-semibold disabled:opacity-60 ${sizeClass} ${buttonClasses.retry}`}
            title="Retry only failed runs"
          >
            <RotateCcw className={iconClass} />
            Retry Failed
          </button>
        )}
      </div>

      {message && (
        <div
          className={`mt-2 rounded-lg border px-2 py-1 text-xs font-medium ${
            message.type === "success"
              ? "border-green-200 bg-green-50 text-green-700"
              : "border-red-200 bg-red-50 text-red-700"
          }`}
        >
          {message.text}
        </div>
      )}

      {restartOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={() => setRestartOpen(false)}>
          <div className="w-full max-w-md rounded-lg bg-white p-5 shadow-xl" onClick={(event) => event.stopPropagation()}>
            <h3 className="text-lg font-semibold text-slate-950">Restart Pipeline?</h3>
            <p className="mt-2 text-sm text-slate-600">
              This will restart {pipeline.name || "this pipeline"} using the saved configuration.
            </p>
            <label className="mt-4 flex items-start gap-2 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
              <input
                type="checkbox"
                checked={keepBaseline}
                onChange={(event) => setKeepBaseline(event.target.checked)}
                className="mt-0.5"
              />
              <span>Keep baseline/default runs where the backend supports it.</span>
            </label>
            <label className="mt-2 flex items-start gap-2 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
              <input
                type="checkbox"
                checked={keepLogSpaceSchedule}
                onChange={(event) => setKeepLogSpaceSchedule(event.target.checked)}
                className="mt-0.5"
              />
              <span>Keep saved log-space multiplier schedule.</span>
            </label>
            <div className="mt-5 flex justify-end gap-2">
              <button
                onClick={() => setRestartOpen(false)}
                disabled={actioning === "restart"}
                className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-60"
              >
                Cancel
              </button>
              <button
                onClick={() => void runRestart()}
                disabled={actioning === "restart"}
                className="inline-flex items-center gap-1.5 rounded-lg bg-orange-600 px-3 py-2 text-sm font-semibold text-white hover:bg-orange-700 disabled:opacity-60"
              >
                <RotateCcw className={`h-4 w-4 ${actioning === "restart" ? "animate-spin" : ""}`} />
                {actioning === "restart" ? "Restarting..." : "Restart"}
              </button>
            </div>
          </div>
        </div>
      )}

      {retryOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={() => setRetryOpen(false)}>
          <div className="w-full max-w-md rounded-lg bg-white p-5 shadow-xl" onClick={(event) => event.stopPropagation()}>
            <h3 className="text-lg font-semibold text-slate-950">Retry Runs?</h3>
            <p className="mt-2 text-sm text-slate-600">
              Retry Failed will rerun {Number(pipeline.failed_runs || 0)} failed run(s). This pipeline also has {hardCapCount} hard-cap run(s).
            </p>
            <label className="mt-4 flex items-start gap-2 rounded-lg border border-orange-200 bg-orange-50 p-3 text-sm text-orange-900">
              <input
                type="checkbox"
                checked={includeHardCapRetries}
                onChange={(event) => setIncludeHardCapRetries(event.target.checked)}
                className="mt-0.5"
              />
              <span>Also retry hard-cap runs. Increase the Gaussian hard cap in Edit Config before using this if the previous cap was too low.</span>
            </label>
            <div className="mt-5 flex justify-end gap-2">
              <button
                onClick={() => setRetryOpen(false)}
                disabled={actioning === "restart"}
                className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-60"
              >
                Cancel
              </button>
              <button
                onClick={() => void runRetryFailed(includeHardCapRetries)}
                disabled={actioning === "restart"}
                className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-2 text-sm font-semibold text-white hover:bg-indigo-700 disabled:opacity-60"
              >
                <RotateCcw className={`h-4 w-4 ${actioning === "restart" ? "animate-spin" : ""}`} />
                {actioning === "restart" ? "Starting..." : "Retry Failed"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
