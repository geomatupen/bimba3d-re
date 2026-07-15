import { RefreshCw } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api } from "../../api/client";

interface PipelineLogsPanelProps {
  pipelineId: string;
  autoRefresh?: boolean;
}

export default function PipelineLogsPanel({ pipelineId, autoRefresh = true }: PipelineLogsPanelProps) {
  const [logs, setLogs] = useState<{ id?: string; project: string; run_id?: string | null; log_path?: string; logs: string; lines: number }[]>([]);
  const [loading, setLoading] = useState(false);
  const [openLogs, setOpenLogs] = useState<Set<string>>(new Set());

  const getLogKey = (log: { id?: string; project: string; run_id?: string | null }, index: number) =>
    log.id || `${log.project}-${log.run_id || "project"}-${index}`;

  const toggleLog = (key: string, open: boolean) => {
    setOpenLogs((current) => {
      const next = new Set(current);
      if (open) {
        next.add(key);
      } else {
        next.delete(key);
      }
      return next;
    });
  };

  const loadLogs = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const res = await api.get(`/api/workflow/pipelines/${pipelineId}/worker-logs`);
      setLogs(res.data?.logs || []);
    } catch (err) {
      console.error("Failed to load pipeline worker logs", err);
      setLogs([]);
    } finally {
      if (!silent) setLoading(false);
    }
  }, [pipelineId]);

  useEffect(() => {
    void loadLogs();
    if (!autoRefresh) return;

    const timer = window.setInterval(() => {
      void loadLogs(true);
    }, 5000);

    return () => window.clearInterval(timer);
  }, [autoRefresh, loadLogs]);

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-950">Logs</h2>
          <p className="mt-1 text-sm text-slate-600">Worker logs are preserved and shown per project/run.</p>
        </div>
        <button
          onClick={() => void loadLogs()}
          disabled={loading}
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-60"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>
      {loading ? (
        <div className="rounded-lg border border-slate-200 p-4 text-sm text-slate-500">Loading logs...</div>
      ) : logs.length === 0 ? (
        <div className="rounded-lg border border-dashed border-slate-300 p-4 text-sm text-slate-500">No worker logs available yet.</div>
      ) : (
        <div className="space-y-3">
          {logs.map((log, index) => {
            const key = getLogKey(log, index);
            return (
            <details
              key={key}
              open={openLogs.has(key)}
              onToggle={(event) => toggleLog(key, event.currentTarget.open)}
              className="rounded-lg border border-slate-200 bg-slate-50"
            >
              <summary className="cursor-pointer px-3 py-2 text-sm font-semibold text-slate-800">
                {log.project} <span className="text-xs font-normal text-slate-500">({log.lines} lines)</span>
                {log.log_path && <span className="ml-2 font-mono text-[10px] font-normal text-slate-400">{log.log_path}</span>}
              </summary>
              <pre className="max-h-[520px] overflow-auto border-t border-slate-200 bg-slate-950 p-3 text-xs text-slate-100">
                {log.logs}
              </pre>
            </details>
            );
          })}
        </div>
      )}
    </section>
  );
}
