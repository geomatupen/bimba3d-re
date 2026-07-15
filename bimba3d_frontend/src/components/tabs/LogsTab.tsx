import { useCallback, useEffect, useRef, useState } from "react";
import { Download, FileText, RefreshCw } from "lucide-react";
import { api } from "../../api/client";

interface LogsTabProps {
  projectId: string;
}

export default function LogsTab({ projectId }: LogsTabProps) {
  const [logs, setLogs] = useState("Loading logs...");
  const [loading, setLoading] = useState(true);
  const [autoScroll, setAutoScroll] = useState(true);
  const logsEndRef = useRef<HTMLDivElement>(null);

  const fetchProcessingLogs = useCallback(async () => {
    const res = await api.get(`/projects/${projectId}/logs?lines=500`);
    setLogs(res.data.logs || "No logs available yet.");
  }, [projectId]);

  const refreshLogs = async () => {
    setLoading(true);
    try {
      await fetchProcessingLogs();
    } catch (err) {
      console.error("Failed to refresh logs:", err);
      setLogs("Failed to load logs.");
    } finally {
      setLoading(false);
    }
  };

  const downloadLogs = () => {
    const blob = new Blob([logs], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${projectId}_logs.txt`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  };

  useEffect(() => {
    const loadLogs = async () => {
      try {
        await fetchProcessingLogs();
      } catch (err) {
        console.error("Failed to fetch logs:", err);
        setLogs("Failed to load logs.");
      } finally {
        setLoading(false);
      }
    };

    void loadLogs();
    const interval = setInterval(loadLogs, 3000);
    return () => clearInterval(interval);
  }, [fetchProcessingLogs]);

  useEffect(() => {
    if (autoScroll) {
      logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, autoScroll]);

  return (
    <div className="p-6">
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileText className="w-5 h-5 text-gray-600" />
            <div>
              <h2 className="text-lg font-semibold text-gray-900">Processing Logs</h2>
              <p className="text-xs text-gray-500 mt-0.5">Live project processing output</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-2 text-xs text-gray-600">
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(event) => setAutoScroll(event.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              Auto-scroll
            </label>
            <button
              onClick={() => void refreshLogs()}
              disabled={loading}
              className="inline-flex items-center gap-1.5 rounded-md bg-gray-100 px-2.5 py-1.5 text-xs font-medium text-gray-700 transition-colors hover:bg-gray-200 disabled:opacity-60"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </button>
            <button
              onClick={downloadLogs}
              className="inline-flex items-center gap-1.5 rounded-md bg-blue-600 px-2.5 py-1.5 text-xs font-medium text-white transition-colors hover:bg-blue-700"
            >
              <Download className="h-3.5 w-3.5" />
              Download Logs
            </button>
          </div>
        </div>

        <pre className="h-[600px] overflow-x-auto overflow-y-auto rounded-b-xl bg-gray-900 p-6 font-mono text-sm text-green-400">
          {logs}
          <div ref={logsEndRef} />
        </pre>
      </div>
    </div>
  );
}
