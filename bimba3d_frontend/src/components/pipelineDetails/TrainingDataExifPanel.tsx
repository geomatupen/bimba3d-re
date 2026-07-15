import { RefreshCw, Square } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api } from "../../api/client";
import type { PipelineDetail } from "./types";

interface TrainingDataExifPanelProps {
  pipeline: PipelineDetail;
}

const modeLabel = () => {
  return "Compact Scene Descriptors";
};

const featureValue = (value: any) => {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  if (value === null || value === undefined || value === "") return "-";
  return String(value);
};

export default function TrainingDataExifPanel({ pipeline }: TrainingDataExifPanelProps) {
  const [extracting, setExtracting] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [progress, setProgress] = useState<any[]>([]);
  const [currentProject, setCurrentProject] = useState("");
  const [totalProjects, setTotalProjects] = useState(0);
  const [results, setResults] = useState<any | null>(null);
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());

  const loadResults = useCallback(async () => {
    try {
      const res = await api.get(`/api/workflow/pipelines/${pipeline.id}/test-exif/results`);
      setResults(res.data?.has_results ? res.data : null);
    } catch (err) {
      console.error("Failed to load EXIF results", err);
      setResults(null);
    }
  }, [pipeline.id]);

  useEffect(() => {
    void loadResults();
  }, [loadResults]);

  const startExtraction = async () => {
    setExtracting(true);
    setProgress([]);
    setResults(null);
    try {
      const res = await api.post(`/api/workflow/pipelines/${pipeline.id}/test-exif/start`);
      const nextTaskId = res.data.task_id;
      setTaskId(nextTaskId);
      setTotalProjects(res.data.total_projects || 0);

      const pollInterval = window.setInterval(async () => {
        try {
          const progressRes = await api.get(`/api/workflow/pipelines/${pipeline.id}/test-exif/progress/${nextTaskId}`);
          const state = progressRes.data;
          setProgress(state.progress || []);
          setCurrentProject(`${state.current_project || 0}/${state.total_projects || 0}`);

          if (state.status === "complete" || state.status === "stopped" || state.status === "error") {
            window.clearInterval(pollInterval);
            setExtracting(false);
            setTaskId(null);
            if (Array.isArray(state.results) && state.results.length > 0) {
              setResults({
                pipeline_id: state.pipeline_id,
                pipeline_name: state.pipeline_name,
                total_projects: state.total_projects,
                test_timestamp: state.started_at,
                results: state.results,
              });
            } else {
              void loadResults();
            }
          }
        } catch (err) {
          console.error("Failed to poll EXIF progress", err);
        }
      }, 500);
    } catch (err) {
      console.error("Failed to start EXIF extraction", err);
      setExtracting(false);
      setTaskId(null);
    }
  };

  const stopExtraction = async () => {
    if (!taskId) return;
    try {
      await api.post(`/api/workflow/pipelines/${pipeline.id}/test-exif/stop/${taskId}`);
    } catch (err) {
      console.error("Failed to stop EXIF extraction", err);
    }
  };

  const toggleRow = (key: string) => {
    setExpandedRows((current) => {
      const next = new Set(current);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  const resultRows = Array.isArray(results?.results) ? results.results : [];
  const modes = ["exif_compact_featurewise"];

  return (
    <section className="rounded-lg border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-200 px-5 py-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold text-slate-950">EXIF</h2>
            <p className="mt-1 text-sm text-slate-600">Test EXIF extraction on all projects with the same extraction logic used by training.</p>
          </div>
          <div className="flex items-center gap-2">
            {extracting && (
              <button
                onClick={() => void stopExtraction()}
                className="inline-flex items-center gap-1.5 rounded-lg bg-red-600 px-3 py-2 text-sm font-semibold text-white hover:bg-red-700"
              >
                <Square className="h-4 w-4" />
                Stop
              </button>
            )}
            <button
              onClick={() => void startExtraction()}
              disabled={extracting}
              className="inline-flex items-center gap-1.5 rounded-lg bg-blue-600 px-3 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-60"
            >
              <RefreshCw className={`h-4 w-4 ${extracting ? "animate-spin" : ""}`} />
              {extracting ? `Extracting ${currentProject}` : "Extract EXIF"}
            </button>
          </div>
        </div>
      </div>

      <div className="space-y-5 p-5">
        {extracting && progress.length > 0 && (
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-900">Extraction Progress</h3>
              <span className="text-xs text-slate-600">Project {currentProject} of {totalProjects}</span>
            </div>
            <div className="max-h-80 space-y-1 overflow-y-auto">
              {progress.slice().reverse().map((item: any, index: number) => (
                <div
                  key={`${item.project_name || "project"}-${index}`}
                  className={`flex items-start gap-2 rounded px-3 py-1.5 text-xs ${
                    item.status === "complete" ? "bg-green-50 text-green-800" :
                    item.status === "error" ? "bg-red-50 text-red-800" :
                    item.status === "warning" ? "bg-yellow-50 text-yellow-800" :
                    item.status === "stopped" ? "bg-slate-100 text-slate-700" :
                    "bg-blue-50 text-blue-800"
                  }`}
                >
                  <span className="shrink-0 font-semibold">{item.status || "running"}</span>
                  <span className="min-w-0 flex-1">
                    {item.project_name && <strong>{item.project_name}</strong>}
                    {item.mode && item.mode !== "detecting" && <span> - {item.mode}</span>}
                    {item.message && <span className="ml-2">{item.message}</span>}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {!extracting && resultRows.length === 0 && (
          <div className="rounded-lg border border-dashed border-slate-300 p-8 text-center">
            <p className="mb-4 text-sm text-slate-600">Click Extract EXIF to test feature extraction on all projects.</p>
            <button
              onClick={() => void startExtraction()}
              className="inline-flex items-center gap-1.5 rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700"
            >
              <RefreshCw className="h-4 w-4" />
              Extract EXIF
            </button>
          </div>
        )}

        {!extracting && resultRows.length > 0 && (
          <div className="space-y-5">
            <div className="grid gap-3 rounded-lg bg-slate-50 p-4 md:grid-cols-4">
              <div>
                <p className="text-xs text-slate-600">Total Projects</p>
                <p className="text-2xl font-bold text-slate-900">{results.total_projects}</p>
              </div>
              <div>
                <p className="text-xs text-slate-600">Tested At</p>
                <p className="text-sm font-medium text-slate-900">{results.test_timestamp ? new Date(results.test_timestamp).toLocaleString() : "-"}</p>
              </div>
              <div>
                <p className="text-xs text-slate-600">Average Completeness</p>
                <p className="text-2xl font-bold text-green-600">
                  {(resultRows.reduce((sum: number, row: any) => sum + (row.completeness_percent || 0), 0) / resultRows.length).toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-600">Total Features</p>
                <p className="text-2xl font-bold text-slate-900">{resultRows[0]?.total_count || 0}</p>
              </div>
            </div>

            {modes.map((mode) => {
              const modeResults = resultRows.filter((row: any) => row.mode === mode);
              if (modeResults.length === 0) return null;

              return (
                <div key={mode} className="overflow-hidden rounded-lg border border-slate-200">
                  <div className="border-b border-slate-200 bg-slate-100 px-4 py-3">
                    <h3 className="text-sm font-semibold text-slate-900">{modeLabel()}</h3>
                    <p className="mt-1 text-xs text-slate-600">{modeResults[0]?.total_count || 0} features extracted per project</p>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-[11px] leading-tight">
                      <thead className="bg-slate-50 text-slate-700">
                        <tr>
                          <th className="px-1.5 py-1 text-left font-semibold">Project</th>
                          <th className="px-1.5 py-1 text-left font-semibold">Camera</th>
                          <th className="px-1.5 py-1 text-center font-semibold">Descriptors</th>
                          <th className="px-1.5 py-1 text-right font-semibold">Time (ms)</th>
                          <th className="px-1.5 py-1 text-center font-semibold">Details</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {modeResults.map((row: any, index: number) => {
                          const key = `${row.project_name}-${mode}-${index}`;
                          const expanded = expandedRows.has(key);
                          const featureKeys = Object.keys(row.features || {});

                          return (
                            <>
                              <tr key={key} className="hover:bg-slate-50">
                                <td className="max-w-[220px] truncate px-1.5 py-1 font-medium text-slate-900" title={row.project_name}>{row.project_name}</td>
                                <td className="px-1.5 py-1 text-slate-700">
                                  <div className="font-medium">{row.camera_make || "-"}</div>
                                  <div className="text-slate-500">{row.camera_model || "-"}</div>
                                </td>
                                <td className="px-1.5 py-1 text-center">
                                  <span className="inline-flex rounded-full bg-blue-100 px-2 py-0.5 text-xs font-semibold text-blue-800">
                                    {featureKeys.length}
                                  </span>
                                </td>
                                <td className="px-1.5 py-1 text-right text-slate-700">{typeof row.extraction_time_ms === "number" ? row.extraction_time_ms.toFixed(1) : "-"}</td>
                                <td className="px-1.5 py-1 text-center">
                                  <button onClick={() => toggleRow(key)} className="text-xs font-semibold text-blue-700 hover:text-blue-900">
                                    {expanded ? "Hide" : "Show"} Values
                                  </button>
                                </td>
                              </tr>
                              {expanded && (
                                <tr key={`${key}-expanded`}>
                                  <td colSpan={5} className="bg-slate-50 px-2 py-2">
                                    <div className="grid gap-2 md:grid-cols-3 lg:grid-cols-4">
                                      {featureKeys.map((featureKey) => {
                                        const value = row.features?.[featureKey];
                                        return (
                                          <div key={featureKey} className="rounded border border-slate-200 bg-white px-2 py-1">
                                            <div className="flex items-start justify-between gap-2">
                                              <span className="min-w-0 truncate font-mono text-[10px] text-slate-500" title={featureKey}>{featureKey}</span>
                                              <span className="shrink-0 font-mono text-[10px] font-semibold text-slate-900">{featureValue(value)}</span>
                                            </div>
                                          </div>
                                        );
                                      })}
                                    </div>
                                  </td>
                                </tr>
                              )}
                            </>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
