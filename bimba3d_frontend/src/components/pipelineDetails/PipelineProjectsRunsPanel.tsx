import { Fragment, useEffect, useMemo, useState } from "react";
import { ExternalLink } from "lucide-react";
import { useLocation, useNavigate } from "react-router-dom";
import type { PipelineDetail } from "./types";

interface PipelineProjectsRunsPanelProps {
  pipeline: PipelineDetail;
}

interface ProjectOption {
  key: string;
  label: string;
  id: string | null;
  index: number;
}

const projectKey = (project: any, index: number) =>
  String(project.name || project.project_name || project.project_id || project.id || `Project ${index + 1}`);

const projectId = (project: any) => project.project_id || project.id || null;

const formatDate = (value?: string | null) => {
  if (!value) return "-";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
};

const statusClass = (status?: string) => {
  const normalized = String(status || "").toLowerCase();
  if (normalized === "success" || normalized === "completed") return "bg-green-100 text-green-700";
  if (normalized === "hard_cap_reached" || normalized === "partial_completed") return "bg-orange-100 text-orange-700";
  if (normalized === "failed" || normalized === "error") return "bg-red-100 text-red-700";
  if (normalized === "running") return "bg-blue-100 text-blue-700";
  return "bg-slate-100 text-slate-700";
};

export default function PipelineProjectsRunsPanel({ pipeline }: PipelineProjectsRunsPanelProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const projects = Array.isArray(pipeline.config?.projects) ? pipeline.config.projects : [];
  const runs = Array.isArray(pipeline.runs) ? pipeline.runs : [];
  const returnTo = encodeURIComponent(`${location.pathname}${location.search}`);

  const projectOptions = useMemo<ProjectOption[]>(
    () =>
      projects.map((project: any, index: number) => ({
        key: projectKey(project, index),
        label: project.name || project.project_name || project.project_id || project.id || `Project ${index + 1}`,
        id: projectId(project),
        index,
      })),
    [projects],
  );

  const [selectedProjects, setSelectedProjects] = useState<string[]>([]);
  const [sortBy, setSortBy] = useState<"project" | "time">("project");

  useEffect(() => {
    setSelectedProjects(projectOptions.map((project) => project.key));
  }, [projectOptions]);

  const selectedSet = useMemo(() => new Set(selectedProjects), [selectedProjects]);
  const selectedRunProjectKeys = useMemo(() => {
    const keys = new Set<string>();
    projectOptions.forEach((project) => {
      if (!selectedSet.has(project.key)) return;
      keys.add(project.key);
      keys.add(project.label);
      if (project.id) keys.add(project.id);
    });
    return keys;
  }, [projectOptions, selectedSet]);
  const filteredRuns = useMemo(() => {
    const visibleRuns = runs.filter((run: any) => {
      const runProject = String(run.project_name || run.project || run.project_id || "");
      return selectedRunProjectKeys.has(runProject);
    });

    const runTime = (run: any) => {
      const value = run.completed_at || run.timestamp || run.created_at || "";
      const parsed = Date.parse(value);
      return Number.isNaN(parsed) ? 0 : parsed;
    };

    if (sortBy === "time") {
      return [...visibleRuns].sort((a: any, b: any) => runTime(b) - runTime(a));
    }

    const projectOrder = new Map<string, number>();
    projectOptions.forEach((project, index) => {
      projectOrder.set(project.key, index);
      projectOrder.set(project.label, index);
      if (project.id) projectOrder.set(project.id, index);
    });
    return [...visibleRuns].sort((a: any, b: any) => {
      const aProject = String(a.project_name || a.project || a.project_id || "");
      const bProject = String(b.project_name || b.project || b.project_id || "");
      const aOrder = projectOrder.get(aProject) ?? Number.MAX_SAFE_INTEGER;
      const bOrder = projectOrder.get(bProject) ?? Number.MAX_SAFE_INTEGER;
      if (aOrder !== bOrder) return aOrder - bOrder;
      return runTime(a) - runTime(b);
    });
  }, [projectOptions, runs, selectedRunProjectKeys, sortBy]);

  const allSelected = projectOptions.length > 0 && selectedProjects.length === projectOptions.length;

  const toggleProject = (key: string) => {
    setSelectedProjects((current) =>
      current.includes(key) ? current.filter((item) => item !== key) : [...current, key],
    );
  };

  return (
    <section className="rounded-lg border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-200 px-4 py-3">
        <h2 className="text-base font-semibold text-slate-950">Pipeline Projects</h2>
        <p className="mt-0.5 text-xs text-slate-600">Select projects on the left to filter the runs on the right.</p>
      </div>

      <div className="grid gap-3 p-3 lg:grid-cols-[minmax(300px,0.8fr)_minmax(0,1.6fr)]">
        <div className="min-w-0">
          <div className="mb-2 flex items-center justify-between gap-3">
            <h3 className="text-sm font-semibold text-slate-800">Projects</h3>
            <div className="flex items-center gap-1 text-[10px] leading-none">
              <button
                onClick={() => setSelectedProjects(projectOptions.map((project) => project.key))}
                className="text-[10px] font-semibold leading-none text-blue-700 hover:text-blue-900"
              >
                Select all
              </button>
              <span className="text-slate-300">|</span>
              <button
                onClick={() => setSelectedProjects([])}
                className="text-[10px] font-semibold leading-none text-blue-700 hover:text-blue-900"
              >
                Deselect all
              </button>
            </div>
          </div>

          <div className="max-h-[620px] overflow-auto rounded-lg border border-slate-200">
            {projectOptions.length === 0 ? (
              <div className="p-3 text-xs text-slate-500">No projects configured for this pipeline.</div>
            ) : (
              projectOptions.map((project) => {
                const isCurrentProject = pipeline.current_project_index === project.index;
                const isChecked = selectedSet.has(project.key);

                return (
                  <div
                    key={project.key}
                    className={`border-b border-slate-100 px-2 py-1.5 last:border-0 ${
                      isCurrentProject ? "bg-indigo-50" : isChecked ? "bg-white" : "bg-slate-50"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onChange={() => toggleProject(project.key)}
                        className="h-3.5 w-3.5 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                        aria-label={`Filter runs for ${project.label}`}
                      />
                      <button
                        onClick={() => toggleProject(project.key)}
                        className="block min-w-0 flex-1 text-left"
                      >
                        <div className="flex w-full flex-wrap items-center justify-start gap-2 text-left">
                          <h4 className="min-w-0 flex-1 truncate text-left text-xs font-semibold text-slate-950" title={project.label}>
                            {project.label}
                          </h4>
                          {isCurrentProject && <span className="rounded bg-indigo-600 px-1 py-0.5 text-[10px] font-medium text-white">Current</span>}
                        </div>
                      </button>
                      {project.id && (
                        <button
                          onClick={() => navigate(`/projects/${project.id}?from=workflow-projects&returnToPipeline=${encodeURIComponent(pipeline.id)}&returnTo=${returnTo}`)}
                          className="shrink-0 rounded border border-indigo-300 px-1.5 py-0.5 text-[10px] font-medium text-indigo-700 hover:bg-indigo-50"
                          title="Open project"
                        >
                          <ExternalLink className="h-3.5 w-3.5" />
                        </button>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>
          <div className="mt-2 text-xs text-slate-500">
            {selectedProjects.length}/{projectOptions.length} projects selected
            {!allSelected && selectedProjects.length > 0 ? ` - showing ${filteredRuns.length} run records` : ""}
          </div>
        </div>

        <div className="min-w-0">
          <div className="mb-2 flex items-center justify-between gap-3">
            <h3 className="text-sm font-semibold text-slate-800">Runs</h3>
            <div className="flex items-center gap-2">
              <label className="text-[11px] font-semibold text-slate-500" htmlFor="runs-sort-by">Sort by</label>
              <select
                id="runs-sort-by"
                value={sortBy}
                onChange={(event) => setSortBy(event.target.value as "project" | "time")}
                className="rounded border border-slate-200 bg-white px-2 py-1 text-[11px] font-semibold text-slate-700"
              >
                <option value="project">Project</option>
                <option value="time">Time</option>
              </select>
              <span className="rounded bg-slate-100 px-2 py-0.5 text-[11px] font-semibold text-slate-600">
                {filteredRuns.length} shown
              </span>
            </div>
          </div>
          <div className="max-h-[620px] overflow-auto rounded-lg border border-slate-200">
            {filteredRuns.length === 0 ? (
              <div className="p-3 text-xs text-slate-500">No run records match the selected projects.</div>
            ) : (
              <table className="w-full border-collapse text-[11px]">
                <thead className="sticky top-0 z-10 bg-slate-50">
                  <tr>
                    <th className="border-b border-slate-200 px-1.5 py-1 text-left font-semibold text-slate-700">#</th>
                    <th className="border-b border-slate-200 px-1.5 py-1 text-left font-semibold text-slate-700">Project</th>
                    <th className="border-b border-slate-200 px-1.5 py-1 text-left font-semibold text-slate-700">Run Name</th>
                    <th className="border-b border-slate-200 px-1.5 py-1 text-left font-semibold text-slate-700">Phase</th>
                    <th className="border-b border-slate-200 px-1.5 py-1 text-left font-semibold text-slate-700">Run</th>
                    <th className="border-b border-slate-200 px-1.5 py-1 text-left font-semibold text-slate-700">Status</th>
                    <th className="border-b border-slate-200 px-1.5 py-1 text-left font-semibold text-slate-700">Score</th>
                    <th className="border-b border-slate-200 px-1.5 py-1 text-left font-semibold text-slate-700">Multipliers</th>
                    <th className="border-b border-slate-200 px-1.5 py-1 text-left font-semibold text-slate-700">Notes</th>
                    <th className="border-b border-slate-200 px-1.5 py-1 text-left font-semibold text-slate-700">Completed</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredRuns.map((run: any, index: number) => {
                    const multipliers = run.group_multipliers || run.selected_multipliers || null;
                    const multiplierText = multipliers
                      ? Object.entries(multipliers)
                          .slice(0, 3)
                          .map(([key, value]: [string, any]) => {
                            const numeric = typeof value === "number" ? value : value?.multiplier;
                            return `${key.replace("_lr", "").replace("_", " ")} ${typeof numeric === "number" ? numeric.toFixed(3) : JSON.stringify(value)}`;
                          })
                          .join(" | ")
                      : "";
                    const notes = run.error || run.last_error || run.remarks || "";
                    const isBaseline = run.phase === 1 || run.is_baseline_row || String(run.run_name || "").toLowerCase().includes("baseline");
                    const projectName = run.project_name || run.project || "-";
                    const previousRun = filteredRuns[index - 1];
                    const previousProjectName = previousRun?.project_name || previousRun?.project || "-";
                    const showProjectGroup = sortBy === "project" && projectName !== previousProjectName;

                    return (
                      <Fragment key={`${run.run_id || run.id || index}-row-group`}>
                      {showProjectGroup && (
                        <tr key={`${projectName}-group-${index}`}>
                          <td colSpan={10} className="border-b border-slate-200 bg-slate-100 px-1.5 py-1 text-[11px] font-bold text-slate-700" title={projectName}>
                            {projectName}
                          </td>
                        </tr>
                      )}
                      <tr key={`${run.run_id || run.id || index}`} className="hover:bg-slate-50">
                        <td className="border-b border-slate-100 px-1.5 py-1 text-slate-900">{index + 1}</td>
                        <td className="max-w-[180px] truncate border-b border-slate-100 px-1.5 py-1 font-medium text-slate-900" title={projectName}>
                          {projectName}
                        </td>
                        <td className="max-w-[220px] border-b border-slate-100 px-1.5 py-1 font-medium text-slate-900">
                          <div className="truncate" title={run.run_name || run.run_id || run.id || "-"}>{run.run_name || run.run_id || run.id || "-"}</div>
                          {isBaseline && <span className="mt-1 inline-block rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-700">Baseline</span>}
                        </td>
                        <td className="border-b border-slate-100 px-1.5 py-1 text-slate-700">{run.phase ?? "-"}</td>
                        <td className="border-b border-slate-100 px-1.5 py-1 text-slate-700">{run.run ?? "-"}</td>
                        <td className="border-b border-slate-100 px-1.5 py-1">
                          {run.status ? (
                            <span className={`inline-block rounded px-1 py-0.5 text-[10px] font-medium ${statusClass(run.status)}`}>
                              {run.status}
                            </span>
                          ) : (
                            "-"
                          )}
                        </td>
                        <td className="border-b border-slate-100 px-1.5 py-1 font-medium text-slate-900">
                          {typeof run.score === "number" ? run.score.toFixed(4) : "N/A"}
                        </td>
                        <td className="max-w-[260px] truncate border-b border-slate-100 px-1.5 py-1 font-mono text-[10px] text-indigo-700" title={multiplierText}>
                          {multiplierText || "-"}
                        </td>
                        <td className="max-w-[220px] truncate border-b border-slate-100 px-1.5 py-1 text-slate-600" title={notes}>
                          {notes || "-"}
                        </td>
                        <td className="border-b border-slate-100 px-1.5 py-1 text-slate-600">{formatDate(run.completed_at || run.timestamp)}</td>
                      </tr>
                      </Fragment>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
