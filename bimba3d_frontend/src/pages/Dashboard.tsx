import { Link, useNavigate } from "react-router-dom";
import {
  Plus,
  FolderOpen,
  Activity,
  CheckCircle2,
  Clock,
  AlertTriangle,
  X,
  MoreVertical,
  Workflow,
  ArrowRight,
  Database,
  Brain,
  FlaskConical,
  ArrowLeft,
  GitCompareArrows,
} from "lucide-react";
import {
  useEffect,
  useMemo,
  useState,
  useCallback,
  type MouseEvent,
} from "react";
import { api } from "../api/client";
import PipelineBreadcrumbs from "../components/pipelineDetails/PipelineBreadcrumbs";

interface Project {
  project_id: string;
  name: string | null;
  status: string;
  progress: number;
  created_at: string | null;
  modified_at?: string | null;
  has_outputs: boolean;
  session_count?: number;
  created_by?: string | null;
  pipeline_id?: string | null;
  pipeline_name?: string | null;
}

interface PipelineSummary {
  id: string;
  pipeline_type?: string | null;
  workflow_stage?: string | null;
  config?: {
    pipeline_type?: string | null;
  } | null;
}

interface DashboardProps {
  view?: "home" | "projects";
}

const getPipelineStage = (pipeline: PipelineSummary) => {
  const stage = String(pipeline.workflow_stage || "").toLowerCase();
  if (stage === "testing" || stage === "testing_pipeline" || stage === "test") return "test";
  if (stage === "model_training") return "train";
  if (stage === "offline_data_preparation" || stage === "training_data") return "data";

  const type = String(pipeline.pipeline_type || pipeline.config?.pipeline_type || "offline_data").toLowerCase();
  if (type === "test" || type.includes("test")) return "test";
  if (type === "model_training" || type === "model" || type.includes("model") || type.includes("train_model")) return "train";
  return "data";
};

const metricToneClasses = {
  amber: "bg-amber-100 text-amber-700 ring-amber-200",
  blue: "bg-blue-100 text-blue-700 ring-blue-200",
  emerald: "bg-emerald-100 text-emerald-700 ring-emerald-200",
  rose: "bg-rose-100 text-rose-700 ring-rose-200",
};

const pipelineToneClasses = {
  amber: "bg-amber-50 text-amber-700 ring-amber-200",
  blue: "bg-blue-50 text-blue-700 ring-blue-200",
  emerald: "bg-emerald-50 text-emerald-700 ring-emerald-200",
};

export default function Dashboard({ view = "home" }: DashboardProps) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [pipelines, setPipelines] = useState<PipelineSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [editingProject, setEditingProject] = useState<Project | null>(null);
  const [editName, setEditName] = useState("");
  const [editSaving, setEditSaving] = useState(false);
  const [editError, setEditError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState<"modified" | "created" | "name" | "sessions" | "status" | "pipeline">("modified");
  const navigate = useNavigate();
  const [confirmProject, setConfirmProject] = useState<Project | null>(null);
  const [toast, setToast] = useState<{ message: string; type: "success" | "error" } | null>(null);

  const showToast = (message: string, type: "success" | "error" = "success") => {
    setToast({ message, type });
    window.setTimeout(() => setToast(null), 3000);
  };

  const loadProjects = useCallback(async () => {
    try {
      const res = await api.get("/projects");
      const payload = res.data;
      const list = Array.isArray(payload) ? payload : payload?.projects;
      setProjects(list || []);
      try {
        const pipelineRes = await api.get("/api/workflow/pipelines?limit=100");
        setPipelines(pipelineRes.data?.items || []);
      } catch (pipelineErr) {
        console.error("Failed to load pipeline counts", pipelineErr);
        setPipelines([]);
      }
    } catch (err) {
      console.error("Failed to load projects", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadProjects();
    const timer = setInterval(loadProjects, 5000);
    return () => clearInterval(timer);
  }, [loadProjects]);

  const stats = useMemo(() => {
    const total = projects.length;
    const processing = projects.filter((p) => p.status === "processing").length;
    const completed = projects.filter((p) => p.status === "completed").length;
    const failed = projects.filter((p) => p.status === "failed").length;
    return { total, processing, completed, failed };
  }, [projects]);

  const pipelineStats = useMemo(() => {
    return pipelines.reduce(
      (acc, pipeline) => {
        acc[getPipelineStage(pipeline)] += 1;
        return acc;
      },
      { data: 0, train: 0, test: 0 },
    );
  }, [pipelines]);

  const visibleProjects = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    const filtered = q
      ? projects.filter((p) => {
          const name = (p.name || "").toLowerCase();
          const id = (p.project_id || "").toLowerCase();
          return name.includes(q) || id.includes(q);
        })
      : [...projects];

    filtered.sort((a, b) => {
      if (sortBy === "name") {
        const aHasName = Boolean(a.name && a.name.trim());
        const bHasName = Boolean(b.name && b.name.trim());
        if (aHasName !== bHasName) {
          // Keep explicitly named projects first when sorting by name.
          return aHasName ? -1 : 1;
        }
        const an = (a.name || "").trim();
        const bn = (b.name || "").trim();
        const byName = an.localeCompare(bn, undefined, { sensitivity: "base", numeric: true });
        if (byName !== 0) return byName;
        return a.project_id.localeCompare(b.project_id, undefined, { sensitivity: "base", numeric: true });
      }
      if (sortBy === "sessions") {
        return (b.session_count ?? 0) - (a.session_count ?? 0);
      }
      if (sortBy === "status") {
        const statusRank: Record<string, number> = {
          processing: 0,
          stopping: 1,
          failed: 2,
          completed: 3,
          done: 3,
          stopped: 4,
          pending: 5,
        };
        const ar = statusRank[(a.status || "").toLowerCase()] ?? 99;
        const br = statusRank[(b.status || "").toLowerCase()] ?? 99;
        if (ar !== br) return ar - br;
        const aTs = a.modified_at ? Date.parse(a.modified_at) : 0;
        const bTs = b.modified_at ? Date.parse(b.modified_at) : 0;
        return bTs - aTs;
      }
      if (sortBy === "pipeline") {
        // Pipeline projects first, then manual projects
        const aPipeline = a.pipeline_name || "";
        const bPipeline = b.pipeline_name || "";
        const aIsPipeline = Boolean(aPipeline);
        const bIsPipeline = Boolean(bPipeline);

        if (aIsPipeline !== bIsPipeline) {
          return aIsPipeline ? -1 : 1;
        }

        // Within pipeline projects, sort by pipeline name
        if (aIsPipeline && bIsPipeline) {
          const pipelineCompare = aPipeline.localeCompare(bPipeline, undefined, { sensitivity: "base" });
          if (pipelineCompare !== 0) return pipelineCompare;
        }

        // Within same group, sort by name
        const aName = (a.name || a.project_id).toLowerCase();
        const bName = (b.name || b.project_id).toLowerCase();
        return aName.localeCompare(bName, undefined, { sensitivity: "base", numeric: true });
      }
      const aDateRaw = sortBy === "modified" ? (a.modified_at || a.created_at) : a.created_at;
      const bDateRaw = sortBy === "modified" ? (b.modified_at || b.created_at) : b.created_at;
      const aTs = aDateRaw ? Date.parse(aDateRaw) : 0;
      const bTs = bDateRaw ? Date.parse(bDateRaw) : 0;
      return bTs - aTs;
    });

    return filtered;
  }, [projects, searchQuery, sortBy]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-emerald-50 text-emerald-700 border-emerald-200";
      case "processing":
        return "bg-amber-50 text-amber-700 border-amber-200";
      case "failed":
        return "bg-rose-50 text-rose-700 border-rose-200";
      default:
        return "bg-slate-50 text-slate-700 border-slate-200";
    }
  };

  const openEdit = (project: Project) => {
    setEditingProject(project);
    setEditName(project.name || "");
    setEditError(null);
  };

  const closeEdit = () => {
    setEditingProject(null);
    setEditName("");
    setEditSaving(false);
    setEditError(null);
  };

  const saveEdit = async () => {
    if (!editingProject) return;
    const trimmed = editName.trim();
    if (!trimmed) {
      setEditError("Name cannot be empty");
      return;
    }
    setEditSaving(true);
    setEditError(null);
    try {
      await api.patch(`/projects/${editingProject.project_id}`, { name: trimmed });
      setProjects((prev) => prev.map((p) => (p.project_id === editingProject.project_id ? { ...p, name: trimmed } : p)));
      closeEdit();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to update project";
      setEditError(msg);
    } finally {
      setEditSaving(false);
    }
  };

  const requestDelete = (project: Project, e?: MouseEvent) => {
    if (e) e.stopPropagation();
    setConfirmProject(project);
  };

  const performDelete = async () => {
    if (!confirmProject) return;
    setDeletingId(confirmProject.project_id);
    try {
      await api.delete(`/projects/${confirmProject.project_id}`);
      setProjects((prev) => prev.filter((p) => p.project_id !== confirmProject!.project_id));
      showToast("Deleted successfully", "success");
    } catch (err) {
      console.error("Failed to delete project", err);
      showToast("Failed to delete project", "error");
    } finally {
      setDeletingId(null);
      setConfirmProject(null);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50">
      {view === "home" ? (
        <header className="bg-gradient-to-r from-blue-600 via-blue-700 to-indigo-700 shadow-xl">
          <div className="max-w-7xl mx-auto px-6 lg:px-8 py-8">
            <div>
              <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/10 px-3 py-1 backdrop-blur-sm">
                <Activity className="h-3 w-3 text-white" />
                <span className="text-xs font-medium uppercase tracking-wider text-white">Gaussian Splatting Platform</span>
              </div>
              <h1 className="mb-2 text-3xl font-bold tracking-tight text-white lg:text-4xl">Bimba3d</h1>
              <p className="max-w-2xl text-base text-blue-100">
                Professional 3D reconstruction pipeline. Upload images, train Gaussian splats, and visualize results in real-time.
              </p>
            </div>
          </div>
        </header>
      ) : (
        <header className="bg-gradient-to-r from-blue-600 via-blue-700 to-indigo-700 shadow-xl">
          <div className="mx-auto max-w-7xl px-6 py-6">
            <div className="flex items-center gap-4">
              <button
                onClick={() => navigate("/")}
                className="inline-flex items-center gap-2 rounded-xl border border-white/20 bg-white/10 px-3 py-2 text-sm font-medium text-white backdrop-blur-sm transition-all duration-200 hover:scale-105 hover:bg-white/20"
              >
                <ArrowLeft className="h-4 w-4 text-white" />
                Back
              </button>
              <div>
                <div className="mb-2">
                  <PipelineBreadcrumbs items={[{ label: "Home", to: "/" }, { label: "Projects" }]} />
                </div>
                <div className="text-xs font-semibold uppercase tracking-wide text-blue-100">Project Workspace</div>
                <h1 className="text-2xl font-bold text-white">Projects</h1>
              </div>
            </div>
          </div>
        </header>
      )}

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 lg:px-8 py-6 space-y-6">
        {/* Stats Cards */}
        {view === "home" && (
        <div className="-mt-12 relative z-10 rounded-2xl border border-slate-200/70 bg-white/95 p-3 shadow-xl shadow-slate-200/70 backdrop-blur">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {[
              { label: "Total Projects", value: stats.total, icon: FolderOpen, color: "blue" },
              { label: "Processing", value: stats.processing, icon: Clock, color: "amber" },
              { label: "Completed", value: stats.completed, icon: CheckCircle2, color: "emerald" },
              { label: "Failed", value: stats.failed, icon: AlertTriangle, color: "rose" },
            ].map(({ label, value, icon: Icon, color }) => (
              <div key={label} className="rounded-xl border border-slate-200 bg-slate-50/70 px-4 py-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-3">
                    <div className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ring-1 ${metricToneClasses[color as keyof typeof metricToneClasses]}`}>
                      <Icon className="h-[18px] w-[18px]" />
                    </div>
                    <span className="truncate text-xs font-bold uppercase tracking-wide text-slate-600">{label}</span>
                  </div>
                  <span className="text-2xl font-bold tabular-nums text-slate-950">{value}</span>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-3 grid gap-3 md:grid-cols-3">
            {[
              { label: "Data Pipelines", value: pipelineStats.data, icon: Database, color: "blue" },
              { label: "Train Pipelines", value: pipelineStats.train, icon: Brain, color: "emerald" },
              { label: "Test Pipelines", value: pipelineStats.test, icon: FlaskConical, color: "amber" },
            ].map(({ label, value, icon: Icon, color }) => (
              <div key={label} className="rounded-xl border border-slate-200 bg-white px-4 py-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-3">
                    <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ring-1 ${pipelineToneClasses[color as keyof typeof pipelineToneClasses]}`}>
                      <Icon className="h-4 w-4" />
                    </div>
                    <span className="truncate text-sm font-semibold text-slate-800">{label}</span>
                  </div>
                  <span className="text-xl font-bold tabular-nums text-slate-950">{value}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
        )}

        {view === "home" && (
          <div className="grid gap-5 lg:grid-cols-3">
            <Link
              to="/projects"
              className="group rounded-2xl border border-slate-200 bg-white p-6 shadow-sm transition hover:border-blue-300 hover:shadow-lg"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex items-start gap-4">
                  <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-blue-600 to-blue-700 text-white shadow-sm">
                    <FolderOpen className="h-6 w-6" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-slate-950">Projects</h2>
                    <p className="mt-2 max-w-md text-sm leading-6 text-slate-600">
                      Open project workspaces, inspect images, run project tests, review sessions, logs, and per-project results.
                    </p>
                  </div>
                </div>
                <ArrowRight className="mt-1 h-5 w-5 shrink-0 text-slate-400 transition group-hover:translate-x-1 group-hover:text-blue-600" />
              </div>
            </Link>
            <Link
              to="/comparison"
              className="group rounded-2xl border border-slate-200 bg-white p-6 shadow-sm transition hover:border-violet-300 hover:shadow-lg"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex items-start gap-4">
                  <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-violet-600 to-violet-700 text-white shadow-sm">
                    <GitCompareArrows className="h-6 w-6" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-slate-950">Comparison</h2>
                    <p className="mt-2 max-w-md text-sm leading-6 text-slate-600">
                      Compare reconstruction runs, model-guided tests, metrics, visual outputs, and results.
                    </p>
                  </div>
                </div>
                <ArrowRight className="mt-1 h-5 w-5 shrink-0 text-slate-400 transition group-hover:translate-x-1 group-hover:text-violet-600" />
              </div>
            </Link>
            <Link
              to="/workflow"
              className="group rounded-2xl border border-slate-200 bg-white p-6 shadow-sm transition hover:border-emerald-300 hover:shadow-lg"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex items-start gap-4">
                  <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-600 to-emerald-700 text-white shadow-sm">
                    <Workflow className="h-6 w-6" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-slate-950">Research Workflow</h2>
                    <p className="mt-2 max-w-md text-sm leading-6 text-slate-600">
                      Prepare offline data, manage final training datasets, train models, and run testing pipelines.
                    </p>
                  </div>
                </div>
                <ArrowRight className="mt-1 h-5 w-5 shrink-0 text-slate-400 transition group-hover:translate-x-1 group-hover:text-emerald-600" />
              </div>
            </Link>
          </div>
        )}

        {view === "projects" && loading ? (
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="rounded-2xl border border-slate-200 bg-white p-6 animate-pulse flex items-center gap-6 shadow-sm">
                <div className="h-24 w-24 bg-slate-100 rounded-xl flex-shrink-0" />
                <div className="flex-1 space-y-3">
                  <div className="h-6 w-2/3 bg-slate-100 rounded" />
                  <div className="h-4 w-1/3 bg-slate-100 rounded" />
                  <div className="h-3 w-full bg-slate-100 rounded" />
                </div>
              </div>
            ))}
          </div>
        ) : view === "projects" && projects.length === 0 ? (
          <div className="text-center py-16 bg-white border-2 border-dashed border-slate-300 rounded-2xl shadow-sm">
            <div className="h-16 w-16 rounded-xl bg-gradient-to-br from-blue-50 to-blue-100 flex items-center justify-center mx-auto mb-4">
              <FolderOpen className="w-8 h-8 text-blue-600" />
            </div>
            <h2 className="text-xl font-bold text-slate-900 mb-2">No projects yet</h2>
            <p className="text-sm text-slate-600 mb-6 max-w-md mx-auto">Create your first 3D reconstruction project to see it tracked here.</p>
            <Link
              to="/create"
              className="inline-flex items-center gap-2 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-semibold px-6 py-3 rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl hover:scale-105"
            >
              <Plus className="w-4 h-4" />
              Create Your First Project
            </Link>
          </div>
        ) : view === "projects" ? (
          <div>
            <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-lg font-bold text-slate-900">Your Projects</h2>
                <p className="text-xs text-slate-500">Auto-refreshing every 5s</p>
              </div>
              <Link
                to="/create"
                className="inline-flex w-fit items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
              >
                <Plus className="h-4 w-4" />
                Add Project
              </Link>
            </div>
            <div className="mb-4 grid grid-cols-1 md:grid-cols-3 gap-3">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search projects by name or ID"
                className="md:col-span-2 px-3 py-2 rounded-lg border border-slate-300 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as "modified" | "created" | "name" | "sessions" | "status" | "pipeline")}
                className="px-3 py-2 rounded-lg border border-slate-300 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="modified">Sort: Date Modified</option>
                <option value="created">Sort: Date Created</option>
                <option value="name">Sort: Project Name</option>
                <option value="sessions">Sort: Sessions in Project</option>
                <option value="status">Sort: Status</option>
                <option value="pipeline">Sort: Pipeline</option>
              </select>
            </div>
            <div className="space-y-4">
              {visibleProjects.map((project) => (
                <div
                  key={project.project_id}
                  className="group relative block rounded-xl border border-slate-300 bg-white hover:shadow-lg transition-all duration-300 shadow-sm overflow-hidden hover:border-blue-400 cursor-pointer"
                  onClick={() => navigate(`/projects/${project.project_id}`)}
                >
                  <div className="flex items-center gap-4 p-4">
                    {/* Thumbnail/Icon */}
                    <div className="flex-shrink-0 h-16 w-16 rounded-lg bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center shadow-md group-hover:scale-105 transition-transform duration-300">
                      <FolderOpen className="w-8 h-8 text-white" />
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0 space-y-2">
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1 min-w-0">
                          <h3 className="text-base font-bold text-slate-900 group-hover:text-blue-600 transition-colors mb-0.5 truncate">
                            {project.name || `Project ${project.project_id.slice(0, 8)}`}
                          </h3>
                          <p className="text-xs text-slate-500 font-mono">ID: {project.project_id.slice(0, 16)}...</p>
                        </div>
                        <div className="flex items-center gap-1">
                          <span
                            className={`flex-shrink-0 px-3 py-1 rounded-full text-xs font-semibold border ${getStatusColor(
                              project.status
                            )}`}
                          >
                            {project.status}
                          </span>
                          <div className="relative">
                            <button
                              className="p-1.5 rounded-md hover:bg-slate-100 text-slate-500 hover:text-slate-700"
                              onClick={(e) => {
                                e.stopPropagation();
                                setMenuOpenId((prev) => (prev === project.project_id ? null : project.project_id));
                              }}
                              aria-label="Project actions"
                            >
                              <MoreVertical className="w-4 h-4" />
                            </button>
                            {menuOpenId === project.project_id && (
                              <div className="absolute right-0 mt-2 w-36 rounded-lg border border-slate-200 bg-white shadow-lg z-20">
                                <button
                                  className="w-full text-left px-3 py-2 text-sm hover:bg-slate-50"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setMenuOpenId(null);
                                    openEdit(project);
                                  }}
                                >
                                  Edit name
                                </button>
                                <button
                                  className="w-full text-left px-3 py-2 text-sm text-rose-600 hover:bg-rose-50 disabled:opacity-60"
                                  onClick={(e) => {
                                    setMenuOpenId(null);
                                    requestDelete(project, e);
                                  }}
                                  disabled={deletingId === project.project_id}
                                >
                                  {deletingId === project.project_id ? "Deleting..." : "Delete"}
                                </button>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>

                      {/* Progress Bar */}
                      {project.status === "processing" && (
                        <div className="space-y-1">
                          <div className="flex justify-between text-xs font-medium text-slate-600">
                            <span>Processing Progress</span>
                            <span className="text-blue-600">{project.progress}%</span>
                          </div>
                          <div className="w-full h-2.5 rounded-full bg-slate-100 overflow-hidden shadow-inner">
                            <div
                              className="h-2.5 rounded-full bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-500 shadow-sm"
                              style={{ width: `${project.progress}%` }}
                            />
                          </div>
                        </div>
                      )}

                      {/* Metadata */}
                      <div className="flex flex-wrap items-center gap-4 text-xs text-slate-500">
                        {project.pipeline_name && (
                          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-purple-50 text-purple-700 font-medium border border-purple-200">
                            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                            </svg>
                            {project.pipeline_name}
                          </span>
                        )}
                        {project.created_at && (
                          <span className="flex items-center gap-1.5">
                            <Clock className="w-3.5 h-3.5" />
                            Created {new Date(project.created_at).toLocaleDateString()}
                          </span>
                        )}
                        {project.modified_at && (
                          <span className="flex items-center gap-1.5">
                            <Clock className="w-3.5 h-3.5" />
                            Modified {new Date(project.modified_at).toLocaleDateString()}
                          </span>
                        )}
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-100 text-slate-700 font-medium border border-slate-200">
                          Sessions: {project.session_count ?? 0}
                        </span>
                        {project.has_outputs && (
                          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-700 font-medium border border-emerald-200">
                            <CheckCircle2 className="w-3.5 h-3.5" /> Outputs ready
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </main>

      {/* Toast */}
      {toast && (
        <div className={`fixed top-4 right-4 z-50 rounded-lg border px-3 py-2 text-sm shadow-lg ${
          toast.type === "success"
            ? "bg-emerald-50 text-emerald-700 border-emerald-200"
            : "bg-rose-50 text-rose-700 border-rose-200"
        }`}
        >
          {toast.message}
        </div>
      )}

      {/* Edit Modal */}
      {editingProject && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4" onClick={closeEdit}>
          <div
            className="w-full max-w-md rounded-2xl bg-white shadow-2xl p-6 space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-lg font-semibold text-slate-900">Edit project name</h3>
                <p className="text-sm text-slate-500">Update the display name for this project.</p>
              </div>
              <button className="text-slate-400 hover:text-slate-600" onClick={closeEdit}>
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-700">Project name</label>
              <input
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-blue-500 focus:ring focus:ring-blue-100"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                placeholder="Enter project name"
              />
              {editError && <p className="text-sm text-rose-600">{editError}</p>}
            </div>

            <div className="flex justify-end gap-3">
              <button className="px-4 py-2 text-sm font-medium text-slate-600 hover:text-slate-800" onClick={closeEdit}>
                Cancel
              </button>
              <button
                className="px-4 py-2 text-sm font-semibold rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-60 disabled:cursor-not-allowed"
                onClick={saveEdit}
                disabled={editSaving || !editName.trim()}
              >
                {editSaving ? "Saving..." : "Save"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirm Modal */}
      {confirmProject && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4" onClick={() => setConfirmProject(null)}>
          <div
            className="w-full max-w-md rounded-2xl bg-white shadow-2xl p-6 space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-lg font-semibold text-slate-900">Delete project?</h3>
                <p className="text-sm text-slate-500">Are you sure want to delete the model? This cannot be undone.</p>
              </div>
              <button className="text-slate-400 hover:text-slate-600" onClick={() => setConfirmProject(null)}>
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="flex justify-end gap-3">
              <button className="px-4 py-2 text-sm font-medium text-slate-600 hover:text-slate-800" onClick={() => setConfirmProject(null)}>
                Cancel
              </button>
              <button
                className="px-4 py-2 text-sm font-semibold rounded-lg bg-rose-600 text-white hover:bg-rose-700 disabled:opacity-60 disabled:cursor-not-allowed"
                onClick={performDelete}
                disabled={deletingId === confirmProject.project_id}
              >
                {deletingId === confirmProject.project_id ? "Deleting..." : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
