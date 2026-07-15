import { useCallback, useEffect, useMemo, useState } from "react";
import { RefreshCw, Search, SlidersHorizontal, Workflow } from "lucide-react";
import { api } from "../../api/client";
import PipelineSummaryList, {
  getPipelineStage,
  type WorkflowPipeline,
  type WorkflowPipelineStage,
} from "./PipelineSummaryList";

type StageFilter = "all" | WorkflowPipelineStage;
type SortBy = "created_desc" | "created_asc" | "name" | "status" | "progress" | "stage";

interface PipelineWorkspacePanelProps {
  detailBasePath?: string;
  subtitle?: string;
  title?: string;
}

const progressValue = (pipeline: WorkflowPipeline) =>
  pipeline.total_runs > 0 ? pipeline.completed_runs / pipeline.total_runs : 0;

export default function PipelineWorkspacePanel({
  detailBasePath = "/all-pipelines/pipelines",
  subtitle = "Data, training, and testing pipelines in one searchable workspace.",
  title = "Pipeline List",
}: PipelineWorkspacePanelProps) {
  const [pipelines, setPipelines] = useState<WorkflowPipeline[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [stageFilter, setStageFilter] = useState<StageFilter>("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [sortBy, setSortBy] = useState<SortBy>("created_desc");

  const loadPipelines = useCallback(async () => {
    setRefreshing(true);
    try {
      const res = await api.get("/api/workflow/pipelines?limit=100");
      setPipelines(res.data?.items || []);
    } catch (err) {
      console.error("Failed to load all pipelines", err);
      setPipelines([]);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void loadPipelines();
  }, [loadPipelines]);

  const statuses = useMemo(
    () => Array.from(new Set(pipelines.map((pipeline) => pipeline.status).filter(Boolean))).sort(),
    [pipelines],
  );

  const visiblePipelines = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    const filtered = pipelines.filter((pipeline) => {
      const stage = getPipelineStage(pipeline);
      const matchesStage = stageFilter === "all" || stage === stageFilter;
      const matchesStatus = statusFilter === "all" || pipeline.status === statusFilter;
      const matchesSearch =
        !query ||
        pipeline.name.toLowerCase().includes(query) ||
        pipeline.id.toLowerCase().includes(query) ||
        pipeline.status.toLowerCase().includes(query) ||
        stage.includes(query);
      return matchesStage && matchesStatus && matchesSearch;
    });

    filtered.sort((a, b) => {
      if (sortBy === "name") return a.name.localeCompare(b.name, undefined, { sensitivity: "base", numeric: true });
      if (sortBy === "status") return a.status.localeCompare(b.status, undefined, { sensitivity: "base" });
      if (sortBy === "progress") return progressValue(b) - progressValue(a);
      if (sortBy === "stage") return getPipelineStage(a).localeCompare(getPipelineStage(b));
      const aTime = new Date(a.created_at).getTime() || 0;
      const bTime = new Date(b.created_at).getTime() || 0;
      return sortBy === "created_asc" ? aTime - bTime : bTime - aTime;
    });

    return filtered;
  }, [pipelines, searchQuery, sortBy, stageFilter, statusFilter]);

  const counts = useMemo(() => {
    return pipelines.reduce(
      (acc, pipeline) => {
        const stage = getPipelineStage(pipeline);
        acc[stage] += 1;
        return acc;
      },
      { training_data: 0, model_training: 0, test: 0 },
    );
  }, [pipelines]);

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-purple-600 to-purple-700 text-white shadow-sm">
            <Workflow className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-slate-950">{title}</h2>
            <p className="mt-1 text-sm text-slate-600">{subtitle}</p>
          </div>
        </div>
        <button
          onClick={() => void loadPipelines()}
          disabled={refreshing}
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-60"
        >
          <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      <div className="mb-4 grid gap-3 lg:grid-cols-[1fr_160px_180px_180px]">
        <label className="relative block">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <input
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            placeholder="Search pipelines by name, id, status, or stage"
            className="w-full rounded-lg border border-slate-200 py-2 pl-9 pr-3 text-sm outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
          />
        </label>
        <label className="relative block">
          <SlidersHorizontal className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <select
            value={stageFilter}
            onChange={(event) => setStageFilter(event.target.value as StageFilter)}
            className="w-full appearance-none rounded-lg border border-slate-200 py-2 pl-9 pr-3 text-sm outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
          >
            <option value="all">Stage: All</option>
            <option value="training_data">Stage: Data</option>
            <option value="model_training">Stage: Train</option>
            <option value="test">Stage: Test</option>
          </select>
        </label>
        <select
          value={statusFilter}
          onChange={(event) => setStatusFilter(event.target.value)}
          className="rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
        >
          <option value="all">Status: All</option>
          {statuses.map((status) => (
            <option key={status} value={status}>
              Status: {status}
            </option>
          ))}
        </select>
        <select
          value={sortBy}
          onChange={(event) => setSortBy(event.target.value as SortBy)}
          className="rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
        >
          <option value="created_desc">Sort: Newest</option>
          <option value="created_asc">Sort: Oldest</option>
          <option value="name">Sort: Name</option>
          <option value="stage">Sort: Stage</option>
          <option value="status">Sort: Status</option>
          <option value="progress">Sort: Progress</option>
        </select>
      </div>

      <div className="mb-4 flex flex-wrap gap-2 text-xs">
        <span className="rounded-full bg-slate-100/60 px-2 py-1 font-semibold text-slate-600 ring-1 ring-slate-200/60">
          All {pipelines.length}
        </span>
        <span className="rounded-full bg-blue-100/45 px-2 py-1 font-semibold text-blue-600 ring-1 ring-blue-200/50">
          Data {counts.training_data}
        </span>
        <span className="rounded-full bg-emerald-100/45 px-2 py-1 font-semibold text-emerald-600 ring-1 ring-emerald-200/50">
          Train {counts.model_training}
        </span>
        <span className="rounded-full bg-amber-100/45 px-2 py-1 font-semibold text-amber-600 ring-1 ring-amber-200/50">
          Test {counts.test}
        </span>
      </div>

      <PipelineSummaryList
        detailBasePath={detailBasePath}
        emptyMessage="No pipelines match the current search and filters."
        loading={loading}
        pipelines={visiblePipelines}
        showStageLabel
        tone="blue"
      />
    </section>
  );
}
