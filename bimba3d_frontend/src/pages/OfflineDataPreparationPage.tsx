import { useCallback, useEffect, useMemo, useState } from "react";
import { Database, FolderSearch, Plus, SlidersHorizontal, Thermometer, Workflow } from "lucide-react";
import { api } from "../api/client";
import PipelineSummaryList, { getPipelineStage, type WorkflowPipeline } from "../components/workflow/PipelineSummaryList";
import WorkflowActionPanel from "../components/workflow/WorkflowActionPanel";
import WorkflowShell from "../components/workflow/WorkflowShell";

export default function OfflineDataPreparationPage() {
  const [pipelines, setPipelines] = useState<WorkflowPipeline[]>([]);
  const [loading, setLoading] = useState(true);

  const loadPipelines = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.get("/api/workflow/pipelines?stage=offline_data_preparation&limit=50");
      setPipelines(res.data?.items || []);
    } catch (err) {
      console.error("Failed to load preparation pipelines", err);
      setPipelines([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadPipelines();
  }, [loadPipelines]);

  const preparationPipelines = useMemo(
    () => pipelines.filter((pipeline) => getPipelineStage(pipeline) === "training_data"),
    [pipelines],
  );

  return (
    <WorkflowShell
      eyebrow="Stage 1"
      title="Prepare Offline Data"
      backTo="/workflow"
      breadcrumbs={[
        { label: "Research Workflow", to: "/workflow" },
        { label: "Prepare Offline Data" },
      ]}
    >
      <div className="space-y-4">
        <WorkflowActionPanel
          title="Prepare Training Data"
          subtitle="Select projects, configure baseline runs, and set exploration runs."
          icon={Workflow}
          actionIcon={Plus}
          actionLabel="Create New"
          actionTo="/workflow/pipeline-builder?type=offline_data"
          tone="blue"
        >
          <div className="grid gap-3 md:grid-cols-3">
            <div className="rounded-lg border border-slate-200 p-3">
              <FolderSearch className="mb-2 h-4 w-4 text-blue-600" />
              <div className="text-sm font-semibold text-slate-900">Datasets</div>
              <div className="text-xs text-slate-500">Directory scan and project selection stay configurable.</div>
            </div>
            <div className="rounded-lg border border-slate-200 p-3">
              <SlidersHorizontal className="mb-2 h-4 w-4 text-blue-600" />
              <div className="text-sm font-semibold text-slate-900">Exploration</div>
              <div className="text-xs text-slate-500">Run count and multiplier ranges remain editable.</div>
            </div>
            <div className="rounded-lg border border-slate-200 p-3">
              <Thermometer className="mb-2 h-4 w-4 text-blue-600" />
              <div className="text-sm font-semibold text-slate-900">Runtime</div>
              <div className="text-xs text-slate-500">Storage and cooldown settings stay in the setup flow.</div>
            </div>
          </div>
        </WorkflowActionPanel>

        <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-start justify-between gap-3">
            <div className="flex items-start gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-600 to-emerald-700 text-white shadow-sm">
                <Database className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-slate-950">Preparation Pipelines</h2>
                <p className="mt-1 text-sm text-slate-600">Baseline, exploration, and prepared dataset runs.</p>
              </div>
            </div>
            <button
              onClick={() => void loadPipelines()}
              className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
            >
              Refresh
            </button>
          </div>
          <PipelineSummaryList
            detailBasePath="/offline-data-preparation/pipelines"
            emptyMessage="No offline preparation pipelines found yet."
            loading={loading}
            pipelines={preparationPipelines}
            tone="blue"
          />
        </section>
      </div>
    </WorkflowShell>
  );
}
