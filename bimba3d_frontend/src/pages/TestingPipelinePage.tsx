import { useCallback, useEffect, useMemo, useState } from "react";
import { FlaskConical, GitCompare, PlayCircle, Plus, SlidersHorizontal } from "lucide-react";
import { api } from "../api/client";
import PipelineSummaryList, { type WorkflowPipeline } from "../components/workflow/PipelineSummaryList";
import WorkflowActionPanel from "../components/workflow/WorkflowActionPanel";
import WorkflowShell from "../components/workflow/WorkflowShell";

export default function TestingPipelinePage() {
  const [pipelines, setPipelines] = useState<WorkflowPipeline[]>([]);
  const [loading, setLoading] = useState(true);

  const loadPipelines = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.get("/api/workflow/pipelines?stage=testing_pipeline&limit=50");
      setPipelines(res.data?.items || []);
    } catch (err) {
      console.error("Failed to load testing pipelines", err);
      setPipelines([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadPipelines();
  }, [loadPipelines]);

  const testingPipelines = useMemo(
    () => pipelines.filter((pipeline) => pipeline.workflow_stage === "testing_pipeline" || pipeline.pipeline_type === "test"),
    [pipelines],
  );

  return (
    <WorkflowShell
      eyebrow="Stage 3"
      title="Run Tests"
      backTo="/workflow"
      breadcrumbs={[
        { label: "Research Workflow", to: "/workflow" },
        { label: "Run Tests" },
      ]}
    >
      <div className="space-y-4">
        <WorkflowActionPanel
          title="Create Testing Pipeline"
          subtitle="Run baseline plus selected featurewise or compact models on test projects."
          icon={FlaskConical}
          actionIcon={Plus}
          actionLabel="Create New"
          actionTo="/workflow/pipeline-builder?type=test"
          tone="amber"
        >
          <div className="grid gap-3 md:grid-cols-3">
            <div className="rounded-lg border border-slate-200 p-3">
              <PlayCircle className="mb-2 h-4 w-4 text-amber-600" />
              <div className="text-sm font-semibold text-slate-900">Default Baseline</div>
              <div className="text-xs text-slate-500">Each test project keeps a default comparison run.</div>
            </div>
            <div className="rounded-lg border border-slate-200 p-3">
              <SlidersHorizontal className="mb-2 h-4 w-4 text-amber-600" />
              <div className="text-sm font-semibold text-slate-900">Predicted Runs</div>
              <div className="text-xs text-slate-500">Featurewise and compact model selection remains configurable.</div>
            </div>
            <div className="rounded-lg border border-slate-200 p-3">
              <GitCompare className="mb-2 h-4 w-4 text-amber-600" />
              <div className="text-sm font-semibold text-slate-900">Evaluation</div>
              <div className="text-xs text-slate-500">Comparison outputs feed final report result tables.</div>
            </div>
          </div>
        </WorkflowActionPanel>
        <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-start justify-between gap-3">
            <div className="flex items-start gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-amber-500 to-amber-600 text-white shadow-sm">
                <FlaskConical className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-slate-950">Testing Pipelines</h2>
                <p className="mt-1 text-sm text-slate-600">Baseline and selected model-predicted evaluation runs.</p>
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
            detailBasePath="/testing-pipeline/pipelines"
            emptyMessage="No testing pipelines found yet."
            loading={loading}
            pipelines={testingPipelines}
            tone="amber"
          />
        </section>
      </div>
    </WorkflowShell>
  );
}
