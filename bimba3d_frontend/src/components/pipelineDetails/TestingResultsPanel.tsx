import type { PipelineDetail } from "./types";
import TrainingDataRowsTable from "../TrainingDataRowsTable";

interface TestingResultsPanelProps {
  pipeline: PipelineDetail;
}

export default function TestingResultsPanel({ pipeline }: TestingResultsPanelProps) {
  const configuredModelIds = Array.isArray(pipeline.config?.source_model_ids)
    ? pipeline.config.source_model_ids.filter(Boolean)
    : pipeline.config?.source_model_id
      ? [pipeline.config.source_model_id]
      : [];

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-slate-950">Training Data Rows</h2>
        <p className="mt-1 text-sm text-slate-600">
          Recorded test rows with model ids, selected multipliers, scores, score terms, and report values.
        </p>
        {configuredModelIds.length > 0 && (
          <p className="mt-1 text-xs text-slate-500">
            Showing rows for all {configuredModelIds.length} configured model{configuredModelIds.length === 1 ? "" : "s"} in this test pipeline.
          </p>
        )}
      </div>
      <TrainingDataRowsTable pipelineId={pipeline.id} showFinalMetricDeltas />
    </section>
  );
}
