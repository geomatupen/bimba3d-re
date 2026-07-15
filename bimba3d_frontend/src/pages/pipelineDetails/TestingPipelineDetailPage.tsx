import { useEffect, useMemo, useState } from "react";
import { Download, Pencil } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "../../api/client";
import PipelineConfigPanel from "../../components/pipelineDetails/PipelineConfigPanel";
import PipelineDetailShell from "../../components/pipelineDetails/PipelineDetailShell";
import PipelineDetailTabs from "../../components/pipelineDetails/PipelineDetailTabs";
import PipelineLogsPanel from "../../components/pipelineDetails/PipelineLogsPanel";
import PipelineLogSpaceValuesPanel from "../../components/pipelineDetails/PipelineLogSpaceValuesPanel";
import PipelineOverviewPanel from "../../components/pipelineDetails/PipelineOverviewPanel";
import PipelineProjectsRunsPanel from "../../components/pipelineDetails/PipelineProjectsRunsPanel";
import PipelineScoreDistributionPanel from "../../components/pipelineDetails/PipelineScoreDistributionPanel";
import TestingModelsPanel from "../../components/pipelineDetails/TestingModelsPanel";
import TestingPredictionsPanel from "../../components/pipelineDetails/TestingPredictionsPanel";
import TestingResultsPanel from "../../components/pipelineDetails/TestingResultsPanel";
import TestingOverviewSection from "../../components/pipelineOverview/TestingOverviewSection";
import PipelineActionControls from "../../components/workflow/PipelineActionControls";
import type { DetailTab, PipelineDetail } from "../../components/pipelineDetails/types";

interface TestingPipelineDetailPageProps {
  breadcrumbs?: { label: string; to?: string }[];
  onRefresh: () => void;
  pipeline: PipelineDetail;
  refreshing?: boolean;
}

const rowModelId = (row: any): string =>
  String(row?.model_id || row?.test_model_id || row?.source_model_id || row?.current_test_model_id || row?.selected_model_id || "");

const hasCandidateScoreChecks = (row: any): boolean => {
  const checks = row?.candidate_score_checks;
  return !!checks && typeof checks === "object" && Object.values(checks).some((items) => Array.isArray(items) && items.length > 0);
};

const asTestingOverviewRow = (row: any, live = false) => ({
  candidate_points: row?.candidate_points,
  candidate_score_checks: row?.candidate_score_checks || {},
  has_signal: row?.has_signal,
  model_id: rowModelId(row),
  n_runs: row?.n_runs,
  project_name: row?.project_name || row?.project || row?.project_id || (live ? "Active project" : "Project"),
  run_id: row?.run_id,
  score_spreads: row?.score_spreads,
  selected_log_multipliers: row?.selected_log_multipliers || {},
  selected_multipliers: row?.selected_multipliers || {},
  status: "ok",
});

const dedupeTestingRows = (rows: any[]) => {
  const seen = new Set<string>();
  const result: any[] = [];
  for (const row of rows) {
    const key = `${row.project_name || ""}::${row.model_id || ""}::${row.run_id || ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(row);
  }
  return result;
};

const tabs: DetailTab[] = [
  { id: "overview", label: "Overview" },
  { id: "projects", label: "Projects & Runs" },
  { id: "models", label: "Models Used" },
  { id: "predictions", label: "Predictions" },
  { id: "results", label: "Training Data Rows" },
  { id: "logs", label: "Logs" },
  { id: "configuration", label: "Configuration" },
];

export default function TestingPipelineDetailPage({
  breadcrumbs,
  onRefresh,
  pipeline,
  refreshing = false,
}: TestingPipelineDetailPageProps) {
  const [activeTab, setActiveTab] = useState("overview");
  const configuredModelIds = useMemo(
    () => Array.isArray(pipeline.config?.source_model_ids)
      ? pipeline.config.source_model_ids.filter(Boolean)
      : pipeline.config?.source_model_id
        ? [pipeline.config.source_model_id]
        : [],
    [pipeline.config?.source_model_id, pipeline.config?.source_model_ids],
  );
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [exportMessage, setExportMessage] = useState<{ text: string; type: "success" | "error" } | null>(null);
  const latestPreviewKey = String(pipeline.latest_prediction_preview_key || "").trim();
  const latestPreview = latestPreviewKey ? pipeline.prediction_previews?.[latestPreviewKey] : null;
  const predictionRows = Array.isArray(latestPreview?.results)
    ? latestPreview.results
    : Array.isArray(latestPreview?.rows)
      ? latestPreview.rows
      : [];
  const testingOverviewRows = useMemo(() => {
    const completedRunRows = Array.isArray(pipeline.runs)
      ? pipeline.runs
          .filter((run: any) => Number(run?.phase || run?.phase_number || 0) > 1)
          .filter(hasCandidateScoreChecks)
          .map((run: any) => asTestingOverviewRow(run))
      : [];
    const activeRunRows = hasCandidateScoreChecks(pipeline.active_run) && Number(pipeline.active_run?.phase || 0) > 1
      ? [asTestingOverviewRow(pipeline.active_run, true)]
      : [];
    const previewRows = predictionRows.filter(hasCandidateScoreChecks).map((row: any) => asTestingOverviewRow(row));
    return dedupeTestingRows([...activeRunRows, ...completedRunRows, ...previewRows]);
  }, [pipeline.active_run, pipeline.runs, predictionRows]);
  const modelFilteredTestingOverviewRows = selectedModelId
    ? testingOverviewRows.filter((row: any) => rowModelId(row) === selectedModelId)
    : testingOverviewRows;
  const previewGeneratedAt = typeof latestPreview?.generated_at === "string" ? latestPreview.generated_at : null;
  const restartVersion = Number.isFinite(Number(pipeline.config?.restart_version)) ? Number(pipeline.config?.restart_version) : 0;
  const restartToken = typeof pipeline.config?.restart_token === "string" ? pipeline.config.restart_token : null;
  const lastRestartAt = typeof pipeline.config?.last_restart_at === "string" ? pipeline.config.last_restart_at : null;

  const showExportMessage = (text: string, type: "success" | "error") => {
    setExportMessage({ text, type });
    window.setTimeout(() => setExportMessage(null), 3500);
  };

  useEffect(() => {
    if (configuredModelIds.length === 0) {
      setSelectedModelId(null);
      return;
    }
    setSelectedModelId((current) => current && configuredModelIds.includes(current) ? current : null);
  }, [configuredModelIds]);

  const exportCurrentTest = async () => {
    if (exporting) return;
    setExporting(true);
    try {
      const res = await api.get(`/api/workflow/pipelines/${pipeline.id}/export-current-test`, {
        responseType: "blob",
      });
      const contentDisposition = String(res.headers?.["content-disposition"] || "");
      const match = /filename=\"?([^\";]+)\"?/i.exec(contentDisposition);
      const fileName = (match?.[1] || `test_pipeline_export_${pipeline.id}.zip`).trim();
      const blobUrl = window.URL.createObjectURL(new Blob([res.data], { type: "application/zip" }));
      const anchor = document.createElement("a");
      anchor.href = blobUrl;
      anchor.download = fileName;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.URL.revokeObjectURL(blobUrl);
      showExportMessage("Current test export downloaded.", "success");
    } catch (err: any) {
      showExportMessage(err.response?.data?.detail || "Failed to export current test bundle.", "error");
    } finally {
      setExporting(false);
    }
  };

  const headerActions = (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <PipelineActionControls onComplete={onRefresh} pipeline={pipeline} />
        <Link
          to={`/workflow/pipeline-builder?edit=${encodeURIComponent(pipeline.id)}`}
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
          title="Edit this pipeline configuration"
        >
          <Pencil className="h-4 w-4" />
          Edit Config
        </Link>
        <button
          onClick={() => void exportCurrentTest()}
          disabled={exporting}
          className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 px-3 py-2 text-sm font-semibold text-white hover:bg-emerald-700 disabled:opacity-60"
          title="Download the current test bundle with prepared rows, predictions, and model references."
        >
          <Download className={`h-4 w-4 ${exporting ? "animate-pulse" : ""}`} />
          Export Current Test
        </button>
      </div>
      {exportMessage && (
        <div
          className={`rounded-lg border px-2 py-1 text-xs font-medium ${
            exportMessage.type === "success"
              ? "border-green-200 bg-green-50 text-green-700"
              : "border-red-200 bg-red-50 text-red-700"
          }`}
        >
          {exportMessage.text}
        </div>
      )}
    </div>
  );

  return (
    <PipelineDetailShell
      actions={headerActions}
      badge="Run Tests"
      breadcrumbs={breadcrumbs || [
        { label: "Research Workflow", to: "/workflow" },
        { label: "Run Tests", to: "/testing-pipeline" },
        { label: pipeline.name },
      ]}
      onRefresh={onRefresh}
      pipeline={pipeline}
      refreshing={refreshing}
    >
      <PipelineDetailTabs activeTab={activeTab} onChange={setActiveTab} tabs={tabs} />

      {activeTab === "overview" && (
        <div className="space-y-4">
          <PipelineOverviewPanel pipeline={pipeline} variant="test" />
          <TestingModelsPanel
            onSelectModel={setSelectedModelId}
            pipeline={pipeline}
            selectedModelId={selectedModelId}
          />
          <PipelineLogSpaceValuesPanel
            pipeline={pipeline}
            predictionRows={predictionRows}
            selectedModelId={selectedModelId}
          />
          <TestingOverviewSection
            rows={modelFilteredTestingOverviewRows}
            loading={false}
            restartVersion={restartVersion}
            restartToken={restartToken}
            lastRestartAt={lastRestartAt}
            previewGeneratedAt={previewGeneratedAt}
          />
          <PipelineScoreDistributionPanel pipelineId={pipeline.id} refreshKey={pipeline.updated_at} selectedModelId={selectedModelId} title="Observed Test Score Distribution" />
        </div>
      )}
      {activeTab === "projects" && <PipelineProjectsRunsPanel pipeline={pipeline} />}
      {activeTab === "models" && <TestingModelsPanel pipeline={pipeline} />}
      {activeTab === "predictions" && <TestingPredictionsPanel pipeline={pipeline} selectedModelId={selectedModelId} />}
      {activeTab === "results" && <TestingResultsPanel pipeline={pipeline} />}
      {activeTab === "logs" && <PipelineLogsPanel pipelineId={pipeline.id} />}
      {activeTab === "configuration" && <PipelineConfigPanel pipeline={pipeline} />}
    </PipelineDetailShell>
  );
}
