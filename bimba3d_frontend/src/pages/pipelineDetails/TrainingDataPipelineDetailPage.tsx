import { useState } from "react";
import { Pencil } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import PipelineConfigPanel from "../../components/pipelineDetails/PipelineConfigPanel";
import PipelineDetailShell from "../../components/pipelineDetails/PipelineDetailShell";
import PipelineDetailTabs from "../../components/pipelineDetails/PipelineDetailTabs";
import PipelineLogsPanel from "../../components/pipelineDetails/PipelineLogsPanel";
import PipelineLogSpaceValuesPanel from "../../components/pipelineDetails/PipelineLogSpaceValuesPanel";
import PipelineOverviewPanel from "../../components/pipelineDetails/PipelineOverviewPanel";
import PipelineProjectsRunsPanel from "../../components/pipelineDetails/PipelineProjectsRunsPanel";
import PipelineScoreDistributionPanel from "../../components/pipelineDetails/PipelineScoreDistributionPanel";
import TrainingDataRowsTable from "../../components/TrainingDataRowsTable";
import TrainingDataExifPanel from "../../components/pipelineDetails/TrainingDataExifPanel";
import PipelineActionControls from "../../components/workflow/PipelineActionControls";
import type { DetailTab, PipelineDetail } from "../../components/pipelineDetails/types";

interface TrainingDataPipelineDetailPageProps {
  breadcrumbs?: { label: string; to?: string }[];
  onRefresh: () => void;
  pipeline: PipelineDetail;
  refreshing?: boolean;
}

const tabs: DetailTab[] = [
  { id: "overview", label: "Overview" },
  { id: "projects", label: "Projects & Runs" },
  { id: "learning", label: "Training Data Rows" },
  { id: "configuration", label: "Configuration" },
  { id: "exif", label: "EXIF" },
  { id: "logs", label: "Logs" },
];

export default function TrainingDataPipelineDetailPage({
  breadcrumbs,
  onRefresh,
  pipeline,
  refreshing = false,
}: TrainingDataPipelineDetailPageProps) {
  const location = useLocation();
  const [activeTab, setActiveTab] = useState("overview");
  const returnTo = encodeURIComponent(`${location.pathname}${location.search}`);
  const headerActions = (
    <div className="flex flex-wrap items-center gap-2">
      <PipelineActionControls onComplete={onRefresh} pipeline={pipeline} />
      <Link
        to={`/workflow/pipeline-builder?edit=${encodeURIComponent(pipeline.id)}&returnTo=${returnTo}`}
        className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
        title="Edit this pipeline configuration"
      >
        <Pencil className="h-4 w-4" />
        Edit Config
      </Link>
    </div>
  );

  return (
    <PipelineDetailShell
      actions={headerActions}
      badge="Prepare Offline Data"
      breadcrumbs={breadcrumbs || [
        { label: "Research Workflow", to: "/workflow" },
        { label: "Prepare Offline Data", to: "/offline-data-preparation" },
        { label: pipeline.name },
      ]}
      onRefresh={onRefresh}
      pipeline={pipeline}
      refreshing={refreshing}
    >
      <PipelineDetailTabs activeTab={activeTab} onChange={setActiveTab} tabs={tabs} />

      {activeTab === "overview" && (
        <div className="space-y-4">
          <PipelineOverviewPanel pipeline={pipeline} variant="training_data" />
          <PipelineLogSpaceValuesPanel pipeline={pipeline} />
          <PipelineScoreDistributionPanel pipelineId={pipeline.id} title="Relative Score Distribution" />
        </div>
      )}
      {activeTab === "projects" && <PipelineProjectsRunsPanel pipeline={pipeline} onRunDeleted={onRefresh} />}
      {activeTab === "learning" && (
        <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4">
            <h2 className="text-lg font-semibold text-slate-950">Training Data Rows</h2>
            <p className="mt-1 text-sm text-slate-600">
              Prepared run rows, selected multipliers, score terms, and report values from this data preparation pipeline.
            </p>
          </div>
          <TrainingDataRowsTable pipelineId={pipeline.id} />
        </section>
      )}
      {activeTab === "logs" && <PipelineLogsPanel pipelineId={pipeline.id} />}
      {activeTab === "configuration" && <PipelineConfigPanel pipeline={pipeline} />}
      {activeTab === "exif" && <TrainingDataExifPanel pipeline={pipeline} />}
    </PipelineDetailShell>
  );
}

