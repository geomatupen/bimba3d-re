import { useState } from "react";
import ModelTrainingArtifactsPanel from "../../components/pipelineDetails/ModelTrainingArtifactsPanel";
import ModelTrainingDetailsPanel from "../../components/pipelineDetails/ModelTrainingDetailsPanel";
import PipelineConfigPanel from "../../components/pipelineDetails/PipelineConfigPanel";
import PipelineDetailShell from "../../components/pipelineDetails/PipelineDetailShell";
import PipelineDetailTabs from "../../components/pipelineDetails/PipelineDetailTabs";
import PipelineLogsPanel from "../../components/pipelineDetails/PipelineLogsPanel";
import PipelineOverviewPanel from "../../components/pipelineDetails/PipelineOverviewPanel";
import PipelineStatusSummary from "../../components/pipelineDetails/PipelineStatusSummary";
import type { DetailTab, PipelineDetail } from "../../components/pipelineDetails/types";

interface ModelTrainingPipelineDetailPageProps {
  breadcrumbs?: { label: string; to?: string }[];
  onRefresh: () => void;
  pipeline: PipelineDetail;
  refreshing?: boolean;
}

const tabs: DetailTab[] = [
  { id: "overview", label: "Overview" },
  { id: "training-data", label: "Training Data" },
  { id: "model-details", label: "Model Details" },
  { id: "metrics", label: "Metrics" },
  { id: "logs", label: "Logs" },
  { id: "artifacts", label: "Artifacts" },
];

export default function ModelTrainingPipelineDetailPage({
  breadcrumbs,
  onRefresh,
  pipeline,
  refreshing = false,
}: ModelTrainingPipelineDetailPageProps) {
  const [activeTab, setActiveTab] = useState("overview");

  return (
    <PipelineDetailShell
      badge="Train Models"
      breadcrumbs={breadcrumbs || [
        { label: "Research Workflow", to: "/workflow" },
        { label: "Train Models", to: "/model-training" },
        { label: pipeline.name },
      ]}
      onRefresh={onRefresh}
      pipeline={pipeline}
      refreshing={refreshing}
    >
      <PipelineDetailTabs activeTab={activeTab} onChange={setActiveTab} tabs={tabs} />

      {activeTab === "overview" && (
        <div className="space-y-4">
          <PipelineOverviewPanel pipeline={pipeline} variant="model_training" />
          <ModelTrainingDetailsPanel pipeline={pipeline} />
        </div>
      )}
      {activeTab === "training-data" && <PipelineConfigPanel pipeline={pipeline} />}
      {activeTab === "model-details" && <ModelTrainingDetailsPanel pipeline={pipeline} />}
      {activeTab === "metrics" && <PipelineStatusSummary pipeline={pipeline} />}
      {activeTab === "logs" && <PipelineLogsPanel pipelineId={pipeline.id} />}
      {activeTab === "artifacts" && <ModelTrainingArtifactsPanel pipeline={pipeline} />}
    </PipelineDetailShell>
  );
}
