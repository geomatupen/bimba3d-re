import TrainingDataDatasetPanel from "../components/pipelineDetails/TrainingDataDatasetPanel";
import WorkflowShell from "../components/workflow/WorkflowShell";

export default function TrainingDataPage() {
  return (
    <WorkflowShell
      eyebrow="Stage 2"
      title="Training Data"
      backTo="/workflow"
      breadcrumbs={[
        { label: "Research Workflow", to: "/workflow" },
        { label: "Training Data" },
      ]}
    >
      <TrainingDataDatasetPanel
        allowPipelineSelection
        description="Final prepared training datasets. Select the offline data pipeline target before building or inspecting rows."
        title="Training Data"
      />
    </WorkflowShell>
  );
}
