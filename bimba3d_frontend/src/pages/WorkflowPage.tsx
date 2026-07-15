import { Brain, Database, FlaskConical, TableProperties } from "lucide-react";
import WorkflowActionPanel from "../components/workflow/WorkflowActionPanel";
import PipelineWorkspacePanel from "../components/workflow/PipelineWorkspacePanel";
import WorkflowShell from "../components/workflow/WorkflowShell";

export default function WorkflowPage() {
  return (
    <WorkflowShell
      eyebrow="Research Workflow"
      title="AI-Guided Gaussian Splatting"
      breadcrumbs={[{ label: "Research Workflow" }]}
    >
      <div className="grid gap-3 lg:grid-cols-4">
        <WorkflowActionPanel
          title="Prepare Offline Data"
          subtitle="Baseline and log-space exploration runs for building the learning dataset."
          icon={Database}
          tone="blue"
          primaryTo="/offline-data-preparation"
          compact
        />
        <WorkflowActionPanel
          title="Training Data"
          subtitle="Final prepared datasets selected by model training and project testing."
          icon={TableProperties}
          tone="purple"
          primaryTo="/training-data"
          compact
        />
        <WorkflowActionPanel
          title="Train Models"
          subtitle="Featurewise and compact Ridge/MLP training from selected Training Data."
          icon={Brain}
          tone="emerald"
          primaryTo="/model-training"
          compact
        />
        <WorkflowActionPanel
          title="Run Tests"
          subtitle="Baseline plus selected featurewise or compact models on test projects."
          icon={FlaskConical}
          tone="amber"
          primaryTo="/testing-pipeline"
          compact
        />
      </div>
      <div className="mt-4">
        <PipelineWorkspacePanel
          detailBasePath="/all-pipelines/pipelines"
          subtitle="Search, filter, sort, and manage data, training, and testing pipelines without leaving the workflow page."
          title="All Pipelines"
        />
      </div>
    </WorkflowShell>
  );
}
