import PipelineWorkspacePanel from "../components/workflow/PipelineWorkspacePanel";
import WorkflowShell from "../components/workflow/WorkflowShell";

export default function AllPipelinesPage() {
  return (
    <WorkflowShell
      eyebrow="Pipeline Workspace"
      title="All Pipelines"
      backTo="/workflow"
      breadcrumbs={[
        { label: "Research Workflow", to: "/workflow" },
        { label: "All Pipelines" },
      ]}
    >
      <PipelineWorkspacePanel />
    </WorkflowShell>
  );
}
