import WorkflowShell from "../components/workflow/WorkflowShell";
import ComparisonTab from "../components/tabs/ComparisonTab";

export default function ComparisonHubPage() {
  return (
    <WorkflowShell
      eyebrow="Analysis"
      title="Comparison"
      backTo="/"
      breadcrumbs={[
        { label: "Home", to: "/" },
        { label: "Comparison" },
      ]}
    >
      <ComparisonTab />
    </WorkflowShell>
  );
}
