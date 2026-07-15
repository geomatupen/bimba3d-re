import { Navigate, Routes, Route } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import ProjectsPage from "./pages/ProjectsPage";
import CreateProject from "./pages/CreateProject";
import Comparison from "./pages/Comparison";
import ComparisonHubPage from "./pages/ComparisonHubPage";
import WorkflowPipelineBuilderPage from "./pages/WorkflowPipelineBuilderPage";
import WorkflowPage from "./pages/WorkflowPage";
import OfflineDataPreparationPage from "./pages/OfflineDataPreparationPage";
import TrainingDataPage from "./pages/TrainingDataPage";
import ModelTrainingPage from "./pages/ModelTrainingPage";
import TestingPipelinePage from "./pages/TestingPipelinePage";
import AllPipelinesPage from "./pages/AllPipelinesPage";
import WorkflowPipelineDetailPage from "./pages/WorkflowPipelineDetailPage";
import ModelArtifactDetailPage from "./pages/ModelArtifactDetailPage";
import ProjectDetailPageV2 from "./pages/projectDetails/ProjectDetailPage";
import { HMRStatusBanner } from "./HMRStatusBanner";

function App() {
  return (
    <>
      {import.meta.env.DEV && <HMRStatusBanner />}
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/projects" element={<ProjectsPage />} />
        <Route path="/create" element={<CreateProject />} />
        <Route path="/projects/:id" element={<ProjectDetailPageV2 />} />
        <Route path="/comparison" element={<ComparisonHubPage />} />
        <Route path="/comparison/:id" element={<Comparison />} />
        <Route path="/workflow" element={<WorkflowPage />} />
        <Route path="/offline-data-preparation" element={<OfflineDataPreparationPage />} />
        <Route path="/offline-data-preparation/pipelines/:id" element={<WorkflowPipelineDetailPage />} />
        <Route path="/training-data" element={<TrainingDataPage />} />
        <Route path="/model-training" element={<ModelTrainingPage />} />
        <Route path="/model-training/new" element={<Navigate to="/model-training" replace />} />
        <Route path="/model-training/models/:id" element={<ModelArtifactDetailPage />} />
        <Route path="/model-training/pipelines/:id" element={<WorkflowPipelineDetailPage />} />
        <Route path="/testing-pipeline" element={<TestingPipelinePage />} />
        <Route path="/testing-pipeline/pipelines/:id" element={<WorkflowPipelineDetailPage />} />
        <Route path="/all-pipelines" element={<AllPipelinesPage />} />
        <Route path="/all-pipelines/pipelines/:id" element={<WorkflowPipelineDetailPage />} />
        <Route path="/workflow/pipelines/:id" element={<WorkflowPipelineDetailPage />} />
        <Route path="/workflow/pipeline-builder" element={<WorkflowPipelineBuilderPage />} />
        <Route path="/pipelines/:id" element={<WorkflowPipelineDetailPage />} />
      </Routes>
    </>
  );
}

export default App;
