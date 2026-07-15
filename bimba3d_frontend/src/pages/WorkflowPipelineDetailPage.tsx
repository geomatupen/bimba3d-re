import { AlertCircle, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { api } from "../api/client";
import { getPipelineStage } from "../components/workflow/PipelineSummaryList";
import type { PipelineDetail } from "../components/pipelineDetails/types";
import ModelTrainingPipelineDetailPage from "./pipelineDetails/ModelTrainingPipelineDetailPage";
import TestingPipelineDetailPage from "./pipelineDetails/TestingPipelineDetailPage";
import TrainingDataPipelineDetailPage from "./pipelineDetails/TrainingDataPipelineDetailPage";

export default function WorkflowPipelineDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const [pipeline, setPipeline] = useState<PipelineDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadPipeline = useCallback(async (isRefresh = false) => {
    if (!id) return;
    if (isRefresh) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }
    setError(null);
    try {
      const res = await api.get(`/api/workflow/pipelines/${id}`);
      setPipeline(res.data);
    } catch (err: any) {
      console.error("Failed to load workflow pipeline", err);
      setError(err.response?.data?.detail || "Failed to load pipeline details.");
      setPipeline(null);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [id]);

  useEffect(() => {
    void loadPipeline(false);
  }, [loadPipeline]);

  useEffect(() => {
    if (!pipeline || !["running", "paused", "stopping"].includes(String(pipeline.status || "").toLowerCase())) {
      return;
    }

    const timer = window.setInterval(() => {
      void loadPipeline(true);
    }, 3000);

    return () => window.clearInterval(timer);
  }, [loadPipeline, pipeline?.status]);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-50">
        <div className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600 shadow-sm">
          <RefreshCw className="h-4 w-4 animate-spin text-blue-600" />
          Loading pipeline details...
        </div>
      </div>
    );
  }

  if (error || !pipeline) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-50 px-6">
        <div className="max-w-md rounded-lg border border-red-200 bg-white p-6 text-center shadow-sm">
          <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-red-50 text-red-700">
            <AlertCircle className="h-5 w-5" />
          </div>
          <h1 className="text-lg font-semibold text-slate-950">Pipeline Not Available</h1>
          <p className="mt-2 text-sm text-slate-600">{error || "Pipeline details could not be loaded."}</p>
          <button
            onClick={() => navigate("/all-pipelines")}
            className="mt-4 rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700"
          >
            Back to All Pipelines
          </button>
        </div>
      </div>
    );
  }

  const refresh = () => void loadPipeline(true);
  const stage = getPipelineStage(pipeline);
  const detailBreadcrumbs = (() => {
    if (location.pathname.startsWith("/offline-data-preparation/")) {
      return [
        { label: "Research Workflow", to: "/workflow" },
        { label: "Prepare Offline Data", to: "/offline-data-preparation" },
        { label: pipeline.name },
      ];
    }
    if (location.pathname.startsWith("/model-training/")) {
      return [
        { label: "Research Workflow", to: "/workflow" },
        { label: "Train Models", to: "/model-training" },
        { label: pipeline.name },
      ];
    }
    if (location.pathname.startsWith("/testing-pipeline/")) {
      return [
        { label: "Research Workflow", to: "/workflow" },
        { label: "Run Tests", to: "/testing-pipeline" },
        { label: pipeline.name },
      ];
    }
    if (location.pathname.startsWith("/all-pipelines/")) {
      return [
        { label: "Research Workflow", to: "/workflow" },
        { label: "All Pipelines", to: "/all-pipelines" },
        { label: pipeline.name },
      ];
    }
    return [
      { label: "Research Workflow", to: "/workflow" },
      { label: pipeline.name },
    ];
  })();

  if (stage === "test") {
    return <TestingPipelineDetailPage breadcrumbs={detailBreadcrumbs} onRefresh={refresh} pipeline={pipeline} refreshing={refreshing} />;
  }

  if (stage === "model_training") {
    return <ModelTrainingPipelineDetailPage breadcrumbs={detailBreadcrumbs} onRefresh={refresh} pipeline={pipeline} refreshing={refreshing} />;
  }

  return <TrainingDataPipelineDetailPage breadcrumbs={detailBreadcrumbs} onRefresh={refresh} pipeline={pipeline} refreshing={refreshing} />;
}
