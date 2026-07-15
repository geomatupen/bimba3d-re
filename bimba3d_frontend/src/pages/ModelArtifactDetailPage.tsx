import { ArrowLeft, Brain, CalendarClock } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../api/client";
import PipelineBreadcrumbs from "../components/pipelineDetails/PipelineBreadcrumbs";
import type { WorkflowModel } from "../components/workflow/ModelSummaryList";

const formatNumber = (value: any, digits = 6) => {
  if (value === null || value === undefined || value === "") return "-";
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return String(value);
  return parsed.toLocaleString(undefined, { maximumFractionDigits: digits });
};

const firstValue = (...values: any[]) => values.find((value) => value !== null && value !== undefined && value !== "");

const metricValue = (rawModel: any, key: string) => firstValue(rawModel?.metrics?.[key], rawModel?.config?.[key], rawModel?.[key]);

const boundsSummary = (bounds: any) => {
  if (!bounds || typeof bounds !== "object") return "-";
  return Object.entries(bounds)
    .map(([key, value]) => {
      const pair = Array.isArray(value) ? value : [];
      return `${key}: ${pair.length >= 2 ? `${formatNumber(pair[0], 4)}-${formatNumber(pair[1], 4)}` : "-"}`;
    })
    .join(" | ");
};

export default function ModelArtifactDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [model, setModel] = useState<WorkflowModel | null>(null);
  const [rawModel, setRawModel] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const loadModel = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.get(`/api/models/${encodeURIComponent(id || "")}`);
      const found = res.data || null;
      setModel(found ? {
        ...found,
        artifact_format: found.artifact_format || found.model_family || "model",
        created_at: found.created_at || found.trained_at || null,
        ai_profile: found.ai_profile || { ai_input_mode: found.model_family || null },
        provenance_summary: found.provenance_summary || { unique_project_count: found.training_samples },
      } : null);
      setRawModel(found);
    } catch (err) {
      console.error("Failed to load model artifact", err);
      setModel(null);
      setRawModel(null);
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    void loadModel();
  }, [loadModel]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-50">
      <header className="bg-gradient-to-r from-blue-600 via-blue-700 to-indigo-700 shadow-xl">
        <div className="mx-auto max-w-7xl px-6 py-6">
          <div className="flex items-start gap-4">
            <button
              onClick={() => navigate(-1)}
              className="inline-flex items-center gap-2 rounded-xl border border-white/20 bg-white/10 px-3 py-2 text-sm font-medium text-white backdrop-blur-sm transition hover:bg-white/20"
            >
              <ArrowLeft className="h-4 w-4" />
              Back
            </button>
            <div className="min-w-0">
              <div className="mb-2">
                <PipelineBreadcrumbs
                  items={[
                    { label: "Research Workflow", to: "/workflow" },
                    { label: "Train Models", to: "/model-training" },
                    { label: model?.model_name || id || "Model" },
                  ]}
                />
              </div>
              <div className="mb-1 inline-flex rounded-full border border-white/20 bg-white/10 px-2 py-0.5 text-xs font-semibold uppercase tracking-wide text-white">
                Model Artifact
              </div>
              <h1 className="truncate text-2xl font-bold text-white">{model?.model_name || id}</h1>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-4 px-6 py-6">
        {loading ? (
          <section className="rounded-lg border border-slate-200 bg-white p-5 text-sm text-slate-500 shadow-sm">Loading model...</section>
        ) : !model ? (
          <section className="rounded-lg border border-red-200 bg-white p-5 text-sm text-red-700 shadow-sm">Model artifact was not found.</section>
        ) : (
          <>
            <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
              <div className="mb-4 flex items-start gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-500 to-emerald-600 text-white shadow-sm">
                  <Brain className="h-5 w-5" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-slate-950">{model.model_name || model.model_id}</h2>
                  <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-500">
                    <span className="inline-flex items-center gap-1">
                      <CalendarClock className="h-3.5 w-3.5" />
                      {model.created_at ? new Date(model.created_at).toLocaleString() : "-"}
                    </span>
                    <span>{model.artifact_format || "model"}</span>
                    {model.ai_profile?.ai_input_mode && <span>{model.ai_profile.ai_input_mode}</span>}
                  </div>
                </div>
              </div>
              <div className="grid gap-3 md:grid-cols-3">
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Model ID</div>
                  <div className="mt-1 truncate text-sm font-semibold text-slate-950">{model.model_id}</div>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Engine</div>
                  <div className="mt-1 text-sm font-semibold text-slate-950">{model.engine || "-"}</div>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Training Samples</div>
                  <div className="mt-1 text-sm font-semibold text-slate-950">{rawModel?.training_samples ?? model.provenance_summary?.unique_project_count ?? "-"}</div>
                </div>
              </div>
            </section>

            <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
              <h2 className="text-lg font-semibold text-slate-950">Model Summary</h2>
              <div className="mt-4 grid gap-3 md:grid-cols-3">
                <SummaryItem label="Model Family" value={rawModel?.model_family || model.artifact_format || "-"} />
                <SummaryItem label="Source Training Data" value={rawModel?.source_training_data_id || "-"} />
                <SummaryItem label="Evaluation Step" value={formatNumber(rawModel?.model_evaluation_step ?? rawModel?.config?.model_evaluation_step, 0)} />
                <SummaryItem label="Candidate Points" value={formatNumber(metricValue(rawModel, "candidate_points"), 0)} />
                <SummaryItem label="Bounds Source" value={rawModel?.config?.log_multiplier_bounds_source || "-"} />
                <SummaryItem label="Multiplier Bounds" value={boundsSummary(rawModel?.config?.log_multiplier_bounds)} wide />
              </div>

              {(rawModel?.model_family === "featurewise_ridge_regression" ||
                rawModel?.model_family === "compact_featurewise_ridge_regression") && (
                <div className="mt-4 grid gap-3 md:grid-cols-3">
                  <SummaryItem label="Selected Lambda" value={formatNumber(metricValue(rawModel, "lambda_selected") ?? metricValue(rawModel, "selected_lambda"), 6)} />
                  <SummaryItem label="Lambda Candidates" value={formatNumber(rawModel?.config?.lambda_search_count ?? rawModel?.metrics?.lambda_search?.length, 0)} />
                  <SummaryItem label="Validation MSE" value={formatNumber(rawModel?.metrics?.train_fit_metrics?.avg_val_mse, 8)} />
                </div>
              )}

              {(rawModel?.model_family === "featurewise_mlp" ||
                rawModel?.model_family === "compact_featurewise_mlp") && (
                <div className="mt-4 grid gap-3 md:grid-cols-4">
                  <SummaryItem
                    label="Architecture"
                    value={
                      rawModel?.model_family === "compact_featurewise_mlp"
                        ? `compact shared model, hidden ${formatNumber(metricValue(rawModel, "hidden"), 0)}`
                        : `8 -> 4 heads, hidden ${formatNumber(metricValue(rawModel, "hidden"), 0)}`
                    }
                  />
                  <SummaryItem label="Dropout" value={formatNumber(metricValue(rawModel, "dropout"), 4)} />
                  <SummaryItem label="Learning Rate" value={formatNumber(metricValue(rawModel, "learning_rate"), 6)} />
                  <SummaryItem label="Weight Decay" value={formatNumber(metricValue(rawModel, "weight_decay"), 6)} />
                  <SummaryItem label="Max Epochs" value={formatNumber(metricValue(rawModel, "max_epochs"), 0)} />
                  <SummaryItem label="Epochs Trained" value={formatNumber(metricValue(rawModel, "epochs_trained"), 0)} />
                  <SummaryItem label="Best Epoch" value={formatNumber(metricValue(rawModel, "best_epoch"), 0)} />
                  <SummaryItem label="Patience" value={formatNumber(metricValue(rawModel, "early_stopping_patience"), 0)} />
                  <SummaryItem label="Best Val Loss" value={formatNumber(metricValue(rawModel, "best_val_loss"), 8)} />
                  <SummaryItem label="Final Train Loss" value={formatNumber(metricValue(rawModel, "final_train_loss"), 8)} />
                  <SummaryItem label="Final Val Loss" value={formatNumber(metricValue(rawModel, "final_val_loss"), 8)} />
                  <SummaryItem label="Parameters" value={formatNumber(metricValue(rawModel, "total_parameters"), 0)} />
                </div>
              )}
            </section>

            <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
              <h2 className="text-lg font-semibold text-slate-950">Raw Metadata</h2>
              <pre className="mt-4 max-h-[620px] overflow-auto rounded-lg bg-slate-950 p-4 text-xs text-slate-100">
                {JSON.stringify(rawModel, null, 2)}
              </pre>
            </section>
          </>
        )}
      </main>
    </div>
  );
}

function SummaryItem({ label, value, wide = false }: { label: string; value: string | number | null | undefined; wide?: boolean }) {
  return (
    <div className={`rounded-lg border border-slate-200 bg-slate-50 p-3 ${wide ? "md:col-span-2" : ""}`}>
      <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 break-words text-sm font-semibold text-slate-950">{value ?? "-"}</div>
    </div>
  );
}
