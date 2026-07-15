import { Brain, CalendarClock, Layers, Sigma, Target, TrendingDown } from "lucide-react";
import type { PipelineDetail } from "./types";

interface ModelTrainingDetailsPanelProps {
  pipeline: PipelineDetail;
}

const modelTypeLabel = (value: unknown) => {
  const type = String(value || "").toLowerCase();
  if (type.includes("compact") && type.includes("ridge")) return "Compact Ridge";
  if (type.includes("compact") && type.includes("mlp")) return "Compact MLP";
  if (type.includes("ridge")) return "Ridge";
  if (type.includes("mlp")) return "MLP";
  return value ? String(value) : "Not specified";
};

const asNumber = (value: unknown): number | null => (typeof value === "number" && Number.isFinite(value) ? value : null);

const formatNumber = (value: unknown, digits = 4) => {
  const parsed = asNumber(value);
  return parsed === null ? "-" : parsed.toFixed(digits);
};

const getModelType = (model: any) => model?.config?.model_type || model?.model_type || model?.type || model?.artifact_format || model?.mode;

const getModelName = (model: any) => model?.model_name || model?.name || model?.model_id || "Unnamed model";

const getModelTime = (model: any) => String(model?.trained_at || model?.created_at || model?.generated_at || "");

const getBestMlpStep = (model: any) =>
  model?.config?.best_model_step ??
  model?.config?.best_step ??
  model?.config?.best_epoch ??
  model?.best_model_step ??
  model?.best_step ??
  model?.best_epoch ??
  null;

export default function ModelTrainingDetailsPanel({ pipeline }: ModelTrainingDetailsPanelProps) {
  const config = pipeline.config || {};
  const models = [
    ...(Array.isArray((pipeline as any).trained_models) ? (pipeline as any).trained_models : []),
    ...(Array.isArray(config.trained_models) ? config.trained_models : []),
    ...(Array.isArray(config.models) ? config.models : []),
  ];
  const trainingDataId = config.training_data_pipeline_id || config.source_pipeline_id || config.source_training_data_id;
  const ridgeModels = models.filter((model: any) => String(getModelType(model) || "").toLowerCase().includes("ridge"));
  const mlpModels = models.filter((model: any) => String(getModelType(model) || "").toLowerCase().includes("mlp"));
  const sortedModels = [...models].sort((a: any, b: any) => getModelTime(b).localeCompare(getModelTime(a)));
  const latestModel = sortedModels[0];
  const latestConfig = latestModel?.config || {};
  const latestLambda = latestConfig.lambda_selected ?? latestConfig.lambda_ridge ?? latestModel?.lambda_selected ?? latestModel?.lambda_ridge;
  const lambdaSearchCount = Array.isArray(latestConfig.lambda_search) ? latestConfig.lambda_search.length : 0;
  const bestValLoss = latestConfig.best_val_loss ?? latestModel?.best_val_loss;
  const finalTrainLoss = latestConfig.final_train_loss ?? latestModel?.final_train_loss;
  const bestMlpStep = getBestMlpStep(latestModel);

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-slate-950">Model Details</h2>
        <p className="mt-1 text-sm text-slate-600">Model-training configuration and outputs for featurewise and compact Ridge/MLP training.</p>
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
          <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg bg-emerald-50 text-emerald-700">
            <Brain className="h-4 w-4" />
          </div>
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Model Type</div>
          <div className="mt-1 text-sm font-bold text-slate-950">
            {latestModel ? modelTypeLabel(getModelType(latestModel)) : modelTypeLabel(config.model_type || config.learner_type)}
          </div>
        </div>
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
          <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg bg-blue-50 text-blue-700">
            <Layers className="h-4 w-4" />
          </div>
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Training Data</div>
          <div className="mt-1 truncate text-sm font-bold text-slate-950" title={trainingDataId || ""}>
            {trainingDataId || "-"}
          </div>
        </div>
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
          <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg bg-amber-50 text-amber-700">
            <Target className="h-4 w-4" />
          </div>
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Outputs</div>
          <div className="mt-1 text-sm font-bold text-slate-950">{models.length || config.model_count || 0} models</div>
        </div>
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-lg border border-slate-200 bg-white p-3">
          <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            <Sigma className="h-3.5 w-3.5" /> Ridge
          </div>
          <div className="text-lg font-bold text-slate-950">{ridgeModels.length}</div>
          <div className="mt-1 text-xs text-slate-600">
            Lambda {formatNumber(latestLambda, 4)} {lambdaSearchCount > 0 ? `(${lambdaSearchCount} searched)` : ""}
          </div>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-3">
          <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            <Brain className="h-3.5 w-3.5" /> MLP
          </div>
          <div className="text-lg font-bold text-slate-950">{mlpModels.length}</div>
          <div className="mt-1 text-xs text-slate-600">Best step/epoch {bestMlpStep ?? "-"}</div>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-3">
          <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            <TrendingDown className="h-3.5 w-3.5" /> MLP Loss
          </div>
          <div className="text-sm font-bold text-slate-950">Best {formatNumber(bestValLoss, 6)}</div>
          <div className="mt-1 text-xs text-slate-600">Final train {formatNumber(finalTrainLoss, 6)}</div>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-3">
          <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            <CalendarClock className="h-3.5 w-3.5" /> Latest
          </div>
          <div className="truncate text-sm font-bold text-slate-950" title={latestModel ? getModelName(latestModel) : ""}>
            {latestModel ? getModelName(latestModel) : "-"}
          </div>
          <div className="mt-1 text-xs text-slate-600">
            {latestModel?.runs_used ?? latestModel?.training_samples ?? latestConfig.training_samples ?? "-"} rows used
          </div>
        </div>
      </div>

      <div className="mt-4 overflow-x-auto rounded-lg border border-slate-200">
        <table className="min-w-[860px] w-full text-xs">
          <thead className="border-b border-slate-200 bg-slate-50">
            <tr>
              <th className="px-3 py-2 text-left font-semibold text-slate-700">Model</th>
              <th className="px-3 py-2 text-left font-semibold text-slate-700">Type</th>
              <th className="px-3 py-2 text-left font-semibold text-slate-700">Rows</th>
              <th className="px-3 py-2 text-left font-semibold text-slate-700">Score Mean</th>
              <th className="px-3 py-2 text-left font-semibold text-slate-700">Lambda</th>
              <th className="px-3 py-2 text-left font-semibold text-slate-700">Best MLP Step</th>
              <th className="px-3 py-2 text-left font-semibold text-slate-700">Best Val Loss</th>
              <th className="px-3 py-2 text-left font-semibold text-slate-700">Trained</th>
            </tr>
          </thead>
          <tbody>
            {sortedModels.length === 0 ? (
              <tr>
                <td colSpan={8} className="px-3 py-4 text-center text-slate-500">
                  No trained model records available yet.
                </td>
              </tr>
            ) : (
              sortedModels.map((model: any, index: number) => {
                const modelConfig = model?.config || {};
                const trainedAt = getModelTime(model);
                return (
                  <tr key={model.model_id || model.path || index} className="border-b border-slate-100 last:border-0">
                    <td className="max-w-[220px] px-3 py-2 font-medium text-slate-900">
                      <div className="truncate" title={getModelName(model)}>
                        {getModelName(model)}
                      </div>
                    </td>
                    <td className="px-3 py-2 text-slate-700">{modelTypeLabel(getModelType(model))}</td>
                    <td className="px-3 py-2 text-slate-700">{model.runs_used ?? model.training_samples ?? modelConfig.training_samples ?? "-"}</td>
                    <td className="px-3 py-2 text-slate-700">{formatNumber(model.score_mean, 4)}</td>
                    <td className="px-3 py-2 text-slate-700">{formatNumber(modelConfig.lambda_selected ?? modelConfig.lambda_ridge ?? model.lambda_ridge, 4)}</td>
                    <td className="px-3 py-2 text-slate-700">{getBestMlpStep(model) ?? "-"}</td>
                    <td className="px-3 py-2 text-slate-700">{formatNumber(modelConfig.best_val_loss ?? model.best_val_loss, 6)}</td>
                    <td className="px-3 py-2 text-slate-600">{trainedAt ? new Date(trainedAt).toLocaleString() : "-"}</td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

