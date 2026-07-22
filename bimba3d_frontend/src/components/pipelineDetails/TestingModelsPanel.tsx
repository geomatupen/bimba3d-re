import { useEffect, useMemo, useState } from "react";
import { BrainCircuit } from "lucide-react";
import { api } from "../../api/client";
import type { PipelineDetail } from "./types";

interface TestingModelsPanelProps {
  onSelectModel?: (modelId: string | null) => void;
  pipeline: PipelineDetail;
  selectedModelId?: string | null;
}

const doneStatuses = new Set(["completed", "success", "done", "ok"]);

const runModelKey = (run: any) =>
  String(
    run.model_id ||
      run.source_model_id ||
      run.test_model_id ||
      run.current_test_model_id ||
      run.selected_model_id ||
      "",
  );

const phaseRunCount = (phase: any): number => {
  const raw = phase?.exploration_runs_per_project ?? phase?.runs_per_project ?? phase?.runs ?? 1;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? Math.max(0, Math.floor(parsed)) : 1;
};

const expectedRunsPerModel = (pipeline: PipelineDetail): number => {
  const configuredProjects = Array.isArray(pipeline.config?.projects) ? pipeline.config.projects.length : 0;
  const phases = Array.isArray(pipeline.config?.phases) ? pipeline.config.phases : [];
  const nonBaselineRunsPerProject = phases
    .filter((phase: any) => Number(phase?.phase_number || 1) > 1)
    .reduce((total: number, phase: any) => total + phaseRunCount(phase), 0);
  return configuredProjects * Math.max(1, nonBaselineRunsPerProject || 1);
};

const addModelLabel = (labels: Map<string, string>, rawId: unknown, rawName: unknown) => {
  const id = String(rawId || "").trim();
  const name = String(rawName || "").trim();
  if (id && name) labels.set(id, name);
};

const collectConfiguredModelLabels = (pipeline: PipelineDetail): Map<string, string> => {
  const labels = new Map<string, string>();
  const config = pipeline.config || {};
  const possibleLists = [
    config.source_models,
    config.models,
    config.model_records,
    config.workflow_models,
    config.selected_models,
  ];
  possibleLists.forEach((list: unknown) => {
    if (!Array.isArray(list)) return;
    list.forEach((item: any) => {
      if (!item || typeof item !== "object") return;
      addModelLabel(labels, item.model_id || item.id || item.source_model_id, item.model_name || item.name || item.label);
    });
  });

  const possibleMaps = [config.source_model_names, config.model_names, config.model_name_by_id];
  possibleMaps.forEach((map: unknown) => {
    if (!map || typeof map !== "object" || Array.isArray(map)) return;
    Object.entries(map as Record<string, unknown>).forEach(([id, name]) => addModelLabel(labels, id, name));
  });

  (Array.isArray(pipeline.runs) ? pipeline.runs : []).forEach((run: any) => {
    addModelLabel(labels, runModelKey(run), run.source_model_name || run.model_name || run.test_model_name);
  });

  return labels;
};

const getModelProgress = (pipeline: PipelineDetail, modelId: string) => {
  const runs = Array.isArray(pipeline.runs) ? pipeline.runs : [];
  // Only count phase > 1 runs (exclude baseline) for model progress
  const modelRuns = runs.filter(
    (run: any) => runModelKey(run) === modelId && Number(run.phase || run.phase_number || 0) > 1,
  );
  const done = modelRuns.filter((run: any) => doneStatuses.has(String(run.status || "").toLowerCase())).length;
  const total = Math.max(expectedRunsPerModel(pipeline), modelRuns.length);
  return { done, total };
};

export default function TestingModelsPanel({ onSelectModel, pipeline, selectedModelId }: TestingModelsPanelProps) {
  const [registryLabels, setRegistryLabels] = useState<Map<string, string>>(new Map());
  const configuredModels = Array.isArray(pipeline.config?.source_model_ids)
    ? pipeline.config.source_model_ids.filter(Boolean)
    : pipeline.config?.source_model_id
      ? [pipeline.config.source_model_id]
      : [];
  const configLabels = useMemo(() => collectConfiguredModelLabels(pipeline), [pipeline]);
  const activeModel = pipeline.active_run?.test_model_id || pipeline.current_test_model_id || configuredModels[0] || null;
  const selectedModel = selectedModelId || null;
  const allSelected = !selectedModelId;

  useEffect(() => {
    let cancelled = false;
    api.get("/api/models")
      .then((res) => {
        if (cancelled) return;
        const labels = new Map<string, string>();
        const items = Array.isArray(res.data?.items) ? res.data.items : [];
        items.forEach((item: any) => addModelLabel(labels, item.model_id, item.model_name || item.name));
        setRegistryLabels(labels);
      })
      .catch(() => {
        if (!cancelled) setRegistryLabels(new Map());
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-slate-950">Models Used</h2>
        <p className="mt-1 text-sm text-slate-600">Models selected for this testing pipeline.</p>
      </div>
      {configuredModels.length === 0 ? (
        <div className="rounded-lg border border-dashed border-slate-300 p-5">
          <div className="mb-2 flex h-10 w-10 items-center justify-center rounded-lg bg-amber-50 text-amber-700">
            <BrainCircuit className="h-5 w-5" />
          </div>
          <h3 className="text-sm font-semibold text-slate-900">No models linked</h3>
          <p className="mt-1 text-sm text-slate-600">Testing pipelines should use models created by the Train Models workflow.</p>
        </div>
      ) : (
        <div className="grid gap-3 md:grid-cols-2">
          {onSelectModel && (
            <button
              type="button"
              onClick={() => onSelectModel(null)}
              aria-pressed={allSelected}
              className={`relative overflow-hidden rounded-lg border p-4 text-left transition ${
                allSelected
                  ? "border-amber-300 bg-amber-50 shadow-sm"
                  : "border-slate-200 bg-slate-50 hover:border-blue-200 hover:bg-blue-50"
              }`}
            >
              <div className="text-sm font-semibold text-slate-950">All configured models</div>
              <div className="mt-1 text-xs text-slate-600">
                Show overview charts and rows for all {configuredModels.length} selected model{configuredModels.length === 1 ? "" : "s"}.
              </div>
            </button>
          )}
          {configuredModels.map((modelId: string) => {
            const progress = getModelProgress(pipeline, modelId);
            const progressLabel = progress.total > 0 ? `${progress.done}/${progress.total} done` : "No runs yet";
            const selected = selectedModel === modelId;
            const modelName = registryLabels.get(modelId) || configLabels.get(modelId) || modelId;
            const showIdLine = modelName !== modelId;
            return (
            <button
              key={modelId}
              type="button"
              onClick={() => onSelectModel?.(modelId)}
              aria-pressed={selected}
              className={`relative overflow-hidden rounded-lg border p-4 pr-28 text-left transition ${
                selected
                  ? "border-amber-300 bg-amber-50 shadow-sm"
                  : "border-slate-200 bg-slate-50 hover:border-blue-200 hover:bg-blue-50"
              } ${onSelectModel ? "cursor-pointer" : "cursor-default"}`}
            >
              <div className="flex min-w-0 items-start gap-3">
                <span
                  aria-hidden="true"
                  className={`mt-1 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border ${
                    selected ? "border-amber-600 bg-amber-600" : "border-slate-400 bg-white"
                  }`}
                >
                  {selected && <span className="h-2 w-2 rounded-full bg-white" />}
                </span>
                <div className="min-w-0 flex-1">
                  <div
                    className="truncate text-sm font-semibold leading-snug text-slate-950"
                    title={modelName}
                  >
                    {modelName}
                  </div>
                  {showIdLine && (
                    <div className="mt-1 truncate font-mono text-[11px] text-slate-500" title={modelId}>
                      ID: {modelId}
                    </div>
                  )}
                  <div className="mt-2 inline-flex rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-semibold text-slate-600 ring-1 ring-slate-200">
                    {progressLabel}
                  </div>
                </div>
              </div>
              <div className="absolute right-4 top-5">
                {activeModel === modelId && (
                  <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-semibold text-amber-700">Active</span>
                )}
                {selected && activeModel !== modelId && (
                  <span className="rounded-full bg-blue-100 px-2 py-0.5 text-xs font-semibold text-blue-700">Selected</span>
                )}
              </div>
            </button>
            );
          })}
        </div>
      )}
    </section>
  );
}
