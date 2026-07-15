import { ExternalLink, GitBranch } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../../api/client";

interface ProjectModelsReferencePanelProps {
  projectId: string;
}

const formatDate = (value?: string | null) => {
  if (!value) return "N/A";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
};

export default function ProjectModelsReferencePanel({ projectId }: ProjectModelsReferencePanelProps) {
  const [models, setModels] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const loadModels = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.get("/api/models");
      setModels(Array.isArray(res.data?.items) ? res.data.items : []);
    } catch (err) {
      console.error("Failed to load available models", err);
      setModels([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadModels();
  }, [loadModels]);

  const visibleModels = useMemo(() => {
    const projectModels = models.filter((model) => model.source_project_id === projectId || model.project_id === projectId);
    return projectModels.length > 0 ? projectModels : models;
  }, [models, projectId]);

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-600 to-emerald-700 text-white">
            <GitBranch className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-slate-950">Available Models</h2>
            <p className="mt-1 text-sm text-slate-600">
              Read-only model references. Model changes belong in the Train Models workflow.
            </p>
          </div>
        </div>
        <Link
          to="/model-training"
          className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 px-3 py-2 text-sm font-semibold text-white hover:bg-emerald-700"
        >
          <ExternalLink className="h-4 w-4" />
          Open Train Models
        </Link>
      </div>

      {loading ? (
        <div className="rounded-lg border border-slate-200 p-4 text-sm text-slate-500">Loading models...</div>
      ) : visibleModels.length === 0 ? (
        <div className="rounded-lg border border-dashed border-slate-300 p-5 text-sm text-slate-500">
          No trained models are available yet.
        </div>
      ) : (
        <div className="grid gap-3 md:grid-cols-2">
          {visibleModels.map((model) => {
            const id = String(model.model_id || model.id || model.name || "model");
            const name = String(model.model_name || model.name || id);
            const source = model.source_training_data_id || model.source_pipeline_id || model.pipeline_id || model.source_project_id || model.project_id || "N/A";
            return (
              <div key={id} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                <div className="min-w-0">
                  <div className="line-clamp-2 break-words text-sm font-semibold leading-snug text-slate-950" title={name}>
                    {name}
                  </div>
                  <div className="mt-1 font-mono text-[10px] text-slate-500" title={id}>{id}</div>
                </div>
                <div className="mt-3 grid gap-2 text-xs text-slate-600">
                  <div className="flex justify-between gap-2">
                    <span className="text-slate-500">Type</span>
                    <span className="font-semibold text-slate-800">{model.model_family || model.type || model.kind || "model"}</span>
                  </div>
                  <div className="flex justify-between gap-2">
                    <span className="text-slate-500">Source</span>
                    <span className="max-w-[220px] truncate font-mono text-[10px] text-slate-800" title={String(source)}>{source}</span>
                  </div>
                  <div className="flex justify-between gap-2">
                    <span className="text-slate-500">Created</span>
                    <span className="text-right text-slate-800">{formatDate(model.created_at || model.modified_at)}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
