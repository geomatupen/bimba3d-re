import { Brain, CalendarClock, ExternalLink, Pencil, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

export interface WorkflowModel {
  model_id: string;
  model_name?: string | null;
  artifact_format?: string | null;
  engine?: string | null;
  created_at?: string | null;
  ai_profile?: {
    ai_input_mode?: string | null;
    ai_selector_strategy?: string | null;
  } | null;
  provenance_summary?: {
    contributor_count?: number;
    unique_project_count?: number;
  } | null;
}

interface ModelSummaryListProps {
  emptyMessage: string;
  loading: boolean;
  models: WorkflowModel[];
  onDeleteModel?: (model: WorkflowModel) => void;
  onRenameModel?: (model: WorkflowModel) => void;
  onModelClick?: (model: WorkflowModel) => void;
}

const formatDate = (value?: string | null) => {
  if (!value) return "-";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
};

export default function ModelSummaryList({ emptyMessage, loading, models, onDeleteModel, onModelClick, onRenameModel }: ModelSummaryListProps) {
  const navigate = useNavigate();
  const [openMenuModelId, setOpenMenuModelId] = useState<string | null>(null);

  useEffect(() => {
    if (!openMenuModelId) return;

    const closeOnOutsideClick = (event: MouseEvent) => {
      const target = event.target as Element | null;
      if (!target?.closest(`[data-model-menu-root="${openMenuModelId}"]`)) {
        setOpenMenuModelId(null);
      }
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpenMenuModelId(null);
      }
    };

    document.addEventListener("mousedown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("mousedown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [openMenuModelId]);

  if (loading) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-5 text-sm text-slate-500 shadow-sm">
        Loading models...
      </div>
    );
  }

  if (models.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-slate-300 bg-white p-5 text-sm text-slate-500 shadow-sm">
        {emptyMessage}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {models.map((model) => {
        const handleClick = () => {
          if (onModelClick) {
            onModelClick(model);
            return;
          }
          navigate(`/model-training/models/${encodeURIComponent(model.model_id)}`);
        };

        return (
        <div
          key={model.model_id}
          role="button"
          tabIndex={0}
          onClick={handleClick}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              handleClick();
            }
          }}
          className="group relative block w-full cursor-pointer rounded-lg border border-slate-200 bg-white p-4 pr-14 text-left shadow-sm transition hover:border-emerald-300 hover:shadow-md"
        >
          {(onRenameModel || onDeleteModel) && (
            <div className="absolute right-3 top-3 z-10" data-model-menu-root={model.model_id}>
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  setOpenMenuModelId((current) => current === model.model_id ? null : model.model_id);
                }}
                className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-slate-300 bg-white text-xl font-bold leading-none text-slate-800 shadow-sm hover:bg-slate-50 hover:text-slate-950"
                title="Model actions"
                aria-label={`Open actions for ${model.model_name || model.model_id}`}
              >
                <span aria-hidden="true">⋮</span>
              </button>
              {openMenuModelId === model.model_id && (
                <div
                  className="absolute right-0 top-9 z-20 w-36 overflow-hidden rounded-lg border border-slate-200 bg-white py-1 shadow-lg"
                  onClick={(event) => event.stopPropagation()}
                >
                  {onRenameModel && (
                    <button
                      type="button"
                      onClick={() => {
                        setOpenMenuModelId(null);
                        onRenameModel(model);
                      }}
                      className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs font-semibold text-slate-700 hover:bg-blue-50 hover:text-blue-700"
                    >
                      <Pencil className="h-3.5 w-3.5" />
                      Edit name
                    </button>
                  )}
                  {onDeleteModel && (
                    <button
                      type="button"
                      onClick={() => {
                        setOpenMenuModelId(null);
                        onDeleteModel(model);
                      }}
                      className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs font-semibold text-rose-700 hover:bg-rose-50"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                      Delete
                    </button>
                  )}
                </div>
              )}
            </div>
          )}
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-500 to-emerald-600 text-white shadow-sm">
              <Brain className="h-5 w-5" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <h3 className="truncate text-sm font-semibold text-slate-950">
                    {model.model_name || model.model_id}
                  </h3>
                  <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-500">
                    <span className="inline-flex items-center gap-1">
                      <CalendarClock className="h-3.5 w-3.5" />
                      {formatDate(model.created_at)}
                    </span>
                    <span>{model.artifact_format || "model"}</span>
                    {model.ai_profile?.ai_input_mode && <span>{model.ai_profile.ai_input_mode}</span>}
                    {model.provenance_summary?.unique_project_count !== undefined && (
                      <span>{model.provenance_summary.unique_project_count} projects</span>
                    )}
                  </div>
                </div>
                <span className="shrink-0 rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-xs font-semibold text-emerald-700">
                  Registered
                </span>
                {!(onRenameModel || onDeleteModel) && (
                  <ExternalLink className="h-4 w-4 shrink-0 text-slate-400 group-hover:text-emerald-600" />
                )}
              </div>
            </div>
          </div>
        </div>
        );
      })}
    </div>
  );
}
