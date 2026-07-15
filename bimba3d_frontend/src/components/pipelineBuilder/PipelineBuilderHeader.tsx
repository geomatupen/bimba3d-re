import { ArrowLeft } from "lucide-react";

interface PipelineBuilderHeaderProps {
  isEditMode: boolean;
  pipelineType: "offline_data" | "test";
  onBack: () => void;
}

export default function PipelineBuilderHeader({
  isEditMode,
  pipelineType,
  onBack,
}: PipelineBuilderHeaderProps) {
  const badge = isEditMode
    ? "Edit Pipeline"
    : pipelineType === "test"
      ? "New Testing Pipeline"
      : "New Offline Preparation Pipeline";
  const title = isEditMode
    ? "Edit Pipeline Configuration"
    : pipelineType === "test"
      ? "Create Testing Pipeline"
      : "Create Offline Preparation Pipeline";
  const subtitle = isEditMode
    ? "Update pipeline settings (requires restart to take effect)"
    : pipelineType === "test"
      ? "Configure default and model-predicted evaluation runs"
      : "Prepare baseline and exploration runs for offline model training";

  return (
    <header className="bg-gradient-to-r from-blue-600 via-blue-700 to-indigo-700 shadow-xl">
      <div className="max-w-7xl mx-auto px-6 lg:px-8 py-7">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={onBack}
              className="inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-white/10 hover:bg-white/20 backdrop-blur-sm border border-white/20 text-white text-sm font-medium transition-all duration-200 hover:scale-105"
            >
              <ArrowLeft className="w-4 h-4 text-white" />
              Back
            </button>
            <div>
              <div className="inline-flex items-center gap-2 px-2 py-0.5 rounded-full bg-white/10 backdrop-blur-sm border border-white/20 mb-1">
                <span className="text-xs font-medium text-white uppercase tracking-wider">{badge}</span>
              </div>
              <h1 className="text-2xl font-bold text-white mb-1">{title}</h1>
              <p className="text-xs text-blue-100">{subtitle}</p>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
