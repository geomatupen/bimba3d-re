import { FileArchive } from "lucide-react";
import type { PipelineDetail } from "./types";

interface ModelTrainingArtifactsPanelProps {
  pipeline: PipelineDetail;
}

export default function ModelTrainingArtifactsPanel({ pipeline }: ModelTrainingArtifactsPanelProps) {
  const artifacts = pipeline.config?.artifacts || pipeline.config?.model_artifacts || pipeline.config?.outputs || [];
  const artifactList = Array.isArray(artifacts) ? artifacts : Object.entries(artifacts || {}).map(([name, value]) => ({ name, value }));

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-slate-950">Artifacts</h2>
        <p className="mt-1 text-sm text-slate-600">Saved model files, metrics, and lineage metadata from model training.</p>
      </div>
      {artifactList.length === 0 ? (
        <div className="rounded-lg border border-dashed border-slate-300 p-5">
          <div className="mb-2 flex h-10 w-10 items-center justify-center rounded-lg bg-emerald-50 text-emerald-700">
            <FileArchive className="h-5 w-5" />
          </div>
          <h3 className="text-sm font-semibold text-slate-900">No artifacts recorded yet</h3>
          <p className="mt-1 text-sm text-slate-600">Model artifact listing will be populated by the new model_training backend flow.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {artifactList.map((artifact: any, index: number) => (
            <div key={artifact.name || artifact.path || index} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="text-sm font-semibold text-slate-900">{artifact.name || artifact.path || `Artifact ${index + 1}`}</div>
              <pre className="mt-2 max-h-[220px] overflow-auto rounded bg-slate-950 p-3 text-xs text-slate-100">
                {JSON.stringify(artifact, null, 2)}
              </pre>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
