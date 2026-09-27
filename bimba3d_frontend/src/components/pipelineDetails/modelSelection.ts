export const CONTROLLED_EXPERIMENTS = ["time_constrained_test", "gaussian_constrained_test"] as const;

export type ControlledExperiment = (typeof CONTROLLED_EXPERIMENTS)[number];

const SEPARATOR = "::additional-experiment::";

export const controlledExperimentLabel = (kind: ControlledExperiment): string =>
  kind === "time_constrained_test" ? "Time constrained" : "Gaussian constrained";

export const makeModelSelectionId = (modelId: string, kind: ControlledExperiment): string =>
  `${modelId}${SEPARATOR}${kind}`;

export const parseModelSelectionId = (
  selectionId: string | null | undefined,
): { controlledExperiment: ControlledExperiment | null; modelId: string | null } => {
  const value = String(selectionId || "");
  const separatorIndex = value.lastIndexOf(SEPARATOR);
  if (separatorIndex < 0) {
    return { controlledExperiment: null, modelId: value || null };
  }
  const modelId = value.slice(0, separatorIndex);
  const kind = value.slice(separatorIndex + SEPARATOR.length);
  const controlledExperiment = CONTROLLED_EXPERIMENTS.find((candidate) => candidate === kind) || null;
  return controlledExperiment && modelId
    ? { controlledExperiment, modelId }
    : { controlledExperiment: null, modelId: value || null };
};
