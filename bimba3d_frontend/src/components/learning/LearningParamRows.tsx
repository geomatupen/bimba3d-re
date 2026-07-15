interface LearningParamRow {
  key: string;
  actual: number | null;
  selected_multiplier: number | null;
  selected_multiplier_raw?: number | null;
  log_multiplier?: number | null;
  jitter?: number | null;
  final_multiplier: number | null;
}

export type LearningParamField = "actual" | "selected_multiplier" | "log_multiplier" | "final_multiplier";

interface LearningParamRowsProps {
  runId: string;
  rows: LearningParamRow[] | null | undefined;
  field: LearningParamField;
}

function formatParamNumber(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "-";
  const fixed = value.toFixed(6);
  const trimmed = fixed.replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
  return trimmed === "-0" ? "0" : trimmed;
}

export default function LearningParamRows({ runId, rows, field }: LearningParamRowsProps) {
  if (!Array.isArray(rows) || rows.length === 0) return <>-</>;

  return (
    <div className="space-y-0.5">
      {rows.map((row) => {
        const value =
          field === "selected_multiplier"
            ? row.selected_multiplier_raw ?? row.selected_multiplier
            : field === "log_multiplier"
            ? row.log_multiplier ?? row.jitter
            : field === "actual"
            ? row.actual
            : row.final_multiplier;

        return (
          <div key={`${runId}-${field}-${row.key}`} className="text-slate-700">
            {row.key}: {formatParamNumber(value)}
          </div>
        );
      })}
    </div>
  );
}