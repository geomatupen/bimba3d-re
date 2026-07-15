import { Check } from "lucide-react";
import { InfoIcon } from "./InfoIcon";

interface ProcessToastsProps {
  configSavedToast: string | null;
  processInfoToast: string | null;
}

export function ProcessToasts({ configSavedToast, processInfoToast }: ProcessToastsProps) {
  return (
    <>
      {configSavedToast && (
        <div className="fixed bottom-4 right-4 z-[1100] pointer-events-none">
          <div className="inline-flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-semibold text-emerald-800 shadow-lg">
            <Check className="w-4 h-4 text-emerald-600" />
            <span>{configSavedToast}</span>
          </div>
        </div>
      )}
      {processInfoToast && (
        <div className={`fixed right-4 z-[1090] pointer-events-none ${configSavedToast ? "bottom-16" : "bottom-4"}`}>
          <div className="inline-flex max-w-[32rem] items-start gap-2 rounded-lg border border-sky-200 bg-sky-50 px-3 py-2 text-xs font-semibold text-sky-900 shadow-lg">
            <InfoIcon className="mt-0.5 w-4 h-4 text-sky-700" />
            <span>{processInfoToast}</span>
          </div>
        </div>
      )}
    </>
  );
}
