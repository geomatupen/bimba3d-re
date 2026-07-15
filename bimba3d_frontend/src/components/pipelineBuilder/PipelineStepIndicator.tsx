interface PipelineStepIndicatorProps {
  currentStep: number;
  pipelineType: "offline_data" | "test";
}

const getStepLabel = (step: number, pipelineType: "offline_data" | "test") => {
  switch (step) {
    case 1:
      return "Setup";
    case 2:
      return "Run Config";
    case 3:
      return pipelineType === "test" ? "Testing" : "Preparation";
    case 4:
      return "Execution";
    case 5:
      return "Review";
    default:
      return "";
  }
};

export default function PipelineStepIndicator({
  currentStep,
  pipelineType,
}: PipelineStepIndicatorProps) {
  return (
    <div className="bg-white border-b border-gray-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        <div className="flex justify-between py-4">
          {[1, 2, 3, 4, 5].map((step) => (
            <div key={step} className="flex flex-col items-center flex-1">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center font-bold mb-2 transition-all ${
                  currentStep >= step
                    ? "bg-blue-600 text-white shadow-lg"
                    : "bg-gray-200 text-gray-400"
                }`}
              >
                {step}
              </div>
              <div className={`text-xs font-medium ${currentStep >= step ? "text-gray-900" : "text-gray-400"}`}>
                {getStepLabel(step, pipelineType)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
