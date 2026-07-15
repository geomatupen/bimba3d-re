export default function PipelineEditWarning() {
  return (
    <div className="bg-yellow-50 border-b-2 border-yellow-200">
      <div className="max-w-7xl mx-auto px-6 lg:px-8 py-4">
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0">
            <svg className="w-6 h-6 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <div className="flex-1">
            <h3 className="text-sm font-semibold text-yellow-800 mb-1">
              Configuration Changes Require Pipeline Restart
            </h3>
            <p className="text-xs text-yellow-700">
              Any changes you make will only take effect after restarting the pipeline. Restarting will delete all training runs except the baseline, keeping only images, COLMAP data, and baseline splats.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
