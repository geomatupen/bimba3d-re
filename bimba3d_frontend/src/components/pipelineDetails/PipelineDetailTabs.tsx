import type { DetailTab } from "./types";

interface PipelineDetailTabsProps {
  activeTab: string;
  onChange: (tab: string) => void;
  tabs: DetailTab[];
}

export default function PipelineDetailTabs({ activeTab, onChange, tabs }: PipelineDetailTabsProps) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-1 shadow-sm">
      <div className="flex flex-wrap gap-1">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onChange(tab.id)}
            className={`rounded-md px-3 py-2 text-sm font-semibold transition ${
              activeTab === tab.id
                ? "bg-blue-600 text-white shadow-sm"
                : "text-slate-600 hover:bg-slate-50 hover:text-slate-950"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
    </div>
  );
}
