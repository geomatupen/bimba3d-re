import { Clock } from "lucide-react";
import type { ComponentType, SVGProps } from "react";

type IconComponent = ComponentType<SVGProps<SVGSVGElement>>;

interface ProjectTabsNavProps<TTab extends string> {
  activeTab: TTab;
  isProcessing?: boolean;
  onTabChange: (tab: TTab) => void;
  tabs: Array<{
    enabled: boolean;
    icon: IconComponent;
    id: TTab;
    label: string;
  }>;
}

export default function ProjectTabsNav<TTab extends string>({
  activeTab,
  isProcessing = false,
  onTabChange,
  tabs,
}: ProjectTabsNavProps<TTab>) {
  return (
    <div className="bg-white border-b-2 border-slate-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        <nav className="flex space-x-1" aria-label="Tabs">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            const isDisabled = !tab.enabled;

            return (
              <button
                key={tab.id}
                onClick={() => !isDisabled && onTabChange(tab.id)}
                disabled={isDisabled}
                className={`
                  flex items-center gap-2 px-6 py-4 text-sm font-semibold border-b-3 transition-all duration-200
                  ${isActive
                    ? "border-blue-600 text-blue-700 bg-blue-50/50"
                    : isDisabled
                      ? "border-transparent text-slate-400 cursor-not-allowed"
                      : "border-transparent text-slate-600 hover:text-slate-900 hover:bg-slate-50 hover:border-slate-200"
                  }
                `}
              >
                <Icon className="w-5 h-5" />
                {tab.label}
                {tab.id === "process" && isProcessing && (
                  <Clock className="w-4 h-4 text-blue-600 animate-pulse" />
                )}
              </button>
            );
          })}
        </nav>
      </div>
    </div>
  );
}
