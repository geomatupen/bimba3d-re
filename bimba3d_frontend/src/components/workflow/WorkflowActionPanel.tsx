import { ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";
import type { ComponentType, SVGProps, ReactNode } from "react";

interface WorkflowActionPanelProps {
  title: string;
  subtitle: string;
  icon: ComponentType<SVGProps<SVGSVGElement>>;
  tone: "blue" | "emerald" | "amber" | "purple";
  primaryTo?: string;
  actionIcon?: ComponentType<SVGProps<SVGSVGElement>>;
  actionLabel?: string;
  actionOnClick?: () => void;
  actionTo?: string;
  secondaryActionIcon?: ComponentType<SVGProps<SVGSVGElement>>;
  secondaryActionLabel?: string;
  secondaryActionOnClick?: () => void;
  children?: ReactNode;
  compact?: boolean;
}

const toneClasses: Record<
  WorkflowActionPanelProps["tone"],
  { gradient: string }
> = {
  blue: {
    gradient: "from-blue-600 to-blue-700",
  },
  emerald: {
    gradient: "from-emerald-600 to-emerald-700",
  },
  amber: {
    gradient: "from-amber-500 to-amber-600",
  },
  purple: {
    gradient: "from-purple-600 to-purple-700",
  },
};

export default function WorkflowActionPanel({
  title,
  subtitle,
  icon: Icon,
  tone,
  primaryTo,
  actionIcon: ActionIcon,
  actionLabel,
  actionOnClick,
  actionTo,
  secondaryActionIcon: SecondaryActionIcon,
  secondaryActionLabel,
  secondaryActionOnClick,
  children,
  compact = false,
}: WorkflowActionPanelProps) {
  const { gradient } = toneClasses[tone];

  const compactContent = (
    <>
      <div className="flex items-center justify-between gap-3">
        <div className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br ${gradient} text-white shadow-sm`}>
          <Icon className="h-[18px] w-[18px]" />
        </div>
        {primaryTo && !actionTo && (
          <div className="inline-flex shrink-0 items-center gap-1 rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-semibold text-slate-600 transition group-hover:border-blue-200 group-hover:bg-blue-50 group-hover:text-blue-700">
            Open
            <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
          </div>
        )}
      </div>
      <div className="mt-4">
        <h2 className="text-[17px] font-semibold leading-snug text-slate-950">{title}</h2>
        <p className="mt-2 text-[13px] leading-5 text-slate-600">{subtitle}</p>
      </div>
      {children && <div className="mt-5">{children}</div>}
    </>
  );

  const content = compact ? compactContent : (
    <>
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-4">
          <div className={`flex h-11 w-11 items-center justify-center rounded-lg bg-gradient-to-br ${gradient} text-white shadow-sm`}>
            <Icon className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-slate-950">{title}</h2>
            <p className="mt-1 text-sm text-slate-600">{subtitle}</p>
          </div>
        </div>
        {primaryTo && !actionTo && (
          <div className="inline-flex shrink-0 items-center gap-1 rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-semibold text-slate-600 transition group-hover:border-blue-200 group-hover:bg-blue-50 group-hover:text-blue-700">
            Open
            <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
          </div>
        )}
        {(actionOnClick && actionLabel) || (secondaryActionOnClick && secondaryActionLabel) ? (
          <div className="flex shrink-0 flex-wrap justify-end gap-2">
            {secondaryActionOnClick && secondaryActionLabel && (
              <button
                type="button"
                onClick={secondaryActionOnClick}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
              >
                {SecondaryActionIcon && <SecondaryActionIcon className="h-4 w-4" />}
                {secondaryActionLabel}
              </button>
            )}
            {actionOnClick && actionLabel && (
              <button
                type="button"
                onClick={actionOnClick}
                className="inline-flex items-center gap-1.5 rounded-lg border border-blue-100 bg-blue-50 px-3 py-2 text-sm font-semibold text-blue-700 hover:bg-white"
              >
                {ActionIcon && <ActionIcon className="h-4 w-4" />}
                {actionLabel}
              </button>
            )}
          </div>
        ) : null}
        {actionTo && actionLabel && !actionOnClick && (
          <Link
            to={actionTo}
            className="inline-flex shrink-0 items-center gap-1.5 rounded-lg border border-blue-100 bg-blue-50 px-3 py-2 text-sm font-semibold text-blue-700 hover:bg-white"
          >
            {ActionIcon && <ActionIcon className="h-4 w-4" />}
            {actionLabel}
          </Link>
        )}
      </div>
      {children && <div className="mt-5">{children}</div>}
    </>
  );

  if (primaryTo) {
    return (
      <Link
        to={primaryTo}
        className={`group block rounded-lg border border-slate-200 bg-white shadow-sm transition hover:border-blue-300 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-blue-200 ${compact ? "min-h-[190px] p-4" : "p-5"}`}
      >
        {content}
      </Link>
    );
  }

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      {content}
    </section>
  );
}
