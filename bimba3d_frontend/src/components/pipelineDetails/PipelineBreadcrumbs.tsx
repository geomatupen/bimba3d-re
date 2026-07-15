import { ChevronRight, Home } from "lucide-react";
import { Link } from "react-router-dom";

interface BreadcrumbItem {
  label: string;
  to?: string;
}

interface PipelineBreadcrumbsProps {
  items: BreadcrumbItem[];
}

export default function PipelineBreadcrumbs({ items }: PipelineBreadcrumbsProps) {
  return (
    <nav className="flex flex-wrap items-center gap-1 text-xs text-blue-100" aria-label="Breadcrumb">
      <Link to="/" className="inline-flex items-center gap-1 rounded px-1.5 py-1 hover:bg-white/10">
        <Home className="h-3.5 w-3.5" />
        Home
      </Link>
      {items.map((item) => (
        <span key={`${item.label}-${item.to || "current"}`} className="inline-flex items-center gap-1">
          <ChevronRight className="h-3.5 w-3.5 text-blue-200" />
          {item.to ? (
            <Link to={item.to} className="rounded px-1.5 py-1 hover:bg-white/10">
              {item.label}
            </Link>
          ) : (
            <span className="rounded px-1.5 py-1 font-semibold text-white">{item.label}</span>
          )}
        </span>
      ))}
    </nav>
  );
}
