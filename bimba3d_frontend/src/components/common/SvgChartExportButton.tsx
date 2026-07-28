import { useEffect, useRef, useState } from "react";

interface SvgChartExportButtonProps {
  filename: string;
  label?: string;
  svgElement?: SVGSVGElement | null;
}

type SvgExportMode = "screen" | "report";

const safeFilename = (name: string) =>
  name
    .trim()
    .replace(/[^a-z0-9._-]+/gi, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 120) || "chart";

const downloadBlob = (blob: Blob, filename: string) => {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
};

const reportColorMap: Record<string, string> = {
  "#2563eb": "#4f76b8",
  "#3b82f6": "#5f86c7",
  "#1d4ed8": "#496da8",
  "#16a34a": "#4f9b6a",
  "#22c55e": "#62ad79",
  "#15803d": "#447f5a",
  "#f97316": "#c9783d",
  "#fb923c": "#d98e57",
  "#ea580c": "#b96534",
  "#f59e0b": "#d2a247",
  "#fbbf24": "#d8b35a",
  "#d97706": "#b88734",
  "#dc2626": "#b75b5b",
  "#ef4444": "#c86c6c",
};

const reportTextColorMap: Record<string, string> = {
  "#0f172a": "#334155",
  "#1e293b": "#475569",
  "#334155": "#5b6778",
  "#475569": "#68758a",
  "#64748b": "#758398",
};

const transparentFillValues = new Set([
  "#fff",
  "#ffffff",
  "white",
  "rgb(255, 255, 255)",
  "rgba(255, 255, 255, 1)",
  "#f8fafc",
  "rgb(248, 250, 252)",
  "rgba(248, 250, 252, 1)",
  "#f9fafb",
  "rgb(249, 250, 251)",
  "rgba(249, 250, 251, 1)",
  "#f1f5f9",
  "rgb(241, 245, 249)",
  "rgba(241, 245, 249, 1)",
]);

const whiteStrokeValues = new Set([
  "#fff",
  "#ffffff",
  "white",
  "rgb(255, 255, 255)",
  "rgba(255, 255, 255, 1)",
]);

const normalizeColorValue = (value: string) => value.trim().toLowerCase();

const toMutedReportColor = (value: string) => {
  let next = value;
  Object.entries(reportColorMap).forEach(([from, to]) => {
    const hexPattern = new RegExp(from.replace("#", "\\#"), "gi");
    next = next.replace(hexPattern, to);
  });
  next = next
    .replace(/rgb\(37,\s*99,\s*235\)/gi, "#4f76b8")
    .replace(/rgb\(59,\s*130,\s*246\)/gi, "#5f86c7")
    .replace(/rgb\(22,\s*163,\s*74\)/gi, "#4f9b6a")
    .replace(/rgb\(34,\s*197,\s*94\)/gi, "#62ad79")
    .replace(/rgb\(249,\s*115,\s*22\)/gi, "#c9783d")
    .replace(/rgb\(251,\s*146,\s*60\)/gi, "#d98e57")
    .replace(/rgb\(245,\s*158,\s*11\)/gi, "#d2a247")
    .replace(/rgb\(251,\s*191,\s*36\)/gi, "#d8b35a")
    .replace(/rgb\(217,\s*119,\s*6\)/gi, "#b88734")
    .replace(/rgb\(220,\s*38,\s*38\)/gi, "#b75b5b")
    .replace(/rgb\(239,\s*68,\s*68\)/gi, "#c86c6c");
  return next;
};

const toMutedReportTextColor = (value: string) => {
  let next = value;
  Object.entries(reportTextColorMap).forEach(([from, to]) => {
    const hexPattern = new RegExp(from.replace("#", "\\#"), "gi");
    next = next.replace(hexPattern, to);
  });
  next = next
    .replace(/rgb\(15,\s*23,\s*42\)/gi, "#334155")
    .replace(/rgb\(30,\s*41,\s*59\)/gi, "#475569")
    .replace(/rgb\(51,\s*65,\s*85\)/gi, "#5b6778")
    .replace(/rgb\(71,\s*85,\s*105\)/gi, "#68758a")
    .replace(/rgb\(100,\s*116,\s*139\)/gi, "#758398");
  return next;
};

const makeExportFill = (value: string, mode: SvgExportMode) => {
  if (mode !== "report") return value;
  const normalized = normalizeColorValue(value);
  if (transparentFillValues.has(normalized)) return "none";
  return toMutedReportColor(value);
};

const applyReportExportStyle = (clone: SVGSVGElement) => {
  clone.style.background = "transparent";
  clone.style.backgroundColor = "transparent";
  clone.setAttribute("style", `${clone.getAttribute("style") || ""};background:transparent;background-color:transparent`);

  [clone, ...Array.from(clone.querySelectorAll("*"))].forEach((node) => {
    if (!(node instanceof SVGElement)) return;
    const isText = node.tagName.toLowerCase() === "text";
    ["fill", "stroke", "color"].forEach((attr) => {
      const value = node.getAttribute(attr);
      if (!value) return;
      const next = isText
        ? toMutedReportTextColor(value)
        : attr === "fill"
          ? makeExportFill(value, "report")
          : toMutedReportColor(value);
      node.setAttribute(attr, next);
    });

    const style = node.getAttribute("style");
    if (style) {
      const colorAdjustedStyle = isText ? toMutedReportTextColor(style) : toMutedReportColor(style);
      const mutedStyle = colorAdjustedStyle.replace(
        /(fill|background|background-color):\s*(#fff(?:fff)?|white|rgb\(255,\s*255,\s*255\)|rgba\(255,\s*255,\s*255,\s*1\)|#f8fafc|rgb\(248,\s*250,\s*252\)|rgba\(248,\s*250,\s*252,\s*1\)|#f9fafb|rgb\(249,\s*250,\s*251\)|rgba\(249,\s*250,\s*251,\s*1\)|#f1f5f9|rgb\(241,\s*245,\s*249\)|rgba\(241,\s*245,\s*249,\s*1\))\s*;?/gi,
        (_match, prop) => (String(prop).startsWith("background") ? `${prop}:transparent;` : "fill:none;"),
      );
      node.setAttribute("style", mutedStyle);
    }

    const stroke = node.getAttribute("stroke") || node.style.getPropertyValue("stroke");
    const strokeWidth = node.getAttribute("stroke-width") || node.style.getPropertyValue("stroke-width");
    const hasWhiteHalo = stroke && whiteStrokeValues.has(normalizeColorValue(stroke));
    if (isText && hasWhiteHalo) {
      const width = Number.parseFloat(strokeWidth || "");
      if (!Number.isFinite(width) || width > 1.2) {
        node.setAttribute("stroke-width", "1.2");
        node.style.setProperty("stroke-width", "1.2");
      }
    }

  });
};

const serializedSvg = (svg: SVGSVGElement, mode: SvgExportMode = "screen") => {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  const sourceNodes = [svg, ...Array.from(svg.querySelectorAll("*"))];
  const cloneNodes = [clone, ...Array.from(clone.querySelectorAll("*"))] as SVGElement[];
  const styleProps = [
    "fill",
    "stroke",
    "stroke-width",
    "stroke-dasharray",
    "font-family",
    "font-size",
    "font-weight",
    "opacity",
    "text-anchor",
    "background-color",
  ];

  sourceNodes.forEach((sourceNode, index) => {
    const targetNode = cloneNodes[index];
    if (!targetNode || !(sourceNode instanceof Element)) return;
    const computed = window.getComputedStyle(sourceNode);
    styleProps.forEach((prop) => {
      const value = computed.getPropertyValue(prop);
      if (!value) return;
      const isText = sourceNode instanceof SVGElement && sourceNode.tagName.toLowerCase() === "text";
      const exportValue =
        prop === "background-color"
          ? mode === "report"
            ? "transparent"
            : value
          : mode === "report" && isText && (prop === "fill" || prop === "stroke" || prop === "color")
            ? toMutedReportTextColor(value)
          : prop === "fill"
          ? makeExportFill(value, mode)
          : mode === "report"
            ? toMutedReportColor(value)
            : value;
      targetNode.style.setProperty(prop, exportValue);
    });
  });

  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", String(svg.viewBox.baseVal.width || svg.clientWidth || 1200));
  clone.setAttribute("height", String(svg.viewBox.baseVal.height || svg.clientHeight || 700));
  if (mode === "report") applyReportExportStyle(clone);
  return new XMLSerializer().serializeToString(clone);
};

const exportSvg = (svg: SVGSVGElement, filename: string) => {
  const source = serializedSvg(svg, "report");
  downloadBlob(new Blob([source], { type: "image/svg+xml;charset=utf-8" }), `${filename}.svg`);
};

const exportPng = async (svg: SVGSVGElement, filename: string) => {
  const source = serializedSvg(svg);
  const blob = new Blob([source], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  try {
    const image = new Image();
    image.decoding = "async";
    const loaded = new Promise<void>((resolve, reject) => {
      image.onload = () => resolve();
      image.onerror = () => reject(new Error("Chart image could not be rendered."));
    });
    image.src = url;
    await loaded;

    const viewBox = svg.viewBox.baseVal;
    const width = Math.max(viewBox.width || svg.clientWidth || 1200, 1);
    const height = Math.max(viewBox.height || svg.clientHeight || 700, 1);
    const scale = 3;
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas export is not available in this browser.");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    const pngBlob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png", 1));
    if (!pngBlob) throw new Error("PNG export failed.");
    downloadBlob(pngBlob, `${filename}.png`);
  } finally {
    URL.revokeObjectURL(url);
  }
};

const exportReportPng = async (svg: SVGSVGElement, filename: string) => {
  const source = serializedSvg(svg, "report");
  const blob = new Blob([source], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  try {
    const image = new Image();
    image.decoding = "async";
    const loaded = new Promise<void>((resolve, reject) => {
      image.onload = () => resolve();
      image.onerror = () => reject(new Error("Chart image could not be rendered."));
    });
    image.src = url;
    await loaded;

    const viewBox = svg.viewBox.baseVal;
    const width = Math.max(viewBox.width || svg.clientWidth || 1200, 1);
    const height = Math.max(viewBox.height || svg.clientHeight || 700, 1);
    const scale = 3;
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas export is not available in this browser.");
    // No background fill here: this PNG stays transparent for report layouts.
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    const pngBlob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png", 1));
    if (!pngBlob) throw new Error("PNG export failed.");
    downloadBlob(pngBlob, `${filename}_report.png`);
  } finally {
    URL.revokeObjectURL(url);
  }
};

export default function SvgChartExportButton({ filename, label = "Export", svgElement }: SvgChartExportButtonProps) {
  const baseName = safeFilename(filename);
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const handlePointerDown = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, [open]);

  const handleExport = async (format: "png" | "reportPng" | "svg") => {
    const svg = svgElement;
    if (!svg) return;
    setOpen(false);
    if (format === "svg") {
      exportSvg(svg, baseName);
      return;
    }
    if (format === "reportPng") {
      await exportReportPng(svg, baseName);
      return;
    }
    await exportPng(svg, baseName);
  };

  return (
    <div ref={menuRef} className="relative inline-flex">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="inline-flex h-6 w-6 items-center justify-center rounded border border-slate-300 bg-slate-50 text-slate-900 shadow-sm hover:bg-white"
        style={{ fontSize: 14, lineHeight: "14px" }}
        title={label}
        aria-label={label}
        aria-expanded={open}
      >
        <span aria-hidden="true" style={{ fontSize: 13, transform: "translateY(-1px)" }}>⇩</span>
      </button>
      {open && (
        <div
          className="absolute right-0 top-7 z-30 min-w-14 overflow-hidden rounded border border-slate-200 bg-white py-0.5 font-medium text-slate-700 shadow-lg"
          style={{ fontSize: 10, lineHeight: "14px" }}
        >
          <button
            type="button"
            onClick={() => void handleExport("png")}
            className="block w-full px-2 py-0.5 text-left hover:bg-slate-50"
            style={{ fontSize: 10, lineHeight: "14px" }}
          >
            PNG
          </button>
          <button
            type="button"
            onClick={() => void handleExport("reportPng")}
            className="block w-full whitespace-nowrap px-2 py-0.5 text-left hover:bg-slate-50"
            style={{ fontSize: 10, lineHeight: "14px" }}
            title="Transparent PNG with muted report colors"
          >
            Report PNG
          </button>
          <button
            type="button"
            onClick={() => void handleExport("svg")}
            className="block w-full px-2 py-0.5 text-left hover:bg-slate-50"
            style={{ fontSize: 10, lineHeight: "14px" }}
          >
            SVG
          </button>
        </div>
      )}
    </div>
  );
}
