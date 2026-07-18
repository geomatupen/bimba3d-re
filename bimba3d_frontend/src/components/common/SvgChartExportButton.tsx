import { useEffect, useRef, useState } from "react";

interface SvgChartExportButtonProps {
  filename: string;
  label?: string;
  svgElement?: SVGSVGElement | null;
}

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

const serializedSvg = (svg: SVGSVGElement) => {
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
  ];

  sourceNodes.forEach((sourceNode, index) => {
    const targetNode = cloneNodes[index];
    if (!targetNode || !(sourceNode instanceof Element)) return;
    const computed = window.getComputedStyle(sourceNode);
    styleProps.forEach((prop) => {
      const value = computed.getPropertyValue(prop);
      if (value) targetNode.style.setProperty(prop, value);
    });
  });

  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", String(svg.viewBox.baseVal.width || svg.clientWidth || 1200));
  clone.setAttribute("height", String(svg.viewBox.baseVal.height || svg.clientHeight || 700));
  return new XMLSerializer().serializeToString(clone);
};

const exportSvg = (svg: SVGSVGElement, filename: string) => {
  const source = serializedSvg(svg);
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

  const handleExport = async (format: "png" | "svg") => {
    const svg = svgElement;
    if (!svg) return;
    setOpen(false);
    if (format === "svg") {
      exportSvg(svg, baseName);
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
