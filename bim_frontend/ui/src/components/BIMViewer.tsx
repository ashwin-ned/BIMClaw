"use client";
import { getIFCUrl } from "@/lib/api";

interface Props {
  modelId: string;
}

export function BIMViewer({ modelId }: Props) {
  return (
    <div className="w-full h-full flex flex-col items-center justify-center gap-3 bg-gray-100 rounded-lg text-sm text-gray-500">
      <p>3D viewer not available — use the Floor Plan (BEV) view.</p>
      <a
        href={getIFCUrl(modelId)}
        download
        className="text-blue-500 hover:underline text-xs"
      >
        Download IFC file
      </a>
    </div>
  );
}
