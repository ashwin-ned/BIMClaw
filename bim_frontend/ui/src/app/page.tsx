"use client";
import { useState } from "react";
import { UploadZone } from "@/components/UploadZone";
import { ChatBox } from "@/components/ChatBox";
import { FloorPlan } from "@/components/FloorPlan";
import { BIMViewer } from "@/components/BIMViewer";
import { UploadResponse } from "@/lib/api";

type ViewMode = "viewer" | "floorplan";

export default function HomePage() {
  const [model, setModel] = useState<UploadResponse | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("floorplan");
  const [renderReady, setRenderReady] = useState(false);

  return (
    <main className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-900">BIMClaw</h1>
          <p className="text-xs text-gray-400">Spatial understanding for BIM / IFC files</p>
        </div>
        {model && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">
              {model.metadata.schema} ·{" "}
              {model.metadata.storey_names.length} floors ·{" "}
              {model.metadata.space_names.length} spaces
            </span>
            <button
              onClick={() => { setModel(null); setRenderReady(false); }}
              className="text-xs text-red-500 hover:text-red-700 ml-4"
            >
              × Close
            </button>
          </div>
        )}
      </header>

      {!model ? (
        /* Upload screen */
        <div className="flex flex-1 items-center justify-center p-8">
          <div className="w-full max-w-md">
            <h2 className="text-lg font-semibold text-gray-700 mb-4 text-center">
              Upload an IFC file to get started
            </h2>
            <UploadZone onUploaded={(res) => { setModel(res); }} />
          </div>
        </div>
      ) : (
        /* Main workspace: viewer left, chat right */
        <div className="flex flex-1 overflow-hidden gap-0">
          {/* Left panel: 3D viewer + floor plan */}
          <div className="flex flex-col w-3/5 h-full border-r border-gray-200 bg-white">
            {/* View toggle */}
            <div className="flex items-center gap-1 px-3 py-2 border-b border-gray-100">
              <button
                className={`text-xs px-3 py-1 rounded-md font-medium transition-colors
                  ${viewMode === "floorplan" ? "bg-blue-100 text-blue-700" : "text-gray-500 hover:text-gray-700"}`}
                onClick={() => setViewMode("floorplan")}
              >
                Floor Plan (BEV)
              </button>
              <button
                className={`text-xs px-3 py-1 rounded-md font-medium transition-colors
                  ${viewMode === "viewer" ? "bg-blue-100 text-blue-700" : "text-gray-500 hover:text-gray-700"}`}
                onClick={() => setViewMode("viewer")}
              >
                3D Viewer
              </button>
              {renderReady && (
                <span className="ml-auto text-xs text-green-600 font-medium">
                  ✓ Render ready
                </span>
              )}
            </div>

            {/* Viewer area */}
            <div className="flex-1 overflow-hidden p-2">
              {viewMode === "floorplan" ? (
                <FloorPlan
                  modelId={model.model_id}
                  onReady={() => setRenderReady(true)}
                />
              ) : (
                <BIMViewer modelId={model.model_id} />
              )}
            </div>
          </div>

          {/* Right panel: chat */}
          <div className="flex flex-col w-2/5 h-full bg-white">
            {/* Metadata summary */}
            <div className="px-4 py-3 border-b border-gray-100 bg-gray-50">
              <p className="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-1">
                Building Summary
              </p>
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-500">
                <span>Schema: {model.metadata.schema}</span>
                <span>Floors: {model.metadata.storey_names.length}</span>
                <span>Spaces: {model.metadata.space_names.length}</span>
                {Object.entries(model.metadata.element_counts)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 4)
                  .map(([cls, cnt]) => (
                    <span key={cls}>{cls.replace("Ifc", "")}: {cnt}</span>
                  ))}
              </div>
            </div>

            {/* Chat */}
            <div className="flex-1 overflow-hidden p-3">
              <ChatBox modelId={model.model_id} disabled={false} />
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
