"use client";
import { useState, useEffect } from "react";
import { getFloorPlanUrl, pollRenderStatus } from "@/lib/api";

interface Props {
  modelId: string;
  onReady?: () => void;
}

export function FloorPlan({ modelId, onReady }: Props) {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imgKey, setImgKey] = useState(0);

  useEffect(() => {
    if (!modelId) return;
    setReady(false);
    setError(null);

    const interval = setInterval(async () => {
      try {
        const status = await pollRenderStatus(modelId);
        if (status.ready) {
          setReady(true);
          setImgKey((k) => k + 1);
          onReady?.();
          clearInterval(interval);
        } else if (status.error) {
          setError(status.error);
          clearInterval(interval);
        }
      } catch (e: any) {
        setError(e.message);
        clearInterval(interval);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [modelId]);

  if (error) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-50 rounded-lg text-sm text-red-500 p-4">
        Render failed: {error}
      </div>
    );
  }

  if (!ready) {
    return (
      <div className="flex flex-col items-center justify-center h-full bg-gray-50 rounded-lg">
        <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-3" />
        <p className="text-sm text-gray-500">Rendering floor plan…</p>
        <p className="text-xs text-gray-400 mt-1">This takes 30–60 seconds</p>
      </div>
    );
  }

  return (
    <div className="w-full h-full flex items-center justify-center bg-gray-50 rounded-lg overflow-hidden">
      <img
        key={imgKey}
        src={getFloorPlanUrl(modelId)}
        alt="BEV Floor Plan"
        className="max-w-full max-h-full object-contain"
      />
    </div>
  );
}
