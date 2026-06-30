"use client";
import { useState, useRef } from "react";
import { uploadIFC, triggerRender, UploadResponse } from "@/lib/api";

interface Props {
  onUploaded: (res: UploadResponse) => void;
}

export function UploadZone({ onUploaded }: Props) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFile(file: File) {
    if (!file.name.endsWith(".ifc")) {
      setError("Please upload a .ifc file");
      return;
    }
    setError(null);
    setUploading(true);
    try {
      const res = await uploadIFC(file);
      onUploaded(res);
      // Kick off background render immediately
      await triggerRender(res.model_id);
    } catch (e: any) {
      setError(e.message || "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div
      className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
        ${dragging ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-blue-400"}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".ifc"
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
      />
      {uploading ? (
        <p className="text-blue-600 font-medium">Uploading and parsing IFC…</p>
      ) : (
        <>
          <p className="text-gray-600 text-lg font-medium">Drop an IFC file here</p>
          <p className="text-gray-400 text-sm mt-1">or click to browse</p>
        </>
      )}
      {error && <p className="text-red-500 mt-2 text-sm">{error}</p>}
    </div>
  );
}
