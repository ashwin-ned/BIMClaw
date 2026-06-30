const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface UploadResponse {
  model_id: string;
  metadata: {
    schema: string;
    storey_names: string[];
    space_names: string[];
    element_counts: Record<string, number>;
    has_spaces: boolean;
    has_materials: boolean;
    has_quantities: boolean;
    error?: string;
  };
}

export interface RenderStatus {
  model_id: string;
  rendering: boolean;
  ready: boolean;
  error: string | null;
}

export interface QueryResponse {
  answer: string;
  session_id: string;
  steps: number;
  termination_reason: string;
}

export async function uploadIFC(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE_URL}/api/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Upload failed: ${await res.text()}`);
  return res.json();
}

export async function triggerRender(
  model_id: string,
  frames = 64,
  num_key_frames = 16
): Promise<void> {
  const res = await fetch(`${BASE_URL}/api/render`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id, frames, num_key_frames }),
  });
  if (!res.ok) throw new Error(`Render trigger failed: ${await res.text()}`);
}

export async function pollRenderStatus(model_id: string): Promise<RenderStatus> {
  const res = await fetch(`${BASE_URL}/api/render/${model_id}`);
  if (!res.ok) throw new Error(`Status poll failed: ${await res.text()}`);
  return res.json();
}

export async function askQuestion(
  model_id: string,
  question: string
): Promise<QueryResponse> {
  const res = await fetch(`${BASE_URL}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id, question }),
  });
  if (!res.ok) throw new Error(`Query failed: ${await res.text()}`);
  return res.json();
}

export function getIFCUrl(model_id: string): string {
  return `${BASE_URL}/api/model/${model_id}`;
}

export function getFloorPlanUrl(model_id: string): string {
  return `${BASE_URL}/api/images/${model_id}/floor_plan`;
}
