"use client";
import { useState, useRef, useEffect } from "react";
import { askQuestion, QueryResponse } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  steps?: number;
  error?: boolean;
}

interface Props {
  modelId: string;
  disabled?: boolean;
}

export function ChatBox({ modelId, disabled }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage() {
    const q = input.trim();
    if (!q || loading || disabled) return;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setLoading(true);
    try {
      const res = await askQuestion(modelId, q);
      const content = res.answer?.trim()
        || (res.termination_reason === "max_failures"
            ? "The agent could not reach the LLM. Check that Ollama is running (`bash start.sh ollama`) and try again."
            : "The agent reached the step limit without finding an answer. Try rephrasing your question.");
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content, steps: res.steps },
      ]);
    } catch (e: any) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${e.message}`, error: true },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-full bg-white rounded-lg border border-gray-200">
      {/* Message history */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 && (
          <p className="text-gray-400 text-sm text-center pt-8">
            Ask a spatial question about this building model
          </p>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[85%] rounded-lg px-4 py-2 text-sm
                ${msg.role === "user"
                  ? "bg-blue-600 text-white"
                  : msg.error
                    ? "bg-red-50 text-red-700 border border-red-200"
                    : "bg-gray-100 text-gray-800"
                }`}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>
              {msg.steps !== undefined && (
                <p className="text-xs mt-1 opacity-60">{msg.steps} reasoning steps</p>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-lg px-4 py-2 text-sm text-gray-500 animate-pulse">
              Thinking…
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="border-t border-gray-200 p-3 flex gap-2">
        <input
          type="text"
          className="flex-1 rounded-md border border-gray-300 px-3 py-2 text-sm
            focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50"
          placeholder={
            disabled
              ? "Waiting for IFC to load…"
              : "e.g. How many windows are on the first floor?"
          }
          value={input}
          disabled={disabled || loading}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button
          onClick={sendMessage}
          disabled={disabled || loading || !input.trim()}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white
            rounded-md px-4 py-2 text-sm font-medium transition-colors"
        >
          Send
        </button>
      </div>
    </div>
  );
}
