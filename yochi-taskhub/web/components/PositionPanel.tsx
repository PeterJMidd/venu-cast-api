"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { format, parseISO } from "date-fns";
import { supabase } from "@/lib/supabase";

interface Position {
  id: string;
  version: number;
  content: string;
  events: { at: string; text: string }[];
  source: string | null;
  created_at: string;
}

export default function PositionPanel({ projectId }: { projectId: string }) {
  const [open, setOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);

  const { data: positions } = useQuery({
    queryKey: ["positions", projectId],
    enabled: open,
    refetchInterval: open ? 15000 : false,
    queryFn: async () => {
      const { data, error } = await supabase
        .from("positions")
        .select("*")
        .eq("project_id", projectId)
        .order("version", { ascending: false })
        .limit(6);
      if (error) throw error;
      return data as Position[];
    },
  });

  const current = positions?.[0];

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm font-semibold text-gray-700 hover:bg-gray-50"
      >
        📍 Position
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4" onClick={() => setOpen(false)}>
          <div
            className="flex max-h-[88vh] w-full max-w-2xl flex-col overflow-hidden rounded-xl bg-white shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-gray-200 px-5 py-3">
              <div>
                <span className="text-sm font-bold">📍 Project position</span>
                {current && (
                  <span className="ml-2 text-xs text-gray-400">
                    v{current.version} · {format(parseISO(current.created_at), "d MMM HH:mm")}
                    {current.source ? ` · ${current.source.split(":")[0]}` : ""}
                  </span>
                )}
              </div>
              <button onClick={() => setOpen(false)} className="text-gray-400 hover:text-gray-600">✕</button>
            </div>

            <div className="flex-1 overflow-y-auto px-5 py-4">
              {!current && (
                <div className="py-8 text-center text-sm text-gray-400">
                  No position yet — it starts building the first time you run automation on this project
                  (🤖 Run all tasks, or any task agent run).
                </div>
              )}
              {current && (
                <>
                  <div className="whitespace-pre-wrap rounded-lg bg-gray-50 p-4 text-sm leading-relaxed text-gray-800">
                    {current.content}
                  </div>
                  {current.events.length > 0 && (
                    <div className="mt-3">
                      <div className="mb-1 text-xs font-semibold text-amber-700">
                        Since this position ({current.events.length} interim event{current.events.length === 1 ? "" : "s"} — folded in on the next run)
                      </div>
                      {current.events.slice(-8).map((e, i) => (
                        <div key={i} className="text-xs text-gray-500">
                          {e.at?.slice(0, 10)} — {e.text}
                        </div>
                      ))}
                    </div>
                  )}
                  {positions && positions.length > 1 && (
                    <div className="mt-4">
                      <button
                        onClick={() => setHistoryOpen(!historyOpen)}
                        className="text-xs font-semibold text-brand-600 hover:underline"
                      >
                        {historyOpen ? "Hide" : "Show"} history ({positions.length - 1} earlier version{positions.length > 2 ? "s" : ""})
                      </button>
                      {historyOpen &&
                        positions.slice(1).map((p) => (
                          <details key={p.id} className="mt-2 rounded-lg border border-gray-200 p-2">
                            <summary className="cursor-pointer text-xs text-gray-500">
                              v{p.version} · {format(parseISO(p.created_at), "d MMM HH:mm")}
                            </summary>
                            <div className="mt-2 whitespace-pre-wrap text-xs text-gray-600">{p.content}</div>
                          </details>
                        ))}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
