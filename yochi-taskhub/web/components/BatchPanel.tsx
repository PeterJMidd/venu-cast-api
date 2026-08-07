"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/lib/supabase";
import { callFn } from "@/lib/fn";

interface BatchRun {
  id: string;
  kind: string;
  status: "queued" | "running" | "done" | "error";
  progress: {
    done?: number;
    total?: number;
    current?: string;
    results?: { task_id: string; title?: string; ok: boolean; summary?: string; note?: string }[];
  };
  summary: string | null;
  error: string | null;
  created_at: string;
}

export default function BatchPanel({ projectId }: { projectId: string }) {
  const qc = useQueryClient();
  const [activeBatch, setActiveBatch] = useState<string | null>(null);
  const [steer, setSteer] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  const { data: batch } = useQuery({
    queryKey: ["batch", activeBatch],
    enabled: !!activeBatch,
    refetchInterval: (q) => {
      const s = (q.state.data as BatchRun | undefined)?.status;
      return s === "done" || s === "error" ? false : 4000;
    },
    queryFn: async () => {
      const { data, error } = await supabase
        .from("batch_runs")
        .select("*")
        .eq("id", activeBatch!)
        .single();
      if (error) throw error;
      return data as BatchRun;
    },
  });

  async function runAll() {
    if (!confirm("Run the AI agent across ALL open tasks in this project? Each task gets a report attached.")) return;
    setBusy(true);
    setErr(null);
    setOpen(true);
    try {
      const r = await callFn<{ batch_id: string }>("project_run", { project_id: projectId });
      setActiveBatch(r.batch_id);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function sendSteer() {
    if (!batch || !steer.trim()) return;
    setBusy(true);
    setErr(null);
    try {
      const r = await callFn<{ batch_id: string }>("project_steer", {
        project_id: projectId,
        parent_batch_id: batch.id,
        steer: steer.trim(),
      });
      setActiveBatch(r.batch_id);
      setSteer("");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const pct =
    batch?.progress?.total && batch.progress.total > 0
      ? Math.round((100 * (batch.progress.done ?? 0)) / batch.progress.total)
      : 0;

  return (
    <>
      <button
        onClick={runAll}
        disabled={busy}
        className="rounded-lg border border-brand-600 px-3 py-1.5 text-sm font-semibold text-brand-700 hover:bg-brand-50 disabled:opacity-50"
      >
        🤖 Run all tasks
      </button>

      {open && (
        <div className="fixed bottom-4 right-4 z-40 w-[420px] rounded-xl border border-gray-200 bg-white shadow-2xl">
          <div className="flex items-center justify-between border-b border-gray-100 px-4 py-2.5">
            <span className="text-sm font-semibold">
              🤖 Batch run {batch?.kind === "steer" ? "(steer)" : ""}
            </span>
            <button onClick={() => setOpen(false)} className="text-gray-400 hover:text-gray-600">✕</button>
          </div>
          <div className="max-h-[60vh] overflow-y-auto p-4">
            {err && <div className="mb-2 text-sm text-red-600">{err}</div>}
            {!batch && !err && <div className="text-sm text-gray-400">Starting…</div>}

            {batch && (batch.status === "queued" || batch.status === "running") && (
              <div>
                <div className="mb-1 flex justify-between text-xs text-gray-500">
                  <span>{batch.status === "queued" ? "Waiting for worker…" : "Working…"}</span>
                  <span>{batch.progress?.done ?? 0}/{batch.progress?.total ?? "?"}</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-gray-100">
                  <div className="h-full rounded-full bg-brand-600 transition-all" style={{ width: `${pct}%` }} />
                </div>
                {batch.progress?.current && (
                  <div className="mt-2 truncate text-xs text-gray-500">Now: {batch.progress.current}</div>
                )}
                <div className="mt-2 space-y-1">
                  {batch.progress?.results?.slice(-4).map((r, i) => (
                    <div key={i} className={`truncate text-xs ${r.ok ? "text-brand-700" : "text-red-600"}`}>
                      {r.ok ? "✓" : "✗"} {r.title ?? r.task_id}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {batch?.status === "error" && (
              <div className="text-sm text-red-600">Batch failed: {batch.error}</div>
            )}

            {batch?.status === "done" && (
              <div className="space-y-3">
                <div className="whitespace-pre-wrap rounded-lg bg-gray-50 p-3 text-xs leading-relaxed text-gray-800">
                  {batch.summary}
                </div>
                <div>
                  <div className="mb-1 text-xs font-semibold text-gray-600">Steer the next pass</div>
                  <textarea
                    value={steer}
                    onChange={(e) => setSteer(e.target.value)}
                    placeholder="e.g. 'Re-run the clearing account reviews with day-by-day detail; ignore the low-priority items'"
                    rows={2}
                    className="w-full rounded-lg border border-gray-300 px-2 py-1.5 text-xs focus:border-brand-500 focus:outline-none"
                  />
                  <button
                    onClick={sendSteer}
                    disabled={busy || !steer.trim()}
                    className="mt-1 rounded-lg bg-brand-600 px-3 py-1 text-xs font-semibold text-white hover:bg-brand-700 disabled:opacity-50"
                  >
                    {busy ? "Queuing…" : "Steer & re-run"}
                  </button>
                </div>
                <button
                  onClick={() => qc.invalidateQueries({ queryKey: ["tasks"] })}
                  className="text-xs text-gray-400 hover:text-brand-600"
                >
                  Refresh board
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
