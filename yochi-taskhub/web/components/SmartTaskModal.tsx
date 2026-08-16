"use client";

// Smart task: describe exactly what the task should DO. The planner maps it
// to lake + feed data, writes an executable DATA PLAN into the task, proposes
// missing feeds, and the agent starts working it immediately.
import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/lib/supabase";
import { useOpenTask } from "@/components/TaskCard";

type Plan = {
  task_id: string;
  title: string;
  data_plan: string;
  feeds_used: string[];
  feeds_to_activate: string[];
  feeds_proposed: string[];
  recurrence: string | null;
};

export default function SmartTaskModal() {
  const qc = useQueryClient();
  const openTask = useOpenTask();
  const [open, setOpen] = useState(false);
  const [spec, setSpec] = useState("");
  const [busy, setBusy] = useState(false);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!spec.trim() || busy) return;
    setBusy(true);
    setErr(null);
    try {
      const { data } = await supabase.auth.getSession();
      const res = await fetch(`${process.env.NEXT_PUBLIC_FN_BASE}/api/smart_task`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${data.session?.access_token}`,
        },
        body: JSON.stringify({ description: spec, execute_now: true }),
      });
      const out = await res.json();
      if (!res.ok) throw new Error(out.error ?? `HTTP ${res.status}`);
      setPlan(out as Plan);
      qc.invalidateQueries({ queryKey: ["tasks"] });
    } catch (e2) {
      setErr(e2 instanceof Error ? e2.message : String(e2));
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    setOpen(false);
    setPlan(null);
    setSpec("");
    setErr(null);
  }

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-xs font-semibold text-gray-600 hover:border-brand-500"
      >
        ✨ Smart task
      </button>
      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={reset}>
          <div
            className="max-h-[85vh] w-full max-w-lg overflow-y-auto rounded-2xl bg-white p-5 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {!plan ? (
              <>
                <h2 className="mb-1 text-base font-bold">Smart task</h2>
                <p className="mb-3 text-xs text-gray-500">
                  Describe exactly what the task should check or do — the system plans it
                  against the data lake and external feeds, creates the task, and the agent
                  starts executing it straight away.
                </p>
                <form onSubmit={submit}>
                  <textarea
                    value={spec}
                    onChange={(e) => setSpec(e.target.value)}
                    rows={5}
                    autoFocus
                    placeholder="e.g. Check we have paid payroll tax on time for each state — use the state due dates, then scan payments in the GL to each state revenue office and flag any late months."
                    className="mb-3 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
                  />
                  <button
                    disabled={busy || !spec.trim()}
                    className="w-full rounded-lg bg-brand-600 px-3 py-2 text-sm font-semibold text-white disabled:opacity-50"
                  >
                    {busy ? "Planning against the data…" : "Plan, create & execute"}
                  </button>
                </form>
              </>
            ) : (
              <>
                <h2 className="mb-1 text-base font-bold">✓ {plan.title}</h2>
                <p className="mb-2 text-xs text-gray-500">
                  Task created and the agent is working it now — results land on the task as a
                  report + comment.
                </p>
                <div className="mb-3 whitespace-pre-wrap rounded-lg bg-gray-50 p-3 text-xs leading-relaxed text-gray-700">
                  {plan.data_plan}
                </div>
                {plan.feeds_to_activate.length > 0 && (
                  <div className="mb-2 rounded-lg bg-amber-50 p-2.5 text-xs text-amber-800">
                    Needs feeds not yet active: <b>{plan.feeds_to_activate.join(", ")}</b> —
                    activate them in Admin → Data feeds.
                  </div>
                )}
                {plan.feeds_proposed.length > 0 && (
                  <div className="mb-2 rounded-lg bg-blue-50 p-2.5 text-xs text-blue-800">
                    New feeds proposed for the learning centre:{" "}
                    <b>{plan.feeds_proposed.join(", ")}</b> — review in Admin → Data feeds.
                  </div>
                )}
                {plan.recurrence && plan.recurrence !== "one-off" && (
                  <div className="mb-2 text-xs text-gray-500">
                    Suggested cadence: {plan.recurrence}
                  </div>
                )}
                <div className="flex gap-2">
                  <button
                    onClick={() => { openTask(plan.task_id); reset(); }}
                    className="flex-1 rounded-lg bg-brand-600 px-3 py-2 text-sm font-semibold text-white"
                  >
                    Open the task
                  </button>
                  <button
                    onClick={reset}
                    className="rounded-lg border border-gray-200 px-3 py-2 text-sm text-gray-600"
                  >
                    Done
                  </button>
                </div>
              </>
            )}
            {err && <div className="mt-2 text-xs text-red-600">{err}</div>}
          </div>
        </div>
      )}
    </>
  );
}
