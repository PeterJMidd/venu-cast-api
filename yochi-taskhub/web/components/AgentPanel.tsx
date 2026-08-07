"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { callFn } from "@/lib/fn";
import { createTask } from "@/lib/mutations";
import type { TaskPriority } from "@/lib/types";

interface PlanQuery {
  label: string;
  sql: string;
}

interface Plan {
  approach: string;
  deliverable_title: string;
  data_queries: PlanQuery[];
  validation: { label: string; ok: boolean; sample_rows?: number; error?: string }[];
  all_valid: boolean;
  used_precedents: string[];
}

interface RunResult {
  summary: string;
  attachment: string;
  files: string[];
  follow_up_tasks: { title: string; description?: string; priority: TaskPriority }[];
  project_id: string;
}

export default function AgentPanel({ taskId }: { taskId: string }) {
  const qc = useQueryClient();
  const [plan, setPlan] = useState<Plan | null>(null);
  const [lastPlan, setLastPlan] = useState<Plan | null>(null);
  const [result, setResult] = useState<RunResult | null>(null);
  const [busy, setBusy] = useState<"propose" | "execute" | "refine" | "email" | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [created, setCreated] = useState<Set<number>>(new Set());
  const [feedback, setFeedback] = useState("");
  const [emailTo, setEmailTo] = useState("");
  const [emailNote, setEmailNote] = useState("");
  const [emailMsg, setEmailMsg] = useState<string | null>(null);

  async function proposePlan() {
    setBusy("propose");
    setErr(null);
    setResult(null);
    try {
      setPlan(await callFn<Plan>("agent_propose", { task_id: taskId }));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  async function runPlan() {
    if (!plan) return;
    setBusy("execute");
    setErr(null);
    try {
      const r = await callFn<RunResult>("agent_execute", { task_id: taskId, plan });
      setResult(r);
      setLastPlan(plan);
      setPlan(null);
      qc.invalidateQueries({ queryKey: ["task", taskId] });
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  async function refine() {
    if (!lastPlan || !result || !feedback.trim()) return;
    setBusy("refine");
    setErr(null);
    try {
      const r = await callFn<RunResult>("agent_execute", {
        task_id: taskId,
        plan: lastPlan,
        feedback: feedback.trim(),
        prior_summary: result.summary,
      });
      setResult(r);
      setFeedback("");
      qc.invalidateQueries({ queryKey: ["task", taskId] });
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  async function emailOutput() {
    if (!result || !emailTo.trim()) return;
    setBusy("email");
    setErr(null);
    setEmailMsg(null);
    try {
      await callFn("send_report", {
        task_id: taskId,
        to: emailTo.trim(),
        note: emailNote.trim(),
        files: result.files,
        subject: `TaskHub report: ${result.files[0]?.replace(/_/g, " ").replace(/\.html$/, "") ?? "task output"}`,
      });
      setEmailMsg(`Sent to ${emailTo.trim()}`);
      setEmailTo("");
      setEmailNote("");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  async function addFollowUp(i: number) {
    if (!result) return;
    const f = result.follow_up_tasks[i];
    await createTask({
      project_id: result.project_id,
      title: f.title,
      description: f.description ?? null,
      priority: f.priority,
    });
    setCreated((s) => new Set(s).add(i));
    qc.invalidateQueries({ queryKey: ["tasks"] });
  }

  return (
    <div className="rounded-xl border border-brand-100 bg-brand-50/50 p-4">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs font-semibold text-brand-700">🤖 Task agent</span>
        {!plan && !result && (
          <button
            onClick={proposePlan}
            disabled={busy !== null}
            className="rounded-lg bg-brand-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-brand-700 disabled:opacity-50"
          >
            {busy === "propose" ? "Planning…" : "Propose how to do this"}
          </button>
        )}
      </div>

      {!plan && !result && !busy && (
        <p className="text-xs text-gray-500">
          The agent reads this task, plans an approach using the data lake (and what worked on
          similar tasks before), shows you the plan, then executes it and attaches the report here.
        </p>
      )}

      {plan && (
        <div className="space-y-2">
          <div className="whitespace-pre-wrap text-sm text-gray-800">{plan.approach}</div>
          {plan.used_precedents.length > 0 && (
            <div className="text-[11px] text-gray-500">
              Informed by earlier runs: {plan.used_precedents.join("; ")}
            </div>
          )}
          <div className="rounded-lg bg-white p-2 text-xs">
            {plan.validation.map((v, i) => (
              <div key={i} className={v.ok ? "text-brand-700" : "text-red-600"}>
                {v.ok ? `✓ ${v.label} (tested, ${v.sample_rows} sample rows)` : `✗ ${v.label}: ${v.error}`}
              </div>
            ))}
          </div>
          <details className="text-xs text-gray-500">
            <summary className="cursor-pointer">Show queries</summary>
            {plan.data_queries.map((q, i) => (
              <pre key={i} className="mt-1 overflow-x-auto rounded bg-gray-50 p-2 text-[10px]">{q.sql}</pre>
            ))}
          </details>
          <div className="flex gap-2">
            <button
              onClick={runPlan}
              disabled={busy !== null || !plan.all_valid}
              className="rounded-lg bg-brand-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-brand-700 disabled:opacity-50"
            >
              {busy === "execute" ? "Working… (can take a minute)" : "Run it"}
            </button>
            <button
              onClick={() => setPlan(null)}
              disabled={busy !== null}
              className="rounded-lg px-3 py-1.5 text-xs text-gray-500 hover:bg-gray-100"
            >
              Discard
            </button>
          </div>
        </div>
      )}

      {result && (
        <div className="space-y-2">
          <div className="whitespace-pre-wrap text-sm text-gray-800">{result.summary}</div>
          <div className="text-xs text-gray-500">
            📎 Full report attached in the <b>Files</b> tab ({result.attachment}); summary posted to comments.
          </div>
          {result.follow_up_tasks.length > 0 && (
            <div className="rounded-lg bg-white p-2">
              <div className="mb-1 text-xs font-semibold text-gray-600">Suggested follow-up tasks</div>
              {result.follow_up_tasks.map((f, i) => (
                <div key={i} className="flex items-center justify-between gap-2 py-1 text-xs">
                  <span>
                    <b className="capitalize">{f.priority}</b> — {f.title}
                  </span>
                  {created.has(i) ? (
                    <span className="shrink-0 text-brand-600">✓ created</span>
                  ) : (
                    <button
                      onClick={() => addFollowUp(i)}
                      className="shrink-0 rounded bg-brand-600 px-2 py-0.5 font-semibold text-white hover:bg-brand-700"
                    >
                      + Create
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
          {/* Iterate on the result */}
          <div className="rounded-lg bg-white p-2">
            <div className="mb-1 text-xs font-semibold text-gray-600">Refine the output</div>
            <textarea
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              placeholder="Tell the agent what to change… e.g. 'focus only on the two genuine feed failures, add a day-by-day gap table, shorter analysis'"
              rows={2}
              className="w-full rounded-lg border border-gray-300 px-2 py-1.5 text-xs focus:border-brand-500 focus:outline-none"
            />
            <button
              onClick={refine}
              disabled={busy !== null || !feedback.trim()}
              className="mt-1 rounded-lg bg-brand-600 px-3 py-1 text-xs font-semibold text-white hover:bg-brand-700 disabled:opacity-50"
            >
              {busy === "refine" ? "Revising…" : "Revise report"}
            </button>
            <span className="ml-2 text-[10px] text-gray-400">Replaces the attached files with the revised version.</span>
          </div>

          {/* Email the output */}
          <div className="rounded-lg bg-white p-2">
            <div className="mb-1 text-xs font-semibold text-gray-600">Email the output</div>
            <div className="flex gap-2">
              <input
                type="email"
                value={emailTo}
                onChange={(e) => setEmailTo(e.target.value)}
                placeholder="recipient@yochi.com.au"
                className="flex-1 rounded-lg border border-gray-300 px-2 py-1.5 text-xs focus:border-brand-500 focus:outline-none"
              />
              <button
                onClick={emailOutput}
                disabled={busy !== null || !emailTo.trim()}
                className="rounded-lg bg-brand-600 px-3 py-1 text-xs font-semibold text-white hover:bg-brand-700 disabled:opacity-50"
              >
                {busy === "email" ? "Sending…" : "Send"}
              </button>
            </div>
            <input
              value={emailNote}
              onChange={(e) => setEmailNote(e.target.value)}
              placeholder="Optional note for the email body…"
              className="mt-1 w-full rounded-lg border border-gray-300 px-2 py-1.5 text-xs focus:border-brand-500 focus:outline-none"
            />
            {emailMsg && <div className="mt-1 text-xs text-brand-600">{emailMsg}</div>}
          </div>

          <button onClick={() => setResult(null)} className="text-xs text-gray-400 hover:text-brand-600">
            Start over
          </button>
        </div>
      )}

      {err && <div className="mt-2 text-xs text-red-600">{err}</div>}
    </div>
  );
}
