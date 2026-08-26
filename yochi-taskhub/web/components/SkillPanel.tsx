"use client";

// The saved routine on a task: describe what it should do in plain English,
// see the proposed plan (brief + validated data queries), save it, and from
// then on it is a button - run on demand or on a schedule, delivering to the
// chosen emails in the chosen formats, refinable in plain English. Distinct
// from the Task agent (one-off plan/execute) and Research (questions): this
// is repeatable, parameterised work.
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { formatDistanceToNow, parseISO } from "date-fns";
import { callFn } from "@/lib/fn";

type Query = { label: string; sql: string };
type Skill = {
  id: string; name: string; prompt: string; data_queries: Query[];
  cadence: string; weekday: number | null;
  recipients: string | null; cc: string | null; formats: string | null;
  ask: string | null; active: boolean; version: number;
  last_run_at: string | null;
  last_result: { headline?: string; sent_to?: string[]; pulls?: Record<string, unknown>;
                 formats?: string[] } | null;
};
type Proposal = {
  task_id: string; ask: string; name: string; prompt: string;
  data_queries: Query[]; validation: { label: string; ok: boolean; error?: string }[];
  cadence: string; weekday: number | null; recipients: string;
  invalid_recipients: string[]; formats: string;
};

const FORMATS = ["pdf", "xlsx", "csv"] as const;
const CADENCES = [
  ["on-demand", "On demand only"],
  ["daily", "Daily"],
  ["weekly", "Weekly"],
  ["monthly", "Monthly"],
] as const;

export default function SkillPanel({ taskId }: { taskId: string }) {
  const qc = useQueryClient();
  const [ask, setAsk] = useState("");
  const [recipients, setRecipients] = useState("");
  const [cc, setCc] = useState("");
  const [formats, setFormats] = useState<string[]>(["pdf"]);
  const [cadence, setCadence] = useState("on-demand");
  const [proposal, setProposal] = useState<Proposal | null>(null);
  const [refining, setRefining] = useState(false);
  const [feedback, setFeedback] = useState("");
  const [runResult, setRunResult] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [editTo, setEditTo] = useState("");
  const [editCc, setEditCc] = useState("");
  const [editCadence, setEditCadence] = useState("on-demand");

  const { data } = useQuery({
    queryKey: ["task", taskId, "skill"],
    queryFn: () => callFn<{ skill: Skill | null }>("task_skill",
                                                   { task_id: taskId, action: "get" }),
  });
  const skill = data?.skill;

  const invalidate = () => qc.invalidateQueries({ queryKey: ["task", taskId, "skill"] });

  const propose = useMutation({
    mutationFn: () => callFn<Proposal>("task_skill", {
      task_id: taskId, action: "propose", ask,
      recipients, formats: formats.join(","), cadence }),
    onSuccess: (p) => setProposal(p),
  });
  const saveIt = useMutation({
    mutationFn: () => callFn("task_skill", { task_id: taskId, action: "save",
                                             proposal: { ...proposal, cc } }),
    onSuccess: () => { setProposal(null); setAsk(""); invalidate(); },
  });
  const run = useMutation({
    mutationFn: () => callFn<{ headline: string; sent_to: string[] }>(
      "task_skill", { task_id: taskId, action: "run" }),
    onSuccess: (r) => {
      setRunResult("%HEAD% — sent to %TO%"
        .replace("%HEAD%", r.headline || "done")
        .replace("%TO%", r.sent_to?.length ? r.sent_to.join(", ") : "no one (no recipients saved)"));
      invalidate();
      qc.invalidateQueries({ queryKey: ["comments", taskId] });
    },
  });
  const saveSettings = useMutation({
    mutationFn: () => callFn("task_skill", {
      task_id: taskId, action: "settings",
      recipients: editTo, cc: editCc, cadence: editCadence }),
    onSuccess: () => { setEditing(false); invalidate(); },
  });
  const refine = useMutation({
    mutationFn: () => callFn("task_skill", { task_id: taskId, action: "refine",
                                             feedback }),
    onSuccess: () => { setRefining(false); setFeedback(""); invalidate(); },
  });

  const err = (m: unknown) =>
    m instanceof Error ? m.message : m ? String(m) : null;

  // ---------- no routine yet: the ask box ----------
  if (!skill && !proposal) {
    return (
      <div className="rounded-xl border border-violet-200 bg-violet-50/40 p-3">
        <div className="mb-1 flex items-center gap-2">
          <span className="text-xs font-semibold text-violet-700">⚡ Saved routine</span>
        </div>
        <p className="mb-2 text-[11px] text-gray-500">
          Describe a repeatable analysis for this task — it becomes a button you can
          run any time (or on a schedule), delivering the output to whoever needs it.
        </p>
        <textarea
          value={ask}
          onChange={(e) => setAsk(e.target.value)}
          rows={2}
          placeholder="e.g. Check Xero across all entities for overdue supplier invoices, grouped by entity and supplier, oldest first"
          className="mb-2 w-full rounded-lg border border-gray-300 px-3 py-2 text-[13px]"
        />
        <div className="mb-2 grid gap-2 sm:grid-cols-2">
          <input
            value={recipients}
            onChange={(e) => setRecipients(e.target.value)}
            placeholder="To (emails, comma-separated)"
            className="rounded-lg border border-gray-300 px-3 py-1.5 text-[12px]"
          />
          <input
            value={cc}
            onChange={(e) => setCc(e.target.value)}
            placeholder="CC (optional)"
            className="rounded-lg border border-gray-300 px-3 py-1.5 text-[12px]"
          />
          <div className="flex items-center gap-3">
            {FORMATS.map((f) => (
              <label key={f} className="flex items-center gap-1 text-[12px] text-gray-700">
                <input type="checkbox" checked={formats.includes(f)}
                  onChange={(e) => setFormats(e.target.checked
                    ? [...formats, f] : formats.filter((x) => x !== f))} />
                {f === "xlsx" ? "Excel" : f.toUpperCase()}
              </label>
            ))}
            <select value={cadence} onChange={(e) => setCadence(e.target.value)}
                    className="ml-auto rounded-lg border border-gray-300 px-2 py-1 text-[12px]">
              {CADENCES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </div>
        </div>
        <button
          onClick={() => propose.mutate()}
          disabled={!ask.trim() || propose.isPending}
          className="rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-violet-700 disabled:opacity-50"
        >
          {propose.isPending ? "Designing & validating…" : "Propose the routine"}
        </button>
        {err(propose.error) && (
          <div className="mt-2 text-[11px] text-red-600">{err(propose.error)}</div>
        )}
      </div>
    );
  }

  // ---------- proposal review ----------
  if (proposal) {
    const allValid = (proposal.validation || []).every((v) => v.ok);
    return (
      <div className="rounded-xl border border-violet-300 bg-violet-50/60 p-3">
        <div className="mb-1 text-xs font-semibold text-violet-700">
          ⚡ Proposed routine — review before saving
        </div>
        <div className="mb-1 text-[13px] font-semibold">{proposal.name}</div>
        <div className="mb-2 max-h-36 overflow-y-auto whitespace-pre-wrap rounded-lg bg-white/70 p-2 text-[12px] text-gray-700">
          {proposal.prompt}
        </div>
        <div className="mb-2 text-[11px] text-gray-600">
          {(proposal.data_queries || []).map((q, i) => {
            const v = proposal.validation?.[i];
            return (
              <div key={i}>
                {v?.ok ? "✓" : "✗"} {q.label}
                {v && !v.ok && <span className="text-red-600"> — {v.error}</span>}
              </div>
            );
          })}
        </div>
        <div className="mb-2 text-[11px] text-gray-500">
          {proposal.recipients ? <>To: {proposal.recipients} · </> : "No recipients · "}
          {proposal.formats.toUpperCase()} · {proposal.cadence}
          {proposal.invalid_recipients?.length ? (
            <span className="text-red-600"> · not emails: {proposal.invalid_recipients.join(", ")}</span>
          ) : null}
        </div>
        <div className="flex gap-2">
          <button onClick={() => saveIt.mutate()}
                  disabled={!allValid || saveIt.isPending}
                  className="rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-50">
            {saveIt.isPending ? "Saving…" : "Save routine"}
          </button>
          <button onClick={() => setProposal(null)}
                  className="rounded-lg px-3 py-1.5 text-xs text-gray-600 hover:bg-gray-100">
            Start again
          </button>
        </div>
        {!allValid && (
          <div className="mt-1 text-[11px] text-amber-700">
            A query failed validation — reword the ask and propose again.
          </div>
        )}
        {err(saveIt.error) && (
          <div className="mt-2 text-[11px] text-red-600">{err(saveIt.error)}</div>
        )}
      </div>
    );
  }

  // ---------- saved: the button ----------
  const s = skill!;
  return (
    <div className="rounded-xl border border-violet-200 bg-violet-50/40 p-3">
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="text-xs font-semibold text-violet-700">
          ⚡ {s.name} <span className="font-normal text-gray-400">v{s.version}</span>
        </span>
        <button
          onClick={() => run.mutate()}
          disabled={run.isPending || !s.active}
          className="rounded-lg bg-violet-600 px-4 py-1.5 text-xs font-bold text-white hover:bg-violet-700 disabled:opacity-50"
        >
          {run.isPending ? "Running…" : "▶ Run now"}
        </button>
      </div>
      <div className="text-[11px] text-gray-500">
        {s.recipients ? <>To {s.recipients}{s.cc ? <> · cc {s.cc}</> : null} · </> : "No recipients · "}
        {(s.formats || "pdf").toUpperCase()} ·{" "}
        {s.cadence === "on-demand" ? "on demand" : s.cadence}
        {s.last_run_at && (
          <> · last ran {formatDistanceToNow(parseISO(s.last_run_at), { addSuffix: true })}</>
        )}
      </div>
      {s.last_result?.headline && !runResult && (
        <div className="mt-1 rounded-lg bg-white/70 px-2 py-1 text-[12px] text-gray-700">
          {s.last_result.headline}
        </div>
      )}
      {runResult && (
        <div className="mt-1 rounded-lg bg-green-50 px-2 py-1 text-[12px] text-green-800">
          {runResult}
        </div>
      )}
      {err(run.error) && (
        <div className="mt-1 text-[11px] text-red-600">{err(run.error)}</div>
      )}
      <div className="mt-2 flex items-center gap-3 text-[11px]">
        <button onClick={() => setRefining(!refining)}
                className="text-violet-700 hover:underline">
          Refine…
        </button>
        <span className="text-gray-300">·</span>
        <button
          onClick={() => {
            setEditTo(s.recipients || "");
            setEditCc(s.cc || "");
            setEditCadence(s.cadence || "on-demand");
            setEditing(!editing);
          }}
          className="text-violet-700 hover:underline"
        >
          Delivery &amp; schedule…
        </button>
        <span className="text-gray-300">·</span>
        <span className="text-gray-400">
          Full result lands as a comment; the approach compounds on the Approach tab
        </span>
      </div>
      {editing && (
        <div className="mt-2 rounded-lg border border-violet-200 bg-white/70 p-2">
          <div className="mb-1 grid gap-1.5 sm:grid-cols-2">
            <input value={editTo} onChange={(e) => setEditTo(e.target.value)}
                   placeholder="To (emails, comma-separated)"
                   className="rounded-lg border border-gray-300 px-2 py-1 text-[12px]" />
            <input value={editCc} onChange={(e) => setEditCc(e.target.value)}
                   placeholder="CC (optional)"
                   className="rounded-lg border border-gray-300 px-2 py-1 text-[12px]" />
          </div>
          <div className="flex items-center gap-2">
            <select value={editCadence} onChange={(e) => setEditCadence(e.target.value)}
                    className="rounded-lg border border-gray-300 px-2 py-1 text-[12px]">
              {CADENCES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
            <button onClick={() => saveSettings.mutate()}
                    disabled={saveSettings.isPending}
                    className="rounded-lg bg-violet-600 px-3 py-1 text-[11px] font-semibold text-white disabled:opacity-50">
              {saveSettings.isPending ? "Saving…" : "Save delivery & schedule"}
            </button>
          </div>
          {err(saveSettings.error) && (
            <div className="mt-1 text-[11px] text-red-600">{err(saveSettings.error)}</div>
          )}
        </div>
      )}
      {refining && (
        <div className="mt-2">
          <textarea
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            rows={2}
            placeholder="Tell it what to change — e.g. also show invoices over 60 days as their own section"
            className="mb-1 w-full rounded-lg border border-gray-300 px-3 py-2 text-[12px]"
          />
          <button onClick={() => refine.mutate()}
                  disabled={!feedback.trim() || refine.isPending}
                  className="rounded-lg bg-violet-600 px-3 py-1 text-[11px] font-semibold text-white disabled:opacity-50">
            {refine.isPending ? "Refining & re-validating…" : "Apply refinement"}
          </button>
          {err(refine.error) && (
            <div className="mt-1 text-[11px] text-red-600">{err(refine.error)}</div>
          )}
        </div>
      )}
    </div>
  );
}
