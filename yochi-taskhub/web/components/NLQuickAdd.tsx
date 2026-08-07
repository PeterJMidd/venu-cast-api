"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { callFn } from "@/lib/fn";
import { createTask } from "@/lib/mutations";
import { useProfiles } from "@/hooks/useProfiles";
import type { TaskPriority } from "@/lib/types";

interface ParsedTask {
  title: string;
  description?: string | null;
  project_id: string;
  assignee_email?: string | null;
  reviewer_email?: string | null;
  due_date?: string | null;
  priority: TaskPriority;
  checklist?: string[];
}

export default function NLQuickAdd() {
  const qc = useQueryClient();
  const { data: profiles } = useProfiles();
  const [text, setText] = useState("");
  const [parsed, setParsed] = useState<ParsedTask | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function parse(e: React.FormEvent) {
    e.preventDefault();
    if (!text.trim()) return;
    setBusy(true);
    setErr(null);
    try {
      setParsed(await callFn<ParsedTask>("nl_task", { text }));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function confirm() {
    if (!parsed) return;
    setBusy(true);
    setErr(null);
    try {
      const byEmail = (em?: string | null) =>
        em ? profiles?.find((p) => p.email.toLowerCase() === em.toLowerCase())?.id ?? null : null;
      const { data: sess } = await (await import("@/lib/supabase")).supabase.auth.getSession();
      await createTask({
        project_id: parsed.project_id,
        title: parsed.title,
        description: parsed.description ?? null,
        assignee_id: byEmail(parsed.assignee_email) ?? sess.session?.user.id ?? null,
        reviewer_id: byEmail(parsed.reviewer_email),
        due_date: parsed.due_date ?? null,
        priority: parsed.priority,
      });
      qc.invalidateQueries({ queryKey: ["tasks"] });
      qc.invalidateQueries({ queryKey: ["my-tasks"] });
      setParsed(null);
      setText("");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="relative">
      <form onSubmit={parse}>
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="✨ Type a task in plain English… (e.g. 'chase BAS lodgement by the 21st, Joy to review')"
          className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
        />
      </form>
      {busy && <div className="absolute right-3 top-2 text-xs text-gray-400">thinking…</div>}
      {err && <div className="mt-1 text-xs text-red-600">{err}</div>}

      {parsed && (
        <div className="absolute left-0 right-0 top-11 z-30 rounded-xl border border-gray-200 bg-white p-4 shadow-xl">
          <div className="mb-1 text-sm font-semibold">{parsed.title}</div>
          {parsed.description && (
            <div className="mb-2 text-xs text-gray-500">{parsed.description}</div>
          )}
          <div className="mb-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-600">
            <span>Assignee: <b>{parsed.assignee_email ?? "me"}</b></span>
            {parsed.reviewer_email && <span>Reviewer: <b>{parsed.reviewer_email}</b></span>}
            {parsed.due_date && <span>Due: <b>{parsed.due_date}</b></span>}
            <span>Priority: <b className="capitalize">{parsed.priority}</b></span>
          </div>
          <div className="flex justify-end gap-2">
            <button
              onClick={() => setParsed(null)}
              className="rounded-lg px-3 py-1.5 text-xs text-gray-500 hover:bg-gray-100"
            >
              Discard
            </button>
            <button
              onClick={confirm}
              disabled={busy}
              className="rounded-lg bg-brand-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-brand-700 disabled:opacity-50"
            >
              Create task
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
