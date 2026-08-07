"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { format, parseISO } from "date-fns";
import { supabase } from "@/lib/supabase";
import { useProfiles } from "@/hooks/useProfiles";
import type { Project, TaskPriority } from "@/lib/types";

interface WatcherRule {
  id: string;
  name: string;
  description: string | null;
  check_sql: string;
  project_id: string;
  assignee_id: string | null;
  priority: TaskPriority;
  active: boolean;
  last_run_at: string | null;
}

export default function AdminWatchRules() {
  const qc = useQueryClient();
  const { data: profiles } = useProfiles();
  const [editing, setEditing] = useState<Partial<WatcherRule> | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const { data: rules } = useQuery({
    queryKey: ["watcher_rules"],
    queryFn: async () => {
      const { data, error } = await supabase.from("watcher_rules").select("*").order("name");
      if (error) throw error;
      return data as WatcherRule[];
    },
  });

  const { data: projects } = useQuery({
    queryKey: ["projects"],
    queryFn: async () => {
      const { data, error } = await supabase.from("projects").select("*").eq("archived", false).order("name");
      if (error) throw error;
      return data as Project[];
    },
  });

  async function toggle(r: WatcherRule) {
    await supabase.from("watcher_rules").update({ active: !r.active }).eq("id", r.id);
    qc.invalidateQueries({ queryKey: ["watcher_rules"] });
  }

  async function save(e: React.FormEvent) {
    e.preventDefault();
    if (!editing) return;
    setErr(null);
    try {
      const row = {
        name: editing.name,
        description: editing.description || null,
        check_sql: editing.check_sql,
        project_id: editing.project_id,
        assignee_id: editing.assignee_id || null,
        priority: editing.priority ?? "high",
        active: editing.active ?? true,
      };
      const q = editing.id
        ? supabase.from("watcher_rules").update(row).eq("id", editing.id)
        : supabase.from("watcher_rules").insert(row);
      const { error } = await q;
      if (error) throw error;
      qc.invalidateQueries({ queryKey: ["watcher_rules"] });
      setEditing(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  const inputCls =
    "w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none";

  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <p className="max-w-lg text-sm text-gray-500">
          Rules run daily at 6:00 AM over the dataSights lake (mart views). Rows returned = a breach;
          one task is raised per rule per month with AI-written context.
        </p>
        <button
          onClick={() => setEditing({ priority: "high", active: true })}
          className="shrink-0 rounded-lg bg-brand-600 px-3 py-1.5 text-sm font-semibold text-white hover:bg-brand-700"
        >
          + New rule
        </button>
      </div>

      <div className="space-y-2">
        {rules?.map((r) => (
          <div key={r.id} className={`rounded-lg border border-gray-200 bg-white px-4 py-3 ${r.active ? "" : "opacity-50"}`}>
            <div className="flex items-center justify-between gap-3">
              <button onClick={() => setEditing(r)} className="text-left text-sm font-semibold hover:text-brand-700">
                {r.name}
              </button>
              <div className="flex items-center gap-3 text-xs">
                {r.last_run_at && (
                  <span className="text-gray-400">last run {format(parseISO(r.last_run_at), "d MMM HH:mm")}</span>
                )}
                <button
                  onClick={() => toggle(r)}
                  className={`font-semibold ${r.active ? "text-brand-600" : "text-gray-400"}`}
                >
                  {r.active ? "Active" : "Paused"}
                </button>
              </div>
            </div>
            {r.description && <div className="mt-1 text-xs text-gray-500">{r.description}</div>}
          </div>
        ))}
      </div>

      {editing && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4" onClick={() => setEditing(null)}>
          <div className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-xl bg-white p-6 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <h2 className="mb-4 text-lg font-bold">{editing.id ? "Edit rule" : "New rule"}</h2>
            <form onSubmit={save} className="space-y-3">
              <input required placeholder="Rule name" value={editing.name ?? ""} onChange={(e) => setEditing({ ...editing, name: e.target.value })} className={inputCls} />
              <textarea placeholder="Description (shown to the AI when writing the task)" value={editing.description ?? ""} onChange={(e) => setEditing({ ...editing, description: e.target.value })} rows={2} className={inputCls} />
              <textarea
                required
                placeholder="DuckDB SQL over the mart views (rows returned = breach)"
                value={editing.check_sql ?? ""}
                onChange={(e) => setEditing({ ...editing, check_sql: e.target.value })}
                rows={8}
                className={`${inputCls} font-mono text-xs`}
              />
              <div className="grid grid-cols-3 gap-3">
                <select required value={editing.project_id ?? ""} onChange={(e) => setEditing({ ...editing, project_id: e.target.value })} className={inputCls}>
                  <option value="">Project…</option>
                  {projects?.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
                </select>
                <select value={editing.assignee_id ?? ""} onChange={(e) => setEditing({ ...editing, assignee_id: e.target.value || null })} className={inputCls}>
                  <option value="">Assignee…</option>
                  {profiles?.map((p) => <option key={p.id} value={p.id}>{p.full_name || p.email}</option>)}
                </select>
                <select value={editing.priority ?? "high"} onChange={(e) => setEditing({ ...editing, priority: e.target.value as TaskPriority })} className={inputCls}>
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                  <option value="critical">Critical</option>
                </select>
              </div>
              {err && <div className="text-sm text-red-600">{err}</div>}
              <div className="flex justify-end gap-2 pt-1">
                <button type="button" onClick={() => setEditing(null)} className="rounded-lg px-4 py-2 text-sm text-gray-600 hover:bg-gray-100">Cancel</button>
                <button type="submit" className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700">Save rule</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
