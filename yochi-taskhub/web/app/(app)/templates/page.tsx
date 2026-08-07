"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/lib/supabase";
import { useProfile } from "@/hooks/useProfile";
import { useProfiles, profileName } from "@/hooks/useProfiles";
import type { Project, TaskPriority, TaskTemplate } from "@/lib/types";

type DueRuleType = "bd" | "dom";

function describeRule(rule: TaskTemplate["due_rule"] & { month?: string }): string {
  const next = rule.month === "next" ? " (following month)" : "";
  if (rule.type === "bd") {
    const n = rule.n;
    if (n < 0) return `${-n === 1 ? "Last" : `${-n} from last`} business day${next}`;
    return `Business day ${n}${next}`;
  }
  return `Day ${rule.day}${rule.roll ? " (roll fwd)" : ""}${next}`;
}

export default function TemplatesPage() {
  const qc = useQueryClient();
  const { data: me } = useProfile();
  const { data: profiles } = useProfiles();
  const [editing, setEditing] = useState<Partial<TaskTemplate> | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const isStaff = me?.role === "admin" || me?.role === "finance";

  const { data: projects } = useQuery({
    queryKey: ["projects"],
    queryFn: async () => {
      const { data, error } = await supabase.from("projects").select("*").eq("archived", false).order("name");
      if (error) throw error;
      return data as Project[];
    },
  });

  const { data: templates } = useQuery({
    queryKey: ["templates"],
    enabled: isStaff,
    queryFn: async () => {
      const { data, error } = await supabase.from("task_templates").select("*").order("title");
      if (error) throw error;
      return data as TaskTemplate[];
    },
  });

  if (me && !isStaff) {
    return <div className="p-8 text-sm text-gray-400">Finance access required.</div>;
  }

  async function save(e: React.FormEvent) {
    e.preventDefault();
    if (!editing) return;
    setErr(null);
    try {
      const row = {
        project_id: editing.project_id,
        title: editing.title,
        description: editing.description || null,
        cadence: editing.cadence ?? "monthly",
        due_rule: editing.due_rule ?? { type: "bd", n: 3 },
        default_assignee_id: editing.default_assignee_id || null,
        default_reviewer_id: editing.default_reviewer_id || null,
        priority: editing.priority ?? "medium",
        requires_signoff: editing.requires_signoff ?? true,
        active: editing.active ?? true,
      };
      const q = editing.id
        ? supabase.from("task_templates").update(row).eq("id", editing.id)
        : supabase.from("task_templates").insert(row);
      const { error } = await q;
      if (error) throw error;
      qc.invalidateQueries({ queryKey: ["templates"] });
      setEditing(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  async function toggleActive(t: TaskTemplate) {
    await supabase.from("task_templates").update({ active: !t.active }).eq("id", t.id);
    qc.invalidateQueries({ queryKey: ["templates"] });
  }

  const rule = (editing?.due_rule ?? { type: "bd", n: 3 }) as TaskTemplate["due_rule"];
  const ruleType: DueRuleType = rule.type;
  const inputCls =
    "w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none";

  return (
    <div className="mx-auto max-w-5xl px-6 py-8">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold">Recurring templates</h1>
          <p className="mt-1 text-sm text-gray-500">
            Tasks are created automatically at the start of each period (daily engine, 00:15 AEST).
          </p>
        </div>
        <button
          onClick={() => setEditing({ cadence: "monthly", due_rule: { type: "bd", n: 3 }, priority: "medium", requires_signoff: true, active: true })}
          className="rounded-lg bg-brand-600 px-3 py-1.5 text-sm font-semibold text-white hover:bg-brand-700"
        >
          + New template
        </button>
      </div>

      <div className="overflow-hidden rounded-xl border border-gray-200 bg-white">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-left text-xs uppercase tracking-wide text-gray-400">
            <tr>
              <th className="px-4 py-2.5">Template</th>
              <th className="px-4 py-2.5">Project</th>
              <th className="px-4 py-2.5">Cadence</th>
              <th className="px-4 py-2.5">Due</th>
              <th className="px-4 py-2.5">Assignee → Reviewer</th>
              <th className="px-4 py-2.5">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {templates?.map((t) => (
              <tr key={t.id} onClick={() => setEditing(t)} className={`cursor-pointer hover:bg-brand-50/40 ${t.active ? "" : "opacity-50"}`}>
                <td className="px-4 py-2.5 font-medium">{t.title}</td>
                <td className="px-4 py-2.5 text-gray-500">{projects?.find((p) => p.id === t.project_id)?.name ?? "—"}</td>
                <td className="px-4 py-2.5 capitalize text-gray-500">{t.cadence}</td>
                <td className="px-4 py-2.5 text-gray-500">{describeRule(t.due_rule)}</td>
                <td className="px-4 py-2.5 text-gray-500">
                  {profileName(profiles, t.default_assignee_id)} → {profileName(profiles, t.default_reviewer_id)}
                </td>
                <td className="px-4 py-2.5">
                  <button
                    onClick={(e) => { e.stopPropagation(); toggleActive(t); }}
                    className={`text-xs font-semibold ${t.active ? "text-brand-600" : "text-gray-400"}`}
                  >
                    {t.active ? "Active" : "Paused"}
                  </button>
                </td>
              </tr>
            ))}
            {templates?.length === 0 && (
              <tr><td colSpan={6} className="px-4 py-8 text-center text-gray-300">No templates yet — create your month-end close checklist here.</td></tr>
            )}
          </tbody>
        </table>
      </div>

      {editing && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4" onClick={() => setEditing(null)}>
          <div className="max-h-[90vh] w-full max-w-lg overflow-y-auto rounded-xl bg-white p-6 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <h2 className="mb-4 text-lg font-bold">{editing.id ? "Edit template" : "New template"}</h2>
            <form onSubmit={save} className="space-y-3">
              <input required placeholder="Task title" value={editing.title ?? ""} onChange={(e) => setEditing({ ...editing, title: e.target.value })} className={inputCls} />
              <textarea placeholder="Description / checklist notes" value={editing.description ?? ""} onChange={(e) => setEditing({ ...editing, description: e.target.value })} rows={2} className={inputCls} />
              <div className="grid grid-cols-2 gap-3">
                <select required value={editing.project_id ?? ""} onChange={(e) => setEditing({ ...editing, project_id: e.target.value })} className={inputCls}>
                  <option value="">Project…</option>
                  {projects?.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
                </select>
                <select value={editing.cadence ?? "monthly"} onChange={(e) => setEditing({ ...editing, cadence: e.target.value as TaskTemplate["cadence"] })} className={inputCls}>
                  <option value="monthly">Monthly</option>
                  <option value="quarterly">Quarterly (Jan/Apr/Jul/Oct)</option>
                  <option value="annual">Annual (July)</option>
                </select>
                <select value={editing.default_assignee_id ?? ""} onChange={(e) => setEditing({ ...editing, default_assignee_id: e.target.value || null })} className={inputCls}>
                  <option value="">Assignee…</option>
                  {profiles?.map((p) => <option key={p.id} value={p.id}>{p.full_name || p.email}</option>)}
                </select>
                <select value={editing.default_reviewer_id ?? ""} onChange={(e) => setEditing({ ...editing, default_reviewer_id: e.target.value || null })} className={inputCls}>
                  <option value="">Reviewer…</option>
                  {profiles?.map((p) => <option key={p.id} value={p.id}>{p.full_name || p.email}</option>)}
                </select>
                <select value={editing.priority ?? "medium"} onChange={(e) => setEditing({ ...editing, priority: e.target.value as TaskPriority })} className={inputCls}>
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                  <option value="critical">Critical</option>
                </select>
              </div>

              <div className="rounded-lg border border-gray-200 p-3">
                <div className="mb-2 text-xs font-semibold text-gray-500">Due date rule (AU business-day aware)</div>
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <select
                    value={ruleType}
                    onChange={(e) =>
                      setEditing({
                        ...editing,
                        due_rule: {
                          ...(e.target.value === "bd"
                            ? { type: "bd" as const, n: 3 }
                            : { type: "dom" as const, day: 21, roll: "forward" as const }),
                          ...((rule as { month?: string }).month === "next" ? { month: "next" } : {}),
                        } as TaskTemplate["due_rule"],
                      })
                    }
                    className="rounded-lg border border-gray-300 px-2 py-1.5"
                  >
                    <option value="bd">Business day of month</option>
                    <option value="dom">Fixed day of month</option>
                  </select>
                  {ruleType === "bd" ? (
                    <>
                      <span>number</span>
                      <input
                        type="number" min={-10} max={23}
                        value={(rule as { n: number }).n}
                        onChange={(e) =>
                          setEditing({
                            ...editing,
                            due_rule: { ...(rule as object), type: "bd", n: Number(e.target.value) } as TaskTemplate["due_rule"],
                          })
                        }
                        className="w-20 rounded-lg border border-gray-300 px-2 py-1.5"
                      />
                      <span className="text-xs text-gray-400">(-1 = last business day)</span>
                    </>
                  ) : (
                    <>
                      <span>day</span>
                      <input
                        type="number" min={1} max={31}
                        value={(rule as { day: number }).day}
                        onChange={(e) =>
                          setEditing({
                            ...editing,
                            due_rule: { ...(rule as object), type: "dom", day: Number(e.target.value), roll: "forward" } as TaskTemplate["due_rule"],
                          })
                        }
                        className="w-20 rounded-lg border border-gray-300 px-2 py-1.5"
                      />
                      <span className="text-xs text-gray-400">(rolls forward past weekends/holidays)</span>
                    </>
                  )}
                </div>
                <label className="mt-2 flex items-center gap-2 text-xs text-gray-600">
                  <input
                    type="checkbox"
                    checked={(rule as { month?: string }).month === "next"}
                    onChange={(e) => {
                      const r = { ...(rule as Record<string, unknown>) };
                      if (e.target.checked) r.month = "next";
                      else delete r.month;
                      setEditing({ ...editing, due_rule: r as TaskTemplate["due_rule"] });
                    }}
                    className="accent-brand-600"
                  />
                  Due in the <b>following</b> month (close tasks — &ldquo;WD+2&rdquo; style)
                </label>
              </div>

              <label className="flex items-center gap-2 text-sm text-gray-600">
                <input type="checkbox" checked={editing.requires_signoff ?? true} onChange={(e) => setEditing({ ...editing, requires_signoff: e.target.checked })} className="accent-brand-600" />
                Requires preparer + reviewer sign-off
              </label>

              {err && <div className="text-sm text-red-600">{err}</div>}
              <div className="flex justify-end gap-2 pt-1">
                <button type="button" onClick={() => setEditing(null)} className="rounded-lg px-4 py-2 text-sm text-gray-600 hover:bg-gray-100">Cancel</button>
                <button type="submit" className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700">Save template</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
