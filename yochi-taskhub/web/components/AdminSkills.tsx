"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { format, parseISO } from "date-fns";
import { supabase } from "@/lib/supabase";
import { callFn } from "@/lib/fn";
import { useProfiles } from "@/hooks/useProfiles";
import LakeCatalog from "@/components/LakeCatalog";
import type { Project } from "@/lib/types";

interface DataQuery {
  label: string;
  sql: string;
}

interface AiSkill {
  id: string;
  name: string;
  prompt: string;
  data_queries: DataQuery[];
  cadence: "daily" | "weekly" | "monthly";
  weekday: number | null;
  project_id: string;
  assignee_id: string | null;
  email_review: boolean;
  active: boolean;
  last_run_at: string | null;
}

interface ValidationResult {
  label: string;
  ok: boolean;
  sample_rows?: number;
  error?: string;
}

const WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

export default function AdminSkills() {
  const qc = useQueryClient();
  const { data: profiles } = useProfiles();
  const [editing, setEditing] = useState<Partial<AiSkill> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [nlText, setNlText] = useState("");
  const [building, setBuilding] = useState(false);
  const [buildErr, setBuildErr] = useState<string | null>(null);
  const [validation, setValidation] = useState<ValidationResult[] | null>(null);
  const [showCatalog, setShowCatalog] = useState(false);

  const { data: skills } = useQuery({
    queryKey: ["ai_skills"],
    queryFn: async () => {
      const { data, error } = await supabase.from("ai_skills").select("*").order("name");
      if (error) throw error;
      return data as AiSkill[];
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

  async function toggle(s: AiSkill) {
    await supabase.from("ai_skills").update({ active: !s.active }).eq("id", s.id);
    qc.invalidateQueries({ queryKey: ["ai_skills"] });
  }

  async function save(e: React.FormEvent) {
    e.preventDefault();
    if (!editing) return;
    setErr(null);
    try {
      const queries = (editing.data_queries ?? []).filter((q) => q.sql.trim());
      if (!queries.length) throw new Error("Add at least one data query");
      const row = {
        name: editing.name,
        prompt: editing.prompt,
        data_queries: queries,
        cadence: editing.cadence ?? "weekly",
        weekday: editing.cadence === "weekly" ? editing.weekday ?? 0 : null,
        project_id: editing.project_id,
        assignee_id: editing.assignee_id || null,
        email_review: editing.email_review ?? false,
        active: editing.active ?? true,
      };
      const q = editing.id
        ? supabase.from("ai_skills").update(row).eq("id", editing.id)
        : supabase.from("ai_skills").insert(row);
      const { error } = await q;
      if (error) throw error;
      qc.invalidateQueries({ queryKey: ["ai_skills"] });
      setEditing(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  async function buildWithAI(e: React.FormEvent) {
    e.preventDefault();
    if (!nlText.trim()) return;
    setBuilding(true);
    setBuildErr(null);
    setValidation(null);
    try {
      const draft = await callFn<Partial<AiSkill> & { validation: ValidationResult[] }>(
        "build_skill",
        { text: nlText }
      );
      setValidation(draft.validation);
      setEditing({
        name: draft.name,
        prompt: draft.prompt,
        data_queries: draft.data_queries,
        cadence: draft.cadence ?? "weekly",
        weekday: draft.weekday ?? 0,
        email_review: draft.email_review ?? false,
        active: true,
      });
      setNlText("");
    } catch (e) {
      setBuildErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBuilding(false);
    }
  }

  const queries = editing?.data_queries ?? [];
  const inputCls =
    "w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none";

  return (
    <div>
      <form onSubmit={buildWithAI} className="mb-5 rounded-xl border border-brand-100 bg-brand-50/50 p-4">
        <div className="mb-2 text-sm font-semibold text-brand-700">✨ Describe a review in plain English</div>
        <textarea
          value={nlText}
          onChange={(e) => setNlText(e.target.value)}
          placeholder={"e.g. Every Monday morning review Xero invoices for duplicates by vendor and invoice number, email me the result…\nThe AI designs the skill, writes the data queries and tests them against the lake before you save."}
          rows={3}
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
        />
        <div className="mt-2 flex items-center justify-between">
          <span className="text-xs text-gray-500">
            You&rsquo;ll get a draft to review — pick the project and assignee, then save.{" "}
            <button
              type="button"
              onClick={() => setShowCatalog(true)}
              className="font-semibold text-brand-600 hover:underline"
            >
              Browse the data lake ↗
            </button>
          </span>
          <button
            disabled={building || !nlText.trim()}
            className="rounded-lg bg-brand-600 px-4 py-1.5 text-sm font-semibold text-white hover:bg-brand-700 disabled:opacity-50"
          >
            {building ? "Building & testing…" : "Build skill with AI"}
          </button>
        </div>
        {buildErr && <div className="mt-2 text-sm text-red-600">{buildErr}</div>}
      </form>

      {showCatalog && <LakeCatalog onClose={() => setShowCatalog(false)} />}

      <div className="mb-3 flex items-center justify-between">
        <p className="max-w-lg text-sm text-gray-500">
          AI skills are recurring reviews: a brief plus data pulls from the Yo-Chi data lake.
          When due (daily runner, 6:30 AM), the AI writes the review and raises a task with it.
        </p>
        <button
          onClick={() => setEditing({ cadence: "weekly", weekday: 0, data_queries: [{ label: "data", sql: "" }], email_review: false, active: true })}
          className="shrink-0 rounded-lg bg-brand-600 px-3 py-1.5 text-sm font-semibold text-white hover:bg-brand-700"
        >
          + New skill
        </button>
      </div>

      <div className="space-y-2">
        {skills?.map((s) => (
          <div key={s.id} className={`rounded-lg border border-gray-200 bg-white px-4 py-3 ${s.active ? "" : "opacity-50"}`}>
            <div className="flex items-center justify-between gap-3">
              <button onClick={() => setEditing(s)} className="text-left text-sm font-semibold hover:text-brand-700">
                {s.name}
              </button>
              <div className="flex items-center gap-3 text-xs">
                <span className="capitalize text-gray-400">
                  {s.cadence}{s.cadence === "weekly" && s.weekday !== null ? ` · ${WEEKDAYS[s.weekday]}` : ""}
                </span>
                {s.last_run_at && (
                  <span className="text-gray-400">last run {format(parseISO(s.last_run_at), "d MMM")}</span>
                )}
                <button onClick={() => toggle(s)} className={`font-semibold ${s.active ? "text-brand-600" : "text-gray-400"}`}>
                  {s.active ? "Active" : "Paused"}
                </button>
              </div>
            </div>
          </div>
        ))}
        {skills?.length === 0 && (
          <div className="rounded-lg border border-dashed border-gray-300 p-6 text-center text-sm text-gray-400">
            No skills yet.
          </div>
        )}
      </div>

      {editing && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4" onClick={() => setEditing(null)}>
          <div className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-xl bg-white p-6 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <h2 className="mb-4 text-lg font-bold">{editing.id ? "Edit skill" : "New skill"}</h2>
            {validation && (
              <div className="mb-3 rounded-lg bg-gray-50 p-3 text-xs">
                <div className="mb-1 font-semibold text-gray-600">Query validation (tested live against the lake)</div>
                {validation.map((v, i) => (
                  <div key={i} className={v.ok ? "text-brand-700" : "text-red-600"}>
                    {v.ok ? `✓ ${v.label}: ok (${v.sample_rows} sample rows)` : `✗ ${v.label}: ${v.error}`}
                  </div>
                ))}
              </div>
            )}
            <form onSubmit={save} className="space-y-3">
              <input required placeholder="Skill name (e.g. Weekly trading review)" value={editing.name ?? ""} onChange={(e) => setEditing({ ...editing, name: e.target.value })} className={inputCls} />
              <textarea
                required
                placeholder="Review brief for the AI — what to look at, what matters, what to flag"
                value={editing.prompt ?? ""}
                onChange={(e) => setEditing({ ...editing, prompt: e.target.value })}
                rows={4}
                className={inputCls}
              />

              <div className="rounded-lg border border-gray-200 p-3">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-xs font-semibold text-gray-500">Data pulls (DuckDB SQL over the lake)</span>
                  <button
                    type="button"
                    onClick={() => setEditing({ ...editing, data_queries: [...queries, { label: "", sql: "" }] })}
                    className="text-xs text-brand-600 hover:underline"
                  >
                    + add query
                  </button>
                </div>
                <div className="space-y-3">
                  {queries.map((q, i) => (
                    <div key={i} className="space-y-1">
                      <div className="flex gap-2">
                        <input
                          placeholder="Label"
                          value={q.label}
                          onChange={(e) => {
                            const next = [...queries];
                            next[i] = { ...q, label: e.target.value };
                            setEditing({ ...editing, data_queries: next });
                          }}
                          className="flex-1 rounded-lg border border-gray-300 px-2 py-1 text-xs"
                        />
                        <button
                          type="button"
                          onClick={() => setEditing({ ...editing, data_queries: queries.filter((_, j) => j !== i) })}
                          className="text-xs text-gray-300 hover:text-red-500"
                        >
                          remove
                        </button>
                      </div>
                      <textarea
                        placeholder="SELECT …"
                        value={q.sql}
                        onChange={(e) => {
                          const next = [...queries];
                          next[i] = { ...q, sql: e.target.value };
                          setEditing({ ...editing, data_queries: next });
                        }}
                        rows={4}
                        className={`${inputCls} font-mono text-xs`}
                      />
                    </div>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-3 gap-3">
                <select value={editing.cadence ?? "weekly"} onChange={(e) => setEditing({ ...editing, cadence: e.target.value as AiSkill["cadence"] })} className={inputCls}>
                  <option value="daily">Daily</option>
                  <option value="weekly">Weekly</option>
                  <option value="monthly">Monthly (1st bus. day)</option>
                </select>
                {editing.cadence === "weekly" && (
                  <select value={editing.weekday ?? 0} onChange={(e) => setEditing({ ...editing, weekday: Number(e.target.value) })} className={inputCls}>
                    {WEEKDAYS.map((d, i) => <option key={i} value={i}>{d}</option>)}
                  </select>
                )}
                <select required value={editing.project_id ?? ""} onChange={(e) => setEditing({ ...editing, project_id: e.target.value })} className={inputCls}>
                  <option value="">Project…</option>
                  {projects?.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
                </select>
                <select value={editing.assignee_id ?? ""} onChange={(e) => setEditing({ ...editing, assignee_id: e.target.value || null })} className={inputCls}>
                  <option value="">Assignee…</option>
                  {profiles?.map((p) => <option key={p.id} value={p.id}>{p.full_name || p.email}</option>)}
                </select>
              </div>

              <label className="flex items-center gap-2 text-sm text-gray-600">
                <input type="checkbox" checked={editing.email_review ?? false} onChange={(e) => setEditing({ ...editing, email_review: e.target.checked })} className="accent-brand-600" />
                Also email the review to the assignee
              </label>

              {err && <div className="text-sm text-red-600">{err}</div>}
              <div className="flex justify-end gap-2 pt-1">
                <button type="button" onClick={() => setEditing(null)} className="rounded-lg px-4 py-2 text-sm text-gray-600 hover:bg-gray-100">Cancel</button>
                <button type="submit" className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700">Save skill</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
