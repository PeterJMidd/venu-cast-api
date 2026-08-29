"use client";

// Email routing rules: which keywords send an inbound email's task to which
// project, allocated to whom. Applied in position order (lowest first) BEFORE
// the AI classifier - a matching rule is deterministic and the model gets no
// vote. '*' as the keywords makes a rule the catch-all for its mailbox; keep
// catch-alls at a high position so specific keyword rules beat them.
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/lib/supabase";
import { useProfiles, profileName } from "@/hooks/useProfiles";

type Rule = {
  id: string; keywords: string; mailbox: string | null;
  project_id: string; assignee_id: string | null; priority: string | null;
  position: number; active: boolean;
};
type Project = { id: string; name: string };

const MAILBOXES = ["finance@yochi.com.au", "solutions@yochi.com.au",
                   "whsclaims@yochi.com.au", "payroll@yochi.com.au"];

const BLANK = { keywords: "", mailbox: "", project_id: "", assignee_id: "",
                priority: "", position: 100 };

export default function AdminEmailRules() {
  const qc = useQueryClient();
  const { data: profiles } = useProfiles();
  const [draft, setDraft] = useState({ ...BLANK });

  const { data: rules } = useQuery({
    queryKey: ["email-rules"],
    queryFn: async () => {
      const { data, error } = await supabase.from("email_rules")
        .select("*").order("position").order("created_at");
      if (error) throw error;
      return data as Rule[];
    },
  });
  const { data: projects } = useQuery({
    queryKey: ["projects-for-rules"],
    queryFn: async () => {
      const { data, error } = await supabase.from("projects")
        .select("id,name").order("name");
      if (error) throw error;
      return data as Project[];
    },
  });

  const refresh = () => qc.invalidateQueries({ queryKey: ["email-rules"] });

  const add = useMutation({
    mutationFn: async () => {
      const { error } = await supabase.from("email_rules").insert({
        keywords: draft.keywords.trim(),
        mailbox: draft.mailbox || null,
        project_id: draft.project_id,
        assignee_id: draft.assignee_id || null,
        priority: draft.priority || null,
        position: draft.position,
      });
      if (error) throw error;
    },
    onSuccess: () => { setDraft({ ...BLANK }); refresh(); },
  });

  const patch = useMutation({
    mutationFn: async (p: { id: string } & Partial<Rule>) => {
      const { id, ...rest } = p;
      const { error } = await supabase.from("email_rules").update(rest).eq("id", id);
      if (error) throw error;
    },
    onSuccess: refresh,
  });

  const remove = useMutation({
    mutationFn: async (id: string) => {
      const { error } = await supabase.from("email_rules").delete().eq("id", id);
      if (error) throw error;
    },
    onSuccess: refresh,
  });

  const projName = (id: string) =>
    projects?.find((p) => p.id === id)?.name || "?";

  return (
    <div className="space-y-4">
      <p className="text-[13px] text-gray-600">
        Emails dropped from the shared mailboxes become tasks. These rules decide
        where each task lands and who gets it: <b>lowest position wins</b>, keywords
        match anywhere in the subject or body, <code>*</code> matches everything
        (the mailbox&apos;s catch-all — keep those at position 900). Mail no rule
        claims is routed by the AI classifier instead.
      </p>

      <div className="rounded-xl border border-gray-200 bg-white p-3">
        <div className="mb-2 text-xs font-bold uppercase tracking-wide text-gray-400">
          Add a rule
        </div>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          <input value={draft.keywords}
                 onChange={(e) => setDraft({ ...draft, keywords: e.target.value })}
                 placeholder="Keywords, comma-separated (or *)"
                 className="rounded-lg border border-gray-300 px-3 py-1.5 text-[13px]" />
          <select value={draft.mailbox}
                  onChange={(e) => setDraft({ ...draft, mailbox: e.target.value })}
                  className="rounded-lg border border-gray-300 px-2 py-1.5 text-[13px]">
            <option value="">Any mailbox</option>
            {MAILBOXES.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
          <select value={draft.project_id}
                  onChange={(e) => setDraft({ ...draft, project_id: e.target.value })}
                  className="rounded-lg border border-gray-300 px-2 py-1.5 text-[13px]">
            <option value="">Project…</option>
            {(projects || []).map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
          <select value={draft.assignee_id}
                  onChange={(e) => setDraft({ ...draft, assignee_id: e.target.value })}
                  className="rounded-lg border border-gray-300 px-2 py-1.5 text-[13px]">
            <option value="">Default allocation…</option>
            {(profiles || []).filter((p) => p.active !== false).map((p) => (
              <option key={p.id} value={p.id}>{profileName(profiles, p.id)}</option>
            ))}
          </select>
          <select value={draft.priority}
                  onChange={(e) => setDraft({ ...draft, priority: e.target.value })}
                  className="rounded-lg border border-gray-300 px-2 py-1.5 text-[13px]">
            <option value="">Priority: let AI decide</option>
            {["low", "medium", "high", "critical"].map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
          <div className="flex items-center gap-2">
            <input type="number" value={draft.position}
                   onChange={(e) => setDraft({ ...draft, position: parseInt(e.target.value || "100", 10) })}
                   className="w-24 rounded-lg border border-gray-300 px-2 py-1.5 text-[13px]"
                   title="Lower fires first" />
            <button onClick={() => add.mutate()}
                    disabled={!draft.keywords.trim() || !draft.project_id || add.isPending}
                    className="rounded-lg bg-brand-600 px-4 py-1.5 text-xs font-semibold text-white disabled:opacity-50">
              {add.isPending ? "Adding…" : "Add rule"}
            </button>
          </div>
        </div>
        {add.error && (
          <div className="mt-1 text-[11px] text-red-600">
            {add.error instanceof Error ? add.error.message : String(add.error)}
          </div>
        )}
      </div>

      <div className="overflow-x-auto rounded-xl border border-gray-200 bg-white">
        <table className="w-full text-[13px]">
          <thead className="bg-gray-50 text-left text-[11px] uppercase tracking-wide text-gray-500">
            <tr>
              <th className="px-3 py-2">Pos</th>
              <th className="px-3 py-2">Keywords</th>
              <th className="px-3 py-2">Mailbox</th>
              <th className="px-3 py-2">Project</th>
              <th className="px-3 py-2">Allocated to</th>
              <th className="px-3 py-2">Priority</th>
              <th className="px-3 py-2"></th>
            </tr>
          </thead>
          <tbody>
            {(rules || []).map((r) => (
              <tr key={r.id} className={`border-t border-gray-100 ${r.active ? "" : "opacity-40"}`}>
                <td className="px-3 py-2 font-mono text-[11px]">{r.position}</td>
                <td className="px-3 py-2 font-medium">
                  {r.keywords === "*" ? <span className="text-gray-400">* (catch-all)</span> : r.keywords}
                </td>
                <td className="px-3 py-2 text-[12px]">{r.mailbox || <span className="text-gray-400">any</span>}</td>
                <td className="px-3 py-2">{projName(r.project_id)}</td>
                <td className="px-3 py-2">
                  {r.assignee_id ? profileName(profiles, r.assignee_id)
                                 : <span className="text-gray-400">sender/admin</span>}
                </td>
                <td className="px-3 py-2 text-[12px]">{r.priority || <span className="text-gray-400">AI</span>}</td>
                <td className="whitespace-nowrap px-3 py-2 text-right text-[11px]">
                  <button onClick={() => patch.mutate({ id: r.id, active: !r.active })}
                          className="mr-2 text-gray-500 hover:underline">
                    {r.active ? "Disable" : "Enable"}
                  </button>
                  <button onClick={() => remove.mutate(r.id)}
                          className="text-red-500 hover:underline">
                    Delete
                  </button>
                </td>
              </tr>
            ))}
            {!rules?.length && (
              <tr><td colSpan={7} className="px-3 py-6 text-center text-gray-400">
                No rules yet — every email is routed by the AI classifier.
              </td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
