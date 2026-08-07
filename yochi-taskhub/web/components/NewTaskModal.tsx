"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/lib/supabase";
import { createTask } from "@/lib/mutations";
import { useProfiles } from "@/hooks/useProfiles";
import type { Project, TaskPriority } from "@/lib/types";

export default function NewTaskModal({
  defaultProjectId,
  onClose,
}: {
  defaultProjectId?: string;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const { data: profiles } = useProfiles();
  const { data: projects } = useQuery({
    queryKey: ["projects"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("projects")
        .select("*")
        .eq("archived", false)
        .order("name");
      if (error) throw error;
      return data as Project[];
    },
  });

  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [projectId, setProjectId] = useState(defaultProjectId ?? "");
  const [assignee, setAssignee] = useState("");
  const [reviewer, setReviewer] = useState("");
  const [due, setDue] = useState("");
  const [priority, setPriority] = useState<TaskPriority>("medium");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!projectId) {
      setError("Pick a project");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await createTask({
        project_id: projectId,
        title,
        description: description || null,
        assignee_id: assignee || null,
        reviewer_id: reviewer || null,
        due_date: due || null,
        priority,
      });
      qc.invalidateQueries({ queryKey: ["tasks"] });
      qc.invalidateQueries({ queryKey: ["my-tasks"] });
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  }

  const inputCls =
    "w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4" onClick={onClose}>
      <div
        className="w-full max-w-lg rounded-xl bg-white p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="mb-4 text-lg font-bold">New task</h2>
        <form onSubmit={submit} className="space-y-3">
          <input autoFocus required placeholder="Task title" value={title} onChange={(e) => setTitle(e.target.value)} className={inputCls} />
          <textarea placeholder="Description (optional)" value={description} onChange={(e) => setDescription(e.target.value)} rows={3} className={inputCls} />
          <div className="grid grid-cols-2 gap-3">
            <select required value={projectId} onChange={(e) => setProjectId(e.target.value)} className={inputCls}>
              <option value="">Project…</option>
              {projects?.map((p) => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
            <select value={priority} onChange={(e) => setPriority(e.target.value as TaskPriority)} className={inputCls}>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
              <option value="critical">Critical</option>
            </select>
            <select value={assignee} onChange={(e) => setAssignee(e.target.value)} className={inputCls}>
              <option value="">Assignee…</option>
              {profiles?.map((p) => (
                <option key={p.id} value={p.id}>{p.full_name || p.email}</option>
              ))}
            </select>
            <select value={reviewer} onChange={(e) => setReviewer(e.target.value)} className={inputCls}>
              <option value="">Reviewer…</option>
              {profiles?.map((p) => (
                <option key={p.id} value={p.id}>{p.full_name || p.email}</option>
              ))}
            </select>
            <input type="date" value={due} onChange={(e) => setDue(e.target.value)} className={inputCls} />
          </div>
          {error && <div className="text-sm text-red-600">{error}</div>}
          <div className="flex justify-end gap-2 pt-2">
            <button type="button" onClick={onClose} className="rounded-lg px-4 py-2 text-sm text-gray-600 hover:bg-gray-100">
              Cancel
            </button>
            <button type="submit" disabled={busy} className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700 disabled:opacity-50">
              {busy ? "Creating…" : "Create task"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
