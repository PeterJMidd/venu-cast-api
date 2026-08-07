"use client";

import { Suspense, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { format, isBefore, parseISO, startOfDay } from "date-fns";
import { supabase } from "@/lib/supabase";
import { useProfile } from "@/hooks/useProfile";
import { useProfiles, profileName } from "@/hooks/useProfiles";
import { useRealtimeTasks } from "@/hooks/useRealtimeTasks";
import { useOpenTask } from "@/components/TaskCard";
import NewTaskModal from "@/components/NewTaskModal";
import {
  PRIORITY_LABELS,
  STATUS_LABELS,
  TASK_STATUSES,
  type Category,
  type Project,
  type Task,
} from "@/lib/types";

function TasksInner() {
  const { data: me } = useProfile();
  const { data: profiles } = useProfiles();
  const openTask = useOpenTask();
  useRealtimeTasks();

  const [search, setSearch] = useState("");
  const [fCategory, setFCategory] = useState("");
  const [fProject, setFProject] = useState("");
  const [fAssignee, setFAssignee] = useState("");
  const [fStatus, setFStatus] = useState("");
  const [overdueOnly, setOverdueOnly] = useState(false);
  const [showNew, setShowNew] = useState(false);

  const isStaff = me?.role === "admin" || me?.role === "finance";

  const { data: categories } = useQuery({
    queryKey: ["categories"],
    queryFn: async () => {
      const { data, error } = await supabase.from("categories").select("*").order("sort");
      if (error) throw error;
      return data as Category[];
    },
    staleTime: Infinity,
  });

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

  const { data: tasks, isLoading } = useQuery({
    queryKey: ["tasks", "all"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("tasks")
        .select("*")
        .order("due_date", { ascending: true, nullsFirst: false })
        .limit(500);
      if (error) throw error;
      return data as Task[];
    },
  });

  const today = startOfDay(new Date());
  const projectById = new Map(projects?.map((p) => [p.id, p]) ?? []);

  const filtered = (tasks ?? []).filter((t) => {
    if (search) {
      const q = search.toLowerCase();
      if (!t.title.toLowerCase().includes(q) && !(t.description ?? "").toLowerCase().includes(q))
        return false;
    }
    if (fProject && t.project_id !== fProject) return false;
    if (fCategory) {
      const proj = projectById.get(t.project_id);
      if (!proj || String(proj.category_id) !== fCategory) return false;
    }
    if (fAssignee && t.assignee_id !== fAssignee) return false;
    if (fStatus && t.status !== fStatus) return false;
    if (overdueOnly) {
      if (!t.due_date || t.status === "done" || !isBefore(parseISO(t.due_date), today)) return false;
    }
    return true;
  });

  const selCls =
    "rounded-lg border border-gray-300 px-2 py-1.5 text-sm focus:border-brand-500 focus:outline-none";

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      <div className="mb-4 flex items-center justify-between">
        <h1 className="text-xl font-bold">All tasks</h1>
        {isStaff && (
          <button
            onClick={() => setShowNew(true)}
            className="rounded-lg bg-brand-600 px-3 py-1.5 text-sm font-semibold text-white hover:bg-brand-700"
          >
            + New task
          </button>
        )}
      </div>

      <div className="mb-4 flex flex-wrap items-center gap-2">
        <input
          placeholder="Search…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className={`${selCls} w-48`}
        />
        <select value={fCategory} onChange={(e) => setFCategory(e.target.value)} className={selCls}>
          <option value="">All categories</option>
          {categories?.map((c) => (
            <option key={c.id} value={String(c.id)}>{c.name}</option>
          ))}
        </select>
        <select value={fProject} onChange={(e) => setFProject(e.target.value)} className={selCls}>
          <option value="">All projects</option>
          {projects?.map((p) => (
            <option key={p.id} value={p.id}>{p.name}</option>
          ))}
        </select>
        <select value={fAssignee} onChange={(e) => setFAssignee(e.target.value)} className={selCls}>
          <option value="">All assignees</option>
          {profiles?.map((p) => (
            <option key={p.id} value={p.id}>{p.full_name || p.email}</option>
          ))}
        </select>
        <select value={fStatus} onChange={(e) => setFStatus(e.target.value)} className={selCls}>
          <option value="">All statuses</option>
          {TASK_STATUSES.map((s) => (
            <option key={s} value={s}>{STATUS_LABELS[s]}</option>
          ))}
        </select>
        <label className="flex items-center gap-1.5 text-sm text-gray-600">
          <input
            type="checkbox"
            checked={overdueOnly}
            onChange={(e) => setOverdueOnly(e.target.checked)}
            className="accent-brand-600"
          />
          Overdue only
        </label>
      </div>

      {isLoading && <div className="text-sm text-gray-400">Loading…</div>}

      <div className="overflow-hidden rounded-xl border border-gray-200 bg-white">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-left text-xs uppercase tracking-wide text-gray-400">
            <tr>
              <th className="px-4 py-2.5">Task</th>
              <th className="px-4 py-2.5">Project</th>
              <th className="px-4 py-2.5">Assignee</th>
              <th className="px-4 py-2.5">Status</th>
              <th className="px-4 py-2.5">Priority</th>
              <th className="px-4 py-2.5">Due</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {filtered.map((t) => {
              const overdue =
                t.due_date && t.status !== "done" && isBefore(parseISO(t.due_date), today);
              return (
                <tr
                  key={t.id}
                  onClick={() => openTask(t.id)}
                  className="cursor-pointer hover:bg-brand-50/40"
                >
                  <td className="px-4 py-2.5 font-medium">
                    {t.parent_id && <span className="mr-1.5 rounded bg-gray-100 px-1 text-[10px] text-gray-500">sub</span>}
                    {t.title}
                  </td>
                  <td className="px-4 py-2.5 text-gray-500">
                    {projectById.get(t.project_id)?.name ?? "—"}
                  </td>
                  <td className="px-4 py-2.5 text-gray-500">
                    {profileName(profiles, t.assignee_id)}
                  </td>
                  <td className="px-4 py-2.5 text-gray-500">{STATUS_LABELS[t.status]}</td>
                  <td className="px-4 py-2.5 text-gray-500">{PRIORITY_LABELS[t.priority]}</td>
                  <td className={`px-4 py-2.5 ${overdue ? "font-semibold text-red-600" : "text-gray-500"}`}>
                    {t.due_date ? format(parseISO(t.due_date), "d MMM yyyy") : "—"}
                  </td>
                </tr>
              );
            })}
            {!isLoading && filtered.length === 0 && (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-gray-300">
                  No tasks match.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {showNew && <NewTaskModal onClose={() => setShowNew(false)} />}
    </div>
  );
}

export default function TasksPage() {
  return (
    <Suspense>
      <TasksInner />
    </Suspense>
  );
}
