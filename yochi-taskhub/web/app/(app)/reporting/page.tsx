"use client";

import { Suspense } from "react";
import { useQuery } from "@tanstack/react-query";
import { isBefore, parseISO, startOfDay, format } from "date-fns";
import { supabase } from "@/lib/supabase";
import { useProfiles, profileName } from "@/hooks/useProfiles";
import { useOpenTask } from "@/components/TaskCard";
import type { Category, Period, Project, Task } from "@/lib/types";

function Stat({ label, value, tone }: { label: string; value: string | number; tone?: "bad" | "good" | "warn" }) {
  const color =
    tone === "bad" ? "text-red-600" : tone === "good" ? "text-brand-600" : tone === "warn" ? "text-amber-600" : "text-gray-900";
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4">
      <div className={`text-2xl font-bold ${color}`}>{value}</div>
      <div className="mt-0.5 text-xs text-gray-500">{label}</div>
    </div>
  );
}

function ReportingInner() {
  const { data: profiles } = useProfiles();
  const openTask = useOpenTask();

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
      const { data, error } = await supabase.from("projects").select("*");
      if (error) throw error;
      return data as Project[];
    },
  });

  const { data: periods } = useQuery({
    queryKey: ["periods"],
    queryFn: async () => {
      const { data, error } = await supabase.from("periods").select("*").order("period_month", { ascending: false }).limit(3);
      if (error) throw error;
      return data as Period[];
    },
  });

  const { data: tasks, isLoading } = useQuery({
    queryKey: ["tasks", "reporting"],
    queryFn: async () => {
      const { data, error } = await supabase.from("tasks").select("*").limit(2000);
      if (error) throw error;
      return data as Task[];
    },
  });

  if (isLoading) return <div className="p-8 text-sm text-gray-400">Loading…</div>;

  const today = startOfDay(new Date());
  const all = tasks ?? [];
  const open = all.filter((t) => t.status !== "done");
  const overdue = open.filter((t) => t.due_date && isBefore(parseISO(t.due_date), today));
  const done = all.filter((t) => t.status === "done");
  const doneOnTime = done.filter(
    (t) => !t.due_date || !t.completed_at || t.completed_at.slice(0, 10) <= t.due_date
  );
  const projById = new Map(projects?.map((p) => [p.id, p]) ?? []);

  const byCategory = (categories ?? []).map((c) => {
    const catTasks = all.filter((t) => projById.get(t.project_id)?.category_id === c.id);
    const catOpen = catTasks.filter((t) => t.status !== "done");
    const catOverdue = catOpen.filter((t) => t.due_date && isBefore(parseISO(t.due_date), today));
    return { cat: c, total: catTasks.length, open: catOpen.length, overdue: catOverdue.length };
  });

  const byAssignee = Object.entries(
    open.reduce<Record<string, { open: number; overdue: number }>>((acc, t) => {
      const key = t.assignee_id ?? "unassigned";
      acc[key] = acc[key] ?? { open: 0, overdue: 0 };
      acc[key].open += 1;
      if (t.due_date && isBefore(parseISO(t.due_date), today)) acc[key].overdue += 1;
      return acc;
    }, {})
  ).sort((a, b) => b[1].open - a[1].open);

  const currentPeriod = periods?.[0];
  const periodTasks = currentPeriod ? all.filter((t) => t.period_id === currentPeriod.id) : [];
  const periodDone = periodTasks.filter((t) => t.status === "done");

  return (
    <div className="mx-auto max-w-5xl px-6 py-8">
      <h1 className="mb-6 text-xl font-bold">Reporting</h1>

      <div className="mb-8 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Open tasks" value={open.length} />
        <Stat label="Overdue" value={overdue.length} tone={overdue.length ? "bad" : "good"} />
        <Stat
          label="On-time completion"
          value={done.length ? `${Math.round((100 * doneOnTime.length) / done.length)}%` : "—"}
          tone={done.length && doneOnTime.length / done.length >= 0.85 ? "good" : "warn"}
        />
        <Stat
          label={currentPeriod ? `${currentPeriod.label} close progress` : "Close progress"}
          value={periodTasks.length ? `${periodDone.length}/${periodTasks.length}` : "—"}
        />
      </div>

      <div className="mb-8 grid gap-6 md:grid-cols-2">
        <div className="overflow-hidden rounded-xl border border-gray-200 bg-white">
          <div className="border-b border-gray-100 px-4 py-3 text-sm font-semibold">By section</div>
          <table className="w-full text-sm">
            <tbody className="divide-y divide-gray-100">
              {byCategory.map(({ cat, open: o, overdue: od, total }) => (
                <tr key={cat.id}>
                  <td className="px-4 py-2.5">{cat.name}</td>
                  <td className="px-4 py-2.5 text-right text-gray-500">{o} open</td>
                  <td className={`px-4 py-2.5 text-right ${od ? "font-semibold text-red-600" : "text-gray-300"}`}>
                    {od} overdue
                  </td>
                  <td className="px-4 py-2.5 text-right text-gray-400">{total} total</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="overflow-hidden rounded-xl border border-gray-200 bg-white">
          <div className="border-b border-gray-100 px-4 py-3 text-sm font-semibold">By assignee (open)</div>
          <table className="w-full text-sm">
            <tbody className="divide-y divide-gray-100">
              {byAssignee.map(([uid, s]) => (
                <tr key={uid}>
                  <td className="px-4 py-2.5">{uid === "unassigned" ? "Unassigned" : profileName(profiles, uid)}</td>
                  <td className="px-4 py-2.5 text-right text-gray-500">{s.open} open</td>
                  <td className={`px-4 py-2.5 text-right ${s.overdue ? "font-semibold text-red-600" : "text-gray-300"}`}>
                    {s.overdue} overdue
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {overdue.length > 0 && (
        <div className="overflow-hidden rounded-xl border border-gray-200 bg-white">
          <div className="border-b border-gray-100 px-4 py-3 text-sm font-semibold text-red-600">
            Overdue tasks ({overdue.length})
          </div>
          <table className="w-full text-sm">
            <tbody className="divide-y divide-gray-100">
              {overdue
                .sort((a, b) => (a.due_date ?? "").localeCompare(b.due_date ?? ""))
                .map((t) => (
                  <tr key={t.id} onClick={() => openTask(t.id)} className="cursor-pointer hover:bg-red-50/40">
                    <td className="px-4 py-2.5 font-medium">{t.title}</td>
                    <td className="px-4 py-2.5 text-gray-500">{projById.get(t.project_id)?.name}</td>
                    <td className="px-4 py-2.5 text-gray-500">{profileName(profiles, t.assignee_id)}</td>
                    <td className="px-4 py-2.5 text-right font-semibold text-red-600">
                      {t.due_date && format(parseISO(t.due_date), "d MMM")}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default function ReportingPage() {
  return (
    <Suspense>
      <ReportingInner />
    </Suspense>
  );
}
