"use client";

import { Suspense } from "react";
import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/lib/supabase";
import { STATUS_LABELS, type Task } from "@/lib/types";
import { format, isBefore, isToday, parseISO, startOfDay } from "date-fns";
import { useOpenTask } from "@/components/TaskCard";
import { useRealtimeTasks } from "@/hooks/useRealtimeTasks";

function TaskRow({ task }: { task: Task }) {
  const openTask = useOpenTask();
  const overdue =
    task.due_date &&
    task.status !== "done" &&
    isBefore(parseISO(task.due_date), startOfDay(new Date())) &&
    !isToday(parseISO(task.due_date));
  return (
    <button
      onClick={() => openTask(task.id)}
      className="flex w-full items-center justify-between rounded-lg border border-gray-200 bg-white px-4 py-2.5 text-left hover:border-brand-500"
    >
      <div className="min-w-0">
        <div className="truncate text-sm font-medium">{task.title}</div>
        <div className="text-xs text-gray-400">{STATUS_LABELS[task.status]}</div>
      </div>
      {task.due_date && (
        <div className={`ml-4 shrink-0 text-xs ${overdue ? "font-semibold text-red-600" : "text-gray-500"}`}>
          {format(parseISO(task.due_date), "d MMM")}
        </div>
      )}
    </button>
  );
}

function MyTasksInner() {
  useRealtimeTasks();
  const { data: tasks, isLoading } = useQuery({
    queryKey: ["my-tasks"],
    queryFn: async () => {
      const { data: sess } = await supabase.auth.getSession();
      const uid = sess.session!.user.id;
      const { data, error } = await supabase
        .from("tasks")
        .select("*")
        .or(`assignee_id.eq.${uid},reviewer_id.eq.${uid}`)
        .neq("status", "done")
        .order("due_date", { ascending: true, nullsFirst: false });
      if (error) throw error;
      return data as Task[];
    },
  });

  const today = startOfDay(new Date());
  const overdue = tasks?.filter((t) => t.due_date && isBefore(parseISO(t.due_date), today)) ?? [];
  const dueToday = tasks?.filter((t) => t.due_date && isToday(parseISO(t.due_date))) ?? [];
  const upcoming =
    tasks?.filter(
      (t) => !t.due_date || (!isBefore(parseISO(t.due_date), today) && !isToday(parseISO(t.due_date)))
    ) ?? [];

  return (
    <div className="mx-auto max-w-3xl px-6 py-8">
      <h1 className="mb-4 text-xl font-bold">My tasks</h1>
      <div className="mb-6 grid grid-cols-3 gap-3">
        <div className="rounded-xl border border-gray-200 bg-white p-3">
          <div className="text-xl font-bold">{tasks?.length ?? "–"}</div>
          <div className="text-xs text-gray-500">Open</div>
        </div>
        <div className="rounded-xl border border-gray-200 bg-white p-3">
          <div className={`text-xl font-bold ${overdue.length ? "text-red-600" : "text-brand-600"}`}>{overdue.length}</div>
          <div className="text-xs text-gray-500">Overdue</div>
        </div>
        <div className="rounded-xl border border-gray-200 bg-white p-3">
          <div className={`text-xl font-bold ${dueToday.length ? "text-amber-600" : ""}`}>{dueToday.length}</div>
          <div className="text-xs text-gray-500">Due today</div>
        </div>
      </div>
      {isLoading && <div className="text-sm text-gray-400">Loading…</div>}
      {!isLoading && tasks?.length === 0 && (
        <div className="rounded-lg border border-dashed border-gray-300 p-8 text-center text-sm text-gray-400">
          Nothing on your plate. 🎉
        </div>
      )}
      {overdue.length > 0 && (
        <section className="mb-6">
          <h2 className="mb-2 text-sm font-semibold text-red-600">Overdue ({overdue.length})</h2>
          <div className="space-y-2">{overdue.map((t) => <TaskRow key={t.id} task={t} />)}</div>
        </section>
      )}
      {dueToday.length > 0 && (
        <section className="mb-6">
          <h2 className="mb-2 text-sm font-semibold text-amber-600">Due today</h2>
          <div className="space-y-2">{dueToday.map((t) => <TaskRow key={t.id} task={t} />)}</div>
        </section>
      )}
      {upcoming.length > 0 && (
        <section>
          <h2 className="mb-2 text-sm font-semibold text-gray-500">Upcoming</h2>
          <div className="space-y-2">{upcoming.map((t) => <TaskRow key={t.id} task={t} />)}</div>
        </section>
      )}
    </div>
  );
}

export default function MyTasksPage() {
  return (
    <Suspense>
      <MyTasksInner />
    </Suspense>
  );
}
