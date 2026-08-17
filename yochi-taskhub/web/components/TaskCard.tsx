"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useRouter, usePathname, useSearchParams } from "next/navigation";
import { format, isBefore, parseISO, startOfDay } from "date-fns";
import type { Profile, Task } from "@/lib/types";
import { PRIORITY_LABELS } from "@/lib/types";
import { profileName } from "@/hooks/useProfiles";
import { updateTask } from "@/lib/mutations";

const PRIORITY_COLORS: Record<string, string> = {
  low: "bg-gray-100 text-gray-600",
  medium: "bg-blue-50 text-blue-700",
  high: "bg-amber-50 text-amber-700",
  critical: "bg-red-50 text-red-700",
};

export function useOpenTask() {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  return (taskId: string | null) => {
    const p = new URLSearchParams(params.toString());
    if (taskId) p.set("task", taskId);
    else p.delete("task");
    router.push(`${pathname}?${p.toString()}`, { scroll: false });
  };
}

/** One-click complete/reopen circle. Reused on cards and in the drawer. */
export function CompleteToggle({ task, size = "sm" }: { task: Task; size?: "sm" | "lg" }) {
  const qc = useQueryClient();
  const [busy, setBusy] = useState(false);
  const done = task.status === "done";
  const dim = size === "lg" ? "h-7 w-7 text-sm" : "h-5 w-5 text-[11px]";

  async function toggle(e: React.MouseEvent) {
    e.stopPropagation();
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    try {
      await updateTask(task.id, { status: done ? "todo" : "done" });
      qc.invalidateQueries();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      alert(
        /review|depend/i.test(msg)
          ? `Can't close yet — ${msg}. Open the task to complete the sign-off or clear the dependency.`
          : msg
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <span
      role="button"
      tabIndex={0}
      title={done ? "Reopen task" : "Mark completed"}
      onClick={toggle}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") toggle(e as unknown as React.MouseEvent); }}
      className={`flex ${dim} shrink-0 cursor-pointer items-center justify-center rounded-full border transition ${
        done
          ? "border-brand-600 bg-brand-600 text-white hover:bg-brand-700"
          : "border-gray-300 text-transparent hover:border-brand-500 hover:text-brand-500"
      } ${busy ? "opacity-50" : ""}`}
    >
      ✓
    </span>
  );
}

export default function TaskCard({
  task,
  profiles,
}: {
  task: Task;
  profiles?: Profile[];
}) {
  const openTask = useOpenTask();
  const overdue =
    task.due_date &&
    task.status !== "done" &&
    isBefore(parseISO(task.due_date), startOfDay(new Date()));
  const done = task.status === "done";

  return (
    // div+role, not <button>: the complete toggle inside would be an illegally nested button
    <div
      role="button"
      tabIndex={0}
      onClick={() => openTask(task.id)}
      onKeyDown={(e) => { if (e.key === "Enter") openTask(task.id); }}
      className="w-full cursor-pointer rounded-lg border border-gray-200 bg-white p-3 text-left shadow-sm transition hover:border-brand-500 hover:shadow"
    >
      <div className="mb-1.5 flex items-start gap-2">
        <CompleteToggle task={task} />
        <span className={`min-w-0 text-sm font-medium leading-snug ${done ? "text-gray-400 line-through" : ""}`}>
          {task.title}
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-2 text-[11px]">
        <span className={`rounded px-1.5 py-0.5 font-medium ${PRIORITY_COLORS[task.priority]}`}>
          {PRIORITY_LABELS[task.priority]}
        </span>
        {task.due_date && (
          <span className={overdue ? "font-semibold text-red-600" : "text-gray-500"}>
            {format(parseISO(task.due_date), "d MMM")}
          </span>
        )}
        {task.recurrence && (
          <span title={`Repeats ${task.recurrence}`} className="text-gray-400">🔁</span>
        )}
        {task.assignee_id && (
          <span className="ml-auto truncate text-gray-400">
            {profileName(profiles, task.assignee_id)}
          </span>
        )}
      </div>
    </div>
  );
}
