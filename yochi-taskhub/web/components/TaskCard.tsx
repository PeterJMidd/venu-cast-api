"use client";

import { useRouter, usePathname, useSearchParams } from "next/navigation";
import { format, isBefore, parseISO, startOfDay } from "date-fns";
import type { Profile, Task } from "@/lib/types";
import { PRIORITY_LABELS } from "@/lib/types";
import { profileName } from "@/hooks/useProfiles";

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

  return (
    <button
      onClick={() => openTask(task.id)}
      className="w-full rounded-lg border border-gray-200 bg-white p-3 text-left shadow-sm transition hover:border-brand-500 hover:shadow"
    >
      <div className="mb-1.5 text-sm font-medium leading-snug">{task.title}</div>
      <div className="flex flex-wrap items-center gap-2 text-[11px]">
        <span className={`rounded px-1.5 py-0.5 font-medium ${PRIORITY_COLORS[task.priority]}`}>
          {PRIORITY_LABELS[task.priority]}
        </span>
        {task.due_date && (
          <span className={overdue ? "font-semibold text-red-600" : "text-gray-500"}>
            {format(parseISO(task.due_date), "d MMM")}
          </span>
        )}
        {task.assignee_id && (
          <span className="ml-auto truncate text-gray-400">
            {profileName(profiles, task.assignee_id)}
          </span>
        )}
      </div>
    </button>
  );
}
