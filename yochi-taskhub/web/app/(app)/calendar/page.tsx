"use client";

import { Suspense, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  addDays, addMonths, endOfMonth, endOfWeek, format, isSameDay, isSameMonth,
  isToday, parseISO, startOfMonth, startOfWeek,
} from "date-fns";
import { supabase } from "@/lib/supabase";
import { useOpenTask } from "@/components/TaskCard";
import { useRealtimeTasks } from "@/hooks/useRealtimeTasks";
import type { Task } from "@/lib/types";

const PRIORITY_DOT: Record<string, string> = {
  low: "bg-gray-300", medium: "bg-blue-400", high: "bg-amber-400", critical: "bg-red-500",
};

function CalendarInner() {
  const [month, setMonth] = useState(() => startOfMonth(new Date()));
  const openTask = useOpenTask();
  useRealtimeTasks();

  const gridStart = startOfWeek(month, { weekStartsOn: 1 });
  const gridEnd = endOfWeek(endOfMonth(month), { weekStartsOn: 1 });

  const { data: tasks } = useQuery({
    queryKey: ["tasks", "calendar", format(month, "yyyy-MM")],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("tasks")
        .select("*")
        .gte("due_date", format(gridStart, "yyyy-MM-dd"))
        .lte("due_date", format(gridEnd, "yyyy-MM-dd"))
        .order("priority");
      if (error) throw error;
      return data as Task[];
    },
  });

  const days: Date[] = [];
  for (let d = gridStart; d <= gridEnd; d = addDays(d, 1)) days.push(d);

  return (
    <div className="mx-auto max-w-6xl px-6 py-6">
      <div className="mb-4 flex items-center gap-3">
        <h1 className="text-xl font-bold">{format(month, "MMMM yyyy")}</h1>
        <div className="ml-auto flex gap-1">
          <button onClick={() => setMonth(addMonths(month, -1))} className="rounded-lg border border-gray-300 px-3 py-1 text-sm hover:bg-gray-50">←</button>
          <button onClick={() => setMonth(startOfMonth(new Date()))} className="rounded-lg border border-gray-300 px-3 py-1 text-sm hover:bg-gray-50">Today</button>
          <button onClick={() => setMonth(addMonths(month, 1))} className="rounded-lg border border-gray-300 px-3 py-1 text-sm hover:bg-gray-50">→</button>
        </div>
      </div>

      <div className="grid grid-cols-7 gap-px overflow-hidden rounded-xl border border-gray-200 bg-gray-200 text-xs">
        {["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map((d) => (
          <div key={d} className="bg-gray-50 px-2 py-1.5 font-semibold text-gray-500">{d}</div>
        ))}
        {days.map((day) => {
          const dayTasks = tasks?.filter((t) => t.due_date && isSameDay(parseISO(t.due_date), day)) ?? [];
          const open = dayTasks.filter((t) => t.status !== "done");
          return (
            <div
              key={day.toISOString()}
              className={`min-h-[92px] bg-white p-1.5 ${isSameMonth(day, month) ? "" : "opacity-40"}`}
            >
              <div className={`mb-1 text-[11px] font-semibold ${isToday(day) ? "inline-block rounded-full bg-brand-600 px-1.5 text-white" : "text-gray-400"}`}>
                {format(day, "d")}
              </div>
              <div className="space-y-0.5">
                {open.slice(0, 4).map((t) => (
                  <button
                    key={t.id}
                    onClick={() => openTask(t.id)}
                    className="flex w-full items-center gap-1 truncate rounded bg-gray-50 px-1 py-0.5 text-left text-[11px] hover:bg-brand-50"
                    title={t.title}
                  >
                    <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${PRIORITY_DOT[t.priority]}`} />
                    <span className="truncate">{t.title}</span>
                  </button>
                ))}
                {open.length > 4 && <div className="px-1 text-[10px] text-gray-400">+{open.length - 4} more</div>}
                {dayTasks.length > 0 && open.length === 0 && (
                  <div className="px-1 text-[10px] text-brand-600">✓ all done</div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function CalendarPage() {
  return (
    <Suspense>
      <CalendarInner />
    </Suspense>
  );
}
