// Shared task search / priority / due-window filtering, so the board and the
// All-tasks list behave identically.
import { endOfMonth, isBefore, parseISO, startOfDay } from "date-fns";
import type { Task, TaskPriority } from "@/lib/types";

export type DueWindow =
  | "" | "overdue" | "today" | "7" | "month" | "30" | "60" | "90" | "90plus" | "none";

export const DUE_WINDOWS: { key: DueWindow; label: string }[] = [
  { key: "", label: "Any due date" },
  { key: "overdue", label: "Overdue" },
  { key: "today", label: "Due today" },
  { key: "7", label: "Within 7 days" },
  { key: "month", label: "Within this month" },
  { key: "30", label: "Within 30 days" },
  { key: "60", label: "Within 60 days" },
  { key: "90", label: "Within 90 days" },
  { key: "90plus", label: "More than 90 days away" },
  { key: "none", label: "No due date" },
];

export const PRIORITY_ORDER: TaskPriority[] = ["critical", "high", "medium", "low"];

function daysAhead(n: number, from: Date) {
  const d = new Date(from);
  d.setDate(d.getDate() + n);
  return d;
}

/** Windows are forward-looking; anything already past its date is "overdue". */
export function matchesDue(task: Task, window: DueWindow, now = new Date()): boolean {
  if (!window) return true;
  const today = startOfDay(now);
  if (!task.due_date) return window === "none";
  if (window === "none") return false;

  const due = startOfDay(parseISO(task.due_date));
  const overdue = isBefore(due, today) && task.status !== "done";
  if (window === "overdue") return overdue;
  if (overdue) return false;              // keep late work out of forward windows

  switch (window) {
    case "today":  return due.getTime() === today.getTime();
    case "7":      return due <= daysAhead(7, today);
    case "month":  return due <= endOfMonth(today);
    case "30":     return due <= daysAhead(30, today);
    case "60":     return due <= daysAhead(60, today);
    case "90":     return due <= daysAhead(90, today);
    case "90plus": return due > daysAhead(90, today);
    default:       return true;
  }
}

export interface TaskFilter {
  search?: string;
  priority?: string;      // "" = any
  due?: DueWindow;
}

export function filterTasks(tasks: Task[], f: TaskFilter, now = new Date()): Task[] {
  const q = (f.search ?? "").trim().toLowerCase();
  return tasks.filter((t) => {
    if (q) {
      const hay = `${t.title} ${t.description ?? ""}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    if (f.priority && t.priority !== f.priority) return false;
    if (!matchesDue(t, (f.due ?? "") as DueWindow, now)) return false;
    return true;
  });
}
