"use client";

import { useSearchParams } from "next/navigation";
import TaskDrawer from "@/components/TaskDrawer";

/** Renders the task drawer whenever ?task=<id> is present — works on any page. */
export default function DrawerHost() {
  const params = useSearchParams();
  const taskId = params.get("task");
  if (!taskId) return null;
  return <TaskDrawer taskId={taskId} />;
}
