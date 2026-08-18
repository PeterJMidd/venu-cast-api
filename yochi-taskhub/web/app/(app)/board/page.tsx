"use client";

import { Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  DndContext,
  DragOverlay,
  PointerSensor,
  useDroppable,
  useSensor,
  useSensors,
  type DragEndEvent,
  type DragStartEvent,
} from "@dnd-kit/core";
import { SortableContext, useSortable, verticalListSortingStrategy } from "@dnd-kit/sortable";
import { supabase } from "@/lib/supabase";
import { updateTask } from "@/lib/mutations";
import { useProfile } from "@/hooks/useProfile";
import { useProfiles } from "@/hooks/useProfiles";
import { useRealtimeTasks } from "@/hooks/useRealtimeTasks";
import TaskCard from "@/components/TaskCard";
import NewTaskModal from "@/components/NewTaskModal";
import BatchPanel from "@/components/BatchPanel";
import PositionPanel from "@/components/PositionPanel";
import {
  PRIORITY_LABELS,
  STATUS_LABELS,
  TASK_STATUSES,
  type Project,
  type Task,
  type TaskStatus,
} from "@/lib/types";
import { DUE_WINDOWS, PRIORITY_ORDER, filterTasks, type DueWindow } from "@/lib/taskFilters";

function DraggableCard({ task, profiles }: { task: Task; profiles: any }) {
  const { attributes, listeners, setNodeRef, isDragging, transform, transition } = useSortable({ id: task.id });
  return (
    <div
      ref={setNodeRef}
      {...listeners}
      {...attributes}
      style={{
        transform: transform ? `translate3d(${transform.x}px, ${transform.y}px, 0)` : undefined,
        transition,
      }}
      className={isDragging ? "opacity-30" : ""}
    >
      <TaskCard task={task} profiles={profiles} />
    </div>
  );
}

function Column({
  status,
  tasks,
  profiles,
}: {
  status: TaskStatus;
  tasks: Task[];
  profiles: any;
}) {
  const { setNodeRef, isOver } = useDroppable({ id: status });
  return (
    <div
      ref={setNodeRef}
      className={`flex w-64 shrink-0 flex-col rounded-xl p-2 ${
        isOver ? "bg-brand-50" : "bg-gray-100"
      }`}
    >
      <div className="mb-2 flex items-center justify-between px-2 pt-1">
        <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">
          {STATUS_LABELS[status]}
        </span>
        <span className="text-xs text-gray-400">{tasks.length}</span>
      </div>
      <SortableContext items={tasks.map((t) => t.id)} strategy={verticalListSortingStrategy}>
        <div className="flex-1 space-y-2 overflow-y-auto">
          {tasks.map((t) => (
            <DraggableCard key={t.id} task={t} profiles={profiles} />
          ))}
        </div>
      </SortableContext>
    </div>
  );
}

function BoardInner() {
  const router = useRouter();
  const params = useSearchParams();
  const projectId = params.get("project") ?? "";
  const qc = useQueryClient();
  const { data: me } = useProfile();
  const { data: profiles } = useProfiles();
  const [dragTask, setDragTask] = useState<Task | null>(null);
  const [showNew, setShowNew] = useState(false);
  const [search, setSearch] = useState("");
  const [fPriority, setFPriority] = useState("");
  const [fDue, setFDue] = useState<DueWindow>("");
  useRealtimeTasks();

  const isStaff = me?.role === "admin" || me?.role === "finance";
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 6 } }));

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

  const { data: tasks } = useQuery({
    queryKey: ["tasks", "board", projectId],
    enabled: !!projectId,
    queryFn: async () => {
      const { data, error } = await supabase
        .from("tasks")
        .select("*")
        .eq("project_id", projectId)
        .is("parent_id", null)
        .order("sort_order", { ascending: true, nullsFirst: false })
        .order("due_date", { ascending: true, nullsFirst: false });
      if (error) throw error;
      return data as Task[];
    },
  });

  function onDragStart(e: DragStartEvent) {
    setDragTask(tasks?.find((t) => t.id === e.active.id) ?? null);
  }

  async function onDragEnd(e: DragEndEvent) {
    setDragTask(null);
    const taskId = String(e.active.id);
    const overId = e.over?.id ? String(e.over.id) : null;
    if (!overId || !tasks) return;
    const task = tasks.find((t) => t.id === taskId);
    if (!task) return;

    // target column + position: dropped on a column, or on a card within one
    let newStatus: TaskStatus;
    let column: Task[];
    let insertAt: number;
    if (TASK_STATUSES.includes(overId as TaskStatus)) {
      newStatus = overId as TaskStatus;
      column = tasks.filter((t) => t.status === newStatus && t.id !== taskId);
      insertAt = column.length; // append
    } else {
      const overTask = tasks.find((t) => t.id === overId);
      if (!overTask) return;
      newStatus = overTask.status;
      column = tasks.filter((t) => t.status === newStatus && t.id !== taskId);
      insertAt = column.findIndex((t) => t.id === overId);
      if (insertAt < 0) insertAt = column.length;
    }
    if (task.status === newStatus && column[insertAt]?.id === undefined && task.sort_order !== null && insertAt === column.length) {
      // no-op drop at same tail
    }
    const before = column[insertAt - 1]?.sort_order ?? null;
    const after = column[insertAt]?.sort_order ?? null;
    let sort_order: number;
    if (before !== null && after !== null) sort_order = (before + after) / 2;
    else if (before !== null) sort_order = before + 1;
    else if (after !== null) sort_order = after - 1;
    else sort_order = 0;

    qc.setQueryData(["tasks", "board", projectId], (old: Task[] | undefined) =>
      old?.map((t) => (t.id === taskId ? { ...t, status: newStatus, sort_order } : t))
    );
    try {
      await updateTask(taskId, { status: newStatus, sort_order } as Partial<Task>);
    } catch (e) {
      alert(e instanceof Error ? e.message : String(e));
    } finally {
      qc.invalidateQueries({ queryKey: ["tasks"] });
    }
  }

  const project = projects?.find((p) => p.id === projectId);
  const filtering = !!(search.trim() || fPriority || fDue);
  const visible = filterTasks(tasks ?? [], { search, priority: fPriority, due: fDue });

  return (
    <div className="flex h-full flex-col px-6 py-6">
      <div className="mb-4 flex items-center gap-3">
        <select
          value={projectId}
          onChange={(e) => router.push(`/board?project=${e.target.value}`)}
          className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm font-semibold focus:border-brand-500 focus:outline-none"
        >
          <option value="">Select a project…</option>
          {projects?.map((p) => (
            <option key={p.id} value={p.id}>{p.name}</option>
          ))}
        </select>
        {project?.description && (
          <span className="truncate text-sm text-gray-400">{project.description}</span>
        )}
        {isStaff && projectId && (
          <div className="ml-auto flex gap-2">
            <PositionPanel projectId={projectId} />
            <BatchPanel projectId={projectId} />
            <button
              onClick={() => setShowNew(true)}
              className="rounded-lg bg-brand-600 px-3 py-1.5 text-sm font-semibold text-white hover:bg-brand-700"
            >
              + New task
            </button>
          </div>
        )}
      </div>

      {projectId && (
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search this board…"
            className="w-56 rounded-lg border border-gray-300 px-3 py-1.5 text-sm focus:border-brand-500 focus:outline-none"
          />
          <select
            value={fPriority}
            onChange={(e) => setFPriority(e.target.value)}
            className="rounded-lg border border-gray-300 px-2 py-1.5 text-sm focus:border-brand-500 focus:outline-none"
          >
            <option value="">Any importance</option>
            {PRIORITY_ORDER.map((p) => (
              <option key={p} value={p}>{PRIORITY_LABELS[p]}</option>
            ))}
          </select>
          <select
            value={fDue}
            onChange={(e) => setFDue(e.target.value as DueWindow)}
            className="rounded-lg border border-gray-300 px-2 py-1.5 text-sm focus:border-brand-500 focus:outline-none"
          >
            {DUE_WINDOWS.map((w) => (
              <option key={w.key} value={w.key}>{w.label}</option>
            ))}
          </select>
          {filtering && (
            <>
              <span className="text-xs text-gray-500">
                {visible.length} of {tasks?.length ?? 0} shown
              </span>
              <button
                onClick={() => { setSearch(""); setFPriority(""); setFDue(""); }}
                className="text-xs font-semibold text-brand-600 hover:underline"
              >
                Clear
              </button>
            </>
          )}
        </div>
      )}

      {!projectId && (
        <div className="rounded-lg border border-dashed border-gray-300 p-10 text-center text-sm text-gray-400">
          Pick a project to see its board.
        </div>
      )}

      {projectId && (
        <DndContext sensors={sensors} onDragStart={onDragStart} onDragEnd={onDragEnd}>
          <div className="flex flex-1 gap-3 overflow-x-auto pb-2">
            {TASK_STATUSES.map((s) => (
              <Column
                key={s}
                status={s}
                profiles={profiles}
                tasks={visible.filter((t) => t.status === s)}
              />
            ))}
          </div>
          <DragOverlay>{dragTask && <TaskCard task={dragTask} profiles={profiles} />}</DragOverlay>
        </DndContext>
      )}

      {showNew && <NewTaskModal defaultProjectId={projectId} onClose={() => setShowNew(false)} />}
    </div>
  );
}

export default function BoardPage() {
  return (
    <Suspense>
      <BoardInner />
    </Suspense>
  );
}
