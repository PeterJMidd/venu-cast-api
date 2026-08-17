"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { format, parseISO } from "date-fns";
import { supabase } from "@/lib/supabase";
import {
  addApproval,
  addComment,
  addDependency,
  createTask,
  deleteAttachment,
  downloadAttachment,
  notify,
  removeDependency,
  updateTask,
  uploadAttachment,
} from "@/lib/mutations";
import { useProfile } from "@/hooks/useProfile";
import { useProfiles, profileName } from "@/hooks/useProfiles";
import { useOpenTask, CompleteToggle } from "@/components/TaskCard";
import AgentPanel from "@/components/AgentPanel";
import {
  RECURRENCE_LABELS,
  STATUS_LABELS,
  TASK_STATUSES,
  type Approval,
  type Recurrence,
  type Attachment,
  type AuditEntry,
  type ChecklistItem,
  type Comment,
  type Task,
  type TaskPriority,
  type TaskStatus,
} from "@/lib/types";

type Tab = "details" | "comments" | "files" | "activity";

const AUDIT_FIELDS = ["status", "assignee_id", "reviewer_id", "due_date", "priority", "title"];

export default function TaskDrawer({ taskId }: { taskId: string }) {
  const qc = useQueryClient();
  const openTask = useOpenTask();
  const { data: me } = useProfile();
  const { data: profiles } = useProfiles();
  const [tab, setTab] = useState<Tab>("details");
  const [commentText, setCommentText] = useState("");
  const [newCheckItem, setNewCheckItem] = useState("");
  const [depPick, setDepPick] = useState("");
  const [newSubtask, setNewSubtask] = useState("");
  const [viewer, setViewer] = useState<{ title: string; html: string } | null>(null);

  const isStaff = me?.role === "admin" || me?.role === "finance";

  const { data: task } = useQuery({
    queryKey: ["task", taskId],
    queryFn: async () => {
      const { data, error } = await supabase.from("tasks").select("*").eq("id", taskId).single();
      if (error) throw error;
      return data as Task;
    },
  });

  const { data: approvals } = useQuery({
    queryKey: ["task", taskId, "approvals"],
    queryFn: async () => {
      const { data, error } = await supabase.from("approvals").select("*").eq("task_id", taskId);
      if (error) throw error;
      return data as Approval[];
    },
  });

  const { data: comments } = useQuery({
    queryKey: ["task", taskId, "comments"],
    enabled: tab === "comments",
    queryFn: async () => {
      const { data, error } = await supabase
        .from("comments")
        .select("*")
        .eq("task_id", taskId)
        .order("created_at");
      if (error) throw error;
      return data as Comment[];
    },
  });

  const { data: attachments } = useQuery({
    queryKey: ["task", taskId, "attachments"],
    enabled: tab === "files",
    queryFn: async () => {
      const { data, error } = await supabase
        .from("attachments")
        .select("*")
        .eq("task_id", taskId)
        .order("created_at");
      if (error) throw error;
      return data as Attachment[];
    },
  });

  const { data: audit } = useQuery({
    queryKey: ["task", taskId, "audit"],
    enabled: tab === "activity",
    queryFn: async () => {
      const { data, error } = await supabase
        .from("audit_log")
        .select("*")
        .eq("table_name", "tasks")
        .eq("row_id", taskId)
        .order("at", { ascending: false })
        .limit(100);
      if (error) throw error;
      return data as AuditEntry[];
    },
  });

  const { data: deps } = useQuery({
    queryKey: ["task", taskId, "deps"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("task_dependencies")
        .select("depends_on_task_id")
        .eq("task_id", taskId);
      if (error) throw error;
      const ids = (data as { depends_on_task_id: string }[]).map((d) => d.depends_on_task_id);
      if (!ids.length) return [] as Task[];
      const { data: dts, error: e2 } = await supabase.from("tasks").select("*").in("id", ids);
      if (e2) throw e2;
      return dts as Task[];
    },
  });

  const { data: projectTasks } = useQuery({
    queryKey: ["tasks", "project-of", task?.project_id],
    enabled: !!task && isStaff,
    queryFn: async () => {
      const { data, error } = await supabase
        .from("tasks")
        .select("id,title,status")
        .eq("project_id", task!.project_id)
        .neq("id", taskId);
      if (error) throw error;
      return data as Pick<Task, "id" | "title" | "status">[];
    },
  });

  // NOTE: every hook must run on every render (React #310) — keep all hooks
  // above this early return.
  const { data: subtasks } = useQuery({
    queryKey: ["task", taskId, "subtasks"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("tasks")
        .select("id,title,status,assignee_id,due_date")
        .eq("parent_id", taskId)
        .order("created_at");
      if (error) throw error;
      return data as Pick<Task, "id" | "title" | "status" | "assignee_id" | "due_date">[];
    },
  });

  if (!task) return null;

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["task", taskId] });
    qc.invalidateQueries({ queryKey: ["tasks"] });
    qc.invalidateQueries({ queryKey: ["my-tasks"] });
  };

  async function patch(p: Partial<Task>) {
    try {
      await updateTask(taskId, p);
      // assignment email is queued by a DB trigger (notify_outbox) so every
      // path notifies, including server-created tasks - see tk_notify.py
    } catch (e) {
      alert(e instanceof Error ? e.message : String(e));
    }
    invalidate();
  }

  const preparerDone = approvals?.some((a) => a.kind === "preparer");
  const reviewerDone = approvals?.some((a) => a.kind === "reviewer");
  const isAssignee = me?.id === task.assignee_id;
  const isReviewer = me?.id === task.reviewer_id;
  const blockedBy = deps?.filter((d) => d.status !== "done") ?? [];

  async function markPrepared() {
    await addApproval(taskId, "preparer");
    await updateTask(taskId, { status: "waiting_review" });
    qc.invalidateQueries({ queryKey: ["task", taskId, "approvals"] });
    invalidate();
  }

  async function signOff() {
    await addApproval(taskId, "reviewer");
    await updateTask(taskId, { status: "done" });
    qc.invalidateQueries({ queryKey: ["task", taskId, "approvals"] });
    invalidate();
  }

  const checklist = task.checklist ?? [];

  async function toggleCheck(i: number) {
    const list = [...checklist];
    list[i] = { ...list[i], done: !list[i].done };
    await patch({ checklist: list });
  }

  async function addCheck(e: React.FormEvent) {
    e.preventDefault();
    if (!newCheckItem.trim()) return;
    await patch({ checklist: [...checklist, { text: newCheckItem.trim(), done: false }] });
    setNewCheckItem("");
  }

  async function removeCheck(i: number) {
    const list = checklist.filter((_, idx) => idx !== i);
    await patch({ checklist: list });
  }

  async function postComment(e: React.FormEvent) {
    e.preventDefault();
    const text = commentText.trim();
    if (!text) return;
    await addComment(taskId, text);
    // @mentions: match "@First" / "@First Last" against profiles
    const mentioned = (profiles ?? []).filter((p) => {
      const name = (p.full_name || p.email.split("@")[0]).toLowerCase();
      const first = name.split(" ")[0];
      const t = text.toLowerCase();
      return t.includes("@" + name) || t.includes("@" + first);
    });
    if (mentioned.length) notify("mention", taskId, mentioned.map((p) => p.id), text);
    // task participants hear about new comments
    notify("comment", taskId,
      [task?.assignee_id, task?.reviewer_id].filter((id) => !mentioned.some((m) => m.id === id)),
      text);
    setCommentText("");
    qc.invalidateQueries({ queryKey: ["task", taskId, "comments"] });
  }

  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    await uploadAttachment(taskId, file);
    qc.invalidateQueries({ queryKey: ["task", taskId, "attachments"] });
    e.target.value = "";
  }

  async function viewReport(a: Attachment) {
    const { data, error } = await supabase.storage
      .from("taskapp-files")
      .createSignedUrl(a.storage_path, 120);
    if (error) {
      alert(error.message);
      return;
    }
    const html = await (await fetch(data.signedUrl)).text();
    setViewer({ title: a.filename, html });
  }

  function renderAudit(entry: AuditEntry) {
    const who = profileName(profiles, entry.actor);
    if (entry.action === "INSERT") return `${who} created this task`;
    if (entry.action === "DELETE") return `${who} deleted this task`;
    const changes: string[] = [];
    for (const f of AUDIT_FIELDS) {
      const a = entry.old_row?.[f];
      const b = entry.new_row?.[f];
      if (JSON.stringify(a) !== JSON.stringify(b)) {
        const fmt = (v: unknown) =>
          f.endsWith("_id")
            ? profileName(profiles, (v as string) ?? null)
            : String(v ?? "—");
        changes.push(`${f.replace("_id", "").replace("_", " ")}: ${fmt(a)} → ${fmt(b)}`);
      }
    }
    return changes.length ? `${who} changed ${changes.join(", ")}` : `${who} updated this task`;
  }

  const selCls =
    "rounded-lg border border-gray-300 px-2 py-1 text-sm focus:border-brand-500 focus:outline-none disabled:bg-gray-50 disabled:text-gray-500";

  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/20" onClick={() => openTask(null)}>
      <div
        className="flex h-full w-full max-w-xl flex-col overflow-hidden bg-white shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="border-b border-gray-200 px-6 py-4">
          <div className="flex items-start justify-between gap-3">
            <div className="flex min-w-0 items-start gap-2.5">
              <div className="mt-0.5"><CompleteToggle task={task} size="lg" /></div>
              <h2 className={`text-lg font-bold leading-snug ${task.status === "done" ? "text-gray-400 line-through" : ""}`}>
                {task.title}
              </h2>
            </div>
            <button onClick={() => openTask(null)} className="text-gray-400 hover:text-gray-600">✕</button>
          </div>
          {blockedBy.length > 0 && (
            <div className="mt-2 rounded-lg bg-amber-50 px-3 py-1.5 text-xs text-amber-800">
              Blocked by {blockedBy.length} open dependenc{blockedBy.length === 1 ? "y" : "ies"}:{" "}
              {blockedBy.map((d) => d.title).join(", ")}
            </div>
          )}
          <div className="mt-3 flex gap-1">
            {(["details", "comments", "files", "activity"] as Tab[]).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`rounded-lg px-3 py-1 text-sm capitalize ${
                  tab === t ? "bg-brand-50 font-semibold text-brand-700" : "text-gray-500 hover:bg-gray-100"
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {tab === "details" && (
            <div className="space-y-5">
              <div className="grid grid-cols-2 gap-3">
                <label className="text-xs text-gray-500">
                  Status
                  <select
                    value={task.status}
                    onChange={(e) => patch({ status: e.target.value as TaskStatus })}
                    className={`mt-1 block w-full ${selCls}`}
                  >
                    {TASK_STATUSES.map((s) => (
                      <option key={s} value={s}>{STATUS_LABELS[s]}</option>
                    ))}
                  </select>
                </label>
                <label className="text-xs text-gray-500">
                  Priority
                  <select
                    value={task.priority}
                    disabled={!isStaff}
                    onChange={(e) => patch({ priority: e.target.value as TaskPriority })}
                    className={`mt-1 block w-full ${selCls}`}
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                    <option value="critical">Critical</option>
                  </select>
                </label>
                <label className="text-xs text-gray-500">
                  Assignee
                  <select
                    value={task.assignee_id ?? ""}
                    disabled={!isStaff}
                    onChange={(e) => patch({ assignee_id: e.target.value || null })}
                    className={`mt-1 block w-full ${selCls}`}
                  >
                    <option value="">Unassigned</option>
                    {profiles?.map((p) => (
                      <option key={p.id} value={p.id}>{p.full_name || p.email}</option>
                    ))}
                  </select>
                </label>
                <label className="text-xs text-gray-500">
                  Reviewer
                  <select
                    value={task.reviewer_id ?? ""}
                    disabled={!isStaff}
                    onChange={(e) => patch({ reviewer_id: e.target.value || null })}
                    className={`mt-1 block w-full ${selCls}`}
                  >
                    <option value="">None</option>
                    {profiles?.map((p) => (
                      <option key={p.id} value={p.id}>{p.full_name || p.email}</option>
                    ))}
                  </select>
                </label>
                <label className="text-xs text-gray-500">
                  Due date
                  <input
                    type="date"
                    value={task.due_date ?? ""}
                    disabled={!isStaff}
                    onChange={(e) => patch({ due_date: e.target.value || null })}
                    className={`mt-1 block w-full ${selCls}`}
                  />
                </label>
                <label className="text-xs text-gray-500">
                  Repeat
                  <select
                    value={task.recurrence ?? ""}
                    disabled={!isStaff}
                    onChange={(e) =>
                      patch({ recurrence: (e.target.value || null) as Recurrence | null })
                    }
                    className={`mt-1 block w-full ${selCls}`}
                  >
                    <option value="">Does not repeat</option>
                    {(Object.keys(RECURRENCE_LABELS) as Recurrence[]).map((r) => (
                      <option key={r} value={r}>{RECURRENCE_LABELS[r]}</option>
                    ))}
                  </select>
                </label>
                {task.recurrence && (
                  <label className="text-xs text-gray-500">
                    Next one is due
                    <select
                      value={task.recurrence_mode ?? "schedule"}
                      disabled={!isStaff}
                      onChange={(e) =>
                        patch({ recurrence_mode: e.target.value as "schedule" | "completion" })
                      }
                      className={`mt-1 block w-full ${selCls}`}
                    >
                      <option value="schedule">On a fixed schedule</option>
                      <option value="completion">After this one is done</option>
                    </select>
                  </label>
                )}
                {task.recurrence && (
                  <label className="text-xs text-gray-500">
                    Repeat until (optional)
                    <input
                      type="date"
                      value={task.recurrence_until ?? ""}
                      disabled={!isStaff}
                      onChange={(e) => patch({ recurrence_until: e.target.value || null })}
                      className={`mt-1 block w-full ${selCls}`}
                    />
                  </label>
                )}
              </div>

              {task.recurrence && (
                <div className="rounded-lg bg-brand-50 px-3 py-2 text-xs text-brand-800">
                  🔁 Repeats <b>{RECURRENCE_LABELS[task.recurrence].toLowerCase()}</b>
                  {task.recurrence_mode === "completion"
                    ? " — the next one appears once this is marked done."
                    : task.due_date
                      ? " — the next one is created automatically about a week ahead."
                      : " — set a due date so the schedule has something to count from."}
                  {task.recurrence_until && ` Ends ${task.recurrence_until}.`}
                </div>
              )}

              {task.description && (
                <div>
                  <div className="mb-1 text-xs font-semibold text-gray-500">Description</div>
                  <p className="whitespace-pre-wrap text-sm text-gray-700">{task.description}</p>
                </div>
              )}

              {isStaff && <AgentPanel taskId={taskId} />}

              {/* Sign-off */}
              <div className="rounded-xl border border-gray-200 p-4">
                <div className="mb-2 text-xs font-semibold text-gray-500">Sign-off</div>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span>
                      Preparer{" "}
                      {preparerDone ? (
                        <span className="text-brand-600">
                          ✓ {profileName(profiles, approvals!.find((a) => a.kind === "preparer")!.approver_id)}
                        </span>
                      ) : (
                        <span className="text-gray-400">pending</span>
                      )}
                    </span>
                    {!preparerDone && (isAssignee || isStaff) && task.status !== "done" && (
                      <button
                        onClick={markPrepared}
                        className="rounded-lg bg-brand-600 px-3 py-1 text-xs font-semibold text-white hover:bg-brand-700"
                      >
                        Mark prepared
                      </button>
                    )}
                  </div>
                  <div className="flex items-center justify-between">
                    <span>
                      Reviewer{" "}
                      {reviewerDone ? (
                        <span className="text-brand-600">
                          ✓ {profileName(profiles, approvals!.find((a) => a.kind === "reviewer")!.approver_id)}
                        </span>
                      ) : (
                        <span className="text-gray-400">pending</span>
                      )}
                    </span>
                    {preparerDone && !reviewerDone && (isReviewer || isStaff) && (
                      <button
                        onClick={signOff}
                        disabled={blockedBy.length > 0}
                        title={blockedBy.length ? "Blocked by open dependencies" : undefined}
                        className="rounded-lg bg-brand-600 px-3 py-1 text-xs font-semibold text-white hover:bg-brand-700 disabled:opacity-40"
                      >
                        Sign off &amp; complete
                      </button>
                    )}
                  </div>
                </div>
              </div>

              {/* Checklist */}
              <div>
                <div className="mb-2 text-xs font-semibold text-gray-500">Checklist</div>
                <div className="space-y-1.5">
                  {checklist.map((item: ChecklistItem, i: number) => (
                    <div key={i} className="group flex items-center gap-2 text-sm">
                      <input type="checkbox" checked={item.done} onChange={() => toggleCheck(i)} className="accent-brand-600" />
                      <span className={item.done ? "text-gray-400 line-through" : ""}>{item.text}</span>
                      <button
                        onClick={() => removeCheck(i)}
                        className="ml-auto hidden text-xs text-gray-300 hover:text-red-500 group-hover:block"
                      >
                        remove
                      </button>
                    </div>
                  ))}
                </div>
                <form onSubmit={addCheck} className="mt-2 flex gap-2">
                  <input
                    placeholder="Add checklist item…"
                    value={newCheckItem}
                    onChange={(e) => setNewCheckItem(e.target.value)}
                    className="flex-1 rounded-lg border border-gray-300 px-3 py-1.5 text-sm focus:border-brand-500 focus:outline-none"
                  />
                  <button className="rounded-lg bg-gray-100 px-3 py-1.5 text-sm hover:bg-gray-200">Add</button>
                </form>
              </div>

              {/* Subtasks */}
              <div>
                <div className="mb-2 text-xs font-semibold text-gray-500">Subtasks</div>
                <div className="space-y-1.5">
                  {subtasks?.length === 0 && <div className="text-sm text-gray-300">None</div>}
                  {subtasks?.map((s) => (
                    <div key={s.id} className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={s.status === "done"}
                        onChange={async () => {
                          try {
                            await updateTask(s.id, { status: s.status === "done" ? "todo" : "done" });
                          } catch (e) {
                            alert(e instanceof Error ? e.message : String(e));
                          }
                          qc.invalidateQueries({ queryKey: ["task", taskId, "subtasks"] });
                          qc.invalidateQueries({ queryKey: ["tasks"] });
                        }}
                        className="accent-brand-600"
                      />
                      <button
                        onClick={() => openTask(s.id)}
                        className={`text-left hover:text-brand-700 ${s.status === "done" ? "text-gray-400 line-through" : ""}`}
                      >
                        {s.title}
                      </button>
                      <span className="ml-auto text-[11px] text-gray-400">
                        {profileName(profiles, s.assignee_id)}{s.due_date ? ` · ${s.due_date.slice(5)}` : ""}
                      </span>
                    </div>
                  ))}
                </div>
                {isStaff && (
                  <form
                    onSubmit={async (e) => {
                      e.preventDefault();
                      if (!newSubtask.trim() || !task) return;
                      await createTask({
                        project_id: task.project_id,
                        title: newSubtask.trim(),
                        parent_id: taskId,
                        assignee_id: task.assignee_id,
                        due_date: task.due_date,
                      });
                      setNewSubtask("");
                      qc.invalidateQueries({ queryKey: ["task", taskId, "subtasks"] });
                    }}
                    className="mt-2 flex gap-2"
                  >
                    <input
                      placeholder="Add subtask…"
                      value={newSubtask}
                      onChange={(e) => setNewSubtask(e.target.value)}
                      className="flex-1 rounded-lg border border-gray-300 px-3 py-1.5 text-sm focus:border-brand-500 focus:outline-none"
                    />
                    <button className="rounded-lg bg-gray-100 px-3 py-1.5 text-sm hover:bg-gray-200">Add</button>
                  </form>
                )}
              </div>

              {/* Dependencies */}
              <div>
                <div className="mb-2 text-xs font-semibold text-gray-500">Depends on</div>
                <div className="space-y-1.5">
                  {deps?.length === 0 && <div className="text-sm text-gray-300">None</div>}
                  {deps?.map((d) => (
                    <div key={d.id} className="flex items-center gap-2 text-sm">
                      <span className={d.status === "done" ? "text-brand-600" : "text-amber-600"}>
                        {d.status === "done" ? "✓" : "○"}
                      </span>
                      <span>{d.title}</span>
                      {isStaff && (
                        <button
                          onClick={async () => {
                            await removeDependency(taskId, d.id);
                            qc.invalidateQueries({ queryKey: ["task", taskId, "deps"] });
                          }}
                          className="ml-auto text-xs text-gray-300 hover:text-red-500"
                        >
                          remove
                        </button>
                      )}
                    </div>
                  ))}
                </div>
                {isStaff && (
                  <div className="mt-2 flex gap-2">
                    <select value={depPick} onChange={(e) => setDepPick(e.target.value)} className={`flex-1 ${selCls}`}>
                      <option value="">Add dependency…</option>
                      {projectTasks
                        ?.filter((t) => !deps?.some((d) => d.id === t.id))
                        .map((t) => (
                          <option key={t.id} value={t.id}>{t.title}</option>
                        ))}
                    </select>
                    <button
                      disabled={!depPick}
                      onClick={async () => {
                        await addDependency(taskId, depPick);
                        setDepPick("");
                        qc.invalidateQueries({ queryKey: ["task", taskId, "deps"] });
                      }}
                      className="rounded-lg bg-gray-100 px-3 py-1.5 text-sm hover:bg-gray-200 disabled:opacity-40"
                    >
                      Add
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}

          {tab === "comments" && (
            <div className="flex h-full flex-col">
              <div className="flex-1 space-y-3">
                {comments?.length === 0 && <div className="text-sm text-gray-300">No comments yet.</div>}
                {comments?.map((c) => (
                  <div key={c.id} className="rounded-lg bg-gray-50 px-3 py-2">
                    <div className="mb-0.5 flex items-baseline gap-2 text-xs">
                      <span className="font-semibold">{profileName(profiles, c.author_id)}</span>
                      <span className="text-gray-400">{format(parseISO(c.created_at), "d MMM HH:mm")}</span>
                    </div>
                    <div className="whitespace-pre-wrap text-sm">{c.body}</div>
                  </div>
                ))}
              </div>
              <form onSubmit={postComment} className="mt-4 flex gap-2">
                <input
                  placeholder="Write a comment…"
                  value={commentText}
                  onChange={(e) => setCommentText(e.target.value)}
                  className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
                />
                <button className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700">
                  Post
                </button>
              </form>
            </div>
          )}

          {tab === "files" && (
            <div className="space-y-3">
              <label className="block cursor-pointer rounded-lg border border-dashed border-gray-300 p-4 text-center text-sm text-gray-500 hover:border-brand-500">
                Upload workpaper / evidence
                <input type="file" className="hidden" onChange={onFile} />
              </label>
              {attachments?.map((a) => (
                <div key={a.id} className="flex items-center justify-between rounded-lg border border-gray-200 px-3 py-2 text-sm">
                  <div className="min-w-0">
                    <div className="truncate font-medium">{a.filename}</div>
                    <div className="text-xs text-gray-400">
                      {profileName(profiles, a.uploaded_by)} · {format(parseISO(a.created_at), "d MMM HH:mm")}
                      {a.size_bytes ? ` · ${(a.size_bytes / 1024).toFixed(0)} KB` : ""}
                    </div>
                  </div>
                  <div className="ml-3 flex shrink-0 gap-3">
                    {a.mime === "text/html" && (
                      <button
                        onClick={() => viewReport(a)}
                        className="text-xs font-semibold text-brand-600 hover:underline"
                      >
                        View
                      </button>
                    )}
                    <button
                      onClick={() => downloadAttachment(a.storage_path)}
                      className="text-xs font-semibold text-brand-600 hover:underline"
                    >
                      Download
                    </button>
                    {(isStaff || a.uploaded_by === me?.id) && (
                      <button
                        onClick={async () => {
                          if (!confirm(`Delete ${a.filename}?`)) return;
                          try {
                            await deleteAttachment(a.id, a.storage_path);
                            qc.invalidateQueries({ queryKey: ["task", taskId, "attachments"] });
                          } catch (e) {
                            alert(e instanceof Error ? e.message : String(e));
                          }
                        }}
                        className="text-xs text-gray-300 hover:text-red-600"
                      >
                        Delete
                      </button>
                    )}
                  </div>
                </div>
              ))}
              {attachments?.length === 0 && <div className="text-sm text-gray-300">No files yet.</div>}
            </div>
          )}

          {tab === "activity" && (
            <div className="space-y-2">
              {audit?.map((e) => (
                <div key={e.id} className="flex gap-3 text-sm">
                  <div className="w-28 shrink-0 text-xs text-gray-400">
                    {format(parseISO(e.at), "d MMM HH:mm")}
                  </div>
                  <div className="text-gray-700">{renderAudit(e)}</div>
                </div>
              ))}
              {audit?.length === 0 && <div className="text-sm text-gray-300">No activity recorded.</div>}
            </div>
          )}
        </div>
      </div>

      {viewer && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          onClick={() => setViewer(null)}
        >
          <div
            className="flex h-[92vh] w-full max-w-4xl flex-col overflow-hidden rounded-xl bg-white shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-gray-200 px-4 py-2.5">
              <span className="truncate text-sm font-semibold">{viewer.title}</span>
              <button onClick={() => setViewer(null)} className="text-gray-400 hover:text-gray-600">✕</button>
            </div>
            <iframe srcDoc={viewer.html} sandbox="" title={viewer.title} className="h-full w-full flex-1 bg-gray-100" />
          </div>
        </div>
      )}
    </div>
  );
}
