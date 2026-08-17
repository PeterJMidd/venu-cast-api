import { supabase } from "@/lib/supabase";
import { callFn } from "@/lib/fn";
import type { Task, TaskPriority, TaskStatus } from "@/lib/types";

/** Fire-and-forget notification email via the FN app. Never blocks the UI. */
export function notify(kind: "assigned" | "comment" | "mention", taskId: string, recipientIds: (string | null | undefined)[], note?: string) {
  const ids = recipientIds.filter(Boolean) as string[];
  if (!ids.length) return;
  callFn("notify", { kind, task_id: taskId, recipient_ids: ids, note }).catch(() => {});
}

export async function updateTask(id: string, patch: Partial<Task>) {
  const upd: Record<string, unknown> = { ...patch };
  if (patch.status === "done" && !patch.completed_at) upd.completed_at = new Date().toISOString();
  if (patch.status && patch.status !== "done") upd.completed_at = null;
  const { error } = await supabase.from("tasks").update(upd).eq("id", id);
  if (error) throw error;
}

export interface NewTaskInput {
  project_id: string;
  title: string;
  description?: string | null;
  assignee_id?: string | null;
  reviewer_id?: string | null;
  due_date?: string | null;
  priority?: TaskPriority;
  status?: TaskStatus;
  parent_id?: string | null;
}

export async function createTask(input: NewTaskInput): Promise<string> {
  const { data: sess } = await supabase.auth.getSession();
  const uid = sess.session?.user.id;
  const { data, error } = await supabase
    .from("tasks")
    .insert({ ...input, created_by: uid })
    .select("id")
    .single();
  if (error) throw error;
  // assignment emails are NOT sent from here: a DB trigger queues every
  // assignee/reviewer change into taskapp.notify_outbox and the function app
  // drains it, so server-created tasks (watcher, register, email drop, agents)
  // notify too. Comments/@mentions still notify directly.
  return (data as { id: string }).id;
}

export async function addComment(taskId: string, body: string) {
  const { data: sess } = await supabase.auth.getSession();
  const { error } = await supabase
    .from("comments")
    .insert({ task_id: taskId, author_id: sess.session!.user.id, body });
  if (error) throw error;
}

export async function addApproval(taskId: string, kind: "preparer" | "reviewer", note?: string) {
  const { data: sess } = await supabase.auth.getSession();
  const { error } = await supabase
    .from("approvals")
    .insert({ task_id: taskId, kind, approver_id: sess.session!.user.id, note: note ?? null });
  if (error) throw error;
}

export async function uploadAttachment(taskId: string, file: File) {
  const path = `${taskId}/${file.name}`;
  const { error: upErr } = await supabase.storage
    .from("taskapp-files")
    .upload(path, file, { upsert: true });
  if (upErr) throw upErr;
  const { data: sess } = await supabase.auth.getSession();
  const { error } = await supabase.from("attachments").insert({
    task_id: taskId,
    kind: "file",
    storage_path: path,
    filename: file.name,
    size_bytes: file.size,
    mime: file.type || null,
    uploaded_by: sess.session!.user.id,
  });
  if (error) throw error;
}

/** Attach a link (SharePoint doc, Xero screen, dashboard, anything on the web). */
export async function addLink(taskId: string, rawUrl: string, label?: string) {
  const url = rawUrl.trim();
  // http(s) only — a javascript:/data: URI in an <a href> is an XSS vector.
  // The database enforces this too; this check is for a friendly message.
  if (!/^https?:\/\/\S+$/i.test(url)) {
    throw new Error("Enter a full web address starting with http:// or https://");
  }
  let name = (label || "").trim();
  if (!name) {
    try {
      const u = new URL(url);
      const last = u.pathname.split("/").filter(Boolean).pop();
      name = decodeURIComponent(last || u.hostname);
    } catch {
      name = url;
    }
  }
  const { data: sess } = await supabase.auth.getSession();
  const { error } = await supabase.from("attachments").insert({
    task_id: taskId,
    kind: "link",
    url,
    filename: name.slice(0, 200),
    uploaded_by: sess.session!.user.id,
  });
  if (error) throw error;
}

export async function downloadAttachment(storagePath: string) {
  const { data, error } = await supabase.storage
    .from("taskapp-files")
    .createSignedUrl(storagePath, 60);
  if (error) throw error;
  window.open(data.signedUrl, "_blank");
}

export async function deleteAttachment(attachmentId: string, storagePath?: string | null) {
  // links have no stored object to remove
  if (storagePath) {
    const { error: sErr } = await supabase.storage.from("taskapp-files").remove([storagePath]);
    if (sErr) throw sErr;
  }
  const { error } = await supabase.from("attachments").delete().eq("id", attachmentId);
  if (error) throw error;
}

export async function addDependency(taskId: string, dependsOn: string) {
  const { error } = await supabase
    .from("task_dependencies")
    .insert({ task_id: taskId, depends_on_task_id: dependsOn });
  if (error) throw error;
}

export async function removeDependency(taskId: string, dependsOn: string) {
  const { error } = await supabase
    .from("task_dependencies")
    .delete()
    .eq("task_id", taskId)
    .eq("depends_on_task_id", dependsOn);
  if (error) throw error;
}
