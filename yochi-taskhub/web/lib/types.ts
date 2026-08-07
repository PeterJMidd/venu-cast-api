// Hand-written domain types for the `taskapp` schema.
// (MCP type generation only covers exposed schemas; regenerate once taskapp is exposed.)

export type UserRole = "admin" | "finance" | "stakeholder";
export type TaskStatus = "todo" | "in_progress" | "waiting_review" | "done" | "blocked";
export type TaskPriority = "low" | "medium" | "high" | "critical";
export type TaskSource = "manual" | "template" | "watcher" | "nl";

export const TASK_STATUSES: TaskStatus[] = [
  "todo",
  "in_progress",
  "waiting_review",
  "done",
  "blocked",
];

export const STATUS_LABELS: Record<TaskStatus, string> = {
  todo: "To do",
  in_progress: "In progress",
  waiting_review: "Waiting review",
  done: "Done",
  blocked: "Blocked",
};

export const PRIORITY_LABELS: Record<TaskPriority, string> = {
  low: "Low",
  medium: "Medium",
  high: "High",
  critical: "Critical",
};

export interface Profile {
  id: string;
  email: string;
  full_name: string | null;
  role: UserRole;
  active: boolean;
  created_at: string;
  updated_at: string;
}

export interface Category {
  id: number;
  name: string;
  sort: number;
}

export interface Project {
  id: string;
  category_id: number;
  name: string;
  description: string | null;
  owner_id: string | null;
  archived: boolean;
  created_at: string;
  updated_at: string;
}

export interface Period {
  id: string;
  period_month: string; // yyyy-mm-01
  label: string;
  status: "open" | "closed";
}

export interface ChecklistItem {
  text: string;
  done: boolean;
}

export interface Task {
  id: string;
  project_id: string;
  template_id: string | null;
  period_id: string | null;
  title: string;
  description: string | null;
  status: TaskStatus;
  priority: TaskPriority;
  assignee_id: string | null;
  reviewer_id: string | null;
  due_date: string | null;
  completed_at: string | null;
  source: TaskSource;
  checklist: ChecklistItem[];
  created_by: string | null;
  parent_id: string | null;
  sort_order: number | null;
  created_at: string;
  updated_at: string;
}

export interface TaskTemplate {
  id: string;
  project_id: string;
  title: string;
  description: string | null;
  cadence: "monthly" | "quarterly" | "annual";
  due_rule: { type: "bd"; n: number } | { type: "dom"; day: number; roll?: "forward" };
  default_assignee_id: string | null;
  default_reviewer_id: string | null;
  priority: TaskPriority;
  requires_signoff: boolean;
  active: boolean;
}

export interface Approval {
  id: string;
  task_id: string;
  kind: "preparer" | "reviewer";
  approver_id: string;
  approved_at: string;
  note: string | null;
}

export interface Comment {
  id: string;
  task_id: string;
  author_id: string;
  body: string;
  created_at: string;
}

export interface Attachment {
  id: string;
  task_id: string;
  storage_path: string;
  filename: string;
  size_bytes: number | null;
  mime: string | null;
  uploaded_by: string | null;
  created_at: string;
}

export interface AuditEntry {
  id: number;
  table_name: string;
  row_id: string | null;
  action: "INSERT" | "UPDATE" | "DELETE";
  actor: string | null;
  old_row: Record<string, unknown> | null;
  new_row: Record<string, unknown> | null;
  at: string;
}
