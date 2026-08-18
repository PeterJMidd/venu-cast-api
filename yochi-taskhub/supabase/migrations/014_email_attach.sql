-- Attaching an email to an EXISTING task (applied live 2026-08-18 as
-- taskapp_email_attach_to_task).
--
-- Two routes reach this:
--   1. the Outlook add-in's "Attach to a task" mode (posts a comment directly)
--   2. the flagged-email pipeline: a reply/forward whose subject matches an
--      open task's thread, or a mail containing a TaskHub task link
alter table taskapp.tasks
  add column if not exists email_thread text;
create index if not exists tasks_email_thread_idx
  on taskapp.tasks (email_thread) where email_thread is not null;

-- Ledger of processed drops. Needed because an email that ATTACHES creates no
-- new task row, so tasks.external_ref can no longer be the only "already
-- handled" record.
create table if not exists taskapp.email_drop_log (
  ref text primary key,
  task_id uuid references taskapp.tasks(id) on delete set null,
  action text not null,                 -- created | attached | skipped
  subject text,
  processed_at timestamptz not null default now()
);
grant all on taskapp.email_drop_log to service_role;
alter table taskapp.email_drop_log enable row level security;
create policy email_drop_log_staff_read on taskapp.email_drop_log
  for select using (taskapp.is_staff());
