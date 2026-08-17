-- Per-task recurrence + guaranteed assignment notification
-- (applied live 2026-08-18 as taskapp_recurrence_and_notify_outbox and
--  taskapp_recurrence_full_unique_index).

-- 1. Recurrence on any task. Distinct from task_templates, which drives the
--    structured close/statutory calendar from bd/dom rules per period; this is
--    the lightweight "repeat this task" set from the task itself.
alter table taskapp.tasks
  add column if not exists recurrence text,
  add column if not exists recurrence_mode text not null default 'schedule',
  add column if not exists recurrence_until date,
  add column if not exists recurrence_parent_id uuid
    references taskapp.tasks(id) on delete set null;

alter table taskapp.tasks drop constraint if exists tasks_recurrence_chk;
alter table taskapp.tasks add constraint tasks_recurrence_chk
  check (recurrence is null or recurrence in
         ('daily','weekly','fortnightly','monthly','quarterly','annual'));
alter table taskapp.tasks drop constraint if exists tasks_recurrence_mode_chk;
alter table taskapp.tasks add constraint tasks_recurrence_mode_chk
  check (recurrence_mode in ('schedule','completion'));

-- One instance per series per due date. NOT partial: PostgREST's on_conflict
-- rejects partial unique indexes (42P10). Safe because non-recurring tasks all
-- have recurrence_parent_id = NULL and NULLs never collide in a unique index.
drop index if exists taskapp.tasks_recurrence_instance_idx;
create unique index if not exists tasks_recurrence_instance_idx
  on taskapp.tasks (recurrence_parent_id, due_date);
create index if not exists tasks_recurrence_active_idx
  on taskapp.tasks (recurrence) where recurrence is not null;

-- 2. Assignment notification outbox. Previously only three web code paths
--    emailed on assignment, so tasks created by the watcher, the compliance
--    register, the email-drop pipeline or an AI agent notified nobody. A
--    trigger now queues every assignee/reviewer change and tk_notify drains it,
--    so no code path can forget.
create table if not exists taskapp.notify_outbox (
  id uuid primary key default gen_random_uuid(),
  task_id uuid not null references taskapp.tasks(id) on delete cascade,
  recipient uuid not null,
  kind text not null default 'assigned',      -- assigned | reviewer
  actor uuid,                                  -- null when a server job did it
  created_at timestamptz not null default now(),
  sent_at timestamptz,
  attempts int not null default 0,
  error text
);
create index if not exists notify_outbox_pending_idx
  on taskapp.notify_outbox (created_at) where sent_at is null;
grant all on taskapp.notify_outbox to service_role;
alter table taskapp.notify_outbox enable row level security;
create policy notify_outbox_staff_read on taskapp.notify_outbox
  for select using (taskapp.is_staff());

create or replace function taskapp.queue_assignment_notification()
returns trigger language plpgsql security definer set search_path = taskapp, public as $$
declare
  actor uuid := auth.uid();   -- null when a server job or timer did it
begin
  if tg_op = 'INSERT' then
    if new.assignee_id is not null and new.assignee_id is distinct from actor then
      insert into taskapp.notify_outbox (task_id, recipient, kind, actor)
      values (new.id, new.assignee_id, 'assigned', actor);
    end if;
    if new.reviewer_id is not null and new.reviewer_id is distinct from actor
       and new.reviewer_id is distinct from new.assignee_id then
      insert into taskapp.notify_outbox (task_id, recipient, kind, actor)
      values (new.id, new.reviewer_id, 'reviewer', actor);
    end if;
  elsif tg_op = 'UPDATE' then
    if new.assignee_id is not null
       and new.assignee_id is distinct from old.assignee_id
       and new.assignee_id is distinct from actor then
      insert into taskapp.notify_outbox (task_id, recipient, kind, actor)
      values (new.id, new.assignee_id, 'assigned', actor);
    end if;
    if new.reviewer_id is not null
       and new.reviewer_id is distinct from old.reviewer_id
       and new.reviewer_id is distinct from actor then
      insert into taskapp.notify_outbox (task_id, recipient, kind, actor)
      values (new.id, new.reviewer_id, 'reviewer', actor);
    end if;
  end if;
  return new;
end $$;

drop trigger if exists trg_queue_assignment_notification on taskapp.tasks;
create trigger trg_queue_assignment_notification
  after insert or update of assignee_id, reviewer_id on taskapp.tasks
  for each row execute function taskapp.queue_assignment_notification();
