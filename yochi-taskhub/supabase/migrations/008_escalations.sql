-- Escalation & nudge engine (applied live 2026-08-16 as taskapp_escalations).
-- One row per nudge actually sent; the engine skips a (task, rule) pair that
-- has a row within the last 3 days so people are chased, not spammed.
create table if not exists taskapp.escalations (
  id uuid primary key default gen_random_uuid(),
  task_id uuid not null references taskapp.tasks(id) on delete cascade,
  rule text not null,
  sent_at timestamptz not null default now()
);
create index if not exists escalations_task_rule_idx
  on taskapp.escalations (task_id, rule, sent_at desc);
grant all on taskapp.escalations to service_role;
alter table taskapp.escalations enable row level security;
create policy escalations_staff_read on taskapp.escalations
  for select using (taskapp.is_staff());
