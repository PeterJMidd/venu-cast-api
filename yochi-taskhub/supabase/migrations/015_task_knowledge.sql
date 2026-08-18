-- Cumulative research knowledge per task (applied live 2026-08-18 as
-- taskapp_task_knowledge). Every finding is kept, and a rolling consolidated
-- understanding is maintained on top, so asking more questions makes the task
-- smarter rather than just longer. The summary is fed back into the next
-- question, and newer sources supersede older findings explicitly.
create table if not exists taskapp.knowledge (
  id uuid primary key default gen_random_uuid(),
  task_id uuid not null references taskapp.tasks(id) on delete cascade,
  question text not null,
  answer text not null,
  engine text,
  depth text,
  recency text,
  created_by uuid,
  created_at timestamptz not null default now()
);
create index if not exists knowledge_task_idx
  on taskapp.knowledge (task_id, created_at desc);
grant all on taskapp.knowledge to service_role;
alter table taskapp.knowledge enable row level security;
create policy knowledge_read on taskapp.knowledge
  for select using (taskapp.can_see_task(task_id));

create table if not exists taskapp.task_knowledge (
  task_id uuid primary key references taskapp.tasks(id) on delete cascade,
  summary text not null default '',
  entry_count int not null default 0,
  updated_at timestamptz not null default now()
);
grant all on taskapp.task_knowledge to service_role;
alter table taskapp.task_knowledge enable row level security;
create policy task_knowledge_read on taskapp.task_knowledge
  for select using (taskapp.can_see_task(task_id));
