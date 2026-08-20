-- The complete compliance register: every obligation-instance from the FY27
-- calendar (12 monthly tabs) plus the Exceptions tab, held as operational
-- state so TaskHub can show status, flow work into tasks, and report what the
-- workbook and the legal documents disagree about.
-- Applied 2026-08-20.
create table if not exists taskapp.compliance_items (
  id            uuid primary key default gen_random_uuid(),
  ref           text not null,
  period        text not null,              -- 'YYYY-MM', 'EXC', or 'DOC'
  stream        text not null default 'calendar',  -- calendar | exception | document
  category      text,
  obligation    text,
  what_to_do    text,
  authority     text,
  frequency     text,
  due_text      text,
  due_date      date,
  due_basis     text,                       -- how due_date was derived
  owner         text,
  risk          text,
  entities      text,                       -- ticked entity columns, comma separated
  sheet_status  text,                       -- Status column as it reads in the workbook
  completed_date text,
  evidence      text,
  notes         text,
  severity      text,                       -- exceptions / document gaps
  action        text,
  task_id       uuid,
  task_status   text,
  writeback     text,                       -- status we would push back to the workbook
  active        boolean not null default true,
  first_seen_at timestamptz not null default now(),
  updated_at    timestamptz not null default now(),
  unique (ref, period)
);

create index if not exists compliance_items_due_idx on taskapp.compliance_items (due_date);
create index if not exists compliance_items_stream_idx on taskapp.compliance_items (stream, active);

create table if not exists taskapp.compliance_runs (
  id              uuid primary key default gen_random_uuid(),
  ran_at          timestamptz not null default now(),
  ran_by          text,
  items           int,
  tasks_added     int,
  tasks_amended   int,
  tasks_linked    int,
  writeback_ready int,
  flags           jsonb,
  detail          jsonb
);

alter table taskapp.compliance_items enable row level security;
alter table taskapp.compliance_runs  enable row level security;

drop policy if exists compliance_items_staff_read on taskapp.compliance_items;
create policy compliance_items_staff_read on taskapp.compliance_items
  for select using (taskapp.is_staff());

drop policy if exists compliance_runs_staff_read on taskapp.compliance_runs;
create policy compliance_runs_staff_read on taskapp.compliance_runs
  for select using (taskapp.is_staff());

-- PostgREST reaches tables through these roles; without the grants every read
-- is 403 "permission denied" regardless of the RLS policies above.
grant select, insert, update, delete on taskapp.compliance_items to authenticated;
grant select, insert, update, delete on taskapp.compliance_runs  to authenticated;
grant all on taskapp.compliance_items to service_role;
grant all on taskapp.compliance_runs  to service_role;
