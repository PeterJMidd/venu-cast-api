-- Daily work-activity intelligence report (applied live 2026-08-16 as
-- taskapp_work_reports). One row per day: prior-day activity stats scanned
-- from the blob estate + data lake + TaskHub, plus the AI narrative.
create table if not exists taskapp.work_reports (
  report_date date primary key,
  stats jsonb not null default '{}'::jsonb,
  narrative text,
  created_at timestamptz not null default now()
);
grant all on taskapp.work_reports to service_role;
alter table taskapp.work_reports enable row level security;
create policy work_reports_staff_read on taskapp.work_reports
  for select using (taskapp.is_staff());
