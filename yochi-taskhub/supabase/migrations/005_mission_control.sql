-- 005_mission_control.sql — CFO cockpit data layer (applied live 2026-08-08 as
-- migrations taskapp_mission_control, taskapp_external_ref_full_unique,
-- taskapp_cash_forecast; recorded here for replication).

-- Outlook calendar mirror (synced by the morning MCP session)
create table if not exists taskapp.calendar_events (
  id uuid primary key default gen_random_uuid(),
  external_ref text unique not null,
  subject text not null,
  starts_at timestamptz not null,
  ends_at timestamptz,
  location text, organizer text, attendees int,
  is_all_day boolean default false,
  synced_at timestamptz default now()
);
create index if not exists cal_starts_idx on taskapp.calendar_events (starts_at);
alter table taskapp.calendar_events enable row level security;
create policy cal_read on taskapp.calendar_events for select
  to authenticated using (taskapp.my_role() in ('admin','finance'));

-- External context signals shown on the cockpit
create table if not exists taskapp.signals (
  id uuid primary key default gen_random_uuid(),
  kind text not null check (kind in ('economy','industry','business','notice')),
  headline text not null,
  detail text, source text,
  created_at timestamptz default now()
);
create index if not exists signals_created_idx on taskapp.signals (created_at desc);
alter table taskapp.signals enable row level security;
create policy signals_read on taskapp.signals for select
  to authenticated using (taskapp.my_role() in ('admin','finance'));

-- Dedup key for imported tasks (asana:<gid>, email:<id>). NOTE: must be a FULL
-- unique index - PostgREST on_conflict cannot use a partial index.
alter table taskapp.tasks add column if not exists external_ref text;
create unique index if not exists tasks_external_ref_idx on taskapp.tasks (external_ref);

-- 13-week cash forecast generations (tk_cash, Monday 06:55 + run_cash)
create table if not exists taskapp.cash_forecast (
  id uuid primary key default gen_random_uuid(),
  generated_at timestamptz default now(),
  week_start date not null,
  receipts numeric, ap numeric, payroll numeric, ato_super numeric,
  net numeric, closing numeric,
  assumptions jsonb
);
create index if not exists cashf_gen_idx on taskapp.cash_forecast (generated_at desc, week_start);
alter table taskapp.cash_forecast enable row level security;
create policy cashf_read on taskapp.cash_forecast for select
  to authenticated using (taskapp.my_role() in ('admin','finance'));

-- Weekly base-case forecast window for the cash model
create or replace view taskapp.v_forecast_weekly as
select date_trunc('week', l.forecast_date)::date as week_start,
       round(sum(l.amount), 0) as forecast_sales
from public.forecast_version_lines l
join public.forecast_versions v on v.id = l.version_id and v.is_base_case
where l.account_code = 'net_sales'
  and l.forecast_date >= current_date
  and l.forecast_date < current_date + 98
group by 1;
grant select on taskapp.v_forecast_weekly to service_role;

-- Also seeded live: projects aaaaaaaa-…18 'CFO 100-day plan' (cat 3),
-- …19 'Finance operations' (cat 3), …20 'People & payroll' (cat 1);
-- 127 Asana tasks imported with external_ref='asana:<gid>'.
