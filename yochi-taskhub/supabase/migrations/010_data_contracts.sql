-- Data contracts (applied live 2026-08-18 as taskapp_data_contracts).
-- Declarative freshness + volume SLAs on the tables the cockpit depends on,
-- checked at 08:15 daily BEFORE the reports run. Prior to this, staleness was
-- policed by two hand-written watcher rules, so when mart_venue_daily stopped
-- at 13-Aug the 16-Aug report simply printed zero sales and said nothing.
create table if not exists taskapp.data_contracts (
  id uuid primary key default gen_random_uuid(),
  table_name text not null,
  label text,
  date_expr text not null,          -- SQL expression yielding a DATE
  max_staleness_days int not null default 2,
  min_rows_recent int not null default 0,
  recent_days int not null default 7,
  weekdays_only boolean not null default false,   -- weekday-only feeds
  active boolean not null default true,
  last_status text,                 -- ok | breach | error
  last_checked_at timestamptz,
  last_detail text,
  created_at timestamptz not null default now()
);
create unique index if not exists data_contracts_table_idx
  on taskapp.data_contracts (table_name);
grant all on taskapp.data_contracts to service_role;
alter table taskapp.data_contracts enable row level security;
create policy data_contracts_staff_read on taskapp.data_contracts
  for select using (taskapp.is_staff());
create policy data_contracts_admin_write on taskapp.data_contracts
  for all using (taskapp.my_role() = 'admin') with check (taskapp.my_role() = 'admin');

-- Seeded contracts (thresholds tuned against real observed cadence 18-Aug-26).
insert into taskapp.data_contracts
  (table_name, label, date_expr, max_staleness_days, min_rows_recent, recent_days, weekdays_only)
values
  ('mart_venue_daily','Sales by venue (daily)','CAST("date" AS DATE)',2,300,7,false),
  ('mart_procedure_daily','Venue procedure completion (daily)','TRY_CAST(day AS DATE)',3,200,7,false),
  ('mart_reviews_weekly','Guest reviews (weekly)','TRY_CAST(week_start AS DATE)',10,20,21,false),
  ('mart_restoke_food_cost','Food cost (monthly)','TRY_CAST("month" || ''-01'' AS DATE)',40,20,90,false),
  ('XeroAccountTransactionsMasterView','Xero GL transactions','TRY_CAST("Date" AS DATE)',6,20,7,false),
  ('Invoices','Xero invoices & bills','TRY_CAST(updateddateutc AS DATE)',3,100,7,false),
  ('restoke_purchasing','Restoke purchasing/receipting','TRY_CAST(received_date AS DATE)',4,500,7,false),
  ('procedure_report','Restoke HQ procedure detail','TRY_CAST(completion_date AS DATE)',3,100,7,false),
  ('AsanaTasks','Asana task mirror','TRY_CAST(ModifiedAt AS DATE)',5,10,7,false),
  ('feed_xero_user_activity','Xero activity by person','TRY_CAST(activity_date AS DATE)',3,1,7,true)
on conflict (table_name) do nothing;
