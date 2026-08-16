-- 007_feeds.sql — external data feed registry (applied live 2026-08-16 as
-- migration taskapp_feeds). The learning data centre: recommended feeds are
-- activated by an admin; tk_feeds researches each active feed on its cadence
-- (web search, cited) and appends rows to datasights-lake/tables/feed_<slug>/
-- with captured_at history, registering the table in the lake catalog so the
-- voice assistant, task agent, smart tasks and watch rules can query external
-- data (award rates, tax rates, due dates, market indicators) alongside
-- internal data. Daily timer 05:45 (weekly feeds run Mondays) + run_feeds.
create table if not exists taskapp.feeds (
  id uuid primary key default gen_random_uuid(),
  slug text unique not null,
  name text not null,
  description text,
  kind text not null check (kind in ('award','tax_rates','due_dates','economy','industry','other')),
  research_prompt text not null,
  columns jsonb not null,
  cadence text not null default 'weekly' check (cadence in ('daily','weekly')),
  status text not null default 'recommended' check (status in ('recommended','active','paused')),
  last_run_at timestamptz,
  last_summary text,
  created_at timestamptz default now()
);
alter table taskapp.feeds enable row level security;
grant select, insert, update, delete on taskapp.feeds to authenticated;
grant all on taskapp.feeds to service_role;
create policy feeds_select on taskapp.feeds for select to authenticated
  using (taskapp.is_staff());
create policy feeds_insert on taskapp.feeds for insert to authenticated
  with check (taskapp.my_role() = 'admin');
create policy feeds_update on taskapp.feeds for update to authenticated
  using (taskapp.my_role() = 'admin') with check (taskapp.my_role() = 'admin');
create policy feeds_delete on taskapp.feeds for delete to authenticated
  using (taskapp.my_role() = 'admin');

-- Seeded live: 8 feeds — state_payroll_tax + ato_key_dates ACTIVE;
-- fast_food_award, rba_economy, abs_retail, wage_decisions, input_costs,
-- franchising_reg RECOMMENDED (activate in Admin -> Data feeds).
-- Smart tasks (tk_smart) may add further 'recommended' feeds automatically.
