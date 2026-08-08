-- 004_automation.sql — automation roadmap items 1-3 (applied live 2026-08-08
-- as migration taskapp_automation_wave1; recorded here for replication).

-- Item 1: per-venue base-case net_sales forecast, trailing 14 days. The
-- watcher materialises this view into its DuckDB lake cache as table
-- forecast_venue_daily (tk_watcher._sync_forecast) so watch rules can join
-- actuals (mart_venue_daily) to the Venu Cast base case. Venue names match
-- the mart exactly ("Yo-Chi Albert St").
create or replace view taskapp.v_forecast_venue_daily as
select l.venue, l.forecast_date as d, round(sum(l.amount), 0) as forecast_sales
from public.forecast_version_lines l
join public.forecast_versions v on v.id = l.version_id and v.is_base_case
where l.account_code = 'net_sales'
  and l.forecast_date >= current_date - 14
  and l.forecast_date <= current_date
group by l.venue, l.forecast_date;
grant select on taskapp.v_forecast_venue_daily to service_role;

-- Item 2: 'triage' batches — the batch worker agent-pre-works a specific list
-- of freshly created watcher/GL-sweep tasks (task_ids) instead of the whole
-- project. Enqueued automatically from the watcher/glsweep timers unless app
-- setting AUTO_TRIAGE=0.
alter table taskapp.batch_runs drop constraint batch_runs_kind_check;
alter table taskapp.batch_runs add constraint batch_runs_kind_check
  check (kind in ('run_all','steer','triage'));
alter table taskapp.batch_runs add column if not exists task_ids jsonb;

-- Item 1 seed rule (inserted live 2026-08-08, id 779f03d1-dcea-46a4-8b21-fc43d6002219):
-- watcher_rules 'Venue >15% behind forecast 3 days running' — flags venues
-- more than 15% behind base case on EACH of the last 3 trading days
-- (forecast_sales > 200 guard excludes near-zero forecast days; inner join
-- skips venues with no base-case forecast). Project 'Data watch', priority
-- high, assignee Peter.

-- Item 3 has no schema change: close_batch_timer (function_app.py) enqueues a
-- kind='run_all' batch for the Month-end close project at 05:00 on the first
-- business day of each month.

-- Items 4-6 (2026-08-08, no schema changes):
-- 4. tk_digest.py + digest_timer (Mon 07:05) — weekly position-delta email to
--    admins, built from taskapp.positions versions/events.
-- 5. auto-refine lessons — agent_execute now records the refine feedback in
--    agent_runs.plan->>'feedback'; tk_agent._precedents surfaces up to 5 past
--    refinements as 'lessons' in both the plan and report prompts.
-- 6. estate health — tk_estate.py + estate_timer (06:10 daily): blob lake-sync
--    freshness, stuck batch_runs, agent error spikes, skills heartbeat ->
--    '[Estate] …' tasks in Data watch (dedup: one open task per title).
--    Plus 2 seeded data-staleness watcher_rules (live 2026-08-08):
--    'Sales feed stale or incomplete' (d63abe6e…) and 'Procedure feed stale'
--    (e5c15485…) in the Data watch project.
