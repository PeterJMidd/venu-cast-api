-- The recommended-approach playbook lives in the data lake, but the lake is not
-- read-your-writes: a parquet appended seconds ago is not reliably visible to
-- the next read, so every run saw "no prior version" and wrote v1 again.
-- Postgres now holds the authoritative pointer (current version per approach)
-- plus the version history; the lake copy stays as the queryable long-term
-- store for the Data lake page, the voice assistant and skills.
-- Applied 2026-08-19.

create table if not exists taskapp.playbook (
  approach_key        text primary key,
  task_title          text,
  project             text,
  method              text,
  queries_that_worked text,
  pitfalls            text,
  improve_next_time   text,
  outcome             text,
  version             integer not null default 1,
  changed_this_run    text,
  updated_at          timestamptz not null default now()
);

create table if not exists taskapp.playbook_versions (
  id                  uuid primary key default gen_random_uuid(),
  approach_key        text not null,
  task_title          text,
  method              text,
  queries_that_worked text,
  pitfalls            text,
  improve_next_time   text,
  outcome             text,
  version             integer not null,
  changed_this_run    text,
  created_at          timestamptz not null default now()
);

create index if not exists playbook_versions_key_idx
  on taskapp.playbook_versions (approach_key, created_at desc);

alter table taskapp.playbook          enable row level security;
alter table taskapp.playbook_versions enable row level security;

drop policy if exists playbook_staff_read on taskapp.playbook;
create policy playbook_staff_read on taskapp.playbook
  for select using (taskapp.is_staff());

drop policy if exists playbook_versions_staff_read on taskapp.playbook_versions;
create policy playbook_versions_staff_read on taskapp.playbook_versions
  for select using (taskapp.is_staff());
