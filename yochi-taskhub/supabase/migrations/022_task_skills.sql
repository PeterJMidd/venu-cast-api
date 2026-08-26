-- Saved, runnable routines attached to a task. Extends ai_skills additively:
-- existing standalone review skills keep working untouched; a row with a
-- task_id is a "task routine" - proposed from a plain-English ask on the task,
-- saved as a button, run on demand or on a cadence, delivering its output to
-- chosen recipients in chosen formats (pdf/xlsx/csv, any mix), and refined
-- over time with each refinement versioned so the routine's history is
-- auditable.
-- Applied 2026-08-26.
alter table taskapp.ai_skills
  add column if not exists task_id     uuid,
  add column if not exists ask         text,
  add column if not exists recipients  text,          -- comma-separated emails
  add column if not exists formats     text,          -- comma-separated: pdf,xlsx,csv
  add column if not exists version     int not null default 1,
  add column if not exists refinements jsonb not null default '[]'::jsonb,
  add column if not exists last_result jsonb;

create index if not exists ai_skills_task_idx on taskapp.ai_skills (task_id)
  where task_id is not null;
