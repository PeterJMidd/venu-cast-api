-- Keyword routing for inbound email -> task: "if the mail (optionally to a
-- given mailbox) mentions any of these words, it belongs in THIS project with
-- THIS person". Deterministic and owner-set, so it is applied BEFORE the AI
-- classifier - the AI remains the fallback for mail no rule claims.
-- Applied 2026-08-27. Seeded catch-alls (keywords='*', position 900):
-- payroll@ -> People & payroll, whsclaims@ -> Insurance risk & regulatory,
-- finance@ -> Finance operations; solutions@ deliberately left to the AI.
create table if not exists taskapp.email_rules (
  id          uuid primary key default gen_random_uuid(),
  keywords    text not null,          -- comma-separated; any match fires; '*' = all
  mailbox     text,                   -- only mail to this address; null = any
  project_id  uuid not null,
  assignee_id uuid,                   -- null = fall back to the drop's owner/admin
  priority    text,                   -- null = let the classifier decide
  position    int not null default 100,  -- lower fires first; catch-alls at 900
  active      boolean not null default true,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

create index if not exists email_rules_active_idx
  on taskapp.email_rules (active, position);

alter table taskapp.email_rules enable row level security;

drop policy if exists email_rules_staff_all on taskapp.email_rules;
create policy email_rules_staff_all on taskapp.email_rules
  for all using (taskapp.is_staff()) with check (taskapp.is_staff());

grant select, insert, update, delete on taskapp.email_rules to authenticated;
grant all on taskapp.email_rules to service_role;

NOTIFY pgrst, 'reload schema';
