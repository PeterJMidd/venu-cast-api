-- TaskHub schema: isolated `taskapp` schema in the shared Supabase project.
-- Venu Cast lives in `public` — nothing here touches it.

create schema if not exists taskapp;

-- ---------------------------------------------------------------- enums
create type taskapp.user_role as enum ('admin','finance','stakeholder');
create type taskapp.task_status as enum ('todo','in_progress','waiting_review','done','blocked');
create type taskapp.task_priority as enum ('low','medium','high','critical');
create type taskapp.task_source as enum ('manual','template','watcher','nl');

-- ---------------------------------------------------------------- tables
create table taskapp.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text not null unique,
  full_name text,
  role taskapp.user_role not null default 'stakeholder',
  active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table taskapp.categories (
  id smallint primary key,
  name text not null unique,
  sort smallint not null
);
insert into taskapp.categories (id, name, sort) values
  (1,'People',1),
  (2,'Compliance',2),
  (3,'Financial control',3),
  (4,'Insight',4),
  (5,'Systems and processes',5),
  (6,'AI evolution',6);

create table taskapp.projects (
  id uuid primary key default gen_random_uuid(),
  category_id smallint not null references taskapp.categories(id),
  name text not null,
  description text,
  owner_id uuid references taskapp.profiles(id),
  archived boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table taskapp.periods (
  id uuid primary key default gen_random_uuid(),
  period_month date not null unique,  -- always the 1st of the month
  label text not null,
  status text not null default 'open' check (status in ('open','closed'))
);

create table taskapp.task_templates (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references taskapp.projects(id),
  title text not null,
  description text,
  cadence text not null check (cadence in ('monthly','quarterly','annual')),
  due_rule jsonb not null,  -- {"type":"bd","n":3} | {"type":"dom","day":21,"roll":"forward"}
  default_assignee_id uuid references taskapp.profiles(id),
  default_reviewer_id uuid references taskapp.profiles(id),
  priority taskapp.task_priority not null default 'medium',
  requires_signoff boolean not null default true,
  active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table taskapp.tasks (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references taskapp.projects(id),
  template_id uuid references taskapp.task_templates(id),
  period_id uuid references taskapp.periods(id),
  title text not null,
  description text,
  status taskapp.task_status not null default 'todo',
  priority taskapp.task_priority not null default 'medium',
  assignee_id uuid references taskapp.profiles(id),
  reviewer_id uuid references taskapp.profiles(id),
  due_date date,
  completed_at timestamptz,
  source taskapp.task_source not null default 'manual',
  checklist jsonb not null default '[]',
  created_by uuid references taskapp.profiles(id),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (template_id, period_id)  -- template instantiation dedup
);

create table taskapp.task_dependencies (
  task_id uuid not null references taskapp.tasks(id) on delete cascade,
  depends_on_task_id uuid not null references taskapp.tasks(id) on delete cascade,
  primary key (task_id, depends_on_task_id),
  check (task_id <> depends_on_task_id)
);

create table taskapp.approvals (
  id uuid primary key default gen_random_uuid(),
  task_id uuid not null references taskapp.tasks(id) on delete cascade,
  kind text not null check (kind in ('preparer','reviewer')),
  approver_id uuid not null references taskapp.profiles(id),
  approved_at timestamptz not null default now(),
  note text,
  unique (task_id, kind)
);

create table taskapp.comments (
  id uuid primary key default gen_random_uuid(),
  task_id uuid not null references taskapp.tasks(id) on delete cascade,
  author_id uuid not null references taskapp.profiles(id),
  body text not null,
  created_at timestamptz not null default now()
);

create table taskapp.attachments (
  id uuid primary key default gen_random_uuid(),
  task_id uuid not null references taskapp.tasks(id) on delete cascade,
  storage_path text not null,  -- '<task_id>/<filename>' in bucket taskapp-files
  filename text not null,
  size_bytes bigint,
  mime text,
  uploaded_by uuid references taskapp.profiles(id),
  created_at timestamptz not null default now()
);

create table taskapp.audit_log (
  id bigint generated always as identity primary key,
  table_name text not null,
  row_id uuid,
  action text not null,
  actor uuid,
  old_row jsonb,
  new_row jsonb,
  at timestamptz not null default now()
);

create table taskapp.watcher_rules (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  description text,
  check_sql text not null,  -- DuckDB SQL over the lake; returned rows = breaches
  comparator text not null default '>',
  threshold numeric,
  project_id uuid not null references taskapp.projects(id),
  assignee_id uuid references taskapp.profiles(id),
  priority taskapp.task_priority not null default 'high',
  active boolean not null default true,
  last_run_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table taskapp.ai_suggestions (
  id uuid primary key default gen_random_uuid(),
  kind text not null check (kind in ('template_change','deadline_change','other')),
  payload jsonb not null,
  rationale text,
  status text not null default 'pending' check (status in ('pending','accepted','dismissed')),
  created_at timestamptz not null default now()
);

create table taskapp.notification_prefs (
  user_id uuid primary key references taskapp.profiles(id) on delete cascade,
  daily_briefing boolean not null default true,
  email_on_assign boolean not null default true,
  updated_at timestamptz not null default now()
);

-- ---------------------------------------------------------------- indexes
create index tasks_assignee_status_idx on taskapp.tasks (assignee_id, status);
create index tasks_project_status_idx on taskapp.tasks (project_id, status);
create index tasks_due_open_idx on taskapp.tasks (due_date) where status <> 'done';
create index tasks_period_idx on taskapp.tasks (period_id);
create index tasks_reviewer_idx on taskapp.tasks (reviewer_id);
create index tasks_created_by_idx on taskapp.tasks (created_by);
create index projects_category_idx on taskapp.projects (category_id);
create index projects_owner_idx on taskapp.projects (owner_id);
create index templates_project_idx on taskapp.task_templates (project_id);
create index templates_assignee_idx on taskapp.task_templates (default_assignee_id);
create index templates_reviewer_idx on taskapp.task_templates (default_reviewer_id);
create index deps_depends_on_idx on taskapp.task_dependencies (depends_on_task_id);
create index approvals_task_idx on taskapp.approvals (task_id);
create index approvals_approver_idx on taskapp.approvals (approver_id);
create index comments_task_idx on taskapp.comments (task_id);
create index comments_author_idx on taskapp.comments (author_id);
create index attachments_task_idx on taskapp.attachments (task_id);
create index attachments_uploader_idx on taskapp.attachments (uploaded_by);
create index audit_log_row_idx on taskapp.audit_log (table_name, row_id);
create index watcher_rules_project_idx on taskapp.watcher_rules (project_id);
create index watcher_rules_assignee_idx on taskapp.watcher_rules (assignee_id);

-- ---------------------------------------------------------------- triggers
create function taskapp.touch_updated_at() returns trigger
language plpgsql as $$
begin
  new.updated_at := now();
  return new;
end $$;

create trigger touch_profiles before update on taskapp.profiles
  for each row execute function taskapp.touch_updated_at();
create trigger touch_projects before update on taskapp.projects
  for each row execute function taskapp.touch_updated_at();
create trigger touch_templates before update on taskapp.task_templates
  for each row execute function taskapp.touch_updated_at();
create trigger touch_tasks before update on taskapp.tasks
  for each row execute function taskapp.touch_updated_at();
create trigger touch_watcher_rules before update on taskapp.watcher_rules
  for each row execute function taskapp.touch_updated_at();
create trigger touch_notification_prefs before update on taskapp.notification_prefs
  for each row execute function taskapp.touch_updated_at();

-- append-only audit trail on the tables that matter for control
create function taskapp.audit() returns trigger
language plpgsql security definer set search_path = taskapp as $$
begin
  insert into taskapp.audit_log (table_name, row_id, action, actor, old_row, new_row)
  values (
    tg_table_name,
    coalesce(
      case when tg_op <> 'DELETE' then new.id end,
      case when tg_op <> 'INSERT' then old.id end
    ),
    tg_op,
    auth.uid(),
    case when tg_op <> 'INSERT' then to_jsonb(old) end,
    case when tg_op <> 'DELETE' then to_jsonb(new) end
  );
  return coalesce(new, old);
end $$;

create trigger audit_tasks after insert or update or delete on taskapp.tasks
  for each row execute function taskapp.audit();
create trigger audit_approvals after insert or update or delete on taskapp.approvals
  for each row execute function taskapp.audit();
create trigger audit_projects after insert or update or delete on taskapp.projects
  for each row execute function taskapp.audit();
create trigger audit_templates after insert or update or delete on taskapp.task_templates
  for each row execute function taskapp.audit();

-- auto-create a profile whenever an auth user is created (invite flow)
create function taskapp.handle_new_user() returns trigger
language plpgsql security definer set search_path = taskapp, public as $$
begin
  insert into taskapp.profiles (id, email, full_name, role)
  values (
    new.id,
    new.email,
    coalesce(new.raw_user_meta_data->>'full_name', split_part(new.email,'@',1)),
    case when new.raw_user_meta_data->>'role' in ('admin','finance','stakeholder')
         then (new.raw_user_meta_data->>'role')::taskapp.user_role
         else 'stakeholder' end
  )
  on conflict (id) do nothing;
  return new;
end $$;

create trigger on_auth_user_created_taskapp after insert on auth.users
  for each row execute function taskapp.handle_new_user();

-- ---------------------------------------------------------------- grants
grant usage on schema taskapp to authenticated, service_role;
grant select, insert, update, delete on all tables in schema taskapp to authenticated;
grant all on all tables in schema taskapp to service_role;
grant usage on all sequences in schema taskapp to service_role;
grant execute on all functions in schema taskapp to authenticated, service_role;

-- audit log is append-only and written only by triggers (security definer)
revoke insert, update, delete, truncate on taskapp.audit_log from authenticated;

-- profile role/active changes go through the service role only (admin FN endpoint);
-- authenticated users may update their own display name only
revoke update on taskapp.profiles from authenticated;
grant update (full_name) on taskapp.profiles to authenticated;

-- categories are fixed; writes via service role only
revoke insert, update, delete on taskapp.categories from authenticated;

-- realtime for the live board
alter publication supabase_realtime add table taskapp.tasks;
