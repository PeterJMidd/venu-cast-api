-- 006_external_access.sql — per-pillar/per-project access for external parties
-- (tax advisors etc.) + groundwork for enforced MFA. Written 2026-08-16.
-- APPLY: Supabase SQL editor (one paste) or via the Supabase MCP when
-- reconnected. Uses ::text comparisons for the new enum value so the whole
-- file is single-transaction safe.

-- 1. new role: 'external' — sees ONLY projects/pillars they are a member of
alter type taskapp.user_role add value if not exists 'external';

-- 2. membership tables (grant a whole pillar via category_members, or a
--    single project via project_members)
create table if not exists taskapp.project_members (
  project_id uuid not null references taskapp.projects(id) on delete cascade,
  user_id uuid not null references taskapp.profiles(id) on delete cascade,
  added_by uuid,
  created_at timestamptz default now(),
  primary key (project_id, user_id)
);
create table if not exists taskapp.category_members (
  category_id int not null references taskapp.categories(id) on delete cascade,
  user_id uuid not null references taskapp.profiles(id) on delete cascade,
  added_by uuid,
  created_at timestamptz default now(),
  primary key (category_id, user_id)
);
alter table taskapp.project_members enable row level security;
alter table taskapp.category_members enable row level security;
grant select, insert, delete on taskapp.project_members, taskapp.category_members
  to authenticated;
grant all on taskapp.project_members, taskapp.category_members to service_role;

create policy pm_select on taskapp.project_members for select to authenticated
  using (taskapp.is_staff() or user_id = auth.uid());
create policy pm_insert on taskapp.project_members for insert to authenticated
  with check (taskapp.my_role() = 'admin');
create policy pm_delete on taskapp.project_members for delete to authenticated
  using (taskapp.my_role() = 'admin');
create policy cm_select on taskapp.category_members for select to authenticated
  using (taskapp.is_staff() or user_id = auth.uid());
create policy cm_insert on taskapp.category_members for insert to authenticated
  with check (taskapp.my_role() = 'admin');
create policy cm_delete on taskapp.category_members for delete to authenticated
  using (taskapp.my_role() = 'admin');

-- 3. visibility helpers. can_see_project = staff, or explicit project/pillar
--    membership. can_see_task extends the existing assignee/reviewer rule
--    with membership — every dependent policy (comments, attachments, deps,
--    approvals, audit, storage) inherits automatically.
create or replace function taskapp.can_see_project(p uuid) returns boolean
language sql stable security definer set search_path = taskapp as $$
  select taskapp.is_staff()
      or exists (select 1 from taskapp.project_members m
                 where m.project_id = p and m.user_id = auth.uid())
      or exists (select 1 from taskapp.projects pr
                 join taskapp.category_members cm on cm.category_id = pr.category_id
                 where pr.id = p and cm.user_id = auth.uid())
$$;
grant execute on function taskapp.can_see_project(uuid) to authenticated, service_role;

create or replace function taskapp.can_see_task(t uuid) returns boolean
language sql stable security definer set search_path = taskapp as $$
  select taskapp.is_staff()
      or exists (select 1 from taskapp.tasks
                 where id = t and (assignee_id = auth.uid() or reviewer_id = auth.uid()))
      or exists (select 1 from taskapp.tasks tk
                 where tk.id = t and taskapp.can_see_project(tk.project_id))
$$;

-- 4. table policies that need the membership path added
drop policy tasks_select on taskapp.tasks;
create policy tasks_select on taskapp.tasks for select to authenticated
  using (taskapp.can_see_task(id));

drop policy projects_select on taskapp.projects;
create policy projects_select on taskapp.projects for select to authenticated
  using (
    taskapp.can_see_project(id)
    or exists (select 1 from taskapp.tasks t
               where t.project_id = projects.id
                 and (t.assignee_id = auth.uid() or t.reviewer_id = auth.uid()))
  );
-- tasks_update / tasks_insert unchanged: externals can only UPDATE tasks
-- actually assigned to them, and never create — they view and comment.

-- 5. guard trigger: externals get the same field lockdown as stakeholders
create or replace function taskapp.guard_task_update() returns trigger
language plpgsql security definer set search_path = taskapp as $$
declare
  actor_role text;
  needs_signoff boolean;
  open_deps int;
begin
  if auth.uid() is null then
    return new;  -- service-role automation
  end if;
  actor_role := taskapp.my_role()::text;
  if actor_role in ('stakeholder', 'external') then
    if new.assignee_id is distinct from old.assignee_id
       or new.reviewer_id is distinct from old.reviewer_id
       or new.project_id is distinct from old.project_id
       or new.template_id is distinct from old.template_id
       or new.period_id is distinct from old.period_id
       or new.due_date is distinct from old.due_date
       or new.priority is distinct from old.priority
       or new.watcher_rule_id is distinct from old.watcher_rule_id then
      raise exception 'stakeholders may only update status, checklist, description and title';
    end if;
  end if;
  if new.status = 'done' and old.status is distinct from 'done' then
    select count(*) into open_deps
    from taskapp.task_dependencies d
    join taskapp.tasks t on t.id = d.depends_on_task_id
    where d.task_id = new.id and t.status <> 'done';
    if open_deps > 0 then
      raise exception 'cannot complete: % open dependenc%', open_deps,
        case when open_deps = 1 then 'y' else 'ies' end;
    end if;
    needs_signoff := coalesce(
      (select tt.requires_signoff from taskapp.task_templates tt where tt.id = new.template_id),
      new.reviewer_id is not null);
    if needs_signoff and not exists (
      select 1 from taskapp.approvals a where a.task_id = new.id and a.kind = 'reviewer') then
      raise exception 'cannot complete: reviewer sign-off required first';
    end if;
  end if;
  return new;
end $$;

-- 6. invited-role handling includes 'external'
create or replace function taskapp.handle_new_user() returns trigger
language plpgsql security definer set search_path = taskapp, public as $$
begin
  insert into taskapp.profiles (id, email, full_name, role, active)
  values (
    new.id,
    new.email,
    coalesce(new.raw_user_meta_data->>'full_name', split_part(new.email,'@',1)),
    case when new.invited_at is not null
              and new.raw_user_meta_data->>'role' in ('admin','finance','stakeholder','external')
         then (new.raw_user_meta_data->>'role')::taskapp.user_role
         else 'stakeholder' end,
    new.invited_at is not null
  )
  on conflict (id) do nothing;
  return new;
end $$;

-- (guard + handle_new_user bodies = 003 verbatim; only the role checks
--  gained 'external')
