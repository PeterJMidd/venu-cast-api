-- Consolidated record of migrations applied live via MCP after 002
-- (names as applied: taskapp_fix_search_path, taskapp_watcher_dedup,
--  taskapp_watcher_dedup_fix, taskapp_admin_categories, taskapp_hardening).

-- ---- taskapp_fix_search_path ----
alter function taskapp.touch_updated_at() set search_path = taskapp;
alter function taskapp.try_uuid(text) set search_path = taskapp;

-- ---- taskapp_watcher_dedup (+ _fix) ----
-- one watcher task per rule per period; a plain unique constraint (NOT a
-- partial index) because PostgREST on_conflict cannot target partial indexes.
alter table taskapp.tasks add column watcher_rule_id uuid references taskapp.watcher_rules(id);
alter table taskapp.tasks add constraint tasks_watcher_dedup unique (watcher_rule_id, period_id);

-- ---- taskapp_admin_categories ----
grant insert, update, delete on taskapp.categories to authenticated;
create policy categories_admin_insert on taskapp.categories for insert to authenticated
  with check (taskapp.my_role() = 'admin');
create policy categories_admin_update on taskapp.categories for update to authenticated
  using (taskapp.my_role() = 'admin') with check (taskapp.my_role() = 'admin');
create policy categories_admin_delete on taskapp.categories for delete to authenticated
  using (taskapp.my_role() = 'admin');

-- ---- taskapp_hardening (post code-review) ----
-- 1. Invite-only role elevation; self-signups become inactive stakeholders.
create or replace function taskapp.handle_new_user() returns trigger
language plpgsql security definer set search_path = taskapp, public as $$
begin
  insert into taskapp.profiles (id, email, full_name, role, active)
  values (
    new.id,
    new.email,
    coalesce(new.raw_user_meta_data->>'full_name', split_part(new.email,'@',1)),
    case when new.invited_at is not null
              and new.raw_user_meta_data->>'role' in ('admin','finance','stakeholder')
         then (new.raw_user_meta_data->>'role')::taskapp.user_role
         else 'stakeholder' end,
    new.invited_at is not null
  )
  on conflict (id) do nothing;
  return new;
end $$;

-- 2. Task guard: stakeholder field protection + done-gate (sign-off, deps).
create function taskapp.guard_task_update() returns trigger
language plpgsql security definer set search_path = taskapp as $$
declare
  actor_role taskapp.user_role;
  needs_signoff boolean;
  open_deps int;
begin
  if auth.uid() is null then
    return new;  -- service-role automation
  end if;
  actor_role := taskapp.my_role();
  if actor_role = 'stakeholder' then
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
create trigger guard_task_update before update on taskapp.tasks
  for each row execute function taskapp.guard_task_update();

-- 3. Segregation of duties on approvals.
create function taskapp.guard_approval() returns trigger
language plpgsql security definer set search_path = taskapp as $$
declare
  t taskapp.tasks;
  preparer uuid;
begin
  if auth.uid() is null then
    return new;
  end if;
  select * into t from taskapp.tasks where id = new.task_id;
  if new.kind = 'reviewer' then
    select approver_id into preparer
      from taskapp.approvals where task_id = new.task_id and kind = 'preparer';
    if preparer is not null and new.approver_id = preparer then
      raise exception 'reviewer sign-off must be a different person to the preparer';
    end if;
    if t.reviewer_id is not null and new.approver_id <> t.reviewer_id
       and taskapp.my_role() <> 'admin' then
      raise exception 'only the assigned reviewer (or an admin) can sign off';
    end if;
  end if;
  return new;
end $$;
create trigger guard_approval before insert on taskapp.approvals
  for each row execute function taskapp.guard_approval();

-- 4. Storage: allow same-name re-upload (upsert).
create policy taskapp_files_update on storage.objects for update to authenticated
  using (
    bucket_id = 'taskapp-files'
    and taskapp.can_see_task(taskapp.try_uuid((storage.foldername(name))[1]))
  )
  with check (
    bucket_id = 'taskapp-files'
    and taskapp.can_see_task(taskapp.try_uuid((storage.foldername(name))[1]))
  );