-- TaskHub RLS: three roles — admin (full), finance (full operational),
-- stakeholder (only tasks where they are assignee/reviewer, plus satellites).

-- ---------------------------------------------------------------- helpers
-- security definer => reads bypass RLS on profiles/tasks (owner = postgres)
create function taskapp.my_role() returns taskapp.user_role
language sql stable security definer set search_path = taskapp as $$
  select role from taskapp.profiles where id = auth.uid() and active
$$;

create function taskapp.is_staff() returns boolean
language sql stable security definer set search_path = taskapp as $$
  select taskapp.my_role() in ('admin','finance')
$$;

create function taskapp.can_see_task(t uuid) returns boolean
language sql stable security definer set search_path = taskapp as $$
  select taskapp.is_staff()
      or exists (
           select 1 from taskapp.tasks
           where id = t and (assignee_id = auth.uid() or reviewer_id = auth.uid())
         )
$$;

-- safe uuid cast for storage paths ('<task_id>/<filename>')
create function taskapp.try_uuid(s text) returns uuid
language plpgsql immutable as $$
begin
  return s::uuid;
exception when others then
  return null;
end $$;

grant execute on function taskapp.my_role(), taskapp.is_staff(),
  taskapp.can_see_task(uuid), taskapp.try_uuid(text) to authenticated, service_role;

-- ---------------------------------------------------------------- enable RLS
alter table taskapp.profiles enable row level security;
alter table taskapp.categories enable row level security;
alter table taskapp.projects enable row level security;
alter table taskapp.periods enable row level security;
alter table taskapp.task_templates enable row level security;
alter table taskapp.tasks enable row level security;
alter table taskapp.task_dependencies enable row level security;
alter table taskapp.approvals enable row level security;
alter table taskapp.comments enable row level security;
alter table taskapp.attachments enable row level security;
alter table taskapp.audit_log enable row level security;
alter table taskapp.watcher_rules enable row level security;
alter table taskapp.ai_suggestions enable row level security;
alter table taskapp.notification_prefs enable row level security;

-- ---------------------------------------------------------------- profiles
create policy profiles_select on taskapp.profiles for select to authenticated
  using (taskapp.is_staff() or id = auth.uid());
-- update restricted to own row; column grant (001) limits it to full_name.
-- role/active changes: service role only.
create policy profiles_update_self on taskapp.profiles for update to authenticated
  using (id = auth.uid()) with check (id = auth.uid());

-- ---------------------------------------------------------------- categories
create policy categories_select on taskapp.categories for select to authenticated
  using (true);

-- ---------------------------------------------------------------- projects
create policy projects_select on taskapp.projects for select to authenticated
  using (
    taskapp.is_staff()
    or exists (
         select 1 from taskapp.tasks t
         where t.project_id = projects.id
           and (t.assignee_id = auth.uid() or t.reviewer_id = auth.uid())
       )
  );
create policy projects_write on taskapp.projects for insert to authenticated
  with check (taskapp.is_staff());
create policy projects_update on taskapp.projects for update to authenticated
  using (taskapp.is_staff()) with check (taskapp.is_staff());
create policy projects_delete on taskapp.projects for delete to authenticated
  using (taskapp.my_role() = 'admin');

-- ---------------------------------------------------------------- periods
create policy periods_select on taskapp.periods for select to authenticated
  using (true);
create policy periods_write on taskapp.periods for insert to authenticated
  with check (taskapp.is_staff());
create policy periods_update on taskapp.periods for update to authenticated
  using (taskapp.is_staff()) with check (taskapp.is_staff());

-- ---------------------------------------------------------------- task_templates
create policy templates_select on taskapp.task_templates for select to authenticated
  using (taskapp.is_staff());
create policy templates_insert on taskapp.task_templates for insert to authenticated
  with check (taskapp.is_staff());
create policy templates_update on taskapp.task_templates for update to authenticated
  using (taskapp.is_staff()) with check (taskapp.is_staff());
create policy templates_delete on taskapp.task_templates for delete to authenticated
  using (taskapp.my_role() = 'admin');

-- ---------------------------------------------------------------- tasks
create policy tasks_select on taskapp.tasks for select to authenticated
  using (taskapp.is_staff() or assignee_id = auth.uid() or reviewer_id = auth.uid());
create policy tasks_insert on taskapp.tasks for insert to authenticated
  with check (taskapp.is_staff());
-- stakeholders may update their own tasks but the WITH CHECK stops them
-- reassigning the task away from themselves
create policy tasks_update on taskapp.tasks for update to authenticated
  using (taskapp.is_staff() or assignee_id = auth.uid() or reviewer_id = auth.uid())
  with check (taskapp.is_staff() or assignee_id = auth.uid() or reviewer_id = auth.uid());
create policy tasks_delete on taskapp.tasks for delete to authenticated
  using (taskapp.is_staff());

-- ---------------------------------------------------------------- dependencies
create policy deps_select on taskapp.task_dependencies for select to authenticated
  using (taskapp.can_see_task(task_id));
create policy deps_insert on taskapp.task_dependencies for insert to authenticated
  with check (taskapp.is_staff());
create policy deps_delete on taskapp.task_dependencies for delete to authenticated
  using (taskapp.is_staff());

-- ---------------------------------------------------------------- approvals
create policy approvals_select on taskapp.approvals for select to authenticated
  using (taskapp.can_see_task(task_id));
create policy approvals_insert on taskapp.approvals for insert to authenticated
  with check (approver_id = auth.uid() and taskapp.can_see_task(task_id));
create policy approvals_delete on taskapp.approvals for delete to authenticated
  using (taskapp.my_role() = 'admin');

-- ---------------------------------------------------------------- comments
create policy comments_select on taskapp.comments for select to authenticated
  using (taskapp.can_see_task(task_id));
create policy comments_insert on taskapp.comments for insert to authenticated
  with check (author_id = auth.uid() and taskapp.can_see_task(task_id));
create policy comments_update_own on taskapp.comments for update to authenticated
  using (author_id = auth.uid()) with check (author_id = auth.uid());
create policy comments_delete on taskapp.comments for delete to authenticated
  using (author_id = auth.uid() or taskapp.my_role() = 'admin');

-- ---------------------------------------------------------------- attachments
create policy attachments_select on taskapp.attachments for select to authenticated
  using (taskapp.can_see_task(task_id));
create policy attachments_insert on taskapp.attachments for insert to authenticated
  with check (uploaded_by = auth.uid() and taskapp.can_see_task(task_id));
create policy attachments_delete on taskapp.attachments for delete to authenticated
  using (uploaded_by = auth.uid() or taskapp.is_staff());

-- ---------------------------------------------------------------- audit_log
create policy audit_select on taskapp.audit_log for select to authenticated
  using (
    taskapp.is_staff()
    or (table_name = 'tasks' and taskapp.can_see_task(row_id))
  );

-- ---------------------------------------------------------------- watcher_rules
create policy watcher_select on taskapp.watcher_rules for select to authenticated
  using (taskapp.is_staff());
create policy watcher_insert on taskapp.watcher_rules for insert to authenticated
  with check (taskapp.my_role() = 'admin');
create policy watcher_update on taskapp.watcher_rules for update to authenticated
  using (taskapp.my_role() = 'admin') with check (taskapp.my_role() = 'admin');
create policy watcher_delete on taskapp.watcher_rules for delete to authenticated
  using (taskapp.my_role() = 'admin');

-- ---------------------------------------------------------------- ai_suggestions
create policy suggestions_select on taskapp.ai_suggestions for select to authenticated
  using (taskapp.is_staff());
create policy suggestions_update on taskapp.ai_suggestions for update to authenticated
  using (taskapp.is_staff()) with check (taskapp.is_staff());
-- inserts come from the learning-loop function via service role only

-- ---------------------------------------------------------------- notification_prefs
create policy prefs_select on taskapp.notification_prefs for select to authenticated
  using (user_id = auth.uid() or taskapp.my_role() = 'admin');
create policy prefs_upsert on taskapp.notification_prefs for insert to authenticated
  with check (user_id = auth.uid());
create policy prefs_update on taskapp.notification_prefs for update to authenticated
  using (user_id = auth.uid()) with check (user_id = auth.uid());

-- ---------------------------------------------------------------- storage
-- private bucket for workpapers; path convention '<task_id>/<filename>'
insert into storage.buckets (id, name, public)
values ('taskapp-files','taskapp-files', false)
on conflict (id) do nothing;

create policy taskapp_files_select on storage.objects for select to authenticated
  using (
    bucket_id = 'taskapp-files'
    and taskapp.can_see_task(taskapp.try_uuid((storage.foldername(name))[1]))
  );
create policy taskapp_files_insert on storage.objects for insert to authenticated
  with check (
    bucket_id = 'taskapp-files'
    and taskapp.try_uuid((storage.foldername(name))[1]) is not null
    and taskapp.can_see_task(taskapp.try_uuid((storage.foldername(name))[1]))
  );
create policy taskapp_files_delete on storage.objects for delete to authenticated
  using (bucket_id = 'taskapp-files' and taskapp.is_staff());
