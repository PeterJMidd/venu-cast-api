-- Comment notifications (applied live 2026-08-18 as taskapp_comment_notifications).
-- Previously mentions were detected in the BROWSER and only matched "@Name", so
-- pasting a colleague's email address into a comment notified nobody, and
-- comments posted by agents/server jobs notified nobody at all. Now a trigger
-- queues into the same notify_outbox that assignments use.
alter table taskapp.notify_outbox
  add column if not exists comment_id uuid references taskapp.comments(id) on delete cascade;

-- one notification per person per comment, however they were tagged
create unique index if not exists notify_outbox_comment_recipient_idx
  on taskapp.notify_outbox (comment_id, recipient);

create or replace function taskapp.queue_comment_notifications()
returns trigger language plpgsql security definer set search_path = taskapp, public as $$
declare
  actor  uuid := new.author_id;
  body_l text := lower(coalesce(new.body, ''));
  p      record;
  t      record;
begin
  -- 1. anyone tagged: by email address (holliew@yochi.com.au) OR by @name
  for p in select id, email, full_name from taskapp.profiles where active loop
    if p.id is distinct from actor
       and (
         position(lower(p.email) in body_l) > 0
         or (coalesce(p.full_name, '') <> '' and (
              position('@' || lower(p.full_name) in body_l) > 0
              or position('@' || lower(split_part(p.full_name, ' ', 1)) in body_l) > 0))
       )
    then
      insert into taskapp.notify_outbox (task_id, recipient, kind, actor, comment_id)
      values (new.task_id, p.id, 'mention', actor, new.id)
      on conflict do nothing;
    end if;
  end loop;

  -- 2. the people carrying the task hear about any new comment
  select assignee_id, reviewer_id into t from taskapp.tasks where id = new.task_id;
  if t.assignee_id is not null and t.assignee_id is distinct from actor then
    insert into taskapp.notify_outbox (task_id, recipient, kind, actor, comment_id)
    values (new.task_id, t.assignee_id, 'comment', actor, new.id)
    on conflict do nothing;
  end if;
  if t.reviewer_id is not null and t.reviewer_id is distinct from actor then
    insert into taskapp.notify_outbox (task_id, recipient, kind, actor, comment_id)
    values (new.task_id, t.reviewer_id, 'comment', actor, new.id)
    on conflict do nothing;
  end if;
  return new;
end $$;

drop trigger if exists trg_queue_comment_notifications on taskapp.comments;
create trigger trg_queue_comment_notifications
  after insert on taskapp.comments
  for each row execute function taskapp.queue_comment_notifications();
