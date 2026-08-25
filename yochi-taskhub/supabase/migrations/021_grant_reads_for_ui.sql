-- These three tables had RLS policies written for the UI but no GRANT, so
-- PostgREST refused every browser read with 403 regardless of the policy.
-- The Knowledge tab reads knowledge/task_knowledge directly, so it had been
-- silently empty since it shipped; the team-activity view hit the same wall
-- on notify_outbox.
--
-- Row visibility is unchanged - the existing policies still decide what each
-- person sees (can_see_task for the knowledge tables, staff-only for the
-- notification log). This only opens the door those policies already guard.
-- Applied 2026-08-25.
grant select on taskapp.knowledge       to authenticated;
grant select on taskapp.task_knowledge  to authenticated;
grant select on taskapp.notify_outbox   to authenticated;
