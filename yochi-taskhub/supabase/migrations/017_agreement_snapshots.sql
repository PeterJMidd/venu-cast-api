
-- PostgREST reaches tables through these roles; without the grants every read
-- is 403 "permission denied" regardless of the RLS policy above.
grant select, insert, update, delete on taskapp.agreement_snapshots to authenticated;
grant all on taskapp.agreement_snapshots to service_role;
