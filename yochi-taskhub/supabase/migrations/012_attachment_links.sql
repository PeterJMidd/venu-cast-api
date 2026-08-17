-- Links on tasks alongside uploaded files (applied live 2026-08-18 as
-- taskapp_attachment_links). A task can now point at a SharePoint document,
-- a Xero screen or a dashboard without copying the file into storage.
alter table taskapp.attachments
  add column if not exists kind text not null default 'file',
  add column if not exists url text;

-- links have no stored object
alter table taskapp.attachments alter column storage_path drop not null;

alter table taskapp.attachments drop constraint if exists attachments_kind_chk;
alter table taskapp.attachments add constraint attachments_kind_chk
  check (kind in ('file', 'link'));

-- A file needs a stored object; a link needs an http(s) URL. The scheme check
-- lives in the database as well as the UI so a javascript:/data: URI can never
-- reach an <a href> even if inserted by something other than the web app.
alter table taskapp.attachments drop constraint if exists attachments_shape_chk;
alter table taskapp.attachments add constraint attachments_shape_chk
  check (
    (kind = 'file' and storage_path is not null)
    or (kind = 'link' and url is not null and url ~* '^https?://')
  );
