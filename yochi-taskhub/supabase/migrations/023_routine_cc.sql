-- Routine delivery: a CC list alongside the To list. Kept as its own column
-- rather than folded into recipients because To and CC mean different things
-- to the people receiving a report - To is "this is yours to action".
-- Applied 2026-08-26. (Includes the pgrst schema reload the 022 deploy taught
-- us to never forget.)
alter table taskapp.ai_skills
  add column if not exists cc text;
NOTIFY pgrst, 'reload schema';
