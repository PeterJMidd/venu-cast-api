import { createClient } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

// All app tables live in the `taskapp` schema (must be in the project's
// "Exposed schemas" list). Auth + storage are schema-independent.
export const supabase = createClient(url, anonKey, {
  db: { schema: "taskapp" },
});
