import { supabase } from "@/lib/supabase";

const FN_BASE = process.env.NEXT_PUBLIC_FN_BASE || "";

export class FnError extends Error {}

/** Call a yochi-taskhub-fn endpoint with the current Supabase session JWT. */
export async function callFn<T = unknown>(route: string, body: unknown): Promise<T> {
  if (!FN_BASE) throw new FnError("Function app URL not configured (NEXT_PUBLIC_FN_BASE)");
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  if (!token) throw new FnError("Not signed in");
  const res = await fetch(`${FN_BASE}/api/${route}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(body),
  });
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new FnError((json as { error?: string }).error || `HTTP ${res.status}`);
  return json as T;
}
