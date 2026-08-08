"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";

interface Prefs {
  daily_briefing: boolean;
  email_on_assign: boolean;
}

export default function PrefsModal({ userId, onClose }: { userId: string; onClose: () => void }) {
  const [prefs, setPrefs] = useState<Prefs | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    supabase
      .from("notification_prefs")
      .select("daily_briefing,email_on_assign")
      .eq("user_id", userId)
      .maybeSingle()
      .then(({ data }) => setPrefs((data as Prefs) ?? { daily_briefing: true, email_on_assign: true }));
  }, [userId]);

  async function toggle(key: keyof Prefs) {
    if (!prefs) return;
    const next = { ...prefs, [key]: !prefs[key] };
    setPrefs(next);
    const { error } = await supabase
      .from("notification_prefs")
      .upsert({ user_id: userId, ...next }, { onConflict: "user_id" });
    if (error) setErr(error.message);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4" onClick={onClose}>
      <div className="w-full max-w-sm rounded-xl bg-white p-5 shadow-xl" onClick={(e) => e.stopPropagation()}>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-base font-bold">Notification settings</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">✕</button>
        </div>
        {!prefs ? (
          <div className="text-sm text-gray-400">Loading…</div>
        ) : (
          <div className="space-y-3">
            <label className="flex items-center justify-between text-sm">
              <span>
                Daily morning briefing
                <span className="block text-xs text-gray-400">7:00 AM email on business days</span>
              </span>
              <input type="checkbox" checked={prefs.daily_briefing} onChange={() => toggle("daily_briefing")} className="h-4 w-4 accent-brand-600" />
            </label>
            <label className="flex items-center justify-between text-sm">
              <span>
                Assignment emails
                <span className="block text-xs text-gray-400">when a task is assigned to you</span>
              </span>
              <input type="checkbox" checked={prefs.email_on_assign} onChange={() => toggle("email_on_assign")} className="h-4 w-4 accent-brand-600" />
            </label>
            <p className="text-xs text-gray-400">@mention and comment emails always send — they mean someone needs you.</p>
          </div>
        )}
        {err && <div className="mt-2 text-sm text-red-600">{err}</div>}
      </div>
    </div>
  );
}
