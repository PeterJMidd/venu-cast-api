"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"signin" | "setpassword">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    // Invite / recovery links land here with a token in the URL hash;
    // supabase-js picks it up and fires PASSWORD_RECOVERY / SIGNED_IN.
    const { data: sub } = supabase.auth.onAuthStateChange((event) => {
      if (event === "PASSWORD_RECOVERY") setMode("setpassword");
    });
    if (typeof window !== "undefined" && /type=(invite|recovery)/.test(window.location.hash)) {
      setMode("setpassword");
    }
    return () => sub.subscription.unsubscribe();
  }, []);

  async function signIn(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    setBusy(false);
    if (error) setError(error.message);
    else router.replace("/my-tasks");
  }

  async function setNewPassword(e: React.FormEvent) {
    e.preventDefault();
    if (password !== confirm) {
      setError("Passwords do not match");
      return;
    }
    setBusy(true);
    setError(null);
    const { error } = await supabase.auth.updateUser({ password });
    setBusy(false);
    if (error) setError(error.message);
    else router.replace("/my-tasks");
  }

  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <div className="w-full max-w-sm rounded-xl border border-gray-200 bg-white p-8 shadow-sm">
        <div className="mb-6 text-center">
          <div className="text-2xl font-bold text-brand-600">Yo-Chi TaskHub</div>
          <div className="mt-1 text-sm text-gray-500">
            {mode === "signin" ? "Sign in to your account" : "Set your password"}
          </div>
        </div>

        <form onSubmit={mode === "signin" ? signIn : setNewPassword} className="space-y-4">
          {mode === "signin" && (
            <input
              type="email"
              required
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
            />
          )}
          <input
            type="password"
            required
            minLength={8}
            placeholder={mode === "signin" ? "Password" : "New password (min 8 chars)"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
          />
          {mode === "setpassword" && (
            <input
              type="password"
              required
              minLength={8}
              placeholder="Confirm password"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
            />
          )}
          {error && <div className="text-sm text-red-600">{error}</div>}
          {info && <div className="text-sm text-brand-600">{info}</div>}
          <button
            type="submit"
            disabled={busy}
            className="w-full rounded-lg bg-brand-600 py-2 text-sm font-semibold text-white hover:bg-brand-700 disabled:opacity-50"
          >
            {busy ? "Working…" : mode === "signin" ? "Sign in" : "Set password"}
          </button>
        </form>
        {mode === "signin" && (
          <button
            type="button"
            onClick={async () => {
              setError(null);
              setInfo(null);
              if (!email) {
                setError("Enter your email above first");
                return;
              }
              const { error } = await supabase.auth.resetPasswordForEmail(email, {
                redirectTo: window.location.origin + "/login",
              });
              if (error) setError(error.message);
              else setInfo("Reset email sent — check your inbox.");
            }}
            className="mt-3 w-full text-center text-xs text-gray-400 hover:text-brand-600"
          >
            Forgot password?
          </button>
        )}
      </div>
    </div>
  );
}
