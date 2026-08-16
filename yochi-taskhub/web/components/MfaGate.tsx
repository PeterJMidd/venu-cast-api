"use client";

// MFA enforcement: admin + finance MUST enrol a TOTP factor (authenticator
// app — Microsoft/Google Authenticator, 1Password) and verify a 6-digit code
// whenever the session is below AAL2. Other roles are untouched (enrolment
// for them can be added later). Blocks the app with a modal until satisfied.
import { useCallback, useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";
import { useProfile } from "@/hooks/useProfile";

type Mode = "checking" | "ok" | "enroll" | "challenge";

export default function MfaGate() {
  const { data: profile } = useProfile();
  const [mode, setMode] = useState<Mode>("checking");
  const [qr, setQr] = useState<string | null>(null);
  const [secret, setSecret] = useState<string | null>(null);
  const [factorId, setFactorId] = useState<string | null>(null);
  const [code, setCode] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const enforced = profile && (profile.role === "admin" || profile.role === "finance");

  const check = useCallback(async () => {
    if (!enforced) {
      setMode("ok");
      return;
    }
    const { data: aal } = await supabase.auth.mfa.getAuthenticatorAssuranceLevel();
    if (aal?.currentLevel === "aal2") {
      setMode("ok");
      return;
    }
    const { data: factors } = await supabase.auth.mfa.listFactors();
    const totp = factors?.totp?.find((f) => f.status === "verified");
    if (totp) {
      setFactorId(totp.id);
      setMode("challenge");
    } else {
      setMode("enroll");
    }
  }, [enforced]);

  useEffect(() => {
    if (profile) check();
  }, [profile, check]);

  async function startEnroll() {
    setBusy(true);
    setErr(null);
    try {
      // clear any half-finished enrolments first
      const { data: factors } = await supabase.auth.mfa.listFactors();
      for (const f of factors?.totp ?? []) {
        if (f.status !== "verified") await supabase.auth.mfa.unenroll({ factorId: f.id });
      }
      const { data, error } = await supabase.auth.mfa.enroll({
        factorType: "totp",
        friendlyName: "Authenticator app",
      });
      if (error) throw error;
      setFactorId(data.id);
      setQr(data.totp.qr_code);
      setSecret(data.totp.secret);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function verify(e: React.FormEvent) {
    e.preventDefault();
    if (!factorId || code.length < 6) return;
    setBusy(true);
    setErr(null);
    try {
      const { data: ch, error: cErr } = await supabase.auth.mfa.challenge({ factorId });
      if (cErr) throw cErr;
      const { error: vErr } = await supabase.auth.mfa.verify({
        factorId,
        challengeId: ch.id,
        code: code.trim(),
      });
      if (vErr) throw vErr;
      setCode("");
      setQr(null);
      setMode("ok");
    } catch (e) {
      setErr("That code didn't verify — check the app and try again.");
    } finally {
      setBusy(false);
    }
  }

  if (mode === "ok" || mode === "checking" || !enforced) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-md rounded-2xl bg-white p-6 shadow-2xl">
        {mode === "enroll" && !qr && (
          <>
            <h2 className="mb-1 text-lg font-bold">Set up two-factor authentication</h2>
            <p className="mb-4 text-sm text-gray-500">
              Your role ({profile?.role}) requires MFA. You&rsquo;ll scan a QR code with an
              authenticator app — Microsoft Authenticator, Google Authenticator or 1Password —
              then enter the 6-digit code it shows.
            </p>
            <button
              onClick={startEnroll}
              disabled={busy}
              className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
            >
              {busy ? "Preparing…" : "Show my QR code"}
            </button>
          </>
        )}

        {mode === "enroll" && qr && (
          <>
            <h2 className="mb-1 text-lg font-bold">Scan with your authenticator app</h2>
            {/* qr_code is an SVG data URI from Supabase */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={qr} alt="MFA QR code" className="mx-auto my-3 h-44 w-44" />
            {secret && (
              <p className="mb-3 break-all text-center text-[11px] text-gray-400">
                Can&rsquo;t scan? Enter this key manually: <code>{secret}</code>
              </p>
            )}
            <form onSubmit={verify} className="flex gap-2">
              <input
                value={code}
                onChange={(e) => setCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                placeholder="6-digit code"
                inputMode="numeric"
                autoFocus
                className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-center text-lg tracking-widest focus:border-brand-500 focus:outline-none"
              />
              <button
                disabled={busy || code.length < 6}
                className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
              >
                Verify
              </button>
            </form>
          </>
        )}

        {mode === "challenge" && (
          <>
            <h2 className="mb-1 text-lg font-bold">Two-factor check</h2>
            <p className="mb-4 text-sm text-gray-500">
              Enter the 6-digit code from your authenticator app.
            </p>
            <form onSubmit={verify} className="flex gap-2">
              <input
                value={code}
                onChange={(e) => setCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                placeholder="000000"
                inputMode="numeric"
                autoFocus
                className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-center text-lg tracking-widest focus:border-brand-500 focus:outline-none"
              />
              <button
                disabled={busy || code.length < 6}
                className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
              >
                Verify
              </button>
            </form>
            <button
              onClick={() => supabase.auth.signOut()}
              className="mt-3 text-xs text-gray-400 hover:text-gray-600"
            >
              Sign out instead
            </button>
          </>
        )}

        {err && <div className="mt-3 text-sm text-red-600">{err}</div>}
      </div>
    </div>
  );
}
