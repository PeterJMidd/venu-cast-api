"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/lib/supabase";
import { callFn } from "@/lib/fn";
import { useProfile } from "@/hooks/useProfile";
import AdminSections from "@/components/AdminSections";
import AdminWatchRules from "@/components/AdminWatchRules";
import AdminSuggestions from "@/components/AdminSuggestions";
import AdminSkills from "@/components/AdminSkills";
import AdminAccess from "@/components/AdminAccess";
import AdminFeeds from "@/components/AdminFeeds";
import type { Profile, UserRole } from "@/lib/types";

type AdminTab = "users" | "access" | "feeds" | "sections" | "rules" | "skills" | "suggestions";

export default function AdminPage() {
  const qc = useQueryClient();
  const { data: me } = useProfile();
  const [tab, setTab] = useState<AdminTab>("users");
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteName, setInviteName] = useState("");
  const [inviteRole, setInviteRole] = useState<UserRole>("finance");
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const { data: users } = useQuery({
    queryKey: ["profiles", "admin"],
    queryFn: async () => {
      const { data, error } = await supabase.from("profiles").select("*").order("email");
      if (error) throw error;
      return data as Profile[];
    },
  });

  if (me && me.role !== "admin") {
    return <div className="p-8 text-sm text-gray-400">Admin access required.</div>;
  }

  async function invite(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    setMsg(null);
    try {
      await callFn("invite_user", {
        email: inviteEmail,
        full_name: inviteName,
        role: inviteRole,
        redirect_to: window.location.origin + "/login",
      });
      setMsg(`Invite sent to ${inviteEmail}`);
      setInviteEmail("");
      setInviteName("");
      qc.invalidateQueries({ queryKey: ["profiles"] });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setErr(
        msg.includes("email_exists") || msg.includes("already been registered")
          ? "That person already has an account — use “Send reset” in the list below to let them (re)set their password."
          : msg
      );
    } finally {
      setBusy(false);
    }
  }

  async function sendReset(email: string) {
    setErr(null);
    setMsg(null);
    const { error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: window.location.origin + "/login",
    });
    if (error) setErr(error.message);
    else setMsg(`Password reset email sent to ${email}`);
  }

  async function updateUser(userId: string, patch: { role?: UserRole; active?: boolean }) {
    setErr(null);
    try {
      await callFn("update_user", { user_id: userId, ...patch });
      qc.invalidateQueries({ queryKey: ["profiles"] });
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  const inputCls =
    "rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none";

  const TABS: { key: AdminTab; label: string }[] = [
    { key: "users", label: "Users & roles" },
    { key: "access", label: "Access" },
    { key: "feeds", label: "Data feeds" },
    { key: "sections", label: "Sections" },
    { key: "rules", label: "Watch rules" },
    { key: "skills", label: "AI skills" },
    { key: "suggestions", label: "AI suggestions" },
  ];

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <h1 className="mb-4 text-xl font-bold">Admin</h1>
      <div className="mb-6 flex gap-1 border-b border-gray-200">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`-mb-px border-b-2 px-3 py-2 text-sm ${
              tab === t.key
                ? "border-brand-600 font-semibold text-brand-700"
                : "border-transparent text-gray-500 hover:text-gray-700"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "access" && <AdminAccess />}
      {tab === "feeds" && <AdminFeeds />}
      {tab === "sections" && <AdminSections />}
      {tab === "rules" && <AdminWatchRules />}
      {tab === "skills" && <AdminSkills />}
      {tab === "suggestions" && <AdminSuggestions />}

      {tab === "users" && (
      <>
      <form onSubmit={invite} className="mb-8 rounded-xl border border-gray-200 bg-white p-4">
        <div className="mb-3 text-sm font-semibold text-gray-600">Invite a user</div>
        <div className="flex flex-wrap gap-2">
          <input
            type="email"
            required
            placeholder="email@yochi.com.au"
            value={inviteEmail}
            onChange={(e) => setInviteEmail(e.target.value)}
            className={`${inputCls} w-64`}
          />
          <input
            placeholder="Full name"
            value={inviteName}
            onChange={(e) => setInviteName(e.target.value)}
            className={`${inputCls} w-48`}
          />
          <select
            value={inviteRole}
            onChange={(e) => setInviteRole(e.target.value as UserRole)}
            className={inputCls}
          >
            <option value="admin">Admin</option>
            <option value="finance">Finance</option>
            <option value="stakeholder">Stakeholder</option>
            <option value="external">External (advisor)</option>
          </select>
          <button
            disabled={busy}
            className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700 disabled:opacity-50"
          >
            {busy ? "Sending…" : "Send invite"}
          </button>
        </div>
        {msg && <div className="mt-2 text-sm text-brand-600">{msg}</div>}
        {err && <div className="mt-2 text-sm text-red-600">{err}</div>}
      </form>

      <div className="overflow-hidden rounded-xl border border-gray-200 bg-white">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-left text-xs uppercase tracking-wide text-gray-400">
            <tr>
              <th className="px-4 py-2.5">User</th>
              <th className="px-4 py-2.5">Role</th>
              <th className="px-4 py-2.5">Status</th>
              <th className="px-4 py-2.5"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {users?.map((u) => (
              <tr key={u.id} className={u.active ? "" : "opacity-50"}>
                <td className="px-4 py-2.5">
                  <div className="font-medium">{u.full_name || "—"}</div>
                  <div className="text-xs text-gray-400">{u.email}</div>
                </td>
                <td className="px-4 py-2.5">
                  <select
                    value={u.role}
                    disabled={u.id === me?.id}
                    onChange={(e) => updateUser(u.id, { role: e.target.value as UserRole })}
                    className="rounded-lg border border-gray-200 px-2 py-1 text-sm disabled:bg-gray-50"
                  >
                    <option value="admin">Admin</option>
                    <option value="finance">Finance</option>
                    <option value="stakeholder">Stakeholder</option>
            <option value="external">External (advisor)</option>
                  </select>
                </td>
                <td className="px-4 py-2.5">
                  <span className={`text-xs font-semibold ${u.active ? "text-brand-600" : "text-gray-400"}`}>
                    {u.active ? "Active" : "Deactivated"}
                  </span>
                </td>
                <td className="px-4 py-2.5 text-right">
                  <button
                    onClick={() => sendReset(u.email)}
                    className="mr-3 text-xs text-gray-400 hover:text-brand-600"
                  >
                    Send reset
                  </button>
                  {u.id !== me?.id && (
                    <button
                      onClick={() => updateUser(u.id, { active: !u.active })}
                      className="text-xs text-gray-400 hover:text-red-600"
                    >
                      {u.active ? "Deactivate" : "Reactivate"}
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      </>
      )}
    </div>
  );
}
