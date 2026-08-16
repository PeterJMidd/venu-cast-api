"use client";

// Admin → Access: grant external users visibility of whole pillars
// (category_members) or individual projects (project_members). RLS enforces
// the actual boundary — this UI just manages the membership rows.
import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/lib/supabase";
import type { Category, Profile, Project } from "@/lib/types";

export default function AdminAccess() {
  const qc = useQueryClient();
  const [userId, setUserId] = useState<string>("");

  const { data: users } = useQuery({
    queryKey: ["profiles", "admin"],
    queryFn: async () => {
      const { data, error } = await supabase.from("profiles").select("*").order("email");
      if (error) throw error;
      return data as Profile[];
    },
  });
  const { data: categories } = useQuery({
    queryKey: ["categories"],
    queryFn: async () => {
      const { data, error } = await supabase.from("categories").select("*").order("sort");
      if (error) throw error;
      return data as Category[];
    },
  });
  const { data: projects } = useQuery({
    queryKey: ["projects", "admin-access"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("projects").select("*").eq("archived", false).order("name");
      if (error) throw error;
      return data as Project[];
    },
  });
  const { data: grants, error: grantsError } = useQuery({
    queryKey: ["access-grants", userId],
    enabled: !!userId,
    queryFn: async () => {
      const [cm, pm] = await Promise.all([
        supabase.from("category_members").select("category_id").eq("user_id", userId),
        supabase.from("project_members").select("project_id").eq("user_id", userId),
      ]);
      if (cm.error) throw cm.error;
      if (pm.error) throw pm.error;
      return {
        cats: new Set((cm.data ?? []).map((r) => r.category_id as number)),
        projs: new Set((pm.data ?? []).map((r) => r.project_id as string)),
      };
    },
  });

  const invalidate = () => qc.invalidateQueries({ queryKey: ["access-grants", userId] });

  async function toggleCat(id: number, on: boolean) {
    const { data: session } = await supabase.auth.getUser();
    if (on) {
      await supabase.from("category_members")
        .insert({ category_id: id, user_id: userId, added_by: session.user?.id });
    } else {
      await supabase.from("category_members")
        .delete().eq("category_id", id).eq("user_id", userId);
    }
    invalidate();
  }
  async function toggleProj(id: string, on: boolean) {
    const { data: session } = await supabase.auth.getUser();
    if (on) {
      await supabase.from("project_members")
        .insert({ project_id: id, user_id: userId, added_by: session.user?.id });
    } else {
      await supabase.from("project_members")
        .delete().eq("project_id", id).eq("user_id", userId);
    }
    invalidate();
  }

  const externals = (users ?? []).filter((u) => u.role === "external");
  const others = (users ?? []).filter((u) => u.role !== "external");

  if (grantsError) {
    return (
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
        Access control isn&rsquo;t activated yet — the 006_external_access migration needs to be
        applied to the database first (see supabase/migrations/006_external_access.sql).
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-gray-200 bg-white p-4">
        <div className="mb-2 text-sm font-semibold text-gray-600">
          Who are you granting access for?
        </div>
        <select
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          className="w-80 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
        >
          <option value="">Select a user…</option>
          {externals.length > 0 && (
            <optgroup label="External users">
              {externals.map((u) => (
                <option key={u.id} value={u.id}>
                  {u.full_name || u.email} ({u.email})
                </option>
              ))}
            </optgroup>
          )}
          <optgroup label="Internal (grants only matter for externals)">
            {others.map((u) => (
              <option key={u.id} value={u.id}>
                {u.full_name || u.email} — {u.role}
              </option>
            ))}
          </optgroup>
        </select>
        <p className="mt-2 text-xs text-gray-400">
          External users see ONLY the pillars/projects granted here (plus anything directly
          assigned to them). Admin and finance always see everything — grants are ignored.
        </p>
      </div>

      {userId && grants && (
        <div className="grid gap-4 md:grid-cols-2">
          <div className="rounded-xl border border-gray-200 bg-white p-4">
            <div className="mb-2 text-sm font-semibold text-gray-600">Whole pillars</div>
            <div className="space-y-1.5">
              {(categories ?? []).map((c) => (
                <label key={c.id} className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={grants.cats.has(c.id)}
                    onChange={(e) => toggleCat(c.id, e.target.checked)}
                  />
                  {c.name}
                </label>
              ))}
            </div>
          </div>
          <div className="rounded-xl border border-gray-200 bg-white p-4">
            <div className="mb-2 text-sm font-semibold text-gray-600">Individual projects</div>
            <div className="max-h-80 space-y-1.5 overflow-y-auto">
              {(projects ?? []).map((p) => {
                const viaPillar = grants.cats.has(p.category_id);
                return (
                  <label
                    key={p.id}
                    className={`flex items-center gap-2 text-sm ${viaPillar ? "opacity-50" : ""}`}
                  >
                    <input
                      type="checkbox"
                      checked={viaPillar || grants.projs.has(p.id)}
                      disabled={viaPillar}
                      onChange={(e) => toggleProj(p.id, e.target.checked)}
                    />
                    {p.name}
                    {viaPillar && <span className="text-[10px] text-gray-400">(via pillar)</span>}
                  </label>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
