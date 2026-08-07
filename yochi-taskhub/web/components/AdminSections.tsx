"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/lib/supabase";
import type { Category } from "@/lib/types";

export default function AdminSections() {
  const qc = useQueryClient();
  const [newName, setNewName] = useState("");
  const [err, setErr] = useState<string | null>(null);

  const { data: categories } = useQuery({
    queryKey: ["categories"],
    queryFn: async () => {
      const { data, error } = await supabase.from("categories").select("*").order("sort");
      if (error) throw error;
      return data as Category[];
    },
  });

  const invalidate = () => qc.invalidateQueries({ queryKey: ["categories"] });

  async function rename(c: Category, name: string) {
    if (!name.trim() || name === c.name) return;
    const { error } = await supabase.from("categories").update({ name: name.trim() }).eq("id", c.id);
    if (error) setErr(error.message);
    invalidate();
  }

  async function move(c: Category, dir: -1 | 1) {
    const list = categories ?? [];
    const idx = list.findIndex((x) => x.id === c.id);
    const other = list[idx + dir];
    if (!other) return;
    setErr(null);
    await supabase.from("categories").update({ sort: other.sort }).eq("id", c.id);
    await supabase.from("categories").update({ sort: c.sort }).eq("id", other.id);
    invalidate();
  }

  async function add(e: React.FormEvent) {
    e.preventDefault();
    if (!newName.trim()) return;
    setErr(null);
    const maxId = Math.max(0, ...(categories ?? []).map((c) => c.id));
    const maxSort = Math.max(0, ...(categories ?? []).map((c) => c.sort));
    const { error } = await supabase
      .from("categories")
      .insert({ id: maxId + 1, name: newName.trim(), sort: maxSort + 1 });
    if (error) setErr(error.message);
    else setNewName("");
    invalidate();
  }

  async function remove(c: Category) {
    setErr(null);
    const { error } = await supabase.from("categories").delete().eq("id", c.id);
    if (error)
      setErr(
        error.message.includes("foreign key")
          ? `"${c.name}" still has projects — move or delete them first.`
          : error.message
      );
    invalidate();
  }

  return (
    <div className="max-w-xl">
      <p className="mb-4 text-sm text-gray-500">
        Sections are the top-level structure of the whole app (sidebar, reporting, AI task routing).
        Rename inline, reorder with the arrows. A section with projects can&rsquo;t be deleted.
      </p>
      <div className="space-y-2">
        {categories?.map((c, i) => (
          <div key={c.id} className="flex items-center gap-2 rounded-lg border border-gray-200 bg-white px-3 py-2">
            <div className="flex flex-col text-xs leading-none text-gray-400">
              <button onClick={() => move(c, -1)} disabled={i === 0} className="hover:text-brand-600 disabled:opacity-20">▲</button>
              <button onClick={() => move(c, 1)} disabled={i === (categories?.length ?? 0) - 1} className="hover:text-brand-600 disabled:opacity-20">▼</button>
            </div>
            <input
              defaultValue={c.name}
              onBlur={(e) => rename(c, e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
              className="flex-1 rounded px-2 py-1 text-sm focus:bg-brand-50 focus:outline-none"
            />
            <button onClick={() => remove(c)} className="text-xs text-gray-300 hover:text-red-600">delete</button>
          </div>
        ))}
      </div>
      <form onSubmit={add} className="mt-3 flex gap-2">
        <input
          placeholder="New section name…"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
        />
        <button className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700">Add</button>
      </form>
      {err && <div className="mt-2 text-sm text-red-600">{err}</div>}
    </div>
  );
}
