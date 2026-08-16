"use client";

// Scoped status report: pick everything / a pillar / a project -> PDF or CSV.
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/lib/supabase";
import type { Category, Project } from "@/lib/types";

export default function ReportButton() {
  const [open, setOpen] = useState(false);
  const [catId, setCatId] = useState<string>("");
  const [projId, setProjId] = useState<string>("");
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const { data: categories } = useQuery({
    queryKey: ["categories"],
    queryFn: async () => {
      const { data, error } = await supabase.from("categories").select("*").order("sort");
      if (error) throw error;
      return data as Category[];
    },
  });
  const { data: projects } = useQuery({
    queryKey: ["projects", "report"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("projects").select("*").eq("archived", false).order("name");
      if (error) throw error;
      return data as Project[];
    },
  });

  const scoped = (projects ?? []).filter((p) => !catId || p.category_id === Number(catId));

  async function download(format: "pdf" | "csv") {
    setBusy(format);
    setErr(null);
    try {
      const { data } = await supabase.auth.getSession();
      const res = await fetch(`${process.env.NEXT_PUBLIC_FN_BASE}/api/report`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${data.session?.access_token}`,
        },
        body: JSON.stringify({
          category_id: catId ? Number(catId) : null,
          project_id: projId || null,
          format,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `taskhub_report_${new Date().toISOString().slice(0, 10)}.${format}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-xs font-semibold text-gray-600 hover:border-brand-500"
      >
        ⬇ Report
      </button>
      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
             onClick={() => setOpen(false)}>
          <div className="w-full max-w-sm rounded-2xl bg-white p-5 shadow-2xl"
               onClick={(e) => e.stopPropagation()}>
            <h2 className="mb-3 text-base font-bold">Status report</h2>
            <label className="mb-1 block text-xs font-semibold text-gray-500">Pillar</label>
            <select
              value={catId}
              onChange={(e) => { setCatId(e.target.value); setProjId(""); }}
              className="mb-3 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
            >
              <option value="">All pillars</option>
              {(categories ?? []).map((c) => (
                <option key={c.id} value={c.id}>{c.name}</option>
              ))}
            </select>
            <label className="mb-1 block text-xs font-semibold text-gray-500">Project</label>
            <select
              value={projId}
              onChange={(e) => setProjId(e.target.value)}
              className="mb-4 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
            >
              <option value="">All projects{catId ? " in pillar" : ""}</option>
              {scoped.map((p) => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
            <div className="flex gap-2">
              <button
                onClick={() => download("pdf")}
                disabled={!!busy}
                className="flex-1 rounded-lg bg-brand-600 px-3 py-2 text-sm font-semibold text-white disabled:opacity-50"
              >
                {busy === "pdf" ? "Building…" : "PDF"}
              </button>
              <button
                onClick={() => download("csv")}
                disabled={!!busy}
                className="flex-1 rounded-lg border border-brand-600 px-3 py-2 text-sm font-semibold text-brand-700 disabled:opacity-50"
              >
                {busy === "csv" ? "Building…" : "CSV"}
              </button>
            </div>
            {err && <div className="mt-2 text-xs text-red-600">{err}</div>}
          </div>
        </div>
      )}
    </>
  );
}
