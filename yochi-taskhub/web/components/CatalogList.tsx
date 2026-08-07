"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/lib/supabase";
import { callFn } from "@/lib/fn";

export interface CatalogTable {
  view?: string;
  name?: string;
  columns?: string[];
  rows?: number;
  source_rows?: number;
  date_col?: string;
  date_min?: string;
  date_max?: string;
  note?: string;
}

interface PeekResult {
  columns: string[];
  rows: unknown[][];
}

export function useCatalog() {
  return useQuery({
    queryKey: ["lake_catalog"],
    queryFn: async () => {
      const { data } = await supabase.auth.getSession();
      const res = await fetch(`${process.env.NEXT_PUBLIC_FN_BASE}/api/lake_catalog`, {
        headers: { Authorization: `Bearer ${data.session?.access_token}` },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return (await res.json()) as { tables: CatalogTable[] };
    },
    staleTime: 30 * 60 * 1000,
  });
}

export default function CatalogList({ search }: { search: string }) {
  const { data: catalog, isLoading, error } = useCatalog();
  const [open, setOpen] = useState<string | null>(null);
  const [peeks, setPeeks] = useState<Record<string, PeekResult | "loading" | { error: string }>>({});

  async function peek(table: string) {
    setPeeks((p) => ({ ...p, [table]: "loading" }));
    try {
      const r = await callFn<PeekResult>("peek_table", { table });
      setPeeks((p) => ({ ...p, [table]: r }));
    } catch (e) {
      setPeeks((p) => ({ ...p, [table]: { error: e instanceof Error ? e.message : String(e) } }));
    }
  }

  const tables = (catalog?.tables ?? [])
    .map((t) => ({ ...t, _name: t.view || t.name || "" }))
    .filter((t) => t._name)
    .filter(
      (t) =>
        !search ||
        t._name.toLowerCase().includes(search.toLowerCase()) ||
        (t.columns ?? []).some((c) => c.toLowerCase().includes(search.toLowerCase()))
    )
    .sort((a, b) => a._name.localeCompare(b._name));

  if (isLoading) return <div className="py-8 text-center text-sm text-gray-400">Loading catalog…</div>;
  if (error)
    return (
      <div className="py-8 text-center text-sm text-red-600">
        {error instanceof Error ? error.message : String(error)}
      </div>
    );

  return (
    <div className="space-y-1.5">
      {tables.map((t) => {
        const p = peeks[t._name];
        return (
          <div key={t._name} className="rounded-lg border border-gray-200 bg-white">
            <button
              onClick={() => setOpen(open === t._name ? null : t._name)}
              className="flex w-full items-center justify-between px-3 py-2 text-left"
            >
              <span className="font-mono text-sm font-semibold text-gray-800">{t._name}</span>
              <span className="ml-3 shrink-0 text-xs text-gray-400">
                {(t.source_rows ?? t.rows ?? 0).toLocaleString()} rows
                {t.date_col &&
                  ` · ${String(t.date_min ?? "").slice(0, 10)} → ${String(t.date_max ?? "").slice(0, 10)}`}
              </span>
            </button>
            {open === t._name && (
              <div className="border-t border-gray-100 px-3 py-2">
                {t.note && <div className="mb-1.5 text-xs italic text-gray-500">{t.note}</div>}
                <div className="mb-2 flex flex-wrap gap-1">
                  {(t.columns ?? []).map((c) => (
                    <span key={c} className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-[11px] text-gray-600">
                      {c}
                    </span>
                  ))}
                </div>
                {!p && (
                  <button onClick={() => peek(t._name)} className="text-xs font-semibold text-brand-600 hover:underline">
                    Peek at sample rows →
                  </button>
                )}
                {p === "loading" && <div className="text-xs text-gray-400">Fetching sample…</div>}
                {p && p !== "loading" && "error" in p && (
                  <div className="text-xs text-red-600">{p.error}</div>
                )}
                {p && p !== "loading" && "rows" in p && (
                  <div className="overflow-x-auto rounded-lg border border-gray-100">
                    <table className="w-full text-[11px]">
                      <thead className="bg-gray-50 text-left text-gray-400">
                        <tr>
                          {p.columns.slice(0, 8).map((c) => (
                            <th key={c} className="px-2 py-1 font-mono">{c}</th>
                          ))}
                          {p.columns.length > 8 && <th className="px-2 py-1">+{p.columns.length - 8} cols</th>}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-50">
                        {p.rows.slice(0, 5).map((r, i) => (
                          <tr key={i}>
                            {r.slice(0, 8).map((v, j) => (
                              <td key={j} className="max-w-[140px] truncate px-2 py-1 text-gray-600">
                                {v === null ? "—" : String(v)}
                              </td>
                            ))}
                            {p.columns.length > 8 && <td className="px-2 py-1 text-gray-300">…</td>}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
      {tables.length === 0 && <div className="py-8 text-center text-sm text-gray-300">No tables match.</div>}
    </div>
  );
}
