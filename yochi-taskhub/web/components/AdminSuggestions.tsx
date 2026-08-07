"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { format, parseISO } from "date-fns";
import { supabase } from "@/lib/supabase";

interface Suggestion {
  id: string;
  kind: string;
  payload: { template_title?: string; change?: string; [k: string]: unknown };
  rationale: string | null;
  status: "pending" | "accepted" | "dismissed";
  created_at: string;
}

export default function AdminSuggestions() {
  const qc = useQueryClient();
  const { data: suggestions } = useQuery({
    queryKey: ["ai_suggestions"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("ai_suggestions")
        .select("*")
        .order("created_at", { ascending: false })
        .limit(100);
      if (error) throw error;
      return data as Suggestion[];
    },
  });

  async function setStatus(id: string, status: "accepted" | "dismissed") {
    await supabase.from("ai_suggestions").update({ status }).eq("id", id);
    qc.invalidateQueries({ queryKey: ["ai_suggestions"] });
  }

  const pending = suggestions?.filter((s) => s.status === "pending") ?? [];
  const resolved = suggestions?.filter((s) => s.status !== "pending") ?? [];

  return (
    <div className="max-w-2xl">
      <p className="mb-4 text-sm text-gray-500">
        On the 1st of each month the learning loop analyses how templated work actually ran and
        proposes improvements. Accepting is a decision record — apply the change on the Templates page.
      </p>
      {pending.length === 0 && (
        <div className="rounded-lg border border-dashed border-gray-300 p-6 text-center text-sm text-gray-400">
          No pending suggestions. The next learning pass runs on the 1st.
        </div>
      )}
      <div className="space-y-3">
        {pending.map((s) => (
          <div key={s.id} className="rounded-xl border border-gray-200 bg-white p-4">
            <div className="mb-1 flex items-center gap-2 text-xs">
              <span className="rounded bg-brand-50 px-1.5 py-0.5 font-semibold text-brand-700">{s.kind.replace("_", " ")}</span>
              <span className="text-gray-400">{format(parseISO(s.created_at), "d MMM yyyy")}</span>
            </div>
            <div className="text-sm font-semibold">{s.payload.template_title}</div>
            <div className="mt-0.5 text-sm">{s.payload.change}</div>
            {s.rationale && <div className="mt-1 text-xs italic text-gray-500">{s.rationale}</div>}
            <div className="mt-3 flex gap-2">
              <button onClick={() => setStatus(s.id, "accepted")} className="rounded-lg bg-brand-600 px-3 py-1 text-xs font-semibold text-white hover:bg-brand-700">Accept</button>
              <button onClick={() => setStatus(s.id, "dismissed")} className="rounded-lg px-3 py-1 text-xs text-gray-500 hover:bg-gray-100">Dismiss</button>
            </div>
          </div>
        ))}
      </div>
      {resolved.length > 0 && (
        <details className="mt-5">
          <summary className="cursor-pointer text-xs text-gray-400">History ({resolved.length})</summary>
          <div className="mt-2 space-y-1">
            {resolved.map((s) => (
              <div key={s.id} className="flex items-center justify-between rounded-lg bg-gray-50 px-3 py-1.5 text-xs text-gray-500">
                <span className="truncate">{s.payload.template_title}: {s.payload.change}</span>
                <span className={`ml-2 shrink-0 font-semibold ${s.status === "accepted" ? "text-brand-600" : "text-gray-400"}`}>{s.status}</span>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
