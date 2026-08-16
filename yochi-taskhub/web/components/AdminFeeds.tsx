"use client";

// Admin → Data feeds: the learning data centre registry. Recommended feeds
// await activation; active feeds research weekly/daily into the lake as
// feed_<slug> tables.
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/lib/supabase";

type Feed = {
  id: string; slug: string; name: string; description: string | null;
  kind: string; cadence: string; status: string;
  last_run_at: string | null; last_summary: string | null;
};

export default function AdminFeeds() {
  const qc = useQueryClient();
  const { data: feeds, error } = useQuery({
    queryKey: ["feeds"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("feeds").select("*").order("status").order("name");
      if (error) throw error;
      return data as Feed[];
    },
  });

  async function patch(id: string, p: Partial<Feed>) {
    await supabase.from("feeds").update(p).eq("id", id);
    qc.invalidateQueries({ queryKey: ["feeds"] });
  }

  if (error)
    return <div className="text-sm text-red-600">Feeds unavailable: {String(error)}</div>;

  const groups: [string, string][] = [
    ["active", "Active — researching into the lake"],
    ["recommended", "Recommended — activate to start collecting"],
    ["paused", "Paused"],
  ];

  return (
    <div className="space-y-5">
      <p className="text-xs text-gray-500">
        Each active feed is researched on its cadence (web sources, always cited) and lands
        in the data lake as <code>feed_&lt;slug&gt;</code> with full history — usable by the
        voice assistant, task agent, smart tasks and watch rules alongside internal data.
        Weekly feeds run Mondays 05:45, daily feeds every morning.
      </p>
      {groups.map(([status, label]) => {
        const rows = (feeds ?? []).filter((f) => f.status === status);
        if (!rows.length) return null;
        return (
          <div key={status}>
            <div className="mb-2 text-xs font-bold uppercase tracking-wide text-gray-400">
              {label}
            </div>
            <div className="overflow-hidden rounded-xl border border-gray-200 bg-white">
              {rows.map((f) => (
                <div key={f.id} className="flex items-start gap-3 border-b border-gray-50 px-4 py-3">
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-semibold">
                      {f.name}{" "}
                      <span className="rounded bg-gray-100 px-1.5 py-px text-[9px] font-bold uppercase text-gray-500">
                        {f.kind}
                      </span>
                    </div>
                    <div className="text-xs text-gray-500">{f.description}</div>
                    <div className="mt-0.5 text-[11px] text-gray-400">
                      feed_{f.slug} · {f.cadence}
                      {f.last_run_at && ` · last run ${f.last_run_at.slice(0, 10)}`}
                      {f.last_summary && ` — ${f.last_summary}`}
                    </div>
                  </div>
                  <div className="flex shrink-0 gap-1.5">
                    {f.status !== "active" && (
                      <button
                        onClick={() => patch(f.id, { status: "active" })}
                        className="rounded-lg bg-brand-600 px-2.5 py-1 text-xs font-semibold text-white"
                      >
                        Activate
                      </button>
                    )}
                    {f.status === "active" && (
                      <button
                        onClick={() => patch(f.id, { status: "paused" })}
                        className="rounded-lg border border-gray-200 px-2.5 py-1 text-xs text-gray-600"
                      >
                        Pause
                      </button>
                    )}
                    <select
                      value={f.cadence}
                      onChange={(e) => patch(f.id, { cadence: e.target.value })}
                      className="rounded-lg border border-gray-200 px-1.5 py-1 text-xs"
                    >
                      <option value="weekly">weekly</option>
                      <option value="daily">daily</option>
                    </select>
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
