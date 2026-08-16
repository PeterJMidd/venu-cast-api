"use client";

// Admin → Data feeds: the learning data centre registry. Recommended feeds
// await activation; active feeds research weekly/daily into the lake as
// feed_<slug> tables.
import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/lib/supabase";

const INTERNAL_LOADS: [string, string][] = [
  ["03:00 / 03:20", "Restoke + OpCentral API exports → lake"],
  ["04:45", "Full dataSights lake sync (all views)"],
  ["05:10", "Marts rebuild (venue daily/monthly, reviews, food cost)"],
  ["05:45", "External data feeds (daily; weekly ones Mondays) → feed_* tables"],
  ["05:50 / 06:20", "SharePoint document mirror + document index"],
  ["06:40", "Asana → TaskHub sync"],
  ["07:45 · 10:30 · 16:30", "Mart top-ups (catch upstream late arrivals — keeps sales current to yesterday)"],
  ["10:45 / 10:55", "Revenue assurance + P&L pulse (post top-up)"],
];

type Feed = {
  id: string; slug: string; name: string; description: string | null;
  kind: string; cadence: string; status: string;
  last_run_at: string | null; last_summary: string | null;
};

export default function AdminFeeds() {
  const qc = useQueryClient();
  const [nfName, setNfName] = useState("");
  const [nfPrompt, setNfPrompt] = useState("");
  const [nfKind, setNfKind] = useState("industry");
  const [nfMsg, setNfMsg] = useState<string | null>(null);
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
      <form
        onSubmit={async (e) => {
          e.preventDefault();
          if (!nfName.trim() || !nfPrompt.trim()) return;
          const slug = nfName.toLowerCase().replace(/[^a-z0-9]+/g, "_").slice(0, 50);
          const { error } = await supabase.from("feeds").insert({
            slug, name: nfName.trim(), kind: nfKind, cadence: "weekly",
            status: "active", description: "Custom feed added by admin",
            research_prompt: nfPrompt.trim(),
            columns: [
              { name: "item", description: "what this row is about" },
              { name: "value", description: "the key number/fact" },
              { name: "as_of", description: "date it applies" },
              { name: "notes", description: "detail + source" },
            ],
          });
          setNfMsg(error ? error.message : `Feed '${slug}' added and active — first run next 05:45.`);
          if (!error) { setNfName(""); setNfPrompt(""); }
          qc.invalidateQueries({ queryKey: ["feeds"] });
        }}
        className="rounded-xl border border-gray-200 bg-white p-4"
      >
        <div className="mb-2 text-xs font-bold uppercase tracking-wide text-gray-400">
          Add a custom feed
        </div>
        <div className="flex flex-wrap gap-2">
          <input
            value={nfName}
            onChange={(e) => setNfName(e.target.value)}
            placeholder="Name (e.g. NSW electricity tariffs)"
            className="w-64 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
          />
          <select
            value={nfKind}
            onChange={(e) => setNfKind(e.target.value)}
            className="rounded-lg border border-gray-300 px-2 py-2 text-sm"
          >
            {["award", "tax_rates", "due_dates", "economy", "industry", "other"].map((k) => (
              <option key={k} value={k}>{k}</option>
            ))}
          </select>
        </div>
        <textarea
          value={nfPrompt}
          onChange={(e) => setNfPrompt(e.target.value)}
          rows={2}
          placeholder="What should be researched each run? Be specific about sources and what facts to capture."
          className="mt-2 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
        />
        <button className="mt-2 rounded-lg bg-brand-600 px-3 py-1.5 text-xs font-semibold text-white">
          Add & activate
        </button>
        {nfMsg && <div className="mt-1 text-xs text-gray-500">{nfMsg}</div>}
      </form>

      <div className="rounded-xl border border-gray-200 bg-white p-4">
        <div className="mb-2 text-xs font-bold uppercase tracking-wide text-gray-400">
          Internal pipeline — what loads automatically (AEST)
        </div>
        <div className="space-y-1">
          {INTERNAL_LOADS.map(([time, what]) => (
            <div key={time} className="flex gap-3 text-xs">
              <span className="w-32 shrink-0 font-semibold text-gray-600">{time}</span>
              <span className="text-gray-600">{what}</span>
            </div>
          ))}
        </div>
        <p className="mt-2 text-[11px] text-gray-400">
          Internal schedules are managed in Azure (app settings on the lake/TaskHub function
          apps). External feeds above are fully self-service.
        </p>
      </div>

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
