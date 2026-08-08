"use client";

import { Suspense } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { format, isBefore, parseISO, startOfDay } from "date-fns";
import { supabase } from "@/lib/supabase";
import { useOpenTask } from "@/components/TaskCard";
import { useProfile } from "@/hooks/useProfile";
import type { Task } from "@/lib/types";

interface DashData {
  trading: {
    series: { d: string; sales: number; ly: number }[];
    latest_day: string | null;
    latest_sales: number | null;
    latest_ly: number | null;
  };
  forecast: { d: string; forecast_sales: number }[];
  procedures: { procs: number; done: number; cancelled: number; day: string } | null;
}

const ESTATE = [
  { name: "Dashboard portal", desc: "Daily decks, Restoke ops reports", url: "https://zealous-stone-02ebae600.7.azurestaticapps.net" },
  { name: "Lake agent", desc: "Chat with all 108 data tables", url: "https://yochi-lake-agent.azurewebsites.net" },
  { name: "Data lake", desc: "Catalog, sample rows, ask-the-lake", url: "/lake" },
  { name: "Reporting", desc: "Task & close metrics", url: "/reporting" },
];

function Tile({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: "bad" | "good" | "warn" }) {
  const color = tone === "bad" ? "text-red-600" : tone === "good" ? "text-brand-600" : tone === "warn" ? "text-amber-600" : "text-gray-900";
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4">
      <div className={`text-2xl font-bold ${color}`}>{value}</div>
      <div className="mt-0.5 text-xs text-gray-500">{label}</div>
      {sub && <div className="text-[11px] text-gray-400">{sub}</div>}
    </div>
  );
}

function HomeInner() {
  const { data: me } = useProfile();
  const openTask = useOpenTask();

  const { data: lake } = useQuery({
    queryKey: ["dashboard_data"],
    staleTime: 10 * 60 * 1000,
    queryFn: async () => {
      const { data } = await supabase.auth.getSession();
      const res = await fetch(`${process.env.NEXT_PUBLIC_FN_BASE}/api/dashboard_data`, {
        headers: { Authorization: `Bearer ${data.session?.access_token}` },
      });
      if (!res.ok) throw new Error(`lake data: HTTP ${res.status}`);
      return (await res.json()) as DashData;
    },
  });

  const { data: tasks } = useQuery({
    queryKey: ["tasks", "home"],
    queryFn: async () => {
      const { data, error } = await supabase.from("tasks").select("*").neq("status", "done").limit(1000);
      if (error) throw error;
      return data as Task[];
    },
  });

  const { data: position } = useQuery({
    queryKey: ["positions", "home"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("positions").select("project_id,version,content,created_at")
        .order("created_at", { ascending: false }).limit(1);
      if (error) throw error;
      return data[0] as { version: number; content: string; created_at: string } | undefined;
    },
  });

  const { data: agentRuns } = useQuery({
    queryKey: ["agent_runs", "home"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("agent_runs").select("task_title,outcome,created_at")
        .order("created_at", { ascending: false }).limit(5);
      if (error) throw error;
      return data as { task_title: string; outcome: string; created_at: string }[];
    },
  });

  const today = startOfDay(new Date());
  const open = tasks ?? [];
  const overdue = open.filter((t) => t.due_date && isBefore(parseISO(t.due_date), today));
  const critical = open.filter((t) => t.priority === "critical");
  const watcherTasks = open.filter((t) => t.source === "watcher");

  const t = lake?.trading;
  const lfl = t?.latest_sales && t?.latest_ly ? ((t.latest_sales - t.latest_ly) / t.latest_ly) * 100 : null;
  const maxSales = Math.max(1, ...(t?.series ?? []).map((s) => s.sales ?? 0));
  const p = lake?.procedures;
  const procPct = p && p.procs ? Math.round((100 * p.done) / p.procs) : null;

  // Venu Cast base case: actual-vs-forecast for the latest actual day + next-7-day outlook
  const fc = lake?.forecast ?? [];
  const fcByDay = new Map(fc.map((f) => [f.d, f.forecast_sales]));
  const latestFc = t?.latest_day ? fcByDay.get(t.latest_day) : undefined;
  const fcVar = latestFc && t?.latest_sales ? ((t.latest_sales - latestFc) / latestFc) * 100 : null;
  const next7 = t?.latest_day
    ? fc.filter((f) => f.d > t.latest_day!).slice(0, 7).reduce((s, f) => s + (f.forecast_sales ?? 0), 0)
    : 0;

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      <h1 className="mb-1 text-xl font-bold">
        {me?.full_name ? `Morning, ${me.full_name.split(" ")[0]}` : "Home"}
      </h1>
      <p className="mb-6 text-sm text-gray-500">The whole operation on one screen — tasks and live data together.</p>

      {/* Task KPIs */}
      <div className="mb-6 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Tile label="Open tasks" value={String(open.length)} />
        <Tile label="Overdue" value={String(overdue.length)} tone={overdue.length ? "bad" : "good"} />
        <Tile label="Critical priority" value={String(critical.length)} tone={critical.length ? "warn" : "good"} />
        <Tile label="Raised by the data" value={String(watcherTasks.length)} sub="watch rules + AI reviews" />
      </div>

      {/* Live lake row */}
      <div className="mb-6 grid gap-3 md:grid-cols-4">
        <div className="rounded-xl border border-gray-200 bg-white p-4 md:col-span-2">
          <div className="mb-1 flex items-baseline justify-between">
            <span className="text-xs font-semibold text-gray-500">
              GROUP NET SALES {t?.latest_day ? `· ${format(parseISO(t.latest_day), "EEE d MMM")}` : ""}
            </span>
            {lfl !== null && (
              <span className={`text-xs font-bold ${lfl >= 0 ? "text-brand-600" : "text-red-600"}`}>
                {lfl >= 0 ? "▲" : "▼"} {Math.abs(lfl).toFixed(1)}% vs LY
              </span>
            )}
          </div>
          <div className="text-2xl font-bold">
            {t?.latest_sales ? `$${(t.latest_sales / 1000).toFixed(0)}k` : "…"}
          </div>
          <div className="mt-2 flex h-16 items-end gap-1">
            {(t?.series ?? []).map((s) => (
              <div key={s.d} className="group relative flex-1">
                <div
                  className={`rounded-t ${s.d === t?.latest_day ? "bg-brand-600" : "bg-brand-100"}`}
                  style={{ height: `${Math.max(4, (64 * (s.sales ?? 0)) / maxSales)}px` }}
                  title={`${format(parseISO(s.d), "EEE d MMM")}: $${((s.sales ?? 0) / 1000).toFixed(0)}k`}
                />
              </div>
            ))}
          </div>
          <div className="mt-1 text-[10px] text-gray-400">last 14 days · nightly lake refresh</div>
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="text-xs font-semibold text-gray-500">VENU CAST · vs base case</div>
          <div className={`mt-1 text-2xl font-bold ${fcVar === null ? "" : fcVar >= 0 ? "text-brand-600" : "text-amber-600"}`}>
            {fcVar === null ? "—" : `${fcVar >= 0 ? "+" : ""}${fcVar.toFixed(1)}%`}
          </div>
          <div className="text-xs text-gray-500">
            {latestFc && t?.latest_sales
              ? `actual $${(t.latest_sales / 1000).toFixed(0)}k vs forecast $${(latestFc / 1000).toFixed(0)}k`
              : "no base case for latest day"}
          </div>
          {next7 > 0 && (
            <div className="mt-1 text-[11px] text-gray-400">next 7 days forecast: ${(next7 / 1e6).toFixed(2)}m</div>
          )}
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="text-xs font-semibold text-gray-500">
            PROCEDURES {p?.day ? `· ${format(parseISO(p.day), "EEE d MMM")}` : ""}
          </div>
          <div className={`mt-1 text-2xl font-bold ${procPct !== null && procPct < 80 ? "text-amber-600" : "text-brand-600"}`}>
            {procPct !== null ? `${procPct}%` : "…"}
          </div>
          <div className="text-xs text-gray-500">completed{p ? ` (${p.done?.toLocaleString()} of ${p.procs?.toLocaleString()})` : ""}</div>
          {p && <div className="mt-1 text-[11px] text-gray-400">{p.cancelled?.toLocaleString()} cancelled</div>}
          <Link href="/lake" className="mt-2 inline-block text-xs font-semibold text-brand-600 hover:underline">
            Explore the lake →
          </Link>
        </div>
      </div>

      {/* Position + agent activity */}
      <div className="grid gap-3 md:grid-cols-2">
        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="mb-2 text-xs font-semibold text-gray-500">
            LATEST POSITION {position ? `· v${position.version} · ${format(parseISO(position.created_at), "d MMM")}` : ""}
          </div>
          {position ? (
            <div className="max-h-48 overflow-y-auto whitespace-pre-wrap text-xs leading-relaxed text-gray-700">
              {position.content.slice(0, 900)}{position.content.length > 900 ? "…" : ""}
            </div>
          ) : (
            <div className="text-sm text-gray-300">No position yet — run 🤖 Run all tasks on a project.</div>
          )}
        </div>
        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="mb-2 text-xs font-semibold text-gray-500">RECENT AGENT ACTIVITY</div>
          {agentRuns?.length ? (
            <div className="space-y-1.5">
              {agentRuns.map((r, i) => (
                <div key={i} className="flex items-center gap-2 text-xs">
                  <span className={r.outcome === "success" ? "text-brand-600" : "text-red-600"}>
                    {r.outcome === "success" ? "✓" : "✗"}
                  </span>
                  <span className="truncate">{r.task_title}</span>
                  <span className="ml-auto shrink-0 text-gray-400">{format(parseISO(r.created_at), "d MMM HH:mm")}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-sm text-gray-300">No agent runs yet.</div>
          )}
        </div>
      </div>

      {/* Estate launcher */}
      <div className="mt-6 grid grid-cols-2 gap-3 md:grid-cols-4">
        {ESTATE.map((e) =>
          e.url.startsWith("/") ? (
            <Link key={e.name} href={e.url} className="rounded-xl border border-gray-200 bg-white p-3 hover:border-brand-500">
              <div className="text-sm font-semibold">{e.name}</div>
              <div className="text-[11px] text-gray-400">{e.desc}</div>
            </Link>
          ) : (
            <a key={e.name} href={e.url} target="_blank" rel="noreferrer" className="rounded-xl border border-gray-200 bg-white p-3 hover:border-brand-500">
              <div className="text-sm font-semibold">{e.name} ↗</div>
              <div className="text-[11px] text-gray-400">{e.desc}</div>
            </a>
          )
        )}
      </div>

      {/* Overdue quick list */}
      {overdue.length > 0 && (
        <div className="mt-6 overflow-hidden rounded-xl border border-gray-200 bg-white">
          <div className="border-b border-gray-100 px-4 py-2.5 text-sm font-semibold text-red-600">
            Overdue — needs eyes ({overdue.length})
          </div>
          {overdue.slice(0, 6).map((task) => (
            <button
              key={task.id}
              onClick={() => openTask(task.id)}
              className="flex w-full items-center justify-between border-b border-gray-50 px-4 py-2 text-left text-sm hover:bg-red-50/40"
            >
              <span className="truncate">{task.title}</span>
              <span className="ml-3 shrink-0 text-xs font-semibold text-red-600">
                {task.due_date && format(parseISO(task.due_date), "d MMM")}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default function HomePage() {
  return (
    <Suspense>
      <HomeInner />
    </Suspense>
  );
}
