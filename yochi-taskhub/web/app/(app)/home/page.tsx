"use client";

import { Suspense } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { format, isBefore, parseISO, startOfDay } from "date-fns";
import { supabase } from "@/lib/supabase";
import { useOpenTask } from "@/components/TaskCard";
import ReportButton from "@/components/ReportButton";
import SmartTaskModal from "@/components/SmartTaskModal";
import { useProfile } from "@/hooks/useProfile";
import type { Task } from "@/lib/types";

interface Cockpit {
  meetings: { subject: string; starts_at: string; ends_at: string | null; location: string | null; organizer: string | null; attendees: number | null; prep: string | null }[];
  cash: {
    weeks: { week_start: string; closing: number }[];
    trough_week: string; trough: number; opening: number | null; floor: number | null;
    dso_days: number | null; dpo_days: number | null; generated_at: string;
  } | null;
  radar: { title: string; due: string; priority: string; project: string | null }[];
  team: { who: string; open: number; overdue: number; critical: number }[];
  checklist: { item: string; ok: boolean; detail: string }[];
  decisions: { title: string; due: string | null; who: string }[];
  signals: { kind: string; headline: string; detail: string | null; source: string | null; created_at: string }[];
  vip: { sender: string; subject: string; snippet: string | null; received_at: string; weblink: string | null }[];
  priority: { title: string; due: string | null; priority: string; who: string }[];
  pillars: { name: string; sort: number; open: number; overdue: number; critical: number }[];
}

interface DashData {
  trading: {
    series: { d: string; sales: number; ly: number; budget: number | null }[];
    latest_day: string | null;
    latest_sales: number | null;
    latest_ly: number | null;
    latest_budget: number | null;
    mtd: { sales: number; budget: number | null } | null;
  };
  forecast: { d: string; forecast_sales: number }[];
  procedures: { procs: number; done: number; cancelled: number; day: string } | null;
  cockpit: Cockpit | null;
}

const fmtM = (v: number | null | undefined) =>
  v == null ? "—" : `$${(Number(v) / 1e6).toFixed(1)}m`;

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
  const maxSales = Math.max(1, ...(t?.series ?? []).flatMap((s) => [s.sales ?? 0, s.budget ?? 0]));
  const budVar = t?.latest_sales && t?.latest_budget ? ((t.latest_sales - t.latest_budget) / t.latest_budget) * 100 : null;
  const mtdVar =
    t?.mtd?.budget && t.mtd.sales ? ((t.mtd.sales - t.mtd.budget) / t.mtd.budget) * 100 : null;
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
      <div className="mb-1 flex items-center justify-between">
        <h1 className="text-xl font-bold">
          {me?.full_name ? `Morning, ${me.full_name.split(" ")[0]}` : "Home"}
        </h1>
        <div className="flex gap-2">
          <SmartTaskModal />
          <ReportButton />
        </div>
      </div>
      <p className="mb-6 text-sm text-gray-500">The whole operation on one screen — tasks and live data together.</p>

      {/* Executive brief */}
      {lake?.cockpit && (
        <div className="mb-6 rounded-2xl border-2 border-brand-200 bg-gradient-to-b from-brand-50/60 to-white p-4">
          <div className="mb-3 flex items-baseline justify-between">
            <span className="text-sm font-bold text-brand-700">EXECUTIVE BRIEF</span>
            <span className="text-[11px] text-gray-400">
              {lfl !== null && t?.latest_sales
                ? `Sales $${(t.latest_sales / 1000).toFixed(0)}k (${lfl >= 0 ? "+" : ""}${lfl.toFixed(1)}% LY)`
                : ""}
              {fcVar !== null ? ` · ${fcVar >= 0 ? "+" : ""}${fcVar.toFixed(1)}% vs base` : ""}
              {lake.cockpit.cash ? ` · cash trough ${fmtM(lake.cockpit.cash.trough)}` : ""}
              {` · ${lake.cockpit.checklist.filter((c) => c.ok).length}/${lake.cockpit.checklist.length} checks green`}
            </span>
          </div>
          <div className="grid gap-3 md:grid-cols-3">
            <div>
              <div className="mb-1.5 text-[11px] font-bold uppercase tracking-wide text-gray-500">Deliver today</div>
              <div className="space-y-1">
                {lake.cockpit.priority.map((p, i) => (
                  <div key={i} className="flex items-center gap-1.5 text-xs">
                    <span
                      className={`h-1.5 w-1.5 shrink-0 rounded-full ${p.priority === "critical" ? "bg-red-500" : "bg-amber-500"}`}
                    />
                    <span className="truncate">{p.title}</span>
                    {p.due && (
                      <span className="ml-auto shrink-0 text-[10px] text-gray-400">{format(parseISO(p.due), "d MMM")}</span>
                    )}
                  </div>
                ))}
                {!lake.cockpit.priority.length && <div className="text-xs text-gray-300">Nothing critical or high open.</div>}
              </div>
            </div>
            <div>
              <div className="mb-1.5 text-[11px] font-bold uppercase tracking-wide text-gray-500">
                Key messages · Janine, J Dog, Brooke
              </div>
              <div className="space-y-1">
                {lake.cockpit.vip.map((m, i) => (
                  <a
                    key={i}
                    href={m.weblink ?? undefined}
                    target="_blank"
                    rel="noreferrer"
                    className={`block text-xs ${m.weblink ? "hover:underline" : "cursor-default"}`}
                  >
                    <span className="font-semibold">{m.sender.split(" ")[0]}:</span>{" "}
                    <span className={/URGENT/i.test(m.subject) ? "font-semibold text-red-600" : ""}>{m.subject}</span>
                    <span className="ml-1 text-[10px] text-gray-400">{format(parseISO(m.received_at), "EEE HH:mm")}</span>
                  </a>
                ))}
                {!lake.cockpit.vip.length && <div className="text-xs text-gray-300">No recent messages.</div>}
              </div>
            </div>
            <div>
              <div className="mb-1.5 text-[11px] font-bold uppercase tracking-wide text-gray-500">Key news & context</div>
              <div className="space-y-1">
                {lake.cockpit.signals.slice(0, 4).map((s, i) => (
                  <div key={i} className="text-xs">
                    <span className="mr-1 rounded bg-gray-100 px-1 py-px text-[9px] font-bold uppercase text-gray-500">
                      {s.kind}
                    </span>
                    {s.headline}
                  </div>
                ))}
                {!lake.cockpit.signals.length && <div className="text-xs text-gray-300">No signals yet.</div>}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* The 5 pillars — the finance model's organising frame */}
      {lake?.cockpit?.pillars?.length ? (
        <div className="mb-6 grid grid-cols-2 gap-3 md:grid-cols-5">
          {lake.cockpit.pillars.map((p) => (
            <Link
              key={p.name}
              href="/tasks"
              className="rounded-xl border border-gray-200 bg-white p-3 hover:border-brand-500"
            >
              <div className="text-[10px] font-bold uppercase leading-tight text-gray-400">
                {p.sort} · {p.name}
              </div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="text-xl font-bold">{p.open}</span>
                {p.overdue > 0 && (
                  <span className="text-[11px] font-bold text-red-600">{p.overdue} overdue</span>
                )}
                {p.critical > 0 && (
                  <span className="text-[11px] font-bold text-amber-600">{p.critical} crit</span>
                )}
              </div>
            </Link>
          ))}
        </div>
      ) : (
        <div className="mb-6 grid grid-cols-2 gap-3 md:grid-cols-4">
          <Tile label="Open tasks" value={String(open.length)} />
          <Tile label="Overdue" value={String(overdue.length)} tone={overdue.length ? "bad" : "good"} />
          <Tile label="Critical priority" value={String(critical.length)} tone={critical.length ? "warn" : "good"} />
          <Tile label="Raised by the data" value={String(watcherTasks.length)} sub="watch rules + AI reviews" />
        </div>
      )}

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
          <div className="flex items-baseline gap-3">
            <div className="text-2xl font-bold">
              {t?.latest_sales ? `$${(t.latest_sales / 1000).toFixed(0)}k` : "…"}
            </div>
            {budVar !== null && (
              <span className={`text-xs font-bold ${budVar >= 0 ? "text-brand-600" : "text-red-600"}`}>
                {budVar >= 0 ? "+" : ""}
                {budVar.toFixed(1)}% vs budget ${((t!.latest_budget ?? 0) / 1000).toFixed(0)}k
              </span>
            )}
          </div>
          <div className="mt-2 flex h-16 items-end gap-1">
            {(t?.series ?? []).map((s) => (
              <div key={s.d} className="group relative h-16 flex-1">
                <div
                  className={`absolute bottom-0 left-0 right-0 rounded-t ${
                    s.budget != null && s.sales != null && s.sales < s.budget
                      ? s.d === t?.latest_day
                        ? "bg-red-500"
                        : "bg-red-200"
                      : s.d === t?.latest_day
                        ? "bg-brand-600"
                        : "bg-brand-100"
                  }`}
                  style={{ height: `${Math.max(4, (64 * (s.sales ?? 0)) / maxSales)}px` }}
                  title={`${format(parseISO(s.d), "EEE d MMM")}: $${((s.sales ?? 0) / 1000).toFixed(0)}k${
                    s.budget != null ? ` vs budget $${(s.budget / 1000).toFixed(0)}k` : ""
                  }`}
                />
                {s.budget != null && (
                  <div
                    className="pointer-events-none absolute left-0 right-0 border-t-2 border-gray-800/70"
                    style={{ bottom: `${Math.max(2, Math.min(64, (64 * s.budget) / maxSales))}px` }}
                  />
                )}
              </div>
            ))}
          </div>
          <div className="mt-1 flex items-center justify-between text-[10px] text-gray-400">
            <span>last 14 days · ─ budget · red = below budget</span>
            {t?.mtd?.budget != null && mtdVar !== null && (
              <span className={mtdVar >= 0 ? "font-semibold text-brand-600" : "font-semibold text-red-600"}>
                MTD ${(t.mtd.sales / 1e6).toFixed(2)}m vs ${(t.mtd.budget / 1e6).toFixed(2)}m ({mtdVar >= 0 ? "+" : ""}
                {mtdVar.toFixed(1)}%)
              </span>
            )}
          </div>
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

      {/* Cockpit: cash + checklist */}
      {lake?.cockpit && (
        <>
          <div className="mb-6 grid gap-3 md:grid-cols-2">
            <div className="rounded-xl border border-gray-200 bg-white p-4">
              <div className="mb-1 flex items-baseline justify-between">
                <span className="text-xs font-semibold text-gray-500">CASH · 13-WEEK FORECAST</span>
                {lake.cockpit.cash && (
                  <span className="text-[11px] text-gray-400">
                    DSO {lake.cockpit.cash.dso_days ?? "—"}d · DPO {lake.cockpit.cash.dpo_days ?? "—"}d
                  </span>
                )}
              </div>
              {lake.cockpit.cash ? (
                <>
                  <div className="flex items-baseline gap-3">
                    <div className="text-2xl font-bold">{fmtM(lake.cockpit.cash.opening)}</div>
                    <div
                      className={`text-sm font-bold ${
                        lake.cockpit.cash.floor != null && Number(lake.cockpit.cash.trough) < Number(lake.cockpit.cash.floor)
                          ? "text-red-600"
                          : "text-brand-600"
                      }`}
                    >
                      trough {fmtM(lake.cockpit.cash.trough)} · wk {format(parseISO(lake.cockpit.cash.trough_week), "d MMM")}
                    </div>
                  </div>
                  <div className="mt-2 flex h-14 items-end gap-1">
                    {lake.cockpit.cash.weeks.map((w) => {
                      const max = Math.max(1, ...lake.cockpit!.cash!.weeks.map((x) => Number(x.closing)));
                      const below = lake.cockpit!.cash!.floor != null && Number(w.closing) < Number(lake.cockpit!.cash!.floor);
                      return (
                        <div
                          key={w.week_start}
                          className={`flex-1 rounded-t ${below ? "bg-red-400" : "bg-brand-200"}`}
                          style={{ height: `${Math.max(4, (56 * Number(w.closing)) / max)}px` }}
                          title={`wk ${format(parseISO(w.week_start), "d MMM")}: ${fmtM(Number(w.closing))}`}
                        />
                      );
                    })}
                  </div>
                  <div className="mt-1 text-[10px] text-gray-400">
                    closing balance by week · refreshed {format(parseISO(lake.cockpit.cash.generated_at), "d MMM")}
                  </div>
                </>
              ) : (
                <div className="text-sm text-gray-300">No cash forecast yet — runs Monday 06:55.</div>
              )}
            </div>

            <div className="rounded-xl border border-gray-200 bg-white p-4">
              <div className="mb-2 text-xs font-semibold text-gray-500">CRITICAL CHECKLIST · data-verified</div>
              <div className="space-y-1.5">
                {lake.cockpit.checklist.map((c) => (
                  <div key={c.item} className="flex items-center gap-2 text-xs">
                    <span className={c.ok ? "text-brand-600" : "text-red-600"}>{c.ok ? "✓" : "✗"}</span>
                    <span className={c.ok ? "" : "font-semibold"}>{c.item}</span>
                    <span className="ml-auto shrink-0 text-gray-400">{c.detail}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Cockpit: meetings + compliance radar */}
          <div className="mb-6 grid gap-3 md:grid-cols-2">
            <div className="rounded-xl border border-gray-200 bg-white p-4">
              <div className="mb-2 text-xs font-semibold text-gray-500">UPCOMING MEETINGS</div>
              {lake.cockpit.meetings.length ? (
                <div className="space-y-1.5">
                  {lake.cockpit.meetings.map((m, i) =>
                    m.prep ? (
                      <details key={i} className="group text-xs">
                        <summary className="flex cursor-pointer list-none items-center gap-2">
                          <span className="w-24 shrink-0 font-semibold text-gray-600">
                            {format(parseISO(m.starts_at), "EEE HH:mm")}
                          </span>
                          <span className="truncate">{m.subject}</span>
                          <span className="ml-auto shrink-0 rounded bg-brand-50 px-1.5 py-px text-[9px] font-bold text-brand-700">
                            PREP ▾
                          </span>
                        </summary>
                        <div className="mt-1.5 max-h-56 overflow-y-auto whitespace-pre-wrap rounded-lg bg-gray-50 p-2.5 text-[11px] leading-relaxed text-gray-700">
                          {m.prep}
                        </div>
                      </details>
                    ) : (
                      <div key={i} className="flex items-center gap-2 text-xs">
                        <span className="w-24 shrink-0 font-semibold text-gray-600">
                          {format(parseISO(m.starts_at), "EEE HH:mm")}
                        </span>
                        <span className="truncate">{m.subject}</span>
                        {m.attendees != null && m.attendees > 1 && (
                          <span className="ml-auto shrink-0 text-gray-400">{m.attendees}p</span>
                        )}
                      </div>
                    )
                  )}
                </div>
              ) : (
                <div className="text-sm text-gray-300">No meetings synced.</div>
              )}
            </div>

            <div className="rounded-xl border border-gray-200 bg-white p-4">
              <div className="mb-2 text-xs font-semibold text-gray-500">COMPLIANCE RADAR · next 60 days</div>
              {lake.cockpit.radar.length ? (
                <div className="space-y-1.5">
                  {lake.cockpit.radar.slice(0, 8).map((r, i) => {
                    const days = Math.round((parseISO(r.due).getTime() - Date.now()) / 864e5);
                    return (
                      <div key={i} className="flex items-center gap-2 text-xs">
                        <span
                          className={`w-16 shrink-0 font-bold ${days < 0 ? "text-red-600" : days <= 7 ? "text-amber-600" : "text-gray-500"}`}
                        >
                          {format(parseISO(r.due), "d MMM")}
                        </span>
                        <span className="truncate">{r.title}</span>
                        <span className="ml-auto shrink-0 text-gray-400">{r.project}</span>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-sm text-gray-300">Nothing due in the next 60 days.</div>
              )}
            </div>
          </div>

          {/* Cockpit: team + decisions */}
          <div className="mb-6 grid gap-3 md:grid-cols-2">
            <div className="rounded-xl border border-gray-200 bg-white p-4">
              <div className="mb-2 text-xs font-semibold text-gray-500">TEAM WORKLOAD</div>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left text-gray-400">
                    <th className="pb-1 font-medium">Person</th>
                    <th className="pb-1 text-right font-medium">Open</th>
                    <th className="pb-1 text-right font-medium">Overdue</th>
                    <th className="pb-1 text-right font-medium">Critical</th>
                  </tr>
                </thead>
                <tbody>
                  {lake.cockpit.team.map((r) => (
                    <tr key={r.who} className="border-t border-gray-50">
                      <td className="py-1">{r.who}</td>
                      <td className="py-1 text-right">{r.open}</td>
                      <td className={`py-1 text-right ${r.overdue ? "font-bold text-red-600" : ""}`}>{r.overdue}</td>
                      <td className={`py-1 text-right ${r.critical ? "font-bold text-amber-600" : ""}`}>{r.critical}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="rounded-xl border border-gray-200 bg-white p-4">
              <div className="mb-2 text-xs font-semibold text-gray-500">DECISIONS PENDING · critical priority</div>
              {lake.cockpit.decisions.length ? (
                <div className="space-y-1.5">
                  {lake.cockpit.decisions.map((d, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs">
                      <span className="truncate">{d.title}</span>
                      <span className="ml-auto shrink-0 text-gray-400">
                        {d.due ? format(parseISO(d.due), "d MMM") : ""}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-gray-300">Nothing awaiting a call from you.</div>
              )}
            </div>
          </div>

          {/* Cockpit: external signals */}
          {lake.cockpit.signals.length > 0 && (
            <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4">
              <div className="mb-2 text-xs font-semibold text-gray-500">CONTEXT · economy, industry, business</div>
              <div className="grid gap-2 md:grid-cols-3">
                {lake.cockpit.signals.map((s, i) => (
                  <div key={i} className="rounded-lg bg-gray-50 p-2.5">
                    <div className="text-[10px] font-bold uppercase text-gray-400">{s.kind}</div>
                    <div className="text-xs font-semibold">{s.headline}</div>
                    {s.detail && <div className="mt-0.5 text-[11px] text-gray-500">{s.detail}</div>}
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

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
