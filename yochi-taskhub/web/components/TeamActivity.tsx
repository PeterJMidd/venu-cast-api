"use client";

// Live team activity: who did what, lately. The Daily report tab next door is
// a batch digest built at 08:30; this is the current picture, straight from
// TaskHub's own tables, filterable to one person.
//
// "Assigned" counts tasks whose assignee is that person and which MOVED in the
// window - a stale task assigned months ago is not this week's activity.
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { format, formatDistanceToNow, parseISO, subDays } from "date-fns";
import { supabase } from "@/lib/supabase";
import { profileName } from "@/hooks/useProfiles";
import type { Profile } from "@/lib/types";

type Task = {
  id: string; title: string; status: string; priority: string;
  assignee_id: string | null; created_by: string | null;
  created_at: string; updated_at: string; completed_at: string | null;
  project_id: string;
};
type Comment = { id: string; task_id: string; author_id: string | null; body: string; created_at: string };
type Knowledge = { id: string; task_id: string; question: string; engine: string | null; created_by: string | null; created_at: string };
type Notify = { id: string; task_id: string; recipient: string; kind: string; actor: string | null; created_at: string; comment_id: string | null };

type Event = {
  at: string;
  kind: "created" | "assigned" | "completed" | "comment" | "research";
  who: string | null;
  taskId: string;
  title: string;
  detail?: string;
};

const WINDOWS = [
  { days: 7, label: "7 days" },
  { days: 14, label: "14 days" },
  { days: 30, label: "30 days" },
];

const KIND_STYLE: Record<Event["kind"], { label: string; cls: string }> = {
  created: { label: "created", cls: "bg-gray-100 text-gray-700" },
  assigned: { label: "assigned", cls: "bg-blue-100 text-blue-700" },
  completed: { label: "completed", cls: "bg-green-100 text-green-700" },
  comment: { label: "commented", cls: "bg-amber-100 text-amber-800" },
  research: { label: "researched", cls: "bg-purple-100 text-purple-700" },
};

function Stat({ n, label, tone }: { n: number; label: string; tone?: string }) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white px-4 py-3">
      <div className={`text-2xl font-bold ${tone || "text-gray-900"}`}>{n}</div>
      <div className="text-[11px] uppercase tracking-wide text-gray-500">{label}</div>
    </div>
  );
}

export default function TeamActivity({ profiles, me }: { profiles?: Profile[]; me?: string }) {
  const [days, setDays] = useState(14);
  const [who, setWho] = useState<string>("all");
  const since = useMemo(() => subDays(new Date(), days).toISOString(), [days]);

  const { data, isLoading, error } = useQuery({
    queryKey: ["team-activity", days],
    queryFn: async () => {
      const [tasks, comments, knowledge, notify] = await Promise.all([
        supabase.from("tasks")
          .select("id,title,status,priority,assignee_id,created_by,created_at,updated_at,completed_at,project_id")
          .or(`created_at.gte.${since},updated_at.gte.${since}`)
          .order("updated_at", { ascending: false }).limit(1500),
        supabase.from("comments").select("id,task_id,author_id,body,created_at")
          .gte("created_at", since).order("created_at", { ascending: false }).limit(500),
        supabase.from("knowledge").select("id,task_id,question,engine,created_by,created_at")
          .gte("created_at", since).order("created_at", { ascending: false }).limit(300),
        supabase.from("notify_outbox").select("id,task_id,recipient,kind,actor,created_at,comment_id")
          .gte("created_at", since).order("created_at", { ascending: false }).limit(500),
      ]);
      for (const r of [tasks, comments, knowledge, notify]) if (r.error) throw r.error;
      return {
        tasks: (tasks.data || []) as Task[],
        comments: (comments.data || []) as Comment[],
        knowledge: (knowledge.data || []) as Knowledge[],
        notify: (notify.data || []) as Notify[],
      };
    },
  });

  const titles = useMemo(() => {
    const m = new Map<string, string>();
    (data?.tasks || []).forEach((t) => m.set(t.id, t.title));
    return m;
  }, [data]);

  // one merged, time-ordered stream
  const events = useMemo<Event[]>(() => {
    if (!data) return [];
    const out: Event[] = [];
    for (const t of data.tasks) {
      if (t.created_at >= since)
        out.push({ at: t.created_at, kind: "created", who: t.created_by, taskId: t.id, title: t.title });
      if (t.completed_at && t.completed_at >= since)
        out.push({ at: t.completed_at, kind: "completed", who: t.assignee_id, taskId: t.id, title: t.title });
      // assignment has no timestamp of its own; the notify row below carries it
    }
    for (const n of data.notify) {
      // recipient/actor are user IDs, not emails
      if (n.kind === "assigned")
        out.push({ at: n.created_at, kind: "assigned", who: n.recipient,
                   taskId: n.task_id, title: titles.get(n.task_id) || "(task)",
                   detail: n.actor ? `by ${profileName(profiles, n.actor)}`
                                   : "by the system" });
    }
    for (const c of data.comments)
      out.push({ at: c.created_at, kind: "comment", who: c.author_id, taskId: c.task_id,
                 title: titles.get(c.task_id) || "(task)", detail: c.body.slice(0, 160) });
    for (const k of data.knowledge)
      out.push({ at: k.created_at, kind: "research", who: k.created_by, taskId: k.task_id,
                 title: titles.get(k.task_id) || "(task)",
                 detail: `${k.engine === "lake" ? "our data" : k.engine || "research"}: ${k.question.slice(0, 130)}` });
    return out.sort((a, b) => (a.at < b.at ? 1 : -1));
  }, [data, profiles, titles, since]);

  const shown = useMemo(
    () => (who === "all" ? events : events.filter((e) => e.who === who)),
    [events, who]);

  // per-person scoreboard over the same window
  const board = useMemo(() => {
    const rows = new Map<string, { id: string; created: number; completed: number;
                                   assigned: number; comments: number; research: number }>();
    const get = (id: string | null) => {
      if (!id) return null;
      if (!rows.has(id)) rows.set(id, { id, created: 0, completed: 0, assigned: 0, comments: 0, research: 0 });
      return rows.get(id)!;
    };
    for (const e of events) {
      const r = get(e.who);
      if (!r) continue;
      if (e.kind === "created") r.created += 1;
      else if (e.kind === "completed") r.completed += 1;
      else if (e.kind === "assigned") r.assigned += 1;
      else if (e.kind === "comment") r.comments += 1;
      else if (e.kind === "research") r.research += 1;
    }
    return Array.from(rows.values())
      .map((r) => ({ ...r, total: r.created + r.completed + r.assigned + r.comments + r.research }))
      .sort((a, b) => b.total - a.total);
  }, [events]);

  // what landed in MY inbox (or the selected person's)
  const forMe = useMemo(() => {
    if (!data) return [];
    const target = who === "all" ? me : who;
    if (!target) return [];
    return data.notify
      .filter((n) => n.recipient === target && n.kind !== "assigned")
      .slice(0, 25);
  }, [data, profiles, who, me]);

  const open = useMemo(() => {
    if (!data) return [];
    const target = who === "all" ? null : who;
    return data.tasks
      .filter((t) => t.status !== "done" && (!target || t.assignee_id === target))
      .sort((a, b) => (a.updated_at < b.updated_at ? 1 : -1))
      .slice(0, 12);
  }, [data, who]);

  if (isLoading) return <div className="p-6 text-sm text-gray-400">Loading activity…</div>;
  if (error) {
    // Supabase errors are plain objects, not Error instances - String() on one
    // renders "[object Object]" and hides the actual problem (it hid a 403 for
    // an hour).
    const e = error as { message?: string; hint?: string; code?: string };
    return (
      <div className="p-6 text-sm text-red-600">
        {e?.message || JSON.stringify(error)}
        {e?.hint ? <div className="mt-1 text-xs text-red-500">{e.hint}</div> : null}
      </div>
    );
  }

  const totals = {
    created: events.filter((e) => e.kind === "created").length,
    completed: events.filter((e) => e.kind === "completed").length,
    assigned: events.filter((e) => e.kind === "assigned").length,
    comments: events.filter((e) => e.kind === "comment").length,
    research: events.filter((e) => e.kind === "research").length,
  };

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        <select value={who} onChange={(e) => setWho(e.target.value)}
                className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm">
          <option value="all">Everyone</option>
          {(profiles || []).filter((p) => p.active !== false).map((p) => (
            <option key={p.id} value={p.id}>
              {profileName(profiles, p.id)}{p.id === me ? " (me)" : ""}
            </option>
          ))}
        </select>
        <div className="flex gap-1">
          {WINDOWS.map((w) => (
            <button key={w.days} onClick={() => setDays(w.days)}
              className={`rounded-full px-3 py-1 text-xs font-medium ${
                days === w.days ? "bg-brand-600 text-white" : "bg-gray-100 text-gray-700 hover:bg-gray-200"}`}>
              {w.label}
            </button>
          ))}
        </div>
        <span className="ml-auto text-[11px] text-gray-400">
          {shown.length} events · since {format(parseISO(since), "d MMM")}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        <Stat n={totals.created} label="tasks added" />
        <Stat n={totals.assigned} label="assigned" />
        <Stat n={totals.completed} label="completed" tone="text-green-700" />
        <Stat n={totals.comments} label="comments" />
        <Stat n={totals.research} label="researched" tone="text-purple-700" />
      </div>

      {!!board.length && (
        <div className="overflow-x-auto rounded-xl border border-gray-200 bg-white">
          <table className="w-full text-[13px]">
            <thead className="bg-gray-50 text-left text-[11px] uppercase tracking-wide text-gray-500">
              <tr>
                <th className="px-3 py-2">Person</th>
                <th className="px-3 py-2 text-right">Added</th>
                <th className="px-3 py-2 text-right">Assigned</th>
                <th className="px-3 py-2 text-right">Completed</th>
                <th className="px-3 py-2 text-right">Comments</th>
                <th className="px-3 py-2 text-right">Research</th>
                <th className="px-3 py-2 text-right">Total</th>
              </tr>
            </thead>
            <tbody>
              {board.map((r) => (
                <tr key={r.id}
                    onClick={() => setWho(r.id === who ? "all" : r.id)}
                    className={`cursor-pointer border-t border-gray-100 hover:bg-gray-50 ${
                      who === r.id ? "bg-brand-50/50" : ""}`}>
                  <td className="px-3 py-2 font-medium">
                    {profileName(profiles, r.id)}{r.id === me ? " (me)" : ""}
                  </td>
                  <td className="px-3 py-2 text-right">{r.created || "—"}</td>
                  <td className="px-3 py-2 text-right">{r.assigned || "—"}</td>
                  <td className="px-3 py-2 text-right font-semibold text-green-700">{r.completed || "—"}</td>
                  <td className="px-3 py-2 text-right">{r.comments || "—"}</td>
                  <td className="px-3 py-2 text-right">{r.research || "—"}</td>
                  <td className="px-3 py-2 text-right font-bold">{r.total}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="px-3 py-2 text-[11px] text-gray-400">
            Click a row to filter everything below to that person.
          </div>
        </div>
      )}

      <div className="grid gap-5 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <div className="mb-2 text-xs font-bold uppercase tracking-wide text-gray-400">
            Activity {who !== "all" ? `— ${profileName(profiles, who)}` : ""}
          </div>
          <div className="divide-y divide-gray-100 rounded-xl border border-gray-200 bg-white">
            {shown.slice(0, 60).map((e, i) => (
              <a key={i} href={`/tasks?task=${e.taskId}`}
                 className="flex items-start gap-3 px-3 py-2 hover:bg-gray-50">
                <span className={`mt-0.5 shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold ${KIND_STYLE[e.kind].cls}`}>
                  {KIND_STYLE[e.kind].label}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-[13px] font-medium text-gray-900">{e.title}</span>
                  {e.detail && (
                    <span className="block truncate text-[11px] text-gray-500">{e.detail}</span>
                  )}
                  <span className="text-[10px] text-gray-400">
                    {e.who ? profileName(profiles, e.who) : "automation"} ·{" "}
                    {formatDistanceToNow(parseISO(e.at), { addSuffix: true })}
                  </span>
                </span>
              </a>
            ))}
            {!shown.length && (
              <div className="px-3 py-8 text-center text-sm text-gray-400">
                No activity in this window.
              </div>
            )}
          </div>
        </div>

        <div className="space-y-5">
          <div>
            <div className="mb-2 text-xs font-bold uppercase tracking-wide text-gray-400">
              {who === "all" ? "Addressed to me" : `Addressed to ${profileName(profiles, who)}`}
            </div>
            <div className="divide-y divide-gray-100 rounded-xl border border-gray-200 bg-white">
              {forMe.map((n) => (
                <a key={n.id} href={`/tasks?task=${n.task_id}`}
                   className="block px-3 py-2 hover:bg-gray-50">
                  <span className="block truncate text-[13px]">{titles.get(n.task_id) || "(task)"}</span>
                  <span className="text-[10px] text-gray-400">
                    {n.kind === "mention" ? "mentioned you" : n.kind} ·{" "}
                    {formatDistanceToNow(parseISO(n.created_at), { addSuffix: true })}
                  </span>
                </a>
              ))}
              {!forMe.length && (
                <div className="px-3 py-6 text-center text-xs text-gray-400">
                  Nothing addressed to you in this window.
                </div>
              )}
            </div>
          </div>

          <div>
            <div className="mb-2 text-xs font-bold uppercase tracking-wide text-gray-400">
              Open, most recently touched
            </div>
            <div className="divide-y divide-gray-100 rounded-xl border border-gray-200 bg-white">
              {open.map((t) => (
                <a key={t.id} href={`/tasks?task=${t.id}`}
                   className="block px-3 py-2 hover:bg-gray-50">
                  <span className="block truncate text-[13px]">{t.title}</span>
                  <span className="text-[10px] text-gray-400">
                    {t.assignee_id ? profileName(profiles, t.assignee_id) : "unassigned"} ·{" "}
                    {t.priority} · {formatDistanceToNow(parseISO(t.updated_at), { addSuffix: true })}
                  </span>
                </a>
              ))}
              {!open.length && (
                <div className="px-3 py-6 text-center text-xs text-gray-400">Nothing open.</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
