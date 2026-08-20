"use client";

// The compliance register: every obligation from the FY27 calendar, the
// Exceptions tab, and anything the legal documents impose that the workbook
// does not carry. The workbook stays the register of record - this page is the
// live operational view of it, and "Review & update" re-reads it.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { callFn } from "@/lib/fn";

type Item = {
  ref: string;
  period: string;
  stream: "calendar" | "exception" | "document";
  category: string | null;
  obligation: string | null;
  authority: string | null;
  frequency: string | null;
  due_date: string | null;
  due_text: string | null;
  owner: string | null;
  risk: string | null;
  entities: string | null;
  sheet_status: string | null;
  task_id: string | null;
  task_status: string | null;
  writeback: string | null;
  evidence: string | null;
  notes: string | null;
  severity: string | null;
};

type Summary = {
  live_link: string;
  counts: {
    total: number;
    calendar: number;
    exceptions: number;
    document_gaps: number;
    in_window: number;
    open: number;
    done: number;
    overdue: number;
    writeback_pending: number;
  };
  last_run: {
    ran_at: string;
    ran_by: string | null;
    items: number | null;
    tasks_added: number | null;
    tasks_amended: number | null;
    tasks_linked: number | null;
    writeback_ready: number | null;
    flags: { kind: string; detail: string }[] | null;
  } | null;
  items: Item[];
};

const STREAMS = [
  { key: "all", label: "Everything" },
  { key: "calendar", label: "Calendar" },
  { key: "exception", label: "Exceptions" },
  { key: "document", label: "Register gaps" },
] as const;

function Stat({ n, label, tone }: { n: number; label: string; tone?: string }) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white px-4 py-3">
      <div className={`text-2xl font-bold ${tone || "text-gray-900"}`}>{n}</div>
      <div className="text-[11px] uppercase tracking-wide text-gray-500">{label}</div>
    </div>
  );
}

export default function CompliancePage() {
  const qc = useQueryClient();
  const [stream, setStream] = useState<string>("all");
  const [q, setQ] = useState("");
  const [openRef, setOpenRef] = useState<string | null>(null);
  const [withDocs, setWithDocs] = useState(true);

  const { data, isLoading, error } = useQuery({
    queryKey: ["compliance"],
    queryFn: () => callFn<Summary>("compliance_summary", {}),
  });

  const review = useMutation({
    mutationFn: () => callFn<{ queued: boolean }>("run_compliance", { documents: withDocs }),
    onSuccess: () => {
      // the worker updates the register in place; poll for a little while
      let tries = 0;
      const t = setInterval(() => {
        qc.invalidateQueries({ queryKey: ["compliance"] });
        if (++tries > 40) clearInterval(t);
      }, 15000);
    },
  });

  const today = new Date().toISOString().slice(0, 10);
  const items = useMemo(() => {
    const all = data?.items || [];
    const needle = q.trim().toLowerCase();
    return all.filter((i) => {
      if (stream !== "all" && i.stream !== stream) return false;
      if (!needle) return true;
      return [i.ref, i.obligation, i.authority, i.entities, i.category, i.owner]
        .filter(Boolean)
        .some((v) => (v as string).toLowerCase().includes(needle));
    });
  }, [data, stream, q]);

  if (isLoading) return <div className="p-6 text-sm text-gray-400">Loading register…</div>;
  if (error)
    return (
      <div className="p-6 text-sm text-red-600">
        {error instanceof Error ? error.message : String(error)}
      </div>
    );

  const c = data?.counts;

  return (
    <div className="mx-auto max-w-7xl p-6">
      <div className="mb-1 flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-xl font-bold">Compliance register</h1>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1.5 text-xs text-gray-600">
            <input
              type="checkbox"
              checked={withDocs}
              onChange={(e) => setWithDocs(e.target.checked)}
            />
            re-read the legal documents
          </label>
          <button
            onClick={() => review.mutate()}
            disabled={review.isPending || review.isSuccess}
            className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700 disabled:opacity-60"
          >
            {review.isPending
              ? "Starting…"
              : review.isSuccess
                ? "Running…"
                : "Review & update"}
          </button>
        </div>
      </div>

      <p className="mb-4 text-[13px] text-gray-500">
        Register of record:{" "}
        <a
          href={data?.live_link}
          target="_blank"
          rel="noreferrer"
          className="font-medium text-brand-700 underline"
        >
          Yochi_Group_Compliance_Calendar_FY27.xlsx
        </a>{" "}
        — this page reads it live. Fix data in the workbook and it flows here on
        the next review.
      </p>

      {review.isSuccess && (
        <div className="mb-4 rounded-lg border border-brand-200 bg-brand-50 px-4 py-2 text-[13px] text-brand-900">
          Review running in the background{withDocs ? " (re-reading 21 legal documents — a few minutes)" : ""}.
          This page refreshes itself as it lands.
        </div>
      )}

      {c && (
        <div className="mb-5 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-7">
          <Stat n={c.total} label="obligations" />
          <Stat n={c.in_window} label="in the year" />
          <Stat n={c.open} label="open" />
          <Stat n={c.overdue} label="overdue" tone={c.overdue ? "text-red-600" : undefined} />
          <Stat n={c.done} label="done" tone="text-green-700" />
          <Stat n={c.exceptions} label="exceptions" tone={c.exceptions ? "text-amber-600" : undefined} />
          <Stat n={c.document_gaps} label="register gaps" tone={c.document_gaps ? "text-amber-600" : undefined} />
        </div>
      )}

      {data?.last_run && (
        <div className="mb-4 rounded-lg border border-gray-200 bg-gray-50 px-4 py-2 text-[12px] text-gray-600">
          Last review {new Date(data.last_run.ran_at).toLocaleString()} —{" "}
          {data.last_run.items} obligations, {data.last_run.tasks_added} tasks added,{" "}
          {data.last_run.tasks_amended} amended
          {!!data.last_run.writeback_ready && (
            <>
              {" "}
              · <span className="text-amber-700">{data.last_run.writeback_ready} completed
              item(s) waiting to be written back to the workbook (needs SharePoint
              write consent)</span>
            </>
          )}
          {data.last_run.flags?.map((f) => (
            <div key={f.kind + f.detail} className="mt-1 text-amber-700">
              {f.detail}
            </div>
          ))}
        </div>
      )}

      <div className="mb-3 flex flex-wrap items-center gap-2">
        {STREAMS.map((s) => (
          <button
            key={s.key}
            onClick={() => setStream(s.key)}
            className={`rounded-full px-3 py-1 text-xs font-medium ${
              stream === s.key
                ? "bg-brand-600 text-white"
                : "bg-gray-100 text-gray-700 hover:bg-gray-200"
            }`}
          >
            {s.label}
          </button>
        ))}
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search ref, obligation, entity, authority…"
          className="ml-auto w-72 rounded-lg border border-gray-300 px-3 py-1.5 text-sm"
        />
      </div>

      <div className="overflow-x-auto rounded-xl border border-gray-200 bg-white">
        <table className="w-full text-[13px]">
          <thead className="bg-gray-50 text-left text-[11px] uppercase tracking-wide text-gray-500">
            <tr>
              <th className="px-3 py-2">Ref</th>
              <th className="px-3 py-2">Obligation</th>
              <th className="px-3 py-2">Entities</th>
              <th className="px-3 py-2">Due</th>
              <th className="px-3 py-2">Status</th>
            </tr>
          </thead>
          <tbody>
            {items.map((i) => {
              const id = `${i.ref}|${i.period}`;
              const overdue =
                (i.due_date || "") < today && (i.task_status || "todo") !== "done";
              return (
                <tr
                  key={id}
                  onClick={() => setOpenRef(openRef === id ? null : id)}
                  className="cursor-pointer border-t border-gray-100 align-top hover:bg-gray-50"
                >
                  <td className="whitespace-nowrap px-3 py-2 font-mono text-[11px] text-gray-600">
                    {i.ref}
                    <div className="text-[10px] text-gray-400">
                      {i.period === "EXC"
                        ? "exception"
                        : i.period === "DOC"
                          ? "gap"
                          : i.period}
                    </div>
                  </td>
                  <td className="px-3 py-2">
                    <div className="font-medium text-gray-900">{i.obligation}</div>
                    <div className="text-[11px] text-gray-500">
                      {i.category} {i.frequency ? `· ${i.frequency}` : ""}
                    </div>
                    {openRef === id && (
                      <div className="mt-2 space-y-1 rounded-lg bg-gray-50 p-3 text-[12px] text-gray-700">
                        {i.authority && (
                          <div>
                            <span className="font-semibold">Authority:</span> {i.authority}
                          </div>
                        )}
                        {i.due_text && (
                          <div>
                            <span className="font-semibold">Due per workbook:</span> {i.due_text}
                          </div>
                        )}
                        {i.owner && (
                          <div>
                            <span className="font-semibold">Owner:</span> {i.owner}
                          </div>
                        )}
                        {i.evidence && (
                          <div>
                            <span className="font-semibold">Evidence:</span> {i.evidence}
                          </div>
                        )}
                        {i.notes && (
                          <div>
                            <span className="font-semibold">Notes:</span> {i.notes}
                          </div>
                        )}
                        {i.task_id && (
                          <a
                            href={`/tasks?task=${i.task_id}`}
                            className="inline-block pt-1 font-medium text-brand-700 underline"
                          >
                            Open the task
                          </a>
                        )}
                      </div>
                    )}
                  </td>
                  <td className="px-3 py-2 text-[11px] text-gray-600">
                    {(i.entities || "").slice(0, 90)}
                  </td>
                  <td className="whitespace-nowrap px-3 py-2">
                    <span className={overdue ? "font-semibold text-red-600" : "text-gray-700"}>
                      {i.due_date || "—"}
                    </span>
                  </td>
                  <td className="whitespace-nowrap px-3 py-2">
                    <span
                      className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${
                        i.task_status === "done"
                          ? "bg-green-100 text-green-800"
                          : i.stream === "document"
                            ? "bg-amber-100 text-amber-800"
                            : overdue
                              ? "bg-red-100 text-red-700"
                              : "bg-gray-100 text-gray-700"
                      }`}
                    >
                      {i.stream === "document"
                        ? "not in workbook"
                        : i.task_status || i.sheet_status || "not started"}
                    </span>
                  </td>
                </tr>
              );
            })}
            {!items.length && (
              <tr>
                <td colSpan={5} className="px-3 py-8 text-center text-gray-400">
                  Nothing matches that filter.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <p className="mt-3 text-[11px] text-gray-400">
        Showing {items.length} of {data?.items.length || 0} register rows. Status shown
        here is TaskHub&apos;s; writing it back into the workbook is built but waiting on
        SharePoint write consent, so the workbook still reads as the team left it.
      </p>
    </div>
  );
}
