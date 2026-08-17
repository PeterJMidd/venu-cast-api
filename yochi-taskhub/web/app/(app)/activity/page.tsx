"use client";

// Daily activity: the work-activity intelligence report. One row per day in
// taskapp.work_reports (built 07:30 by the FN from the blob estate + data
// lake + TaskHub audit trail) - per-person activity where the source data
// carries names, by type/entity where it doesn't (Xero has no user data).
import { useState, Suspense } from "react";
import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/lib/supabase";

type Row = Record<string, string | number | null>;
type Report = {
  report_date: string;
  narrative: string | null;
  stats: {
    lake: {
      xero_by_source: Row[]; invoices_touched: Row[];
      restoke_receipting: Row[]; procedures_by_person: Row[];
      asana_completed_by_person: Row[]; sales_context: Row[];
      xero_by_user?: Row[];
    };
    taskhub: { people: Row[]; agent_runs: { total: number; success: number };
               batch_runs: { kind: string; status: string | null }[] };
    blob: { container: string; files_changed: number; mb: number;
            areas: Record<string, number>; samples: string[] }[];
  };
};

function Table({ rows, cols, labels }: { rows: Row[]; cols: string[]; labels?: string[] }) {
  if (!rows?.length)
    return <div className="text-xs text-gray-400">No activity recorded.</div>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-[10px] uppercase tracking-wide text-gray-400">
            {(labels ?? cols).map((c) => <th key={c} className="py-1 pr-4">{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-gray-50">
              {cols.map((c) => (
                <td key={c} className="py-1 pr-4">
                  {typeof r[c] === "number" ? r[c]?.toLocaleString() : r[c] ?? ""}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Card({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4">
      <div className="mb-2 text-xs font-bold uppercase tracking-wide text-gray-400">
        {title}{note && <span className="ml-2 font-normal normal-case text-gray-300">{note}</span>}
      </div>
      {children}
    </div>
  );
}

function ActivityInner() {
  const [selected, setSelected] = useState<string | null>(null);
  const { data: reports, isLoading } = useQuery({
    queryKey: ["work-reports"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("work_reports")
        .select("report_date,narrative,stats")
        .order("report_date", { ascending: false })
        .limit(30);
      if (error) throw error;
      return data as Report[];
    },
  });

  const report = reports?.find((r) => r.report_date === selected) ?? reports?.[0];

  if (isLoading) return <div className="p-6 text-sm text-gray-400">Loading…</div>;
  if (!report)
    return (
      <div className="p-6 text-sm text-gray-500">
        No reports yet — the first one is generated at 08:30 tomorrow covering today,
        or an admin can trigger one now via <code>run_workreport</code>.
      </div>
    );

  const s = report.stats;
  const sales = s.lake.sales_context?.[0];

  return (
    <div className="mx-auto max-w-5xl space-y-4 p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold">Daily activity</h1>
          <p className="text-xs text-gray-500">
            Everything done and changed across the business — scanned each morning
            from the blob estate, data lake and TaskHub.
          </p>
        </div>
        <select
          value={report.report_date}
          onChange={(e) => setSelected(e.target.value)}
          className="rounded-lg border border-gray-300 px-2 py-1.5 text-sm"
        >
          {reports!.map((r) => (
            <option key={r.report_date} value={r.report_date}>{r.report_date}</option>
          ))}
        </select>
      </div>

      {sales && (
        <div className="flex flex-wrap gap-3">
          <div className="rounded-xl border border-gray-200 bg-white px-4 py-2 text-sm">
            <span className="text-gray-400">Net sales </span>
            <span className="font-bold">${Number(sales.net_sales ?? 0).toLocaleString()}</span>
            <span className="text-gray-400"> · {sales.venues_traded} venues</span>
          </div>
          <div className="rounded-xl border border-gray-200 bg-white px-4 py-2 text-sm">
            <span className="text-gray-400">Agent runs </span>
            <span className="font-bold">{s.taskhub.agent_runs.success}/{s.taskhub.agent_runs.total} ok</span>
          </div>
          <div className="rounded-xl border border-gray-200 bg-white px-4 py-2 text-sm">
            <span className="text-gray-400">Blob containers changed </span>
            <span className="font-bold">{s.blob.length}</span>
          </div>
        </div>
      )}

      {report.narrative && (
        <div
          className="prose prose-sm max-w-none rounded-xl border border-brand-200 bg-brand-50/40 p-4 [&_h3]:mb-1 [&_h3]:mt-3 [&_h3]:text-sm [&_h3]:font-bold [&_h3]:text-brand-800 [&_ul]:my-1 [&_ul]:pl-5"
          dangerouslySetInnerHTML={{ __html: report.narrative }}
        />
      )}

      <div className="grid gap-4 md:grid-cols-2">
        <Card title="TaskHub — by person">
          <Table rows={s.taskhub.people} cols={["person", "created", "updated", "completed", "comments"]} />
        </Card>
        <Card title="Venue procedures completed — by person">
          <Table rows={s.lake.procedures_by_person} cols={["person", "procedures", "venues"]} />
        </Card>
        <Card title="Asana tasks completed — by person">
          <Table rows={s.lake.asana_completed_by_person} cols={["person", "completed"]} />
        </Card>
        <Card title="Xero — by person" note="from Xero History & Notes (morning scrape)">
          <Table rows={s.lake.xero_by_user ?? []}
                 cols={["xero_user", "entity", "item_type", "action", "items"]}
                 labels={["person", "entity", "type", "action", "items"]} />
        </Card>
        <Card title="Xero postings" note="by type & entity">
          <Table rows={s.lake.xero_by_source} cols={["source", "org", "lines", "net"]}
                 labels={["source", "entity", "lines", "net $"]} />
        </Card>
        <Card title="Invoices touched">
          <Table rows={s.lake.invoices_touched} cols={["typ", "n", "total"]}
                 labels={["type", "count", "total $"]} />
        </Card>
        <Card title="Restoke receipting — by venue">
          <Table rows={s.lake.restoke_receipting} cols={["venue", "orders", "lines", "value"]}
                 labels={["venue", "orders", "lines", "value $"]} />
        </Card>
      </div>

      <Card title="Blob estate — what the systems produced">
        <div className="space-y-2">
          {s.blob.map((c) => (
            <div key={c.container} className="border-t border-gray-50 pt-2 first:border-0 first:pt-0">
              <div className="text-xs font-semibold">
                {c.container}
                <span className="ml-2 font-normal text-gray-400">
                  {c.files_changed.toLocaleString()} files · {c.mb} MB
                </span>
              </div>
              <div className="text-[11px] text-gray-500">
                {Object.entries(c.areas).map(([k, v]) => `${k} (${v})`).join(" · ")}
              </div>
            </div>
          ))}
          {!s.blob.length && <div className="text-xs text-gray-400">No changes recorded.</div>}
        </div>
      </Card>
    </div>
  );
}

export default function ActivityPage() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-gray-400">Loading…</div>}>
      <ActivityInner />
    </Suspense>
  );
}
