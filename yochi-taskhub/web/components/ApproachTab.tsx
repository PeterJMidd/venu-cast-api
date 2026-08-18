"use client";

// Recommended approach for this KIND of task, held in the data lake and amended
// every time the agent runs. The newest version is the standing approach; older
// versions stay as the record of how it got there.
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { callFn } from "@/lib/fn";

type Version = {
  approach_key: string;
  task_title: string;
  project: string | null;
  method: string;
  queries_that_worked: string | null;
  pitfalls: string | null;
  improve_next_time: string | null;
  outcome: string | null;
  version: string | null;
  changed_this_run: string | null;
  run_at: string;
};

type Playbook = {
  approach_key: string;
  current: Version | null;
  history: Version[];
};

function Block({ label, text }: { label: string; text?: string | null }) {
  if (!text) return null;
  return (
    <div className="mt-2">
      <div className="text-[11px] font-bold uppercase tracking-wide text-gray-400">
        {label}
      </div>
      <div className="whitespace-pre-wrap text-[13px] leading-relaxed text-gray-800">
        {text}
      </div>
    </div>
  );
}

export default function ApproachTab({ taskId }: { taskId: string }) {
  const [open, setOpen] = useState<string | null>(null);
  const { data, isLoading, error } = useQuery({
    queryKey: ["task", taskId, "playbook"],
    queryFn: () => callFn<Playbook>("task_playbook", { task_id: taskId }),
  });

  if (isLoading) return <div className="text-sm text-gray-400">Loading…</div>;
  if (error)
    return (
      <div className="text-sm text-red-600">
        {error instanceof Error ? error.message : String(error)}
      </div>
    );

  const cur = data?.current;

  return (
    <div className="space-y-4">
      {cur ? (
        <div className="rounded-xl border border-brand-200 bg-brand-50/40 p-4">
          <div className="mb-1 flex flex-wrap items-baseline gap-2">
            <span className="text-xs font-bold uppercase tracking-wide text-brand-800">
              Recommended approach
            </span>
            <span className="text-[11px] text-gray-500">
              version {cur.version || "1"} · {cur.approach_key} ·{" "}
              {(cur.run_at || "").slice(0, 10)}
              {cur.outcome ? ` · last run ${cur.outcome}` : ""}
            </span>
          </div>
          <div className="whitespace-pre-wrap text-[13px] leading-relaxed text-gray-800">
            {cur.method}
          </div>
          <Block label="Data that works" text={cur.queries_that_worked} />
          <Block label="Pitfalls" text={cur.pitfalls} />
          <Block label="Improve next time" text={cur.improve_next_time} />
          {cur.changed_this_run && (
            <div className="mt-3 rounded-lg bg-white/70 px-3 py-2 text-[12px] text-gray-600">
              <span className="font-semibold">Latest run changed:</span>{" "}
              {cur.changed_this_run}
            </div>
          )}
        </div>
      ) : (
        <div className="rounded-xl border border-dashed border-gray-300 p-5 text-center text-sm text-gray-400">
          No approach recorded yet for this kind of task. Click{" "}
          <span className="font-semibold">Propose how to do this</span> on the
          Details tab — the approach is saved here as soon as it&apos;s proposed,
          and sharpened every time the agent runs.
        </div>
      )}

      {!!data?.history?.length && (
        <div>
          <div className="mb-2 text-xs font-bold uppercase tracking-wide text-gray-400">
            How it evolved ({data.history.length} earlier{" "}
            {data.history.length === 1 ? "version" : "versions"})
          </div>
          <div className="space-y-2">
            {data.history.map((v) => {
              const id = `${v.version}-${v.run_at}`;
              const isOpen = open === id;
              return (
                <div key={id} className="rounded-lg border border-gray-200 bg-white">
                  <button
                    onClick={() => setOpen(isOpen ? null : id)}
                    className="flex w-full items-start gap-2 p-3 text-left"
                  >
                    <span className="mt-0.5 shrink-0 text-gray-300">
                      {isOpen ? "▾" : "▸"}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block text-sm font-medium">
                        Version {v.version || "?"}
                        <span className="ml-2 font-normal text-[11px] text-gray-400">
                          {(v.run_at || "").slice(0, 10)}
                          {v.outcome ? ` · ${v.outcome}` : ""}
                        </span>
                      </span>
                      {v.changed_this_run && (
                        <span className="mt-0.5 block text-[11px] text-gray-500">
                          {v.changed_this_run}
                        </span>
                      )}
                    </span>
                  </button>
                  {isOpen && (
                    <div className="border-t border-gray-100 px-3 py-2">
                      <div className="whitespace-pre-wrap text-[13px] leading-relaxed text-gray-700">
                        {v.method}
                      </div>
                      <Block label="Data that worked" text={v.queries_that_worked} />
                      <Block label="Pitfalls" text={v.pitfalls} />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      <p className="text-[11px] text-gray-400">
        Held in the data lake as <code>agent_playbook</code>, keyed by the kind of
        task rather than this one task — so a similar task next month starts from
        here. Also queryable from the Data lake page and the voice assistant.
      </p>
    </div>
  );
}
