"use client";

// Learnings & knowledge for one task: the living "what we know" summary on top,
// the ask-box, then the full history of findings underneath. Every research run
// is merged into the summary, so asking more questions makes the task smarter
// rather than just longer - and nothing is ever lost.
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { format, parseISO } from "date-fns";
import { supabase } from "@/lib/supabase";
import { profileName } from "@/hooks/useProfiles";
import type { Profile } from "@/lib/types";
import ResearchPanel from "@/components/ResearchPanel";

type Entry = {
  id: string;
  question: string;
  answer: string;
  engine: string | null;
  depth: string | null;
  created_by: string | null;
  created_at: string;
};

type Knowledge = {
  task_id: string;
  summary: string;
  entry_count: number;
  updated_at: string;
};

const ENGINE_LABEL: Record<string, string> = {
  claude: "standard search",
  perplexity: "Perplexity",
  compare: "both engines",
};

export default function KnowledgeTab({
  taskId,
  profiles,
}: {
  taskId: string;
  profiles?: Profile[];
}) {
  const [open, setOpen] = useState<string | null>(null);

  const { data: knowledge } = useQuery({
    queryKey: ["task", taskId, "knowledge"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("task_knowledge")
        .select("*")
        .eq("task_id", taskId)
        .maybeSingle();
      if (error) throw error;
      return data as Knowledge | null;
    },
  });

  const { data: entries } = useQuery({
    queryKey: ["task", taskId, "knowledge-entries"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("knowledge")
        .select("id,question,answer,engine,depth,created_by,created_at")
        .eq("task_id", taskId)
        .order("created_at", { ascending: false });
      if (error) throw error;
      return data as Entry[];
    },
  });

  return (
    <div className="space-y-4">
      {knowledge?.summary ? (
        <div className="rounded-xl border border-brand-200 bg-brand-50/40 p-4">
          <div className="mb-2 flex flex-wrap items-baseline gap-2">
            <span className="text-xs font-bold uppercase tracking-wide text-brand-800">
              What we know
            </span>
            <span className="text-[11px] text-gray-500">
              built from {knowledge.entry_count}{" "}
              {knowledge.entry_count === 1 ? "finding" : "findings"} · updated{" "}
              {format(parseISO(knowledge.updated_at), "d MMM HH:mm")}
            </span>
          </div>
          <div className="whitespace-pre-wrap text-[13px] leading-relaxed text-gray-800">
            {knowledge.summary}
          </div>
        </div>
      ) : (
        <div className="rounded-xl border border-dashed border-gray-300 p-5 text-center text-sm text-gray-400">
          Nothing learned on this task yet. Ask a question below and the answer
          becomes the start of this task&apos;s knowledge.
        </div>
      )}

      <ResearchPanel taskId={taskId} />

      {!!entries?.length && (
        <div>
          <div className="mb-2 text-xs font-bold uppercase tracking-wide text-gray-400">
            Findings ({entries.length}) — newest first
          </div>
          <div className="space-y-2">
            {entries.map((e) => {
              const isOpen = open === e.id;
              return (
                <div key={e.id} className="rounded-lg border border-gray-200 bg-white">
                  <button
                    onClick={() => setOpen(isOpen ? null : e.id)}
                    className="flex w-full items-start gap-2 p-3 text-left"
                  >
                    <span className="mt-0.5 shrink-0 text-gray-300">
                      {isOpen ? "▾" : "▸"}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block text-sm font-medium leading-snug">
                        {e.question}
                      </span>
                      <span className="mt-0.5 block text-[11px] text-gray-400">
                        {format(parseISO(e.created_at), "d MMM HH:mm")}
                        {e.engine ? ` · ${ENGINE_LABEL[e.engine] ?? e.engine}` : ""}
                        {e.depth ? ` · ${e.depth}` : ""}
                        {e.created_by ? ` · ${profileName(profiles, e.created_by)}` : ""}
                      </span>
                    </span>
                  </button>
                  {isOpen && (
                    <div className="border-t border-gray-100 px-3 py-2">
                      <div className="whitespace-pre-wrap text-[13px] leading-relaxed text-gray-700">
                        {e.answer}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
