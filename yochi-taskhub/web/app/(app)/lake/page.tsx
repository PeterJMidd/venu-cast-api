"use client";

import { useState } from "react";
import { callFn } from "@/lib/fn";
import { useProfile } from "@/hooks/useProfile";
import CatalogList from "@/components/CatalogList";

interface ChatMsg {
  role: "user" | "assistant";
  content: string;
}

const LAKE_AGENT_URL = "https://yochi-lake-agent.azurewebsites.net/";

export default function LakePage() {
  const { data: me } = useProfile();
  const [search, setSearch] = useState("");
  const [question, setQuestion] = useState("");
  const [thread, setThread] = useState<ChatMsg[]>([]);
  const [asking, setAsking] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  if (me && me.role === "stakeholder") {
    return <div className="p-8 text-sm text-gray-400">Finance access required.</div>;
  }

  async function ask(e: React.FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (!q || asking) return;
    setAsking(true);
    setErr(null);
    setQuestion("");
    const next: ChatMsg[] = [...thread, { role: "user", content: q }];
    setThread(next);
    try {
      const r = await callFn<{ answer: string }>("ask_lake", {
        question: q,
        history: thread,
      });
      setThread([...next, { role: "assistant", content: r.answer }]);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setThread(thread);
      setQuestion(q);
    } finally {
      setAsking(false);
    }
  }

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <div className="mb-1 flex items-center justify-between">
        <h1 className="text-xl font-bold">Data lake</h1>
        <a
          href={LAKE_AGENT_URL}
          target="_blank"
          rel="noreferrer"
          className="text-xs font-semibold text-brand-600 hover:underline"
        >
          Open the Lake Agent ↗
        </a>
      </div>
      <p className="mb-6 text-sm text-gray-500">
        Everything TaskHub&rsquo;s checks and AI skills can see — refreshed nightly from dataSights.
        Ask below what data exists, how to query it, or how to phrase a new skill.
      </p>

      {/* Ask the lake */}
      <div className="mb-8 rounded-xl border border-brand-100 bg-brand-50/50 p-4">
        {thread.length > 0 && (
          <div className="mb-3 max-h-96 space-y-3 overflow-y-auto">
            {thread.map((m, i) => (
              <div
                key={i}
                className={`rounded-lg px-3 py-2 text-sm ${
                  m.role === "user"
                    ? "ml-10 bg-brand-600 text-white"
                    : "mr-4 whitespace-pre-wrap border border-gray-200 bg-white text-gray-800"
                }`}
              >
                {m.content}
              </div>
            ))}
            {asking && <div className="text-xs text-gray-400">thinking…</div>}
          </div>
        )}
        <form onSubmit={ask} className="flex gap-2">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="✨ Ask about the data… e.g. 'What do we have on refunds?' or 'How would I build a skill to watch gift card sales?'"
            className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
          />
          <button
            disabled={asking || !question.trim()}
            className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700 disabled:opacity-50"
          >
            Ask
          </button>
        </form>
        {err && <div className="mt-2 text-sm text-red-600">{err}</div>}
      </div>

      {/* Catalog */}
      <input
        placeholder="Search tables or columns…"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="mb-3 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
      />
      <CatalogList search={search} />
    </div>
  );
}
