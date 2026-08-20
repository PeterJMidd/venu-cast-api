"use client";

// Ask a question on a task and get it answered, saved to the Knowledge tab and
// pointed to from the comments. Two kinds of question, chosen by engine:
// OUTWARD (Claude / Perplexity / compare both) searches the web and cites it -
// "compare" earns its keep on decisions, where agreement is corroboration and
// disagreement is the signal to dig. INWARD ("Our data lake") runs the agentic
// SQL loop over our own parquet and answers with figures, showing the queries
// it ran. Distinct from the Task agent, which plans and executes the whole task.
import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { callFn } from "@/lib/fn";

type Depth = "quick" | "standard" | "deep";
type Engine = "claude" | "perplexity" | "compare" | "lake";
type Recency = "" | "week" | "month" | "year";

const DEPTHS: { key: Depth; label: string }[] = [
  { key: "quick", label: "Quick" },
  { key: "standard", label: "Standard" },
  { key: "deep", label: "Deep" },
];

const ENGINES: { key: Engine; label: string; hint: string }[] = [
  { key: "claude", label: "Standard search", hint: "fastest, good default" },
  { key: "perplexity", label: "Perplexity", hint: "different index, strong on what changed" },
  { key: "compare", label: "Compare both", hint: "corroborates, flags conflicts" },
  { key: "lake", label: "Our data lake", hint: "answers from OUR numbers, not the web" },
];

const SUGGESTIONS = [
  "What are the current rules and rates, and when did they last change?",
  "What are the deadlines and who enforces them?",
  "How do comparable AU franchise groups handle this?",
  "What are the risks or common mistakes here?",
];

// asking OUR data is a different kind of question from asking the web
const LAKE_SUGGESTIONS = [
  "Summarise the transaction flows behind this, by entity and month.",
  "What have we actually paid or received on this in the last 12 months?",
  "Which entities and venues does this touch, and how much is involved?",
  "Show the trend over the last 6 months and flag anything unusual.",
];

export default function ResearchPanel({ taskId }: { taskId: string }) {
  const qc = useQueryClient();
  const [question, setQuestion] = useState("");
  const [depth, setDepth] = useState<Depth>("standard");
  const [engine, setEngine] = useState<Engine>("claude");
  const [recency, setRecency] = useState<Recency>("");
  const [busy, setBusy] = useState(false);
  const [answer, setAnswer] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const lake = engine === "lake";
  const slow = engine === "compare" && depth === "deep";

  async function ask(q?: string) {
    const text = (q ?? question).trim();
    if (!text || busy) return;
    setBusy(true);
    setError(null);
    setAnswer(null);
    try {
      const out = await callFn<{ answer: string; engine: Engine }>("task_research", {
        task_id: taskId,
        question: text,
        depth,
        engine,
        recency: recency || undefined,
      });
      setAnswer(out.answer);
      setQuestion("");
      qc.invalidateQueries({ queryKey: ["task", taskId, "comments"] });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4">
      <div className="mb-1 flex items-center gap-2">
        <span className="text-sm font-semibold">
          {lake ? "📊 Ask our data" : "🔍 Research this"}
        </span>
        <span className="text-[11px] text-gray-400">
          {lake
            ? "queries the data lake and answers with figures"
            : "live web search, always cited"}
        </span>
      </div>
      <p className="mb-2 text-xs text-gray-500">
        Ask about rules, rates, deadlines or what others do. The answer is saved to
        this task&apos;s comments. For questions about <em>our own</em> numbers, use the
        Task agent above.
      </p>

      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask();
        }}
        rows={2}
        placeholder={
          lake
            ? "e.g. Summarise the transaction flows behind the agreements for this market"
            : "e.g. What withholding tax applies to royalties from Singapore to Australia?"
        }
        className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none"
      />

      <div className="mt-2 flex flex-wrap items-center gap-2">
        <select
          value={engine}
          onChange={(e) => setEngine(e.target.value as Engine)}
          title={ENGINES.find((x) => x.key === engine)?.hint}
          className="rounded-lg border border-gray-300 px-2 py-1.5 text-xs"
        >
          {ENGINES.map((x) => (
            <option key={x.key} value={x.key}>{x.label}</option>
          ))}
        </select>
        {!lake && (
          <>
            <select
              value={depth}
              onChange={(e) => setDepth(e.target.value as Depth)}
              className="rounded-lg border border-gray-300 px-2 py-1.5 text-xs"
            >
              {DEPTHS.map((d) => (
                <option key={d.key} value={d.key}>{d.label}</option>
              ))}
            </select>
            <select
              value={recency}
              onChange={(e) => setRecency(e.target.value as Recency)}
              title="Only consider recent sources — useful for 'what changed'"
              className="rounded-lg border border-gray-300 px-2 py-1.5 text-xs"
            >
              <option value="">Any date</option>
              <option value="week">Last week</option>
              <option value="month">Last month</option>
              <option value="year">Last year</option>
            </select>
          </>
        )}
        <button
          onClick={() => ask()}
          disabled={busy || !question.trim()}
          className="rounded-lg bg-brand-600 px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-50"
        >
          {busy ? (lake ? "Analysing…" : "Researching…") : lake ? "Analyse" : "Research"}
        </button>
        {busy && (
          <span className="text-[11px] text-gray-400">
            {lake
              ? "querying the data lake…"
              : engine === "compare"
                ? "running both engines and reconciling…"
                : "searching the web and reading sources…"}
          </span>
        )}
      </div>

      <div className="mt-1 text-[11px] text-gray-400">
        {lake
          ? "Queries our own data lake and answers with figures — it shows the SQL it ran, and the finding is saved to the Knowledge tab."
          : engine === "compare"
            ? "Runs both search engines, then reports what they agree on, where they differ, and what to verify."
            : ENGINES.find((x) => x.key === engine)?.hint}
        {slow && " — deep + compare can take a couple of minutes."}
      </div>

      {!answer && !busy && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {(lake ? LAKE_SUGGESTIONS : SUGGESTIONS).map((s) => (
            <button
              key={s}
              onClick={() => ask(s)}
              className="rounded-full border border-gray-200 px-2.5 py-1 text-[11px] text-gray-600 hover:border-brand-500 hover:text-brand-700"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {error && (
        <div className="mt-2 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-700">{error}</div>
      )}

      {answer && (
        <div className="mt-3 rounded-lg border border-gray-100 bg-gray-50 p-3">
          <div className="whitespace-pre-wrap text-[13px] leading-relaxed text-gray-800">
            {answer}
          </div>
          <div className="mt-2 text-[11px] text-brand-700">
            Saved to this task&apos;s comments.
          </div>
        </div>
      )}
    </div>
  );
}
