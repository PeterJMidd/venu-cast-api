"use client";

// Floating voice assistant: hold-to-talk mic (Web Speech API) or typed
// question -> /api/ask (agentic answer over the data lake + TaskHub state).
// Answers can be read aloud with speechSynthesis.
import { useEffect, useRef, useState } from "react";
import { supabase } from "@/lib/supabase";

type Msg = { role: "user" | "assistant"; content: string };

declare global {
  interface Window {
    webkitSpeechRecognition?: any;
    SpeechRecognition?: any;
  }
}

export default function VoicePanel() {
  const [open, setOpen] = useState(false);
  const [listening, setListening] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [interim, setInterim] = useState("");
  const [input, setInput] = useState("");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [speak, setSpeak] = useState(true);
  const recRef = useRef<any>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const supported =
    typeof window !== "undefined" && !!(window.SpeechRecognition || window.webkitSpeechRecognition);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [msgs, thinking]);

  async function ask(question: string) {
    const q = question.trim();
    if (!q || thinking) return;
    setInterim("");
    setInput("");
    setMsgs((m) => [...m, { role: "user", content: q }]);
    setThinking(true);
    try {
      const { data } = await supabase.auth.getSession();
      const res = await fetch(`${process.env.NEXT_PUBLIC_FN_BASE}/api/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${data.session?.access_token}`,
        },
        body: JSON.stringify({ question: q, history: msgs.slice(-6) }),
      });
      const out = await res.json();
      const answer: string = out.answer ?? out.error ?? "Something went wrong.";
      setMsgs((m) => [...m, { role: "assistant", content: answer }]);
      if (speak && "speechSynthesis" in window) {
        const u = new SpeechSynthesisUtterance(answer.slice(0, 500));
        u.lang = "en-AU";
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
      }
    } catch (e) {
      setMsgs((m) => [...m, { role: "assistant", content: "Request failed - try again." }]);
    } finally {
      setThinking(false);
    }
  }

  function startListening() {
    if (!supported || listening) return;
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const rec = new SR();
    rec.lang = "en-AU";
    rec.interimResults = true;
    rec.continuous = false;
    let finalText = "";
    rec.onresult = (e: any) => {
      let txt = "";
      for (let i = 0; i < e.results.length; i++) txt += e.results[i][0].transcript;
      setInterim(txt);
      if (e.results[e.results.length - 1].isFinal) finalText = txt;
    };
    rec.onend = () => {
      setListening(false);
      if (finalText.trim()) ask(finalText);
      else setInterim("");
    };
    rec.onerror = () => setListening(false);
    recRef.current = rec;
    setListening(true);
    setOpen(true);
    rec.start();
  }

  function stopListening() {
    recRef.current?.stop();
  }

  return (
    <>
      {/* Floating mic button */}
      <button
        onClick={() => (listening ? stopListening() : open ? startListening() : (setOpen(true), startListening()))}
        title="Ask anything (voice)"
        className={`fixed bottom-5 left-5 z-40 flex h-12 w-12 items-center justify-center rounded-full shadow-lg transition ${
          listening ? "animate-pulse bg-red-600 text-white" : "bg-brand-600 text-white hover:bg-brand-700"
        }`}
      >
        {listening ? "◼" : "🎤"}
      </button>

      {open && (
        <div className="fixed bottom-20 left-5 z-40 flex max-h-[70vh] w-[22rem] flex-col overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-2xl">
          <div className="flex items-center justify-between border-b border-gray-100 px-3 py-2">
            <span className="text-xs font-bold text-gray-600">ASK THE COCKPIT</span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  if (speak) window.speechSynthesis?.cancel();
                  setSpeak(!speak);
                }}
                className={`text-xs ${speak ? "" : "opacity-40"}`}
                title="Read answers aloud"
              >
                🔊
              </button>
              <button onClick={() => setOpen(false)} className="text-sm text-gray-400 hover:text-gray-600">
                ✕
              </button>
            </div>
          </div>

          <div ref={scrollRef} className="flex-1 space-y-2 overflow-y-auto p-3">
            {msgs.length === 0 && !listening && (
              <div className="text-xs leading-relaxed text-gray-400">
                Ask about anything across the systems — sales, cash, suppliers, tasks, venues, meetings.
                <br />
                <span className="text-gray-300">
                  e.g. &ldquo;How did WA trade last week vs budget?&rdquo; · &ldquo;Which suppliers went up in
                  price?&rdquo; · &ldquo;What&rsquo;s my cash trough?&rdquo;
                </span>
              </div>
            )}
            {msgs.map((m, i) => (
              <div
                key={i}
                className={`whitespace-pre-wrap rounded-xl px-3 py-2 text-xs leading-relaxed ${
                  m.role === "user" ? "ml-6 bg-brand-50 text-gray-800" : "mr-2 bg-gray-50 text-gray-800"
                }`}
              >
                {m.content}
              </div>
            ))}
            {listening && (
              <div className="ml-6 rounded-xl bg-red-50 px-3 py-2 text-xs italic text-red-700">
                {interim || "Listening…"}
              </div>
            )}
            {thinking && (
              <div className="mr-2 rounded-xl bg-gray-50 px-3 py-2 text-xs text-gray-400">
                Checking the data<span className="animate-pulse">…</span>
              </div>
            )}
          </div>

          <form
            onSubmit={(e) => {
              e.preventDefault();
              ask(input);
            }}
            className="flex items-center gap-2 border-t border-gray-100 p-2"
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={supported ? "Type or tap the mic…" : "Type a question…"}
              className="min-w-0 flex-1 rounded-lg border border-gray-200 px-2.5 py-1.5 text-xs focus:border-brand-500 focus:outline-none"
            />
            <button
              type="submit"
              disabled={thinking || !input.trim()}
              className="rounded-lg bg-brand-600 px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-40"
            >
              Ask
            </button>
          </form>
        </div>
      )}
    </>
  );
}
