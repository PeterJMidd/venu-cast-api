# -*- coding: utf-8 -*-
"""Perplexity (Sonar) as a SECOND research engine.

Value is not "another AI" - it is a different retrieval stack. Two independent
search engines agreeing on a treaty rate is corroboration; disagreeing is the
signal to dig or ask an advisor. Used for the deep and compare modes only; the
everyday default stays on the existing engine so we are not paying twice for
easy questions.

Model names are overridable by app setting (PPLX_MODEL_*) so a rename upstream
is a config change, not a deploy."""
import json
import logging
import os
import urllib.error
import urllib.request

LOG = logging.getLogger("tk_perplexity")

API = "https://api.perplexity.ai/chat/completions"
DEFAULT_MODELS = {"quick": "sonar",
                  "standard": "sonar-pro",
                  "deep": "sonar-reasoning-pro"}
TIMEOUT = 180


def enabled():
    return bool(os.environ.get("PERPLEXITY_API_KEY"))


def _model(depth):
    return os.environ.get("PPLX_MODEL_" + depth.upper()) \
        or DEFAULT_MODELS.get(depth, DEFAULT_MODELS["standard"])


def ask(system, question, depth="standard", recency=None, max_tokens=3000):
    """Returns (answer_text, [citation urls]). Raises on transport/auth error."""
    key = os.environ.get("PERPLEXITY_API_KEY")
    if not key:
        raise RuntimeError("PERPLEXITY_API_KEY is not set")
    body = {
        "model": _model(depth),
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": question}],
    }
    if recency in ("day", "week", "month", "year"):
        body["search_recency_filter"] = recency
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise RuntimeError("Perplexity %s: %s" % (e.code, detail))
    choices = data.get("choices") or []
    text = ""
    if choices:
        text = (choices[0].get("message") or {}).get("content", "")
    # citations come back either as a flat list or as search_results
    cites = data.get("citations") or []
    if not cites:
        cites = [s.get("url") for s in (data.get("search_results") or [])
                 if s.get("url")]
    return (text or "").strip(), [c for c in cites if c][:20]


def ask_formatted(system, question, depth="standard", recency=None,
                  max_tokens=3000):
    """Answer with a Sources block appended, matching the other engine's shape.
    <think> blocks from reasoning models are stripped."""
    text, cites = ask(system, question, depth=depth, recency=recency,
                      max_tokens=max_tokens)
    if "</think>" in text:                      # reasoning models emit these
        text = text.split("</think>", 1)[1].strip()
    if cites and "http" not in text:
        text += "\n\nSources:\n" + "\n".join("- " + c for c in cites)
    return text
