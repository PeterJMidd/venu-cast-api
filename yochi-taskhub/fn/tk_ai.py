# -*- coding: utf-8 -*-
"""Claude API helper (stdlib urllib, same pattern as yochi-daily-insights).
Supports plain text calls and forced tool-use for structured JSON output."""
import os
import json
import logging
import urllib.request

LOG = logging.getLogger("tk_ai")
API_URL = "https://api.anthropic.com/v1/messages"


def _model(cheap=False):
    if cheap:
        return os.environ.get("ANTHROPIC_MODEL_CHEAP", "claude-haiku-4-5-20251001")
    return os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")


def _call(body):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not configured")
    req = urllib.request.Request(API_URL, data=json.dumps(body).encode(), headers={
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }, method="POST")
    with urllib.request.urlopen(req, timeout=420) as r:
        return json.loads(r.read().decode())


def text(system, user, cheap=False, max_tokens=2000):
    data = _call({
        "model": _model(cheap),
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    })
    out = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
    if out.startswith("```"):
        out = out.split("\n", 1)[1] if "\n" in out else out
        if out.rstrip().endswith("```"):
            out = out.rstrip()[:-3]
    return out.strip()


def structured(system, user, tool_name, tool_schema, cheap=False, max_tokens=2000):
    """Force a single tool call and return its validated-by-API input dict."""
    data = _call({
        "model": _model(cheap),
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "tools": [{"name": tool_name, "description": "Return the structured result.",
                   "input_schema": tool_schema}],
        "tool_choice": {"type": "tool", "name": tool_name},
    })
    for block in data.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == tool_name:
            return block["input"]
    raise RuntimeError("no tool_use block in AI response")
