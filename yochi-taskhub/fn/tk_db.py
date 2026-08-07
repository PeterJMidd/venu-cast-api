# -*- coding: utf-8 -*-
"""Thin PostgREST + Auth-admin client for the taskapp schema (stdlib urllib).
Uses the SERVICE ROLE key - server-side only, bypasses RLS."""
import os
import json
import logging
import urllib.request
import urllib.parse

LOG = logging.getLogger("tk_db")


def _base():
    return os.environ["SUPABASE_URL"].rstrip("/")


def _key():
    return os.environ["SUPABASE_SERVICE_ROLE_KEY"]


def _headers(profile_headers=True, prefer=None):
    h = {
        "apikey": _key(),
        "Authorization": "Bearer " + _key(),
        "Content-Type": "application/json",
    }
    if profile_headers:
        h["Accept-Profile"] = "taskapp"
        h["Content-Profile"] = "taskapp"
    if prefer:
        h["Prefer"] = prefer
    return h


def _req(method, url, body=None, headers=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:500]
        raise RuntimeError("%s %s -> %s: %s" % (method, url.split("?")[0], e.code, detail))


def get(table, params=None):
    qs = urllib.parse.urlencode(params or {}, doseq=True)
    return _req("GET", "%s/rest/v1/%s?%s" % (_base(), table, qs), headers=_headers())


def insert(table, rows, on_conflict=None, ignore_duplicates=False, returning=False):
    params = {}
    prefer = []
    if on_conflict:
        params["on_conflict"] = on_conflict
    if ignore_duplicates:
        prefer.append("resolution=ignore-duplicates")
    prefer.append("return=representation" if returning else "return=minimal")
    qs = urllib.parse.urlencode(params)
    url = "%s/rest/v1/%s%s" % (_base(), table, "?" + qs if qs else "")
    return _req("POST", url, body=rows, headers=_headers(prefer=",".join(prefer)))


def patch(table, params, body):
    qs = urllib.parse.urlencode(params, doseq=True)
    return _req("PATCH", "%s/rest/v1/%s?%s" % (_base(), table, qs),
                body=body, headers=_headers(prefer="return=representation"))


# ---------------------------------------------------------------- auth admin
def invite_user(email, role, full_name=None, redirect_to=None):
    """Send a Supabase invite email, then set the profile's role/active directly.
    (The auth trigger creates the profile, but GoTrue stamps invited_at a few ms
    AFTER the user insert, so the trigger's invite gate can't be relied on for
    role elevation - the service role sets it explicitly here instead.)"""
    body = {"email": email, "data": {"role": role, "full_name": full_name or ""}}
    url = "%s/auth/v1/invite" % _base()
    if redirect_to:
        url += "?" + urllib.parse.urlencode({"redirect_to": redirect_to})
    resp = _req("POST", url, body=body, headers=_headers(profile_headers=False))
    uid = (resp or {}).get("id")
    if uid:
        patch("profiles", {"id": "eq." + uid},
              {"role": role, "active": True, "full_name": full_name or None})
    return resp
