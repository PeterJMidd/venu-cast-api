# -*- coding: utf-8 -*-
"""Verify Supabase user JWTs sent by the SPA (Authorization: Bearer <token>).
Supports the new asymmetric signing keys (JWKS, ES256/RS256) with an HS256
legacy-secret fallback. The caller's role is ALWAYS re-read from taskapp.profiles
server-side - never trusted from the client."""
import os
import json
import time
import logging
import urllib.request

import jwt
from jwt import PyJWKClient

import tk_db

LOG = logging.getLogger("tk_auth")
_JWKS_CLIENT = None
_JWKS_TS = 0


class AuthError(Exception):
    pass


def _jwks_client():
    global _JWKS_CLIENT, _JWKS_TS
    if _JWKS_CLIENT is None or time.time() - _JWKS_TS > 3600:
        url = os.environ["SUPABASE_URL"].rstrip("/") + "/auth/v1/.well-known/jwks.json"
        _JWKS_CLIENT = PyJWKClient(url, cache_keys=True)
        _JWKS_TS = time.time()
    return _JWKS_CLIENT


def verify(req):
    """Returns (user_id, role, email). Raises AuthError on any failure."""
    auth = req.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise AuthError("missing bearer token")
    token = auth[7:]

    header = jwt.get_unverified_header(token)
    alg = header.get("alg", "")
    try:
        if alg == "HS256":
            secret = os.environ.get("SUPABASE_JWT_SECRET")
            if not secret:
                raise AuthError("HS256 token but SUPABASE_JWT_SECRET not configured")
            claims = jwt.decode(token, secret, algorithms=["HS256"], audience="authenticated")
        else:
            key = _jwks_client().get_signing_key_from_jwt(token)
            claims = jwt.decode(token, key.key, algorithms=["ES256", "RS256"],
                                audience="authenticated")
    except jwt.PyJWTError as e:
        raise AuthError("invalid token: %s" % e)

    uid = claims.get("sub")
    if not uid:
        raise AuthError("no sub claim")

    rows = tk_db.get("profiles", {"id": "eq." + uid, "select": "role,active,email"})
    if not rows or not rows[0]["active"]:
        raise AuthError("no active profile")
    return uid, rows[0]["role"], rows[0]["email"]
