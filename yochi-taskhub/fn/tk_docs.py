# -*- coding: utf-8 -*-
"""Semantic search over the document lake.

docs-lake/index/parts/*.parquet is the SharePoint + Canva corpus chunked and
embedded (path, title, ext, mtime, chunk_id, page, text, embedding[512]) by
yochi-data-lake/doc_extract.py. Keyword search finds a clause only if you
guess its wording; this finds it by meaning - "what do we owe on termination"
lands on the termination clause whatever it calls itself.

The query MUST be embedded with the same model and dimensions the index was
built with (OpenAI text-embedding-3-small @ 512) or the vectors are not
comparable and the scores are noise."""
import json
import logging
import os
import re
import urllib.request

import lake_reader

LOG = logging.getLogger("tk_docs")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIMS = int(os.environ.get("EMBED_DIMS", "512"))
INDEX = "az://docs-lake/index/parts/*.parquet"


def enabled():
    return bool(os.environ.get("OPENAI_API_KEY"))


def _embed(text):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not configured on this app, so "
                           "semantic document search is unavailable")
    body = json.dumps({"model": EMBED_MODEL, "input": [text[:20000]],
                       "dimensions": EMBED_DIMS}).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings", data=body,
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())["data"][0]["embedding"]


def _con():
    con = lake_reader._new_con()
    lake_reader._load_azure(con)
    con.execute("CREATE OR REPLACE SECRET lake (TYPE AZURE, CONNECTION_STRING '%s')"
                % os.environ["BLOB_CONNECTION_STRING"].replace("'", "''"))
    return con


def search(query, top_k=8, path_like=None, chars=700):
    """The passages that mean what the question means, best first.

    path_like narrows to a folder or document (SQL LIKE, case-insensitive) -
    use it to ask a question OF one agreement rather than of everything."""
    vec = _embed(query)
    where = ""
    params = [vec]
    if path_like:
        where = "WHERE lower(path) LIKE ?"
        params.append("%" + str(path_like).lower().strip("%") + "%")
    # over-fetch, because deduplication below removes repeats of the same
    # passage and we still want top_k distinct answers
    params.append(int(top_k) * 4)
    sql = ("SELECT path, title, ext, page, substr(text, 1, %d) AS snippet, "
           "array_cosine_similarity(embedding::FLOAT[%d], ?::FLOAT[%d]) AS score "
           "FROM read_parquet('%s', union_by_name=true) %s "
           "ORDER BY score DESC LIMIT ?"
           % (int(chars), EMBED_DIMS, EMBED_DIMS, INDEX, where))
    con = _con()
    try:
        rows = con.execute(sql, params).fetchall()
    finally:
        con.close()
    # the corpus contains the same passage more than once - a file indexed
    # twice, or a tracked-changes copy alongside the clean one - and three
    # identical hits crowd out three different answers
    out, seen = [], set()
    for p, t, e, pg, sn, s in rows:
        if len(out) >= int(top_k):
            break
        key = re.sub(r"\W+", "", (sn or "")[:160]).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"path": p, "title": t, "ext": e, "page": pg,
                    "score": round(float(s), 4), "snippet": sn})
    return out


def as_text(hits, limit=8):
    """Hits formatted for a prompt - named, so the model can cite them."""
    if not hits:
        return "(no matching passages)"
    out = []
    for h in hits[:limit]:
        where = h.get("title") or h.get("path") or "document"
        if h.get("page"):
            where += " p.%s" % int(h["page"])
        out.append("[%s | relevance %.2f]\n%s"
                   % (where, h.get("score") or 0, (h.get("snippet") or "").strip()))
    return "\n\n".join(out)


def document_text(path_like, max_chars=18000):
    """One document reassembled from its chunks, in order.

    Used by the compliance audit: the corpus is already extracted in the lake,
    so re-downloading and re-parsing the file from SharePoint is wasted work."""
    con = _con()
    try:
        rows = con.execute(
            "SELECT path, any_value(title) AS title, "
            "string_agg(text, '\n' ORDER BY chunk_id) AS body "
            "FROM read_parquet('%s', union_by_name=true) "
            "WHERE lower(path) LIKE ? GROUP BY path "
            "ORDER BY length(body) DESC LIMIT 1" % INDEX,
            ["%" + str(path_like).lower().strip("%") + "%"]).fetchall()
    finally:
        con.close()
    if not rows:
        return None
    path, title, body = rows[0]
    return {"path": path, "title": title, "text": (body or "")[:max_chars]}


def documents_under(prefix, exts=("docx", "pdf"), max_docs=40, max_chars=16000):
    """Every document under a folder, reassembled - the compliance audit's
    source material, one round trip instead of one download per file."""
    con = _con()
    try:
        rows = con.execute(
            "SELECT path, any_value(title) AS title, any_value(ext) AS ext, "
            "string_agg(text, '\n' ORDER BY chunk_id) AS body "
            "FROM read_parquet('%s', union_by_name=true) "
            "WHERE lower(path) LIKE ? AND lower(ext) IN (%s) "
            "GROUP BY path HAVING length(body) > 400 "
            "ORDER BY length(body) DESC LIMIT ?"
            % (INDEX, ",".join("'%s'" % e.lower() for e in exts)),
            ["%" + str(prefix).lower().strip("%") + "%", int(max_docs)]).fetchall()
    finally:
        con.close()
    return [{"path": p, "title": t or os.path.basename(p), "ext": e,
             "text": (b or "")[:max_chars]} for p, t, e, b in rows]
