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
# where the index actually lives is decided per environment by
# lake_reader.docs_index_source() - never hard-code the az:// path


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


def _con(force_local=False):
    """A connection that can read the index in EITHER mode."""
    src = lake_reader.docs_index_source(force_local=force_local)
    if not src:
        raise RuntimeError("the document index is not available in this "
                           "environment (no direct blob access and nothing cached)")
    con = lake_reader._new_con()
    if src.startswith("az://"):
        lake_reader._load_azure(con)
        con.execute("CREATE OR REPLACE SECRET lake (TYPE AZURE, CONNECTION_STRING '%s')"
                    % os.environ["BLOB_CONNECTION_STRING"].replace("'", "''"))
    return con, src


def _run(build_sql, params):
    """Run against the blob, and on any non-SQL failure run it again from the
    local cache.

    The Function host advertises direct blob access - the probe passes - but
    individual reads then fail on SSL CA verification. lake_reader.query has
    always retried per query for exactly this reason; document reads have to
    do the same or they work everywhere except production."""
    for force_local in (False, True):
        con, src = _con(force_local=force_local)
        try:
            return con.execute(build_sql(src), params).fetchall()
        except Exception as e:
            if force_local or type(e).__name__ in lake_reader._SQL_ERRORS:
                raise
            LOG.warning("document read failed against blob (%s) - retrying "
                        "from the local cache", str(e)[:160])
        finally:
            con.close()
    return []


# What a document IS matters as much as what it says. A redline of the IP
# licence scored 0.625 against the final's 0.626 - indistinguishable - so an
# answer could quote a superseded clause with total confidence. Rank by
# authority, and keep competitor material out of "what are WE obliged to do"
# unless it is asked for by name.
_DRAFT_MARKERS = ("draft", "redline", "marked up", "markedup", "tracked",
                  "copy of", "/old/", "_old", "working copy", " v1", " v2",
                  "superseded")
_EXTERNAL_MARKERS = ("gong cha", "orange leaf", "clean juice", "competitor",
                     "benchmark")
AUTHORITY_WEIGHT = {"final": 1.0, "draft": 0.82, "external": 0.55}


def classify(path):
    """final | draft | external - from the path, which is all we can trust."""
    low = (path or "").lower()
    if any(m in low for m in _EXTERNAL_MARKERS):
        return "external"
    if any(m in low for m in _DRAFT_MARKERS):
        return "draft"
    return "final"


def search(query, top_k=8, path_like=None, chars=700, include_drafts=True,
           include_external=False, per_doc=1):
    """The passages that mean what the question means, best first.

    Results are ranked by relevance WEIGHTED BY AUTHORITY and capped at
    `per_doc` chunks per document, so eight results are eight documents rather
    than one document eight times.

    path_like narrows to a folder or document (SQL LIKE, case-insensitive) -
    use it to ask a question OF one agreement rather than of everything."""
    vec = _embed(query)
    where = ""
    params = [vec]
    if path_like:
        where = "WHERE lower(path) LIKE ?"
        params.append("%" + str(path_like).lower().strip("%") + "%")
    # over-fetch: authority weighting and per-document capping both discard
    # candidates, and we still want top_k distinct documents back
    params.append(int(top_k) * 8)
    rows = _run(lambda src: (
        "SELECT path, title, ext, page, substr(text, 1, %d) AS snippet, "
        "array_cosine_similarity(embedding::FLOAT[%d], ?::FLOAT[%d]) AS score "
        "FROM read_parquet('%s', union_by_name=true) %s "
        "ORDER BY score DESC LIMIT ?"
        % (int(chars), EMBED_DIMS, EMBED_DIMS, src, where)), params)
    cand = []
    seen_text = set()
    for p, t, e, pg, sn, sc in rows:
        # the corpus holds the same passage twice - a file indexed twice, or a
        # tracked-changes copy beside the clean one
        key = re.sub(r"\W+", "", (sn or "")[:160]).lower()
        if key in seen_text:
            continue
        seen_text.add(key)
        kind = classify(p)
        if kind == "external" and not include_external:
            continue
        if kind == "draft" and not include_drafts:
            continue
        cand.append({"path": p, "title": t, "ext": e, "page": pg,
                     "authority": kind, "raw_score": round(float(sc), 4),
                     "score": round(float(sc) * AUTHORITY_WEIGHT[kind], 4),
                     "snippet": sn})
    cand.sort(key=lambda h: -h["score"])
    out, per = [], {}
    for h in cand:
        if per.get(h["path"], 0) >= max(1, int(per_doc)):
            continue
        per[h["path"]] = per.get(h["path"], 0) + 1
        out.append(h)
        if len(out) >= int(top_k):
            break
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
        mark = "" if h.get("authority") in (None, "final") else \
            "  [%s - NOT the executed version, say so if you rely on it]" \
            % h["authority"].upper()
        out.append("[%s | relevance %.2f]%s\n%s"
                   % (where, h.get("score") or 0, mark,
                      (h.get("snippet") or "").strip()))
    return "\n\n".join(out)


def document_text(path_like, max_chars=18000):
    """One document reassembled from its chunks, in order.

    Used by the compliance audit: the corpus is already extracted in the lake,
    so re-downloading and re-parsing the file from SharePoint is wasted work."""
    rows = _run(lambda src: (
        "SELECT path, any_value(title) AS title, "
        "string_agg(text, '\n' ORDER BY chunk_id) AS body "
        "FROM read_parquet('%s', union_by_name=true) "
        "WHERE lower(path) LIKE ? GROUP BY path "
        "ORDER BY length(body) DESC LIMIT 1" % src),
        ["%" + str(path_like).lower().strip("%") + "%"])
    if not rows:
        return None
    path, title, body = rows[0]
    return {"path": path, "title": title, "text": (body or "")[:max_chars]}


def documents_under(prefix, exts=("docx", "pdf"), max_docs=40, max_chars=16000):
    """Every document under a folder, reassembled - the compliance audit's
    source material, one round trip instead of one download per file."""
    rows = _run(lambda src: (
        "SELECT path, any_value(title) AS title, any_value(ext) AS ext, "
        "string_agg(text, '\n' ORDER BY chunk_id) AS body "
        "FROM read_parquet('%s', union_by_name=true) "
        "WHERE lower(path) LIKE ? AND lower(ext) IN (%s) "
        "GROUP BY path HAVING length(body) > 400 "
        "ORDER BY length(body) DESC LIMIT ?"
        % (src, ",".join("'%s'" % e.lower() for e in exts))),
        ["%" + str(prefix).lower().strip("%") + "%", int(max_docs)])
    return [{"path": p, "title": t or os.path.basename(p), "ext": e,
             "text": (b or "")[:max_chars]} for p, t, e, b in rows]


# ------------------------------------------------------- hybrid retrieval

_STOP = {"the", "and", "for", "with", "what", "when", "which", "that", "this",
         "from", "into", "our", "we", "us", "are", "is", "was", "were", "do",
         "does", "did", "how", "why", "who", "whom", "any", "all", "can",
         "should", "would", "must", "have", "has", "had", "about", "under",
         "over", "per", "each", "their", "they", "them", "there", "then",
         # question verbs: they are how a person asks, never what to match on
         "find", "show", "list", "tell", "give", "need", "want", "please",
         "search", "look", "explain", "summarise", "summarize", "check"}


def _terms(query):
    """The tokens worth matching literally.

    Vector search is poor at exact strings - an ABN, an invoice number, a
    clause reference, an entity number - because an embedding blurs precisely
    the detail that makes them useful. Those come back through keyword search
    instead."""
    out = []
    for raw in re.findall(r'"([^"]+)"', query or ""):        # quoted phrases
        if len(raw.strip()) > 2:
            out.append(raw.strip().lower())
    for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9._/-]{2,}", query or ""):
        low = tok.lower()
        if low in _STOP or low in out:
            continue
        if any(c.isdigit() for c in low) or len(low) >= 4:
            out.append(low)
    return out[:8]


def _keyword_rows(terms, path_like, chars, limit):
    """Documents containing the literal terms, most-terms-matched first."""
    if not terms:
        return []
    score = " + ".join("CASE WHEN lower(text) LIKE ? THEN 1 ELSE 0 END"
                       for _ in terms)
    params = ["%" + t + "%" for t in terms]
    where = "(%s)" % " OR ".join("lower(text) LIKE ?" for _ in terms)
    params += ["%" + t + "%" for t in terms]
    if path_like:
        where += " AND lower(path) LIKE ?"
        params.append("%" + str(path_like).lower().strip("%") + "%")
    params.append(int(limit))
    return _run(lambda src: (
        "SELECT path, title, ext, page, substr(text, 1, %d) AS snippet, "
        "(%s) AS hits FROM read_parquet('%s', union_by_name=true) "
        "WHERE %s ORDER BY hits DESC LIMIT ?"
        % (int(chars), score, src, where)), params)


def hybrid_search(query, top_k=8, path_like=None, chars=700,
                  include_drafts=True, include_external=False, per_doc=1,
                  source="ask"):
    """Semantic and literal search, fused.

    Neither alone is enough: embeddings find the termination clause you could
    not name, keyword finds the invoice number embeddings smear away. Fused by
    reciprocal rank - a passage both methods like beats one that only either
    likes - then weighted by authority, same as the semantic path."""
    sem = search(query, top_k=top_k * 3, path_like=path_like, chars=chars,
                 include_drafts=include_drafts,
                 include_external=include_external, per_doc=99)
    terms = _terms(query)
    kw = []
    try:
        for p, t, e, pg, sn, hits in _keyword_rows(terms, path_like, chars,
                                                   top_k * 6):
            if int(hits or 0) <= 0:
                continue
            kind = classify(p)
            if kind == "external" and not include_external:
                continue
            if kind == "draft" and not include_drafts:
                continue
            kw.append({"path": p, "title": t, "ext": e, "page": pg,
                       "authority": kind, "snippet": sn,
                       "terms_matched": int(hits or 0)})
    except Exception:
        LOG.exception("keyword leg failed - answering on the semantic leg alone")

    fused = {}

    def _key(h):
        return (h["path"], h.get("page"), (h.get("snippet") or "")[:80])

    for rank, h in enumerate(sem):
        row = fused.setdefault(_key(h), dict(h, rrf=0.0, found_by=[]))
        row["rrf"] += 1.0 / (60 + rank)
        row["found_by"].append("meaning")
    for rank, h in enumerate(kw):
        row = fused.setdefault(_key(h), dict(h, rrf=0.0, found_by=[],
                                             score=0.0, raw_score=0.0))
        row["rrf"] += 1.0 / (60 + rank)
        row["found_by"].append("exact terms")

    boosts = _boosts(list({h["path"] for h in fused.values()}))
    ranked = sorted(fused.values(),
                    key=lambda h: -h["rrf"]
                    * AUTHORITY_WEIGHT.get(h.get("authority", "final"), 1.0)
                    * boosts.get(h["path"], 1.0))
    out, per = [], {}
    for h in ranked:
        if per.get(h["path"], 0) >= max(1, int(per_doc)):
            continue
        per[h["path"]] = per.get(h["path"], 0) + 1
        h["found_by"] = ", ".join(sorted(set(h["found_by"])))
        h["boost"] = boosts.get(h["path"], 1.0)
        out.append(h)
        if len(out) >= int(top_k):
            break
    _log_search(query, out, path_like=path_like, source=source)
    return out


# --------------------------------------------------------- corpus map

_MAP_CACHE = {"at": 0.0, "map": None}


def corpus_map(max_age=3600, top=18):
    """What is IN the corpus, by folder.

    The agent used to search blind - it could not know a 'Signed Countries' or
    an 'AUDIT' folder existed, so it could not aim at one. Handing it the shape
    of the corpus turns a broad sweep into a targeted read."""
    now = __import__("time").time()
    if _MAP_CACHE["map"] and now - _MAP_CACHE["at"] < max_age:
        return _MAP_CACHE["map"]
    try:
        rows = _run(lambda src: (
            "SELECT regexp_replace(path, '^(sharepoint/|canva/)', '') AS p, "
            "count(DISTINCT path) AS docs, max(mtime) AS newest "
            "FROM read_parquet('%s', union_by_name=true) GROUP BY p" % src), [])
    except Exception:
        LOG.exception("corpus map unavailable")
        return {"folders": [], "documents": 0}
    folders, total = {}, 0
    for p, docs, newest in rows:
        total += 1
        parts = (p or "").split("/")
        key = "/".join(parts[:3]) if len(parts) > 3 else "/".join(parts[:-1])
        f = folders.setdefault(key or "(root)", {"folder": key or "(root)",
                                                 "documents": 0, "newest": ""})
        f["documents"] += 1
        if newest and str(newest) > f["newest"]:
            f["newest"] = str(newest)[:10]
    out = {"documents": total,
           "folders": sorted(folders.values(), key=lambda f: -f["documents"])[:top]}
    _MAP_CACHE.update({"at": now, "map": out})
    return out


def map_text(limit=14):
    m = corpus_map()
    if not m.get("folders"):
        return ""
    lines = ["Document corpus: %d documents indexed. Largest folders "
             "(use path_like to aim at one):" % m["documents"]]
    for f in m["folders"][:limit]:
        lines.append("  %-58s %4d docs%s" % (
            f["folder"][:58], f["documents"],
            "  newest %s" % f["newest"] if f.get("newest") else ""))
    return "\n".join(lines)


# ------------------------------------------------------------- learning

# Calibrated, not guessed: over 5 real finance/legal questions and 4 nonsense
# ones, real questions scored 0.508-0.689 and nonsense 0.235-0.463. 0.49 sits
# in the gap. Re-run the calibration in tests/ if the embedding model changes -
# scores from a different model are on a different scale entirely.
USEFUL_SCORE = 0.49


def _log_search(question, hits, path_like=None, source="ask"):
    """Record what was asked and whether the corpus could answer it.

    A search that finds nothing is the most useful signal the system produces -
    it names a document we have not indexed, or do not hold at all - and until
    now it vanished silently."""
    try:
        import tk_db
        top = hits[0] if hits else None
        tk_db.insert("doc_search_log", [{
            "question": (question or "")[:500],
            "terms": ", ".join(_terms(question))[:300],
            "path_like": (path_like or "")[:200] or None,
            "hits": len(hits or []),
            "top_score": float(top["score"]) if top else None,
            "top_path": (top or {}).get("path", "")[:400] or None,
            "authorities": ",".join(sorted({h.get("authority", "final")
                                            for h in (hits or [])}))[:60],
            "answered": bool(top and float(top.get("score") or 0) >= USEFUL_SCORE),
            "source": source,
        }])
    except Exception:
        LOG.exception("could not log the document search (search itself was fine)")


def _boosts(paths):
    """How useful each of these documents has proven. 1.0 = neutral."""
    if not paths:
        return {}
    try:
        import tk_db
        rows = tk_db.get("doc_usefulness", {
            "path": "in.(%s)" % ",".join('"%s"' % p.replace('"', "") for p in paths),
            "select": "path,boost"})
        return {r["path"]: float(r.get("boost") or 1.0) for r in rows}
    except Exception:
        LOG.exception("usefulness boosts unavailable - ranking without them")
        return {}


def learn(days=30, max_boost=1.25):
    """Turn the search log into per-document usefulness.

    A document that keeps answering questions should surface sooner; one that
    never does should not crowd the top. The boost is deliberately small - it
    nudges ranking, it must never override what a question actually means."""
    import tk_db
    since = (__import__("datetime").datetime.utcnow()
             - __import__("datetime").timedelta(days=days)).isoformat() + "Z"
    rows = tk_db.get("doc_search_log", {
        "asked_at": "gte." + since, "answered": "eq.true",
        "select": "top_path,asked_at", "limit": "5000"})
    counts = {}
    for r in rows:
        p = r.get("top_path")
        if p:
            c = counts.setdefault(p, {"n": 0, "last": ""})
            c["n"] += 1
            c["last"] = max(c["last"], r.get("asked_at") or "")
    if not counts:
        return {"documents": 0, "note": "no answered searches yet to learn from"}
    busiest = max(c["n"] for c in counts.values())
    out = []
    for path, c in counts.items():
        # log-ish curve: being useful twice matters much more than the
        # difference between twenty and forty times
        frac = c["n"] / float(busiest)
        out.append({"path": path[:400], "times_top": c["n"],
                    "last_used_at": c["last"] or None,
                    "boost": round(1.0 + (max_boost - 1.0) * (frac ** 0.5), 4)})
    for i in range(0, len(out), 200):
        tk_db.insert("doc_usefulness", out[i:i + 200],
                     on_conflict="path", merge_duplicates=True)
    LOG.info("learned usefulness for %d document(s)", len(out))
    return {"documents": len(out), "top": sorted(
        out, key=lambda r: -r["times_top"])[:5]}


def unanswered(days=14, limit=25):
    """Questions the corpus could not answer - the to-index list, in priority
    order by how often they are asked."""
    import tk_db
    since = (__import__("datetime").datetime.utcnow()
             - __import__("datetime").timedelta(days=days)).isoformat() + "Z"
    rows = tk_db.get("doc_search_log", {
        "asked_at": "gte." + since, "answered": "eq.false",
        "select": "question,asked_at,hits", "limit": "1000"})
    seen = {}
    for r in rows:
        q = (r.get("question") or "").strip().lower()[:160]
        if not q:
            continue
        e = seen.setdefault(q, {"question": r["question"][:200], "asked": 0,
                                "last": ""})
        e["asked"] += 1
        e["last"] = max(e["last"], r.get("asked_at") or "")
    return sorted(seen.values(), key=lambda e: -e["asked"])[:limit]
