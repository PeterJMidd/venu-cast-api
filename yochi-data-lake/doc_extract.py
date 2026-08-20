# -*- coding: utf-8 -*-
"""Extract text from docs-lake documents, embed it, and build the semantic index.

Pipeline (resumable, time-budgeted like sp_mirror):
  docs-lake/sharepoint/** + canva/**  ->  text chunks  ->  OpenAI embeddings
  ->  docs-lake/index/parts/<n>.parquet   (path, title, ext, mtime, chunk_id, page, text, embedding)
  state: docs-lake/index/_state.json.gz   {done: {path: etag}, next_part: n}

Extractors: pdf (pypdf), docx, pptx, xlsx (openpyxl values), msg (extract_msg),
csv/txt/md (plain). No OCR - scanned PDFs yield little/no text and are skipped
if empty. Size/char caps keep pathological files cheap.

Embeddings: text-embedding-3-small @ EMBED_DIMS (default 512) - cheap + fast
brute-force cosine in DuckDB on the query side.
"""
import gzip
import io
import json
import logging
import os
import time
import urllib.request

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from azure.storage.blob import BlobServiceClient

LOG = logging.getLogger("doc_extract")
CONTAINER = os.environ.get("DOCS_CONTAINER", "docs-lake")
STATE_BLOB = "index/_state.json.gz"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIMS = int(os.environ.get("EMBED_DIMS", "512"))
CHUNK_CHARS = 4000
CHUNK_OVERLAP = 300
MAX_DOC_CHARS = 200_000
MAX_FILE_BYTES = 80 * 1024 * 1024
EMBED_BATCH = 400

# 'msg' excluded - saved emails are not mirrored or indexed (owner decision 2026-07).
TEXT_EXTS = {"pdf", "docx", "pptx", "xlsx", "xlsm", "csv", "txt", "md", "rtf"}


def _svc():
    return BlobServiceClient.from_connection_string(os.environ["BLOB_CONNECTION_STRING"])


def _load_state(cc):
    try:
        raw = cc.download_blob(STATE_BLOB).readall()
        return json.loads(gzip.decompress(raw))
    except Exception:
        return {"done": {}, "next_part": 0}


def _save_state(cc, state):
    cc.upload_blob(STATE_BLOB, gzip.compress(json.dumps(state).encode()), overwrite=True)


# ---------------- text extraction ----------------

def _pdf_text(data):
    from pypdf import PdfReader
    out = []
    reader = PdfReader(io.BytesIO(data))
    for i, page in enumerate(reader.pages[:400]):
        try:
            out.append((i + 1, page.extract_text() or ""))
        except Exception:
            continue
        if sum(len(t) for _, t in out) > MAX_DOC_CHARS:
            break
    return out


def _docx_text(data):
    import docx
    d = docx.Document(io.BytesIO(data))
    text = "\n".join(p.text for p in d.paragraphs if p.text.strip())
    for tbl in d.tables[:50]:
        for row in tbl.rows:
            text += "\n" + " | ".join(c.text for c in row.cells)
    return [(None, text[:MAX_DOC_CHARS])]


def _pptx_text(data):
    from pptx import Presentation
    prs = Presentation(io.BytesIO(data))
    out = []
    for i, slide in enumerate(prs.slides):
        bits = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                bits.append(shape.text_frame.text)
        out.append((i + 1, "\n".join(bits)))
        if sum(len(t) for _, t in out) > MAX_DOC_CHARS:
            break
    return out


def _xlsx_text(data):
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    out, total = [], 0
    for ws in wb.worksheets[:15]:
        rows = []
        for row in ws.iter_rows(max_row=300, values_only=True):
            vals = [str(v) for v in row if v is not None]
            if vals:
                rows.append(" | ".join(vals))
            if sum(len(r) for r in rows) > 15_000:
                break
        text = "Sheet: %s\n%s" % (ws.title, "\n".join(rows))
        out.append((None, text))
        total += len(text)
        if total > 60_000:
            break
    wb.close()
    return out


def _msg_text(data):
    import extract_msg
    m = extract_msg.openMsg(io.BytesIO(data))
    text = "From: %s\nTo: %s\nDate: %s\nSubject: %s\n\n%s" % (
        m.sender or "", m.to or "", m.date or "", m.subject or "", (m.body or "")[:40_000])
    return [(None, text)]


def _plain_text(data):
    return [(None, data.decode("utf-8", errors="replace")[:MAX_DOC_CHARS])]


EXTRACTORS = {"pdf": _pdf_text, "docx": _docx_text, "pptx": _pptx_text,
              "xlsx": _xlsx_text, "xlsm": _xlsx_text, "msg": _msg_text,
              "csv": _plain_text, "txt": _plain_text, "md": _plain_text,
              "rtf": _plain_text}


def _chunks(pages):
    """pages: [(page_no or None, text)] -> [(page_no, chunk_text)]"""
    out = []
    for page_no, text in pages:
        text = " ".join(text.split())
        if not text:
            continue
        i = 0
        while i < len(text):
            out.append((page_no, text[i:i + CHUNK_CHARS]))
            i += CHUNK_CHARS - CHUNK_OVERLAP
    return out


# ---------------- embeddings ----------------

def _embed_call(batch):
    key = os.environ["OPENAI_API_KEY"]
    body = json.dumps({"model": EMBED_MODEL, "input": batch,
                       "dimensions": EMBED_DIMS}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/embeddings", data=body,
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode())
    return [d["embedding"] for d in data["data"]]


# OpenAI caps an embeddings request at 300k tokens across all inputs. Batching
# by COUNT alone sent ~400k tokens once the corpus reached dense audit PDFs, so
# every request returned 400 and the index stopped dead on 17 Jul 2026. Batch by
# estimated tokens instead, and keep well under the cap.
EMBED_TOKEN_BUDGET = 200_000
EMBED_CHARS_PER_TOKEN = 3.5


def _batches(texts):
    """Sub-batches that respect BOTH the item cap and the token cap."""
    batch, budget = [], 0.0
    for t in texts:
        t = (t or "")[:24_000]
        cost = len(t) / EMBED_CHARS_PER_TOKEN
        if batch and (len(batch) >= EMBED_BATCH
                      or budget + cost > EMBED_TOKEN_BUDGET):
            yield batch
            batch, budget = [], 0.0
        batch.append(t)
        budget += cost
    if batch:
        yield batch


def _embed(texts):
    """Embed texts. A failing batch falls back to per-item embedding with halving
    truncation; an item that still fails gets a zero vector. One poison chunk must
    NEVER kill the run - that stalled the whole index (no parts written) for a month."""
    vectors = []
    for batch in _batches(texts):
        done = False
        for attempt in range(3):
            try:
                vectors.extend(_embed_call(batch))
                done = True
                break
            except Exception as e:
                LOG.warning("embed batch failed (attempt %d): %s", attempt + 1, str(e)[:150])
                time.sleep(8 * (attempt + 1))
        if done:
            continue
        # per-item salvage: halve the text on each failure; zero-vector as last resort
        for t in batch:
            vec = None
            for cut in (24_000, 12_000, 6_000, 3_000):
                try:
                    vec = _embed_call([t[:cut]])[0]
                    break
                except Exception:
                    continue
            if vec is None:
                LOG.warning("chunk unembeddable even at 3k chars - zero vector used")
                vec = [0.0] * EMBED_DIMS
            vectors.append(vec)
    return vectors


# Indexing order. The corpus is ~93k documents and a night only buys ~100
# minutes, so what gets indexed FIRST decides what the assistant can answer for
# the next few months. These are the folders that answer questions: agreements,
# entities, statutory and financial records. Store designs (15k files) and team
# files (9k) are last on purpose - they are bulk, not knowledge.
PRIORITY_PREFIXES = [
    "sharepoint/ALL-SHARES/Finance/GROUP STRUCTURE & CORPORATE/",
    "sharepoint/ALL-SHARES/1.International Confidential/",
    "sharepoint/ALL-SHARES/Finance/",
    "sharepoint/ALL-SHARES/ACCOUNTS/",
    "sharepoint/ALL-SHARES/Payroll/",
    "canva/",
    "sharepoint/",                      # everything else, once the above are done
]
# Only skipped during the catch-all pass - name one of these as an explicit
# prefix and it is indexed normally.
DEPRIORITISED = ("/store designs/", "/team files/", "/marketing/", "/old/",
                 "/lightboxes/", "/ligthboxes/")


def _ordered_prefixes(prefix):
    if prefix:
        return [prefix]
    return list(PRIORITY_PREFIXES)


def _deprioritised(name, prefix_i):
    """True when this blob should wait for the catch-all pass."""
    if prefix_i != "sharepoint/":
        return False
    low = name.lower()
    return any(d in low for d in DEPRIORITISED)


# ---------------- main run ----------------

def run_extract(minutes=90, prefix="sharepoint/"):
    deadline = time.time() + minutes * 60
    svc = _svc()
    cc = svc.get_container_client(CONTAINER)
    state = _load_state(cc)
    done = state["done"]

    rows = []
    files_done = files_skipped = errors = 0
    t0 = time.time()

    def flush():
        nonlocal rows
        if not rows:
            return
        texts = [r["text"] for r in rows]
        vecs = _embed(texts)
        for r, v in zip(rows, vecs):
            r["embedding"] = v
        df = pd.DataFrame(rows)
        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf, compression="zstd")
        part = state.get("next_part", 0)
        cc.upload_blob("index/parts/%06d.parquet" % part, buf.getvalue(), overwrite=True)
        state["next_part"] = part + 1
        _save_state(cc, state)
        LOG.info("index part %06d: %d chunks", part, len(df))
        rows = []

    for prefix_i in _ordered_prefixes(prefix):
        if time.time() > deadline:
            break
        for b in cc.list_blobs(name_starts_with=prefix_i):
            if time.time() > deadline:
                break
            name = b.name
            if name.startswith("index/") or name.endswith("_state.json"):
                continue
            if _deprioritised(name, prefix_i):
                continue
            ext = name.rsplit(".", 1)[1].lower() if "." in name else ""
            if ext not in TEXT_EXTS or b.size > MAX_FILE_BYTES:
                continue
            etag = b.etag.replace('"', "")
            if done.get(name) == etag:
                files_skipped += 1
                continue
            try:
                data = cc.download_blob(name).readall()
                pages = EXTRACTORS[ext](data)
                chks = _chunks(pages)
                title = name.rsplit("/", 1)[-1]
                for ci, (page_no, text) in enumerate(chks[:80]):
                    rows.append({"path": name, "title": title, "ext": ext,
                                 "mtime": str(b.last_modified), "chunk_id": ci,
                                 "page": page_no, "text": text})
                done[name] = etag
                files_done += 1
            except Exception as e:
                errors += 1
                done[name] = etag  # don't retry poison files every run
                if errors <= 15:
                    LOG.warning("extract failed %s: %s", name, str(e)[:150])
            if len(rows) >= 4000:
                flush()
        if time.time() > deadline:
            break

    flush()
    _save_state(cc, state)
    return {"files_indexed": files_done, "skipped_unchanged": files_skipped,
            "errors": errors, "parts": state.get("next_part", 0),
            "secs": round(time.time() - t0)}
