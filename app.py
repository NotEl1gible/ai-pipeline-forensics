"""
FastAPI layer for the Failure Forensics Tool.

- POST /flag        : feedback loop — flag a bad run; the case is appended to the
                      growing eval dataset (eval_dataset.jsonl).
- GET  /replay      : re-run the eval dataset as a regression check.
- GET  /traces      : list saved traces.
- GET  /trace/{id}  : HTML trace-tree viewer, root cause highlighted.

Run:  uvicorn app:app --reload
"""
import json
from pathlib import Path

from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse

import forensics as F

TRACES = Path("traces")
DATASET = "eval_dataset.jsonl"
DOCS = "docs.jsonl"

app = FastAPI(title="Failure Forensics")


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.post("/flag")
def flag(body: dict = Body(...)):
    F.flag_case(DATASET, body["query"], body.get("break"), body["expected_culprit"])
    return {"flagged": True, "dataset": DATASET}


@app.get("/replay")
def replay():
    passed, total, rows = F.replay(DATASET, F.load_docs(DOCS), "mock")
    return {"passed": passed, "total": total,
            "cases": [{"break": c.get("break"), "expected": c["expected_culprit"],
                       "got": got, "ok": ok} for c, got, ok in rows]}


@app.get("/traces")
def traces():
    return {"traces": sorted(p.stem for p in TRACES.glob("*.json"))} if TRACES.exists() else {"traces": []}


@app.get("/trace/{trace_id}", response_class=HTMLResponse)
def trace_view(trace_id: str):
    path = TRACES / f"{trace_id}.json"
    if not path.exists():
        return HTMLResponse("<h3>trace not found</h3>", status_code=404)
    return HTMLResponse(_render(json.loads(path.read_text(encoding="utf-8"))))


def _render(doc: dict) -> str:
    children: dict = {}
    for s in doc["spans"]:
        children.setdefault(s["parent_span_id"], []).append(s)
    rc = doc.get("root_cause_span_id")

    def li(s):
        is_rc = s["span_id"] == rc
        color = "#c0392b" if (s["status"] == "error" or is_rc) else "#2c3e50"
        badge = " <b style='color:#c0392b'>&lt;== ROOT CAUSE</b>" if is_rc else ""
        kids = "".join(li(c) for c in children.get(s["span_id"], []))
        return (f"<li style='color:{color}'>{s['name']} "
                f"<span style='color:#888'>{s['duration_ms']}ms [{s['status']}]</span>{badge}"
                f"<ul>{kids}</ul></li>")

    roots = "".join(li(s) for s in children.get(None, []))
    return (f"<html><body style='font-family:system-ui;max-width:640px;margin:2rem auto'>"
            f"<h2>trace {doc['trace_id']}</h2>"
            f"<p><b>query:</b> {doc.get('query')}</p>"
            f"<p><b>root cause:</b> {doc.get('root_cause') or 'none'}</p>"
            f"<ul>{roots}</ul></body></html>")
