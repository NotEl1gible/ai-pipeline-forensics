#!/usr/bin/env python3
"""
Failure Forensics Tool for AI pipelines.

Traces every step of a multi-step AI pipeline as OpenTelemetry-shaped spans,
then — when the final output is bad — localizes the FIRST step that broke
(the root cause), accounting for error propagation (a later step fails because
an earlier one fed it garbage).

Single file on purpose (Karpathy-style): working end-to-end first, grow later.

Usage:
    python forensics.py demo --provider mock
    python forensics.py demo --provider mock --break retrieve   # inject a failure
    ANTHROPIC_API_KEY=... python forensics.py demo --provider anthropic
"""
from __future__ import annotations

import argparse
import contextlib
import contextvars
import json
import os
import re
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ----------------------------------------------------------------------------
# Span / Trace model (OpenTelemetry-shaped + explicit input/output)
# ----------------------------------------------------------------------------
@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start: float
    end: float | None = None
    status: str = "ok"                 # "ok" | "error"
    status_message: str = ""
    attributes: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    input: dict = field(default_factory=dict)
    output: object = None

    @property
    def duration_ms(self) -> float:
        return 0.0 if self.end is None else round((self.end - self.start) * 1000, 1)


class Tracer:
    """Collects spans for one trace; contextvars carry the current parent."""

    def __init__(self):
        self.trace_id = uuid.uuid4().hex[:8]
        self.spans: list[Span] = []
        self._current: contextvars.ContextVar = contextvars.ContextVar("current_span", default=None)

    @contextlib.contextmanager
    def span(self, name: str, **inputs):
        sp = Span(trace_id=self.trace_id, span_id=uuid.uuid4().hex[:8],
                  parent_span_id=self._current.get(), name=name,
                  start=time.perf_counter(), input=inputs)
        self.spans.append(sp)
        token = self._current.set(sp.span_id)
        try:
            yield sp
        except Exception as e:
            sp.status, sp.status_message = "error", f"{type(e).__name__}: {e}"
            raise
        finally:
            sp.end = time.perf_counter()
            self._current.reset(token)


# ----------------------------------------------------------------------------
# Keyword helpers (used by retrieval + checks)
# ----------------------------------------------------------------------------
_STOP = {"how", "do", "i", "get", "a", "for", "the", "to", "my", "is", "of", "and",
         "what", "in", "on", "an", "me", "you", "with", "can", "please"}


def keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 2}


def overlap(a: str, b: str) -> int:
    return len(keywords(a) & keywords(b))


def load_docs(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


# ----------------------------------------------------------------------------
# Instrumented pipeline steps  (retrieve -> llm_answer -> parse_json)
# ----------------------------------------------------------------------------
def retrieve(tracer, query, docs, broken=False):
    with tracer.span("retrieve", query=query) as sp:
        if broken:
            hits = [d for d in docs if d["id"] in ("d3", "d4")]        # deliberately off-topic
        else:
            hits = sorted(docs, key=lambda d: overlap(query, d["text"]), reverse=True)[:3]
        sp.output = hits
        sp.attributes["doc_ids"] = [d["id"] for d in hits]
        return hits


def llm_answer(tracer, query, docs, provider, broken=False):
    with tracer.span("llm_answer", query=query, doc_ids=[d["id"] for d in docs]) as sp:
        if broken:
            draft = "Sure! Here is a friendly little summary of what I found for you."  # prose, not JSON
        elif provider == "mock":
            top = docs[0]["text"] if docs else ""
            draft = json.dumps({"answer": top, "sources": [d["id"] for d in docs]})
        else:
            draft = _call_claude(query, docs)
        sp.output = draft
        sp.attributes["chars"] = len(draft)
        return draft


def parse_json(tracer, draft, broken=False):
    with tracer.span("parse_json", draft=draft[:80]) as sp:
        if broken:
            raise ValueError("forced parse failure")
        text = draft.strip()
        if text.startswith("```"):        # strip markdown code fences LLMs often add
            text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
        obj = json.loads(text)            # raises if it's prose -> span recorded as error
        sp.output = obj
        return obj


def _call_claude(query, docs):
    import anthropic
    ctx = "\n".join(f"[{d['id']}] {d['text']}" for d in docs)
    prompt = ('Answer the question using ONLY the context. Return ONLY JSON of the form '
              '{"answer": string, "sources": [doc ids]}.\n\n'
              f"Context:\n{ctx}\n\nQuestion: {query}")
    resp = anthropic.Anthropic().messages.create(
        model="claude-haiku-4-5", max_tokens=300,
        messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in resp.content if b.type == "text")


def run_pipeline(tracer, query, docs, provider="mock", break_step=None):
    """Run the pipeline. Any step may fail (incl. real provider/infra errors); each
    failure is captured on its span, and the run continues so the localizer can
    point at it — a forensics tool must never crash on the failure it is tracing."""
    with tracer.span("pipeline", query=query) as root:
        hits, draft, result = [], "", None
        try:
            hits = retrieve(tracer, query, docs, broken=(break_step == "retrieve"))
        except Exception:
            pass
        try:
            draft = llm_answer(tracer, query, hits, provider, broken=(break_step == "llm"))
        except Exception:
            pass
        try:
            result = parse_json(tracer, draft, broken=(break_step == "parse"))
        except Exception:
            pass
        root.output = result
        return result


# ----------------------------------------------------------------------------
# Per-step checks + failure localizer
# ----------------------------------------------------------------------------
def _relevance(sp, query):
    best = max((overlap(query, d["text"]) for d in (sp.output or [])), default=0)
    return best >= 1, f"best retrieved-doc keyword overlap with query = {best}"


def _on_topic(sp, query):
    text = sp.output or ""
    ov = overlap(query, text)
    return (len(text) > 0 and ov >= 1), f"answer/query keyword overlap = {ov}"


def _valid_json(sp, query):
    ok = sp.status != "error"
    return ok, "output is valid JSON" if ok else f"parse raised: {sp.status_message}"


CHECKS = {"retrieve": _relevance, "llm_answer": _on_topic, "parse_json": _valid_json}


def evaluate_checks(tracer, query) -> dict:
    """Run every instrumented span's check -> {span_id: (passed, reason)}."""
    out = {}
    for sp in tracer.spans:
        if sp.name not in CHECKS:
            continue
        out[sp.span_id] = (False, f"raised {sp.status_message}") if sp.status == "error" \
            else CHECKS[sp.name](sp, query)
    return out


def localize(tracer, query) -> dict:
    """First instrumented span that errored or failed its check = the root cause."""
    for sp in tracer.spans:                       # spans are in execution order
        if sp.name not in CHECKS:                 # skip the root "pipeline" span
            continue
        if sp.status == "error":
            return {"culprit": sp.name, "span_id": sp.span_id, "reason": f"raised {sp.status_message}"}
        ok, reason = CHECKS[sp.name](sp, query)
        if not ok:
            return {"culprit": sp.name, "span_id": sp.span_id, "reason": reason}
    return {"culprit": None, "span_id": None, "reason": "all steps passed their checks"}


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def print_tree(tracer, culprit_span_id=None, checks=None):
    checks = checks or {}
    children: dict = {}
    for sp in tracer.spans:
        children.setdefault(sp.parent_span_id, []).append(sp)

    def walk(sp, depth):
        mark = "   <== ROOT CAUSE" if sp.span_id == culprit_span_id else ""
        status = "OK   " if sp.status == "ok" else "ERROR"
        chk = ""
        if sp.span_id in checks:
            chk = "  check:PASS" if checks[sp.span_id][0] else "  check:FAIL"
        print(f"{'  ' * depth}- {sp.name:<12} {sp.duration_ms:>6.1f}ms  [{status}]{chk}{mark}")
        for c in children.get(sp.span_id, []):
            walk(c, depth + 1)

    for root in children.get(None, []):
        walk(root, 0)


# ----------------------------------------------------------------------------
# Storage: JSON trace files (inspectable, git-friendly) + SQLite audit trail
# ----------------------------------------------------------------------------
def span_to_dict(sp) -> dict:
    return {"trace_id": sp.trace_id, "span_id": sp.span_id, "parent_span_id": sp.parent_span_id,
            "name": sp.name, "duration_ms": sp.duration_ms, "status": sp.status,
            "status_message": sp.status_message, "attributes": sp.attributes,
            "input": sp.input, "output": sp.output}


def save_trace(tracer, verdict=None, query=None, out_dir="traces") -> str:
    Path(out_dir).mkdir(exist_ok=True)
    path = Path(out_dir) / f"{tracer.trace_id}.json"
    doc = {"trace_id": tracer.trace_id, "query": query,
           "root_cause": (verdict or {}).get("culprit"),
           "root_cause_span_id": (verdict or {}).get("span_id"),
           "spans": [span_to_dict(s) for s in tracer.spans]}
    path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    return str(path)


def log_sqlite(db, tracer, verdict) -> None:
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE IF NOT EXISTS spans (
        trace_id TEXT, span_id TEXT, parent TEXT, name TEXT,
        duration_ms REAL, status TEXT, is_root_cause INT)""")
    for s in tracer.spans:
        con.execute("INSERT INTO spans VALUES (?,?,?,?,?,?,?)",
                    (s.trace_id, s.span_id, s.parent_span_id, s.name, s.duration_ms, s.status,
                     1 if s.span_id == verdict.get("span_id") else 0))
    con.commit()
    con.close()


# ----------------------------------------------------------------------------
# Feedback loop: flag a failure -> growing eval dataset -> replay as regression
# ----------------------------------------------------------------------------
def flag_case(dataset, query, break_step, expected_culprit) -> None:
    with open(dataset, "a", encoding="utf-8") as f:
        f.write(json.dumps({"query": query, "break": break_step,
                            "expected_culprit": expected_culprit}) + "\n")


def replay(dataset, docs, provider="mock"):
    cases = [json.loads(l) for l in Path(dataset).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows, passed = [], 0
    for c in cases:
        tr = Tracer()
        run_pipeline(tr, c["query"], docs, provider, c.get("break"))
        got = localize(tr, c["query"])["culprit"]
        ok = got == c["expected_culprit"]
        passed += ok
        rows.append((c, got, ok))
    return passed, len(cases), rows


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def cmd_demo(args):
    docs = load_docs(args.docs)
    tracer = Tracer()
    run_pipeline(tracer, args.query, docs, args.provider, args.break_step)
    v = localize(tracer, args.query)
    checks = evaluate_checks(tracer, args.query)

    print(f"\n=== trace {tracer.trace_id}   query: {args.query!r} ===")
    print_tree(tracer, v["span_id"], checks)
    print(f"\nVERDICT: root cause = {v['culprit'] or 'none'}   ({v['reason']})")
    if v["culprit"] and v["culprit"] != tracer.spans[-1].name:
        print("note: later steps failed downstream of this one (error propagation).")

    path = save_trace(tracer, v, args.query)
    log_sqlite(args.db, tracer, v)
    print(f"(trace saved: {path})")


def cmd_flag(args):
    if not args.expect:
        raise SystemExit("flag needs --expect <culprit step>")
    flag_case(args.dataset, args.query, args.break_step, args.expect)
    print(f"flagged -> {args.dataset}: query={args.query!r} break={args.break_step} "
          f"expected_culprit={args.expect}")


def cmd_replay(args):
    docs = load_docs(args.docs)
    passed, total, rows = replay(args.dataset, docs, args.provider)
    print(f"=== replay {total} eval cases (harvested from flagged failures) ===")
    for c, got, ok in rows:
        print(f"[{'PASS' if ok else 'FAIL'}] break={str(c.get('break')):<9} "
              f"expected={c['expected_culprit']:<11} got={got}")
    print(f"\nreplay: {passed}/{total} localized correctly")
    return 0 if passed == total else 1


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Failure Forensics Tool for AI pipelines")
    ap.add_argument("command", choices=["demo", "flag", "replay"])
    ap.add_argument("--provider", choices=["mock", "anthropic"], default="mock")
    ap.add_argument("--break", dest="break_step", choices=["retrieve", "llm", "parse"], default=None)
    ap.add_argument("--query", default="How do I get a refund for a damaged item?")
    ap.add_argument("--expect", help="expected root-cause step (for `flag`)")
    ap.add_argument("--docs", default="docs.jsonl")
    ap.add_argument("--dataset", default="eval_dataset.jsonl")
    ap.add_argument("--db", default="forensics.db")
    args = ap.parse_args()

    load_dotenv()
    return {"demo": cmd_demo, "flag": cmd_flag, "replay": cmd_replay}[args.command](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
