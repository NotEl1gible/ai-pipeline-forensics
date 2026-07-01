"""Offline unit tests for the Failure Forensics Tool (no API calls, mock provider)."""
import json
from pathlib import Path

import forensics as F

HERE = Path(__file__).parent
DOCS = F.load_docs(str(HERE / "docs.jsonl"))
QUERY = "How do I get a refund for a damaged item?"


def _culprit(break_step):
    tr = F.Tracer()
    F.run_pipeline(tr, QUERY, DOCS, "mock", break_step)
    return F.localize(tr, QUERY)["culprit"]


# --- tracing model: contextvars build the parent tree ---
def test_span_tree_parenting():
    tr = F.Tracer()
    with tr.span("root"):
        with tr.span("child"):
            pass
    root = next(s for s in tr.spans if s.name == "root")
    child = next(s for s in tr.spans if s.name == "child")
    assert root.parent_span_id is None
    assert child.parent_span_id == root.span_id
    assert child.trace_id == root.trace_id == tr.trace_id


def test_span_records_error_status():
    tr = F.Tracer()
    try:
        with tr.span("boom"):
            raise ValueError("x")
    except ValueError:
        pass
    assert tr.spans[0].status == "error" and "ValueError" in tr.spans[0].status_message


# --- keyword overlap ---
def test_overlap():
    assert F.overlap("refund for a damaged item", "refund policy damaged items") >= 1
    assert F.overlap("refund damaged", "password reset login") == 0


# --- failure localization (each injected break -> correct root cause) ---
def test_localize_healthy_and_breaks():
    assert _culprit(None) is None
    assert _culprit("retrieve") == "retrieve"
    assert _culprit("llm") == "llm_answer"
    assert _culprit("parse") == "parse_json"


# --- error propagation: parse visibly errors, but llm is the ROOT cause ---
def test_propagation_root_is_first_failure():
    tr = F.Tracer()
    F.run_pipeline(tr, QUERY, DOCS, "mock", "llm")
    parse = next(s for s in tr.spans if s.name == "parse_json")
    assert parse.status == "error"                       # the visible error is in parse
    assert F.localize(tr, QUERY)["culprit"] == "llm_answer"   # but llm is blamed


# --- feedback loop: flag failures -> replay as regression ---
def test_flag_and_replay(tmp_path):
    ds = str(tmp_path / "eval.jsonl")
    F.flag_case(ds, QUERY, "retrieve", "retrieve")
    F.flag_case(ds, QUERY, "parse", "parse_json")
    passed, total, _ = F.replay(ds, DOCS, "mock")
    assert passed == total == 2


# --- storage: trace serializes to inspectable JSON ---
def test_save_trace(tmp_path):
    tr = F.Tracer()
    F.run_pipeline(tr, QUERY, DOCS, "mock", None)
    path = F.save_trace(tr, out_dir=str(tmp_path))
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    spans = data["spans"]
    assert len(spans) == 4                               # pipeline + 3 steps
    assert {d["name"] for d in spans} == {"pipeline", "retrieve", "llm_answer", "parse_json"}
    assert data["trace_id"] == tr.trace_id
