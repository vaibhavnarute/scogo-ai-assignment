from __future__ import annotations

import json

from harness.trace import EventTracer, sanitize_for_log


def test_sanitizer_redacts_keys_and_values():
    value = sanitize_for_log({"api_key": "abc", "message": "Authorization: Bearer topsecret", "nested": {"password": "hello"}})
    assert value["api_key"] == "[REDACTED]"
    assert "topsecret" not in value["message"]
    assert value["nested"]["password"] == "[REDACTED]"


def test_trace_is_append_only_jsonl_and_flushed(tmp_path):
    path = tmp_path / "trace.jsonl"
    with EventTracer(path, "run-1") as tracer:
        tracer.record("run.started", task="repair")
        assert path.read_text(encoding="utf-8").count("\n") == 1
        tracer.record("tool.completed", api_key="never-log-this")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["run.started", "tool.completed"]
    assert records[1]["api_key"] == "[REDACTED]"

