"""Tests for scripts/latency_report.py — latency stats and report generation.

The script itself drives HTTP load and parses logs; we test the pure
stat-computation and report-formatting logic in isolation (no live proxy needed).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.latency_report import (
    RequestResult,
    compute_stats,
    format_report,
    parse_susfactor_log_line,
)

# ---------------------------------------------------------------------------
# RequestResult
# ---------------------------------------------------------------------------


def test_request_result_success() -> None:
    r = RequestResult(
        index=0,
        elapsed_ms=120.5,
        status_code=200,
        guardrail_latency_ms=42.0,
        decision="flag",
        score=0.87,
        error=None,
    )
    assert r.success is True
    assert r.elapsed_ms == pytest.approx(120.5)


def test_request_result_failure() -> None:
    r = RequestResult(
        index=1,
        elapsed_ms=5000.0,
        status_code=None,
        guardrail_latency_ms=None,
        decision=None,
        score=None,
        error="Connection refused",
    )
    assert r.success is False


def test_request_result_http_error() -> None:
    r = RequestResult(
        index=2,
        elapsed_ms=200.0,
        status_code=500,
        guardrail_latency_ms=None,
        decision=None,
        score=None,
        error="Internal Server Error",
    )
    assert r.success is False


# ---------------------------------------------------------------------------
# parse_susfactor_log_line
# ---------------------------------------------------------------------------


def test_parse_log_line_extracts_latency_and_score() -> None:
    line = (
        "INFO susfactor decision=flag score=0.9137 label=suspicious threshold=0.50 "
        "enforcement=flag input_type=request latency_ms=42.3 backend=sdk model=0dinai/susfactor-e5-large"
    )
    result = parse_susfactor_log_line(line)
    assert result is not None
    assert result["latency_ms"] == pytest.approx(42.3)
    assert result["score"] == pytest.approx(0.9137)
    assert result["decision"] == "flag"


def test_parse_log_line_safe_prompt() -> None:
    line = (
        "INFO susfactor decision=allow score=0.0199 label=safe threshold=0.50 "
        "enforcement=flag input_type=request latency_ms=38.1 backend=sdk model=0dinai/susfactor-e5-large"
    )
    result = parse_susfactor_log_line(line)
    assert result is not None
    assert result["decision"] == "allow"
    assert result["score"] == pytest.approx(0.0199)


def test_parse_log_line_returns_none_for_non_susfactor_line() -> None:
    line = "INFO litellm: Proxy initialized with Config"
    assert parse_susfactor_log_line(line) is None


def test_parse_log_line_returns_none_for_empty_line() -> None:
    assert parse_susfactor_log_line("") is None


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------


def _make_results(latencies: list[float]) -> list[RequestResult]:
    return [
        RequestResult(
            index=i,
            elapsed_ms=ms,
            status_code=200,
            guardrail_latency_ms=ms * 0.35,
            decision="flag",
            score=0.1,
            error=None,
        )
        for i, ms in enumerate(latencies)
    ]


def test_compute_stats_basic_percentiles() -> None:
    # 10 evenly spaced values: 10, 20, ..., 100
    results = _make_results([float(i * 10) for i in range(1, 11)])
    stats = compute_stats(results)

    assert stats.count == 10
    assert stats.success_count == 10
    assert stats.error_count == 0
    assert stats.p50_ms == pytest.approx(55.0)  # median of 10–100
    assert stats.p95_ms == pytest.approx(95.0, abs=2.0)
    assert stats.p99_ms == pytest.approx(99.0, abs=2.0)
    assert stats.min_ms == pytest.approx(10.0)
    assert stats.max_ms == pytest.approx(100.0)


def test_compute_stats_counts_errors() -> None:
    successes = _make_results([50.0, 60.0, 70.0])
    failure = RequestResult(
        index=3,
        elapsed_ms=5000.0,
        status_code=None,
        guardrail_latency_ms=None,
        decision=None,
        score=None,
        error="timeout",
    )
    stats = compute_stats(successes + [failure])
    assert stats.count == 4
    assert stats.success_count == 3
    assert stats.error_count == 1


def test_compute_stats_guardrail_inference_latency() -> None:
    results = _make_results([100.0, 200.0, 300.0])
    stats = compute_stats(results)
    # guardrail_latency_ms = elapsed * 0.35
    assert stats.guardrail_p50_ms is not None
    assert stats.guardrail_p50_ms == pytest.approx(70.0)  # median of 35, 70, 105


def test_compute_stats_single_result() -> None:
    results = _make_results([42.0])
    stats = compute_stats(results)
    assert stats.p50_ms == pytest.approx(42.0)
    assert stats.p95_ms == pytest.approx(42.0)
    assert stats.p99_ms == pytest.approx(42.0)


def test_compute_stats_empty_raises() -> None:
    with pytest.raises(ValueError, match="No results"):
        compute_stats([])


def test_compute_stats_all_failures_raises() -> None:
    """All-failure batch raises rather than returning misleading 0.0 percentiles."""
    failures = [
        RequestResult(
            index=i,
            elapsed_ms=5000.0,
            status_code=None,
            guardrail_latency_ms=None,
            decision=None,
            score=None,
            error="Connection refused",
        )
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="All.*failed"):
        compute_stats(failures)


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------


def test_format_report_contains_key_fields() -> None:
    results = _make_results([50.0, 100.0, 150.0, 200.0, 500.0])
    stats = compute_stats(results)
    report = format_report(stats, concurrency=5, total_requests=5)

    assert "P50" in report
    assert "P95" in report
    assert "P99" in report
    assert "5" in report  # concurrency or count
    assert "error" in report.lower() or "Error" in report


def test_format_report_omits_guardrail_row_when_no_guardrail_data() -> None:
    """Live-load results (no guardrail_latency_ms) omit the guardrail row and show a note."""
    results = [
        RequestResult(
            index=i,
            elapsed_ms=100.0 + i * 10,
            status_code=200,
            guardrail_latency_ms=None,  # live-load: not available in headers
            decision="flag",
            score=0.1,
            error=None,
        )
        for i in range(5)
    ]
    stats = compute_stats(results)
    report = format_report(stats, concurrency=5, total_requests=5)

    assert "Guardrail (ms)" not in report
    assert "from-logs" in report  # note directing user to log-parse mode


def test_format_report_json_output() -> None:
    results = _make_results([50.0, 100.0])
    stats = compute_stats(results)
    report_json = format_report(stats, concurrency=2, total_requests=2, output_format="json")

    parsed = json.loads(report_json)
    assert "p50_ms" in parsed
    assert "p95_ms" in parsed
    assert "p99_ms" in parsed
    assert "success_count" in parsed
    assert "error_count" in parsed


def test_format_report_writes_file(tmp_path: Path) -> None:
    results = _make_results([80.0, 90.0, 100.0])
    stats = compute_stats(results)
    out_file = tmp_path / "report.json"
    format_report(
        stats, concurrency=3, total_requests=3, output_format="json", output_file=out_file
    )

    assert out_file.exists()
    parsed = json.loads(out_file.read_text())
    assert parsed["p50_ms"] == pytest.approx(90.0)
