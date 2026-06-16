"""Phase 3 tests — observability: structured logging, guardrail_information payload,
response header injection.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from susfactor_guardrail.backend import SusFactorUnavailable, SusResult
from susfactor_guardrail.guardrail import SusFactorGuardrail

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SUSPICIOUS = SusResult(
    score=0.92, label="suspicious", model="0dinai/susfactor-e5-large",
    threshold=0.5, timing_ms=41.0, backend="sdk",
)
SAFE = SusResult(
    score=0.08, label="safe", model="0dinai/susfactor-e5-large",
    threshold=0.5, timing_ms=38.0, backend="sdk",
)


def _guardrail(action: str = "flag", fail_open: bool = True) -> SusFactorGuardrail:
    g = SusFactorGuardrail(
        guardrail_name="susfactor",
        enforcement=action,
        threshold=0.5,
        fail_open=fail_open,
        backend="sdk",
        model_cache_dir=None,
        device=None,
    )
    g._backend = MagicMock()
    return g


def _inputs(text: str) -> dict[str, Any]:
    return {"texts": [text], "images": None, "tool_calls": None}


# ---------------------------------------------------------------------------
# P3-1: susfactor metadata recorded in request_data for every action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_records_susfactor_metadata_on_suspicious() -> None:
    g = _guardrail(action="flag")
    g._backend.score = AsyncMock(return_value=SUSPICIOUS)
    request_data: dict[str, Any] = {}

    await g.apply_guardrail(inputs=_inputs("jailbreak"), request_data=request_data, input_type="request")

    records = request_data.get("metadata", {}).get("susfactor", [])
    assert len(records) == 1
    rec = records[0]
    assert rec["score"] == pytest.approx(0.92)
    assert rec["label"] == "suspicious"
    assert rec["decision"] == "flag"
    assert rec["input_type"] == "request"
    assert rec["enforcement"] == "flag"
    assert rec["timing_ms"] == pytest.approx(41.0)
    assert rec["backend"] == "sdk"


@pytest.mark.asyncio
async def test_shadow_records_susfactor_metadata() -> None:
    g = _guardrail(action="shadow")
    g._backend.score = AsyncMock(return_value=SUSPICIOUS)
    request_data: dict[str, Any] = {}

    await g.apply_guardrail(inputs=_inputs("jailbreak"), request_data=request_data, input_type="request")

    records = request_data.get("metadata", {}).get("susfactor", [])
    assert len(records) == 1
    assert records[0]["decision"] == "shadow"


@pytest.mark.asyncio
async def test_block_records_susfactor_metadata_before_raising() -> None:
    from litellm.exceptions import GuardrailRaisedException

    g = _guardrail(action="block")
    g._backend.score = AsyncMock(return_value=SUSPICIOUS)
    request_data: dict[str, Any] = {}

    with pytest.raises(GuardrailRaisedException):
        await g.apply_guardrail(inputs=_inputs("jailbreak"), request_data=request_data, input_type="request")

    records = request_data.get("metadata", {}).get("susfactor", [])
    assert len(records) == 1
    assert records[0]["decision"] == "block"


@pytest.mark.asyncio
async def test_safe_result_records_allow_decision() -> None:
    g = _guardrail(action="block")
    g._backend.score = AsyncMock(return_value=SAFE)
    request_data: dict[str, Any] = {}

    await g.apply_guardrail(inputs=_inputs("safe text"), request_data=request_data, input_type="request")

    records = request_data.get("metadata", {}).get("susfactor", [])
    assert len(records) == 1
    assert records[0]["decision"] == "allow"  # block action but safe → allow


@pytest.mark.asyncio
async def test_multiple_texts_records_one_entry_per_text() -> None:
    g = _guardrail(action="flag")
    g._backend.score = AsyncMock(side_effect=[SAFE, SUSPICIOUS])
    request_data: dict[str, Any] = {}

    await g.apply_guardrail(
        inputs={"texts": ["hello", "jailbreak"], "images": None, "tool_calls": None},
        request_data=request_data,
        input_type="request",
    )

    records = request_data.get("metadata", {}).get("susfactor", [])
    assert len(records) == 2


# ---------------------------------------------------------------------------
# P3-2: standard_logging_guardrail_information added to request_data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_standard_logging_guardrail_info_added_on_flag() -> None:
    g = _guardrail(action="flag")
    g._backend.score = AsyncMock(return_value=SUSPICIOUS)
    request_data: dict[str, Any] = {"metadata": {}}

    await g.apply_guardrail(inputs=_inputs("jailbreak"), request_data=request_data, input_type="request")

    slg_list = request_data["metadata"].get("standard_logging_guardrail_information", [])
    assert len(slg_list) >= 1
    slg = slg_list[-1]
    assert slg.get("guardrail_name") == "susfactor"
    # status is "guardrail_intervened" for suspicious, "success" for safe
    assert slg.get("guardrail_status") in ("guardrail_intervened", "success")


@pytest.mark.asyncio
async def test_standard_logging_guardrail_info_includes_timing() -> None:
    g = _guardrail(action="flag")
    g._backend.score = AsyncMock(return_value=SUSPICIOUS)
    request_data: dict[str, Any] = {"metadata": {}}

    await g.apply_guardrail(inputs=_inputs("jailbreak"), request_data=request_data, input_type="request")

    slg_list = request_data["metadata"].get("standard_logging_guardrail_information", [])
    slg = slg_list[-1]
    # duration should be recorded (non-negative float)
    assert slg.get("duration") is not None
    assert slg["duration"] >= 0


@pytest.mark.asyncio
async def test_standard_logging_guardrail_info_on_unavailable_fail_open() -> None:
    """Backend unavailable + fail_open → still records guardrail_failed_to_respond."""
    g = _guardrail(action="block", fail_open=True)
    g._backend.score = AsyncMock(side_effect=SusFactorUnavailable("Model not found"))
    request_data: dict[str, Any] = {"metadata": {}}

    await g.apply_guardrail(inputs=_inputs("test"), request_data=request_data, input_type="request")

    slg_list = request_data["metadata"].get("standard_logging_guardrail_information", [])
    assert len(slg_list) >= 1
    assert slg_list[-1].get("guardrail_status") == "guardrail_failed_to_respond"


# ---------------------------------------------------------------------------
# P3-3: structured log emission (latency captured)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_structured_log_emitted_with_score_and_latency(caplog: pytest.LogCaptureFixture) -> None:
    g = _guardrail(action="flag")
    g._backend.score = AsyncMock(return_value=SUSPICIOUS)

    with caplog.at_level(logging.INFO, logger="susfactor_guardrail.guardrail"):
        await g.apply_guardrail(inputs=_inputs("jailbreak"), request_data={}, input_type="request")

    # At least one log record should mention score or decision
    log_text = " ".join(caplog.messages)
    assert "0.92" in log_text or "suspicious" in log_text or "flag" in log_text


# ---------------------------------------------------------------------------
# P3-4: response header hook returns X-SusFactor-Decision
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_header_set_after_flag_decision() -> None:
    from litellm.proxy._types import UserAPIKeyAuth

    g = _guardrail(action="flag")
    g._backend.score = AsyncMock(return_value=SUSPICIOUS)
    request_data: dict[str, Any] = {}

    # Simulate the guardrail running first (populates metadata)
    await g.apply_guardrail(inputs=_inputs("jailbreak"), request_data=request_data, input_type="request")

    headers = await g.async_post_call_response_headers_hook(
        data=request_data,
        user_api_key_dict=UserAPIKeyAuth(),
        response=MagicMock(),
    )

    assert headers is not None
    assert "x-susfactor-decision" in {k.lower() for k in headers}


@pytest.mark.asyncio
async def test_response_header_absent_when_no_decision_recorded() -> None:
    from litellm.proxy._types import UserAPIKeyAuth

    g = _guardrail(action="flag")
    # No apply_guardrail call — no decision recorded in request_data
    request_data: dict[str, Any] = {}

    headers = await g.async_post_call_response_headers_hook(
        data=request_data,
        user_api_key_dict=UserAPIKeyAuth(),
        response=MagicMock(),
    )

    # Either None or an empty dict — no susfactor header
    if headers:
        assert "x-susfactor-decision" not in {k.lower() for k in headers}
