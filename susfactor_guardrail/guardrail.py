"""SusFactorGuardrail — LiteLLM CustomGuardrail implementation.

Scores each text through SusFactor and applies one of three actions:

- ``block``  — raise to reject the request (returns an error to the client).
- ``flag``   — allow through, annotate decision in logs. Default for POCs.
- ``shadow`` — allow through, record only (no annotation to client).

On ``SusFactorUnavailable`` the ``fail_open`` setting decides whether to allow
(``True``, default) or block (``False``).

Configuration via ``config.yaml`` ``litellm_params``::

    guardrails:
      - guardrail_name: "susfactor"
        litellm_params:
          guardrail: susfactor_guardrail.SusFactorGuardrail
          mode: "during_call"
          enforcement: "flag"
          threshold: 0.5
          fail_open: true
          backend: "sdk"
          model_cache_dir: os.environ/SIGNATURE_SDK_MODEL_CACHE
          device: null
          scan_output: false
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Literal, Optional

from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth

from susfactor_guardrail.backend import SdkBackend, SusFactorUnavailable

logger = logging.getLogger(__name__)

# Valid action values
ACTIONS = frozenset({"block", "flag", "shadow"})


class SusFactorGuardrail(CustomGuardrail):
    """LiteLLM guardrail that runs the SusFactor jailbreak/prompt-injection classifier.

    Args:
        enforcement: What to do when a suspicious text is detected.
            ``"block"`` raises an exception; ``"flag"`` annotates and allows;
            ``"shadow"`` records only (never blocks).
        threshold: Decision threshold. Score >= threshold ⇒ suspicious.
        fail_open: If ``True`` (default), allow the request when the backend is
            unavailable. If ``False``, block on unavailability.
        backend: Backend implementation to use. Currently only ``"sdk"`` is
            supported (in-process ``odin-prompt-toolkit``).
        model_cache_dir: Override for the model cache directory. ``None`` uses
            the SDK default or the ``SIGNATURE_SDK_MODEL_CACHE`` env var.
        device: Torch device (``"cuda"``, ``"mps"``, ``"cpu"``). ``None`` auto-detects.
        scan_output: If ``True``, also score model outputs via ``post_call``.
        **kwargs: Forwarded to :class:`~litellm.integrations.custom_guardrail.CustomGuardrail`.
    """

    def __init__(
        self,
        enforcement: str = "flag",
        threshold: float = 0.5,
        fail_open: bool = True,
        backend: str = "sdk",
        model_cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        scan_output: bool = False,
        **kwargs: Any,
    ) -> None:
        if enforcement not in ACTIONS:
            raise ValueError(f"enforcement must be one of {sorted(ACTIONS)}, got {enforcement!r}")

        self._enforcement = enforcement
        self._threshold = threshold
        self._fail_open = fail_open
        self._scan_output = scan_output

        # Resolve model_cache_dir from env if not explicitly passed
        resolved_cache_dir = model_cache_dir or os.environ.get("SIGNATURE_SDK_MODEL_CACHE")

        # Build backend — only "sdk" supported in v1; seam for "heimdall" later
        if backend == "sdk":
            self._backend: SdkBackend = SdkBackend(
                model_cache_dir=resolved_cache_dir,
                threshold=threshold,
                device=device,
            )
        else:
            raise ValueError(f"Unknown backend {backend!r}. Supported: 'sdk'")

        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Core hook — called by LiteLLM for pre_call / during_call / post_call
    # ------------------------------------------------------------------

    async def apply_guardrail(  # type: ignore[override]
        self,
        inputs: dict[str, Any],
        request_data: dict[str, Any],
        input_type: Literal["request", "response"] = "request",
        logging_obj: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Score each text in ``inputs["texts"]`` and apply the configured action.

        Empty or whitespace-only texts are skipped (no backend call).
        If any text is suspicious and enforcement is ``"block"``, raises an exception.
        For ``"flag"`` and ``"shadow"``, always returns ``inputs`` unchanged.

        Args:
            inputs: Dict with ``texts`` (list[str]), optional ``images``, optional ``tool_calls``.
            request_data: LiteLLM request data dict (passed through unchanged).
            input_type: ``"request"`` or ``"response"``.
            logging_obj: Optional LiteLLM logging object.

        Returns:
            ``inputs`` unchanged (possibly annotated in ``request_data["metadata"]``).

        Raises:
            Exception: When enforcement is ``"block"`` and a suspicious text is detected
                (or when ``fail_open=False`` and the backend is unavailable).
        """
        texts: list[str] = inputs.get("texts") or []

        for text in texts:
            if not text or not text.strip():
                continue
            await self._score_and_act(text=text, request_data=request_data, input_type=input_type)

        # Score tool-call argument strings — LiteLLM does not flatten these into texts.
        # Each tool call's function.arguments is a JSON string that may contain
        # injected instructions (e.g. prompt injection via a search result returned
        # as a tool argument).
        tool_calls = inputs.get("tool_calls") or []
        for tool_call in tool_calls:
            args = self._extract_tool_call_args(tool_call)
            if not args or not args.strip():
                continue
            await self._score_and_act(text=args, request_data=request_data, input_type=input_type)

        return inputs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _score_and_act(
        self,
        text: str,
        request_data: dict[str, Any],
        input_type: Literal["request", "response"],
    ) -> None:
        """Score one text, record observability data, and apply the action."""
        start = time.monotonic()

        # --- Backend call ---
        try:
            result = await self._backend.score(text)
        except SusFactorUnavailable as exc:
            elapsed_s = time.monotonic() - start
            logger.warning(
                "susfactor decision=error enforcement=%s fail_open=%s latency_ms=%.1f error=%s",
                self._enforcement,
                self._fail_open,
                elapsed_s * 1000,
                exc,
            )
            self._record_slg(
                request_data=request_data,
                status="guardrail_failed_to_respond",
                duration=elapsed_s,
                response={"error": str(exc)},
            )
            if self._fail_open:
                return
            raise GuardrailRaisedException(
                message=f"SusFactor guardrail unavailable and fail_open=False: {exc}",
                guardrail_name=self.guardrail_name,
            ) from exc

        elapsed_s = time.monotonic() - start
        is_suspicious = result.score >= self._threshold

        # Determine decision label for logging/metadata
        if self._enforcement == "block" and is_suspicious:
            decision = "block"
        elif self._enforcement == "shadow":
            decision = "shadow"
        elif self._enforcement == "flag":
            decision = "flag"
        else:
            decision = "allow"

        # --- Structured INFO log ---
        logger.info(
            "susfactor decision=%s score=%.4f label=%s threshold=%.2f "
            "enforcement=%s input_type=%s latency_ms=%.1f backend=%s model=%s",
            decision,
            result.score,
            result.label,
            self._threshold,
            self._enforcement,
            input_type,
            elapsed_s * 1000,
            result.backend,
            result.model,
        )

        # --- Always record metadata (safe + suspicious) ---
        self._record(
            result=result,
            request_data=request_data,
            input_type=input_type,
            decision=decision,
        )

        # --- StandardLoggingGuardrailInformation ---
        slg_status = "guardrail_intervened" if is_suspicious else "success"
        self._record_slg(
            request_data=request_data,
            status=slg_status,
            duration=elapsed_s,
            response={
                "decision": decision,
                "score": result.score,
                "label": result.label,
                "model": result.model,
                "threshold": self._threshold,
                "timing_ms": result.timing_ms,
            },
        )

        # --- Apply action ---
        if self._enforcement in ("shadow", "flag"):
            return  # never block

        # enforcement == "block"
        if is_suspicious:
            raise GuardrailRaisedException(
                message=(
                    f"SusFactor: {input_type} blocked — suspicious content detected "
                    f"(score={result.score:.4f} >= threshold={self._threshold})"
                ),
                guardrail_name=self.guardrail_name,
            )

    @staticmethod
    def _extract_tool_call_args(tool_call: Any) -> str:
        """Extract the function.arguments string from a tool call object or dict.

        Handles both ``ChatCompletionMessageToolCall`` (attribute access) and plain
        ``dict`` representations. Returns an empty string if arguments cannot be found.
        """
        try:
            # Attribute access path (ChatCompletionMessageToolCall / OpenAIObject)
            function = getattr(tool_call, "function", None)
            if function is not None:
                args = getattr(function, "arguments", None)
                if args is not None:
                    return str(args)

            # Dict path
            if isinstance(tool_call, dict):
                func = tool_call.get("function", {})
                if isinstance(func, dict):
                    return str(func.get("arguments", ""))
        except Exception:  # noqa: BLE001 — never crash the guardrail on extraction
            pass
        return ""

    def _record(
        self,
        result: Any,
        request_data: dict[str, Any],
        input_type: Literal["request", "response"],
        decision: str = "allow",
    ) -> None:
        """Write per-text decision metadata into request_data["metadata"]["susfactor"]."""
        meta = request_data.setdefault("metadata", {})
        susfactor_records: list[dict[str, Any]] = meta.setdefault("susfactor", [])
        susfactor_records.append(
            {
                "score": result.score,
                "label": result.label,
                "model": result.model,
                "threshold": self._threshold,
                "enforcement": self._enforcement,
                "input_type": input_type,
                "timing_ms": result.timing_ms,
                "backend": result.backend,
                "decision": decision,
            }
        )

    def _record_slg(
        self,
        request_data: dict[str, Any],
        status: str,
        duration: float,
        response: dict[str, Any],
    ) -> None:
        """Write a StandardLoggingGuardrailInformation entry via the base-class helper."""
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response=response,
            request_data=request_data,
            guardrail_status=status,  # type: ignore[arg-type]
            duration=duration,
            guardrail_provider="susfactor",
        )

    async def async_post_call_response_headers_hook(
        self,
        data: dict[str, Any],
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_headers: Optional[Dict[str, str]] = None,
        litellm_call_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, str]]:
        """Inject ``X-SusFactor-Decision`` header summarising the last decision.

        Reads the ``susfactor`` list from ``data["metadata"]`` (written by
        ``apply_guardrail``) and sets the header to the most severe decision
        across all scored texts (block > flag > shadow > allow).
        Returns ``None`` when no susfactor records exist.
        """
        records: list[dict[str, Any]] = data.get("metadata", {}).get("susfactor", [])
        if not records:
            return None

        # Severity order for summary header
        _severity: dict[str, int] = {"block": 3, "flag": 2, "shadow": 1, "allow": 0, "error": -1}
        worst = max(records, key=lambda r: _severity.get(r.get("decision", "allow"), 0))
        decision = worst.get("decision", "allow")
        score = worst.get("score")

        header_value = decision
        if score is not None:
            header_value = f"{decision};score={score:.4f}"

        return {"X-SusFactor-Decision": header_value}
