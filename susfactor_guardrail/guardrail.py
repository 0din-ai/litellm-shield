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
          action: "flag"
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
from typing import Any, Literal, Optional

from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import CustomGuardrail

from susfactor_guardrail.backend import SdkBackend, SusFactorUnavailable

logger = logging.getLogger(__name__)

# Valid action values
ACTIONS = frozenset({"block", "flag", "shadow"})


class SusFactorGuardrail(CustomGuardrail):
    """LiteLLM guardrail that runs the SusFactor jailbreak/prompt-injection classifier.

    Args:
        action: What to do when a suspicious text is detected.
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
        action: str = "flag",
        threshold: float = 0.5,
        fail_open: bool = True,
        backend: str = "sdk",
        model_cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        scan_output: bool = False,
        **kwargs: Any,
    ) -> None:
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {sorted(ACTIONS)}, got {action!r}")

        self._action = action
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
        If any text is suspicious and action is ``"block"``, raises an exception.
        For ``"flag"`` and ``"shadow"``, always returns ``inputs`` unchanged.

        Args:
            inputs: Dict with ``texts`` (list[str]), optional ``images``, optional ``tool_calls``.
            request_data: LiteLLM request data dict (passed through unchanged).
            input_type: ``"request"`` or ``"response"``.
            logging_obj: Optional LiteLLM logging object.

        Returns:
            ``inputs`` unchanged (possibly annotated in ``request_data["metadata"]``).

        Raises:
            Exception: When action is ``"block"`` and a suspicious text is detected
                (or when ``fail_open=False`` and the backend is unavailable).
        """
        texts: list[str] = inputs.get("texts") or []

        for text in texts:
            if not text or not text.strip():
                continue

            await self._score_and_act(text=text, request_data=request_data, input_type=input_type)

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
        """Score one text and apply the action. Raises on block."""
        try:
            result = await self._backend.score(text)
        except SusFactorUnavailable as exc:
            logger.warning(
                "susfactor: backend unavailable (fail_open=%s): %s",
                self._fail_open,
                exc,
            )
            if self._fail_open:
                return
            raise GuardrailRaisedException(
                message=f"SusFactor guardrail unavailable and fail_open=False: {exc}",
                guardrail_name=self.guardrail_name,
            ) from exc

        is_suspicious = result.score >= self._threshold

        logger.debug(
            "susfactor: score=%.4f label=%s threshold=%.2f action=%s input_type=%s",
            result.score,
            result.label,
            self._threshold,
            self._action,
            input_type,
        )

        if self._action == "shadow":
            # Record only — never block
            self._record(result=result, request_data=request_data, input_type=input_type)
            return

        if self._action == "flag":
            self._record(result=result, request_data=request_data, input_type=input_type)
            return

        # action == "block"
        if is_suspicious:
            self._record(result=result, request_data=request_data, input_type=input_type)
            raise GuardrailRaisedException(
                message=(
                    f"SusFactor: {input_type} blocked — suspicious content detected "
                    f"(score={result.score:.4f} >= threshold={self._threshold})"
                ),
                guardrail_name=self.guardrail_name,
            )

    def _record(
        self,
        result: Any,
        request_data: dict[str, Any],
        input_type: Literal["request", "response"],
    ) -> None:
        """Write decision metadata into request_data for downstream logging."""
        meta = request_data.setdefault("metadata", {})
        susfactor_records: list[dict[str, Any]] = meta.setdefault("susfactor", [])
        susfactor_records.append(
            {
                "score": result.score,
                "label": result.label,
                "model": result.model,
                "threshold": self._threshold,
                "action": self._action,
                "input_type": input_type,
                "timing_ms": result.timing_ms,
                "backend": result.backend,
                "decision": "block" if (result.score >= self._threshold and self._action == "block") else self._action,
            }
        )
