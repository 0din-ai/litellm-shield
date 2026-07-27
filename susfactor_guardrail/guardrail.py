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
          backend: "sdk"             # sdk | sdk-onnx | hosted
          model_cache_dir: os.environ/SIGNATURE_SDK_MODEL_CACHE
          device: null
          scan_output: false
          # hosted-backend-only settings (ignored unless backend: "hosted",
          # or auto-selected when an access token is present and backend is unset):
          portal_url: null           # default https://0din.ai, or ODIN_PORTAL_URL
          susfactor_url: null        # default https://defense.0din.ai, or ODIN_SUSFACTOR_URL
          access_token: os.environ/ODIN_ACCESS_TOKEN
          timeout_s: 10.0             # HTTP timeout for hosted scoring/JWT-mint requests
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Literal, Optional

from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth

from susfactor_guardrail.backend import (
    HostedBackend,
    OnnxSdkBackend,
    SdkBackend,
    SusFactorUnavailable,
)

logger = logging.getLogger(__name__)

# Valid action values
ACTIONS = frozenset({"block", "flag", "shadow"})

# Valid backend values
BACKENDS = frozenset({"sdk", "sdk-onnx", "hosted"})

_DEFAULT_PORTAL_URL = "https://0din.ai"
_DEFAULT_SUSFACTOR_URL = "https://defense.0din.ai"

# Captured at import time (not read off HostedBackend at call time) so tests that
# patch susfactor_guardrail.guardrail.HostedBackend don't turn this into a MagicMock.
_DEFAULT_HOSTED_TIMEOUT_S = HostedBackend._DEFAULT_TIMEOUT_S


class SusFactorGuardrail(CustomGuardrail):
    """LiteLLM guardrail that runs the SusFactor jailbreak/prompt-injection classifier.

    Args:
        enforcement: What to do when a suspicious text is detected.
            ``"block"`` raises an exception; ``"flag"`` annotates and allows;
            ``"shadow"`` records only (never blocks).
        threshold: Decision threshold. Score >= threshold ⇒ suspicious.
        fail_open: If ``True`` (default), allow the request when the backend is
            unavailable. If ``False``, block on unavailability.
        backend: Backend implementation to use — ``"sdk"`` (in-process torch),
            ``"sdk-onnx"`` (in-process ONNX Runtime), or ``"hosted"`` (HTTP
            calls to 0DIN's hosted SusFactor API; prompts leave the process
            over HTTPS). ``None`` (default) auto-selects: ``"hosted"`` if an
            access token is resolved (see ``access_token``), else ``"sdk"``.
            An explicit value is always honored as-is.
        model_cache_dir: Override for the model cache directory. ``None`` uses
            the SDK default or the ``SIGNATURE_SDK_MODEL_CACHE`` env var.
            Only used by ``"sdk"``/``"sdk-onnx"``.
        device: Torch device (``"cuda"``, ``"mps"``, ``"cpu"``). ``None`` auto-detects.
            Only used by ``"sdk"``/``"sdk-onnx"``.
        scan_output: If ``True``, also score model outputs via ``post_call``.
        portal_url: Base URL of the 0DIN Portal, used by the ``"hosted"``
            backend to mint JWTs. ``None`` uses ``ODIN_PORTAL_URL`` or
            ``"https://0din.ai"``.
        susfactor_url: Base URL of the hosted SusFactor scoring API. ``None``
            uses ``ODIN_SUSFACTOR_URL`` or ``"https://defense.0din.ai"``.
        access_token: 0DIN Portal API access token. ``None`` falls back to
            ``ODIN_ACCESS_TOKEN``, then ``ODIN_API_TOKEN``. Presence of a
            resolved token drives backend auto-selection when ``backend`` is
            unset; required (directly or via env) when ``backend="hosted"``.
        timeout_s: HTTP timeout (seconds) for the hosted backend's JWT-mint
            and scoring requests. Only used by ``"hosted"``. ``None`` uses
            ``HostedBackend``'s documented default.
        **kwargs: Forwarded to :class:`~litellm.integrations.custom_guardrail.CustomGuardrail`.
    """

    def __init__(
        self,
        enforcement: str = "flag",
        threshold: float = 0.5,
        fail_open: bool = True,
        backend: Optional[str] = None,
        model_cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        scan_output: bool = False,
        portal_url: Optional[str] = None,
        susfactor_url: Optional[str] = None,
        access_token: Optional[str] = None,
        timeout_s: Optional[float] = None,
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

        resolved_portal_url = portal_url or os.environ.get("ODIN_PORTAL_URL") or _DEFAULT_PORTAL_URL
        resolved_susfactor_url = (
            susfactor_url or os.environ.get("ODIN_SUSFACTOR_URL") or _DEFAULT_SUSFACTOR_URL
        )
        resolved_token = (
            access_token or os.environ.get("ODIN_ACCESS_TOKEN") or os.environ.get("ODIN_API_TOKEN")
        )

        resolved_backend = self._resolve_backend(
            backend=backend, token=resolved_token, susfactor_url=resolved_susfactor_url
        )
        self._backend_kind = resolved_backend

        if resolved_backend == "sdk":
            self._backend: SdkBackend | OnnxSdkBackend | HostedBackend = SdkBackend(
                model_cache_dir=resolved_cache_dir,
                threshold=threshold,
                device=device,
            )
        elif resolved_backend == "sdk-onnx":
            self._backend = OnnxSdkBackend(
                model_cache_dir=resolved_cache_dir,
                threshold=threshold,
                device=device,
            )
        else:
            assert resolved_token is not None  # enforced by _resolve_backend
            resolved_timeout_s = timeout_s if timeout_s is not None else _DEFAULT_HOSTED_TIMEOUT_S
            self._backend = HostedBackend(
                portal_url=resolved_portal_url,
                susfactor_url=resolved_susfactor_url,
                access_token=resolved_token,
                threshold=threshold,
                timeout_s=resolved_timeout_s,
            )

        super().__init__(**kwargs)

    @staticmethod
    def _resolve_backend(backend: Optional[str], token: Optional[str], susfactor_url: str) -> str:
        """Apply the backend/token precedence rule from 0DIN-1847.

        See the class docstring's ``backend`` entry for the full decision table.
        """
        if backend is None:
            if token:
                logger.warning(
                    "susfactor: ODIN access token detected — auto-selecting hosted "
                    "backend (prompts will be sent to %s). Set backend explicitly to "
                    "'sdk' or 'sdk-onnx' to keep prompts in-process.",
                    susfactor_url,
                )
                return "hosted"
            return "sdk"

        if backend not in BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}. Supported: {sorted(BACKENDS)}")

        if backend in ("sdk", "sdk-onnx") and token:
            logger.info(
                "susfactor: access token present but backend explicitly set to %r — "
                "ignoring token, prompts stay in-process.",
                backend,
            )

        if backend == "hosted" and not token:
            raise ValueError(
                "backend='hosted' requires an access token — pass access_token= or set "
                "ODIN_ACCESS_TOKEN (or ODIN_API_TOKEN)."
            )

        return backend

    async def apply_guardrail(  # type: ignore[override]
        self,
        inputs: dict[str, Any],
        request_data: dict[str, Any],
        input_type: Literal["request", "response"] = "request",
        logging_obj: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Score texts and tool-call args, applying the configured enforcement action."""
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

    async def _score_and_act(
        self,
        text: str,
        request_data: dict[str, Any],
        input_type: Literal["request", "response"],
    ) -> None:
        """Score one text, record observability data, and apply the action."""
        start = time.monotonic()

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
            self._record(
                result=None,
                request_data=request_data,
                input_type=input_type,
                decision="error",
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
        if self._enforcement == "block":
            decision = "block" if is_suspicious else "allow"
        else:
            decision = self._enforcement  # "flag" or "shadow"

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

        self._record(
            result=result, request_data=request_data, input_type=input_type, decision=decision
        )
        self._record_slg(
            request_data=request_data,
            status="guardrail_intervened" if is_suspicious else "success",
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

        if self._enforcement in ("shadow", "flag"):
            return

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
        """Return function.arguments from a tool call object or dict, or empty string."""
        # Attribute path: ChatCompletionMessageToolCall / OpenAIObject
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
        return ""

    def _record(
        self,
        result: Optional[Any],
        request_data: dict[str, Any],
        input_type: Literal["request", "response"],
        decision: str = "allow",
    ) -> None:
        """Write per-text decision metadata into request_data["metadata"]["susfactor"].

        ``result`` is ``None`` for the ``"error"`` decision (backend raised
        ``SusFactorUnavailable`` — no score was produced) so the timeout/error
        is still visible in the decision record instead of being silently
        dropped (0DIN-1874).
        """
        meta = request_data.setdefault("metadata", {})
        susfactor_records: list[dict[str, Any]] = meta.setdefault("susfactor", [])
        susfactor_records.append(
            {
                "score": result.score if result is not None else None,
                "label": result.label if result is not None else None,
                "model": result.model if result is not None else None,
                "threshold": self._threshold,
                "enforcement": self._enforcement,
                "input_type": input_type,
                "timing_ms": result.timing_ms if result is not None else None,
                "backend": result.backend if result is not None else self._backend_kind,
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
