"""Backend seam for SusFactor scoring.

Defines the ``SusFactorBackend`` protocol and concrete implementations:

- ``SdkBackend``: in-process torch inference via ``odin-prompt-toolkit[susfactor]``.
- ``OnnxSdkBackend``: in-process ONNX Runtime inference via ``odin-prompt-toolkit[onnx]``.
  Faster on CPU (~3-5× vs torch); no torch dependency at inference time.
- ``HostedBackend``: HTTP calls to 0DIN's hosted SusFactor API. Prompts leave
  the local process over HTTPS — use this when self-hosting the model isn't
  desired. Mints short-lived JWTs from the 0DIN Portal and refreshes them
  out-of-band in the background so ``score()`` never blocks on a mint.

- ``SusFactorUnavailable``: raised when the backend cannot produce a score
  (model missing, load failure, inference error, network error). Callers use
  ``fail_open`` to decide whether to allow or block on this error.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import httpx

# Import at module level so unittest.mock.patch can resolve the name.
# Optional deps; ImportError is surfaced as SusFactorUnavailable at runtime.
_SDK_AVAILABLE = False
try:
    from odin_prompt_toolkit.providers.model_cache import ModelCache
    from odin_prompt_toolkit.susfactor import SusFactorClassifier

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass

_ONNX_SDK_AVAILABLE = False
try:
    from odin_prompt_toolkit.susfactor.onnx_classifier import SusFactorOnnxClassifier

    _ONNX_SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)


@dataclass
class SusResult:
    """Result of a SusFactor scoring call.

    Mirrors the shape of both ``SusFactorResult`` (prompt-toolkit SDK) and
    the Heimdall ``/api/v1/sus`` response so backends are interchangeable.

    Fields:
        score: P(suspicious) ∈ [0, 1].
        label: ``"suspicious"`` if ``score >= threshold`` else ``"safe"``.
        model: Model identifier reported by the backend.
        threshold: Decision threshold used to derive ``label``.
        timing_ms: Inference time in milliseconds, if measured.
        backend: Which backend produced this result (``"sdk"``, ``"sdk-onnx"``, or ``"hosted"``).
    """

    score: float
    label: str
    model: str
    threshold: float
    timing_ms: Optional[float]
    backend: str


class SusFactorUnavailable(Exception):
    """Raised when a backend cannot produce a score.

    Signals transient or permanent unavailability (model not cached, load
    failure, inference error, network error for HTTP backends). The guardrail
    uses ``fail_open`` to decide whether to allow or block on this error.
    """


class SusFactorBackend(Protocol):
    """Structural type for SusFactor scoring backends — implement ``score``."""

    async def score(self, text: str) -> SusResult: ...


class SdkBackend:
    """In-process SusFactor backend using ``odin-prompt-toolkit[susfactor]``.

    Loads ``SusFactorClassifier`` once, lazily, on the first ``score()`` call.
    An ``asyncio.Lock`` prevents double-loading under concurrent proxy requests.

    Args:
        model_cache_dir: Override for the model cache directory. ``None`` uses
            the SDK default (``~/.cache/signature-sdk/models/``) or the
            ``SIGNATURE_SDK_MODEL_CACHE`` env var.
        threshold: Decision threshold passed to the classifier.
        device: Torch device (``"cuda"``, ``"mps"``, ``"cpu"``).
            ``None`` auto-detects.
    """

    def __init__(
        self,
        model_cache_dir: Optional[str],
        threshold: float,
        device: Optional[str],
    ) -> None:
        self._model_cache_dir = model_cache_dir
        self._threshold = threshold
        self._device = device

        self._classifier: Any = None
        self._loaded: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        """Load the classifier exactly once, even under concurrent calls."""
        if self._loaded:
            return

        async with self._lock:
            # Re-check inside the lock in case another task loaded while we waited.
            if self._loaded:
                return

            if not _SDK_AVAILABLE:
                raise SusFactorUnavailable(
                    "Self-hosted SusFactor inference requires the internal "
                    "odin-prompt-toolkit[susfactor] package, which is not available in the "
                    "public 0din-litellm-shield distribution. Use backend='hosted' instead, or "
                    "install the internal self-hosted extra if you have access to 0DIN's "
                    "private package registry."
                )

            try:
                cache = ModelCache(self._model_cache_dir)
                logger.info(
                    "susfactor.sdk: loading classifier (threshold=%.2f, device=%s)",
                    self._threshold,
                    self._device or "auto",
                )
                self._classifier = await SusFactorClassifier.new(
                    cache,
                    threshold=self._threshold,
                    device=self._device,
                )
                self._loaded = True
                logger.info("susfactor.sdk: classifier loaded")

            except SusFactorUnavailable:
                raise
            except Exception as exc:
                raise SusFactorUnavailable(str(exc)) from exc

    async def score(self, text: str) -> SusResult:
        """Classify ``text``, raising ``SusFactorUnavailable`` on any failure."""
        await self._ensure_loaded()

        try:
            sdk_result = await self._classifier.classify(text)
            return SusResult(
                score=sdk_result.score,
                label=sdk_result.label,
                model=sdk_result.model,
                threshold=sdk_result.threshold,
                timing_ms=sdk_result.timing_ms,
                backend="sdk",
            )
        except SusFactorUnavailable:
            raise
        except Exception as exc:
            raise SusFactorUnavailable(str(exc)) from exc


class OnnxSdkBackend:
    """In-process ONNX Runtime backend via ``odin-prompt-toolkit[onnx]``.

    Runs the SusFactor classifier via ONNX Runtime instead of PyTorch —
    ~3-5× faster on CPU and no torch dependency at inference time. Requires
    the ONNX model artifact from ``0dinai/susfactor-e5-large-onnx``.

    Args:
        model_cache_dir: Override for the model cache directory.
        threshold: Decision threshold passed to the classifier.
        device: Accepted for API parity; ONNX Runtime selects its own providers.
    """

    def __init__(
        self,
        model_cache_dir: Optional[str],
        threshold: float,
        device: Optional[str],
    ) -> None:
        self._model_cache_dir = model_cache_dir
        self._threshold = threshold
        self._device = device

        self._classifier: Any = None
        self._loaded: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        async with self._lock:
            if self._loaded:
                return

            if not _ONNX_SDK_AVAILABLE:
                raise SusFactorUnavailable(
                    "Self-hosted SusFactor inference requires the internal "
                    "odin-prompt-toolkit[onnx] package, which is not available in the public "
                    "0din-litellm-shield distribution. Use backend='hosted' instead, or install the "
                    "internal self-hosted extra if you have access to 0DIN's private package "
                    "registry."
                )

            try:
                cache = ModelCache(self._model_cache_dir)
                logger.info(
                    "susfactor.sdk-onnx: loading classifier (threshold=%.2f)",
                    self._threshold,
                )
                self._classifier = await SusFactorOnnxClassifier.new(
                    cache,
                    threshold=self._threshold,
                    device=self._device,
                )
                self._loaded = True
                logger.info("susfactor.sdk-onnx: classifier loaded")

            except SusFactorUnavailable:
                raise
            except Exception as exc:
                raise SusFactorUnavailable(str(exc)) from exc

    async def score(self, text: str) -> SusResult:
        """Classify ``text``, raising ``SusFactorUnavailable`` on any failure."""
        await self._ensure_loaded()

        try:
            sdk_result = await self._classifier.classify(text)
            return SusResult(
                score=sdk_result.score,
                label=sdk_result.label,
                model=sdk_result.model,
                threshold=sdk_result.threshold,
                timing_ms=sdk_result.timing_ms,
                backend="sdk-onnx",
            )
        except SusFactorUnavailable:
            raise
        except Exception as exc:
            raise SusFactorUnavailable(str(exc)) from exc


# Response bodies are logged/embedded in exceptions truncated to this length —
# long enough for debugging, short enough to avoid dumping huge error pages.
_RESPONSE_BODY_TRUNCATE = 500

# Initial and max backoff for background JWT refresh retries after a failure.
_REFRESH_BACKOFF_INITIAL_S = 5.0
_REFRESH_BACKOFF_MAX_S = 60.0


class HostedBackend:
    """Hosted SusFactor backend — HTTP calls to 0DIN's hosted API.

    Two-step flow: mint a short-lived JWT from the 0DIN Portal using a
    long-lived access token, then call the hosted SusFactor scoring endpoint
    with the JWT as a Bearer token. The JWT is refreshed out-of-band by a
    background task so steady-state ``score()`` calls never await a mint —
    they read the already-fresh token.

    Args:
        portal_url: Base URL of the 0DIN Portal (mints JWTs).
        susfactor_url: Base URL of the hosted SusFactor scoring API.
        access_token: 0DIN Portal API access token (long-lived).
        threshold: Decision threshold, kept for API parity with the other
            backends (the hosted API reports its own threshold per response).
        refresh_buffer_s: Seconds before expiry to trigger the background
            refresh — e.g. with the default 900s token lifetime and a 120s
            buffer, refresh fires at the 780s mark.
        timeout_s: HTTP timeout (connect/read/write/pool, seconds) for both
            the JWT mint and scoring requests. httpx.AsyncClient() with no
            timeout= silently gets an implicit 5s default — indistinguishable
            from a deliberate choice and easy to blow through on a cold
            backend or a slow network hop. We use a longer, explicit default
            so transient slowness doesn't masquerade as unavailability.
    """

    # See timeout_s docstring above for why this isn't httpx's implicit 5s default.
    _DEFAULT_TIMEOUT_S = 10.0

    def __init__(
        self,
        portal_url: str,
        susfactor_url: str,
        access_token: str,
        threshold: float,
        refresh_buffer_s: int = 120,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._portal_url = portal_url.rstrip("/")
        self._susfactor_url = susfactor_url.rstrip("/")
        self._access_token = access_token
        self._threshold = threshold
        self._refresh_buffer_s = refresh_buffer_s

        self._client = httpx.AsyncClient(timeout=timeout_s)
        self._jwt: Optional[str] = None
        self._jwt_expires_at: float = 0.0
        self._lock: asyncio.Lock = asyncio.Lock()
        self._refresh_task: Optional[asyncio.Task[None]] = None
        self._refresh_backoff_s: float = _REFRESH_BACKOFF_INITIAL_S

    async def _mint_jwt(self) -> tuple[str, float]:
        """POST to the Portal's ``/api/v1/access_tokens``, return ``(jwt, expires_in)``."""
        try:
            resp = await self._client.post(
                f"{self._portal_url}/api/v1/access_tokens",
                headers={"Authorization": self._access_token},
            )
        except httpx.HTTPError as exc:
            raise SusFactorUnavailable(
                f"susfactor.hosted: JWT mint request to Portal failed: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise SusFactorUnavailable(
                f"susfactor.hosted: JWT mint failed with status {resp.status_code}: "
                f"{resp.text[:_RESPONSE_BODY_TRUNCATE]}"
            )

        try:
            data = resp.json()
            token = data["token"]
            expires_in = data["expires_in"]
        except (ValueError, KeyError, TypeError) as exc:
            raise SusFactorUnavailable(
                f"susfactor.hosted: unexpected JWT mint response shape: "
                f"{resp.text[:_RESPONSE_BODY_TRUNCATE]}"
            ) from exc

        return token, expires_in

    async def _ensure_jwt(self) -> None:
        """Mint the JWT if missing or expired, even under concurrent calls; (re)start refresh."""
        if self._jwt is not None and time.monotonic() < self._jwt_expires_at:
            return

        async with self._lock:
            # Re-check inside the lock in case another task minted while we waited.
            if self._jwt is not None and time.monotonic() < self._jwt_expires_at:
                return

            token, expires_in = await self._mint_jwt()
            self._jwt = token
            self._jwt_expires_at = time.monotonic() + expires_in
            logger.info("susfactor.hosted: JWT minted (expires_in=%.0fs)", expires_in)

            # Only spawn a new background refresh loop if one isn't already running
            # (avoids a duplicate loop when this inline mint happens because the
            # existing loop is stuck in its own retry/backoff cycle).
            if self._refresh_task is None or self._refresh_task.done():
                self._refresh_task = asyncio.ensure_future(self._refresh_loop())

    async def _refresh_loop(self) -> None:
        """Re-mint the JWT out-of-band before it expires, retrying with backoff on failure."""
        while True:
            delay = max(self._jwt_expires_at - time.monotonic() - self._refresh_buffer_s, 0)
            await asyncio.sleep(delay)

            try:
                token, expires_in = await self._mint_jwt()
                self._jwt = token
                self._jwt_expires_at = time.monotonic() + expires_in
                self._refresh_backoff_s = _REFRESH_BACKOFF_INITIAL_S
                logger.info("susfactor.hosted: JWT refreshed (expires_in=%.0fs)", expires_in)
            except Exception as exc:
                logger.warning(
                    "susfactor.hosted: background JWT refresh failed, "
                    "keeping current token, retrying in %.0fs: %s",
                    self._refresh_backoff_s,
                    exc,
                )
                await asyncio.sleep(self._refresh_backoff_s)
                self._refresh_backoff_s = min(self._refresh_backoff_s * 2, _REFRESH_BACKOFF_MAX_S)

    async def score(self, text: str) -> SusResult:
        """Score ``text`` via the hosted API, raising ``SusFactorUnavailable`` on any failure."""
        await self._ensure_jwt()

        try:
            resp = await self._client.post(
                f"{self._susfactor_url}/api/v1/sus",
                headers={"Authorization": f"Bearer {self._jwt}"},
                json={"prompt": text},
            )
        except httpx.HTTPError as exc:
            raise SusFactorUnavailable(f"susfactor.hosted: scoring request failed: {exc}") from exc

        if resp.status_code != 200:
            raise SusFactorUnavailable(
                f"susfactor.hosted: scoring failed with status {resp.status_code}: "
                f"{resp.text[:_RESPONSE_BODY_TRUNCATE]}"
            )

        try:
            data = resp.json()
            return SusResult(
                score=data["score"],
                label=data["label"],
                model=data["model"],
                threshold=data["threshold"],
                timing_ms=data.get("timing_ms"),
                backend="hosted",
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise SusFactorUnavailable(
                f"susfactor.hosted: unexpected scoring response shape: "
                f"{resp.text[:_RESPONSE_BODY_TRUNCATE]}"
            ) from exc
