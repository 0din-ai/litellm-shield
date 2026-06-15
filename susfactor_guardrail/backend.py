"""Backend seam for SusFactor scoring.

Defines the ``SusFactorBackend`` protocol and concrete implementations:

- ``SdkBackend``: in-process inference via ``odin-prompt-toolkit[susfactor]``.
  Loads the e5-large encoder + MLP head once, lazily, behind an asyncio.Lock.

- ``SusFactorUnavailable``: raised when the backend cannot produce a score
  (model missing, load failure, inference error). Callers use ``fail_open``
  to decide whether to allow or block on this error.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

# Import at module level so unittest.mock.patch can resolve the name.
# These are optional deps (torch/transformers); ImportError is surfaced as
# SusFactorUnavailable at runtime rather than at import time.
_SDK_AVAILABLE = False
try:
    from odin_prompt_toolkit.providers.model_cache import ModelCache
    from odin_prompt_toolkit.susfactor import SusFactorClassifier

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover — torch/transformers not installed
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class SusResult:
    """Result of a SusFactor scoring call.

    Mirrors the shape of both ``SusFactorResult`` (prompt-toolkit SDK) and
    the Heimdall ``/api/v1/sus`` response so backends are interchangeable.
    """

    score: float
    """P(suspicious) ∈ [0, 1]."""

    label: str
    """``"suspicious"`` if ``score >= threshold`` else ``"safe"``."""

    model: str
    """Model identifier reported by the backend."""

    threshold: float
    """Decision threshold used to derive ``label``."""

    timing_ms: Optional[float]
    """Inference time in milliseconds, if measured."""

    backend: str
    """Which backend produced this result (``"sdk"`` or ``"heimdall"``)."""

    @property
    def is_suspicious(self) -> bool:
        """Whether the text was classified as suspicious."""
        return self.label == "suspicious"


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class SusFactorUnavailable(Exception):
    """Raised when a backend cannot produce a score.

    Signals transient or permanent unavailability (model not cached, load
    failure, inference error, network error for HTTP backends). The guardrail
    uses ``fail_open`` to decide whether to allow or block on this error.
    """


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SusFactorBackend(Protocol):
    """One-method interface for SusFactor scoring backends."""

    async def score(self, text: str) -> SusResult:
        """Score a single text string.

        Args:
            text: The text to classify.

        Returns:
            A :class:`SusResult` with the suspicion probability and label.

        Raises:
            SusFactorUnavailable: If the backend cannot produce a score.
        """
        ...


# ---------------------------------------------------------------------------
# SDK backend
# ---------------------------------------------------------------------------


class SdkBackend:
    """In-process SusFactor backend using ``odin-prompt-toolkit[susfactor]``.

    Loads ``SusFactorClassifier`` once, lazily, on the first ``score()`` call.
    Subsequent calls reuse the loaded classifier. An ``asyncio.Lock`` prevents
    double-loading under concurrent proxy requests.

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
                    "odin-prompt-toolkit[susfactor] is not installed. "
                    "Install it with: pip install 'odin-prompt-toolkit[susfactor]'"
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
                # Surface as SusFactorUnavailable regardless of the underlying error type
                # (SusFactorError for missing model, any other Exception for load failures).
                raise SusFactorUnavailable(str(exc)) from exc

    async def score(self, text: str) -> SusResult:
        """Classify ``text`` using the in-process SusFactor classifier.

        Args:
            text: The text to classify.

        Returns:
            A :class:`SusResult`.

        Raises:
            SusFactorUnavailable: If the model is not cached or inference fails.
        """
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
