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
from typing import Any, Optional, Protocol

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
        backend: Which backend produced this result (``"sdk"`` or ``"heimdall"``).
    """

    score: float
    label: str
    model: str
    threshold: float
    timing_ms: Optional[float]
    backend: str

    @property
    def is_suspicious(self) -> bool:
        return self.label == "suspicious"


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
