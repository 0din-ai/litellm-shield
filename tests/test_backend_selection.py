"""Backend precedence/resolution tests for SusFactorGuardrail.

Covers the backend/token precedence table from 0DIN-1847: whether ``backend``
is explicitly set and whether an access token is resolved together determine
which concrete backend gets instantiated. All backend construction is mocked
— no real HTTP client or classifier gets built.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from susfactor_guardrail.guardrail import SusFactorGuardrail


def _make_guardrail(
    backend: Optional[str] = None,
    access_token: Optional[str] = None,
    **kwargs: object,
) -> SusFactorGuardrail:
    return SusFactorGuardrail(
        guardrail_name="susfactor",
        backend=backend,
        access_token=access_token,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _mock_backends():
    """Patch backend constructors so no real classifier/HTTP client gets built."""
    with (
        patch("susfactor_guardrail.guardrail.SdkBackend", return_value=MagicMock()) as sdk,
        patch("susfactor_guardrail.guardrail.OnnxSdkBackend", return_value=MagicMock()) as onnx,
        patch("susfactor_guardrail.guardrail.HostedBackend", return_value=MagicMock()) as hosted,
    ):
        yield {"sdk": sdk, "sdk-onnx": onnx, "hosted": hosted}


# ---------------------------------------------------------------------------
# Row: backend=None, no token → "sdk" (preserves today's default)
# ---------------------------------------------------------------------------


def test_unset_backend_no_token_resolves_to_sdk(_mock_backends, monkeypatch) -> None:
    monkeypatch.delenv("ODIN_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("ODIN_API_TOKEN", raising=False)

    g = _make_guardrail(backend=None, access_token=None)

    assert g._backend_kind == "sdk"
    _mock_backends["sdk"].assert_called_once()
    _mock_backends["sdk-onnx"].assert_not_called()
    _mock_backends["hosted"].assert_not_called()


# ---------------------------------------------------------------------------
# Row: backend=None, token resolved → "hosted", with a warning log
# ---------------------------------------------------------------------------


def test_unset_backend_with_token_auto_selects_hosted(
    _mock_backends, monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("ODIN_API_TOKEN", raising=False)

    with caplog.at_level("WARNING"):
        g = _make_guardrail(backend=None, access_token="secret-token")

    assert g._backend_kind == "hosted"
    _mock_backends["hosted"].assert_called_once()
    _mock_backends["sdk"].assert_not_called()
    assert any("auto-selecting hosted backend" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Row: backend explicit sdk/sdk-onnx, token resolved → honor explicit choice, info log
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("explicit_backend", ["sdk", "sdk-onnx"])
def test_explicit_sdk_backend_with_token_ignores_token(
    explicit_backend: str, _mock_backends, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO"):
        g = _make_guardrail(backend=explicit_backend, access_token="secret-token")

    assert g._backend_kind == explicit_backend
    _mock_backends[explicit_backend].assert_called_once()
    _mock_backends["hosted"].assert_not_called()
    assert any("ignoring token" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Row: backend explicit sdk/sdk-onnx, no token → honor explicit choice (unchanged)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("explicit_backend", ["sdk", "sdk-onnx"])
def test_explicit_sdk_backend_no_token(explicit_backend: str, _mock_backends, monkeypatch) -> None:
    monkeypatch.delenv("ODIN_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("ODIN_API_TOKEN", raising=False)

    g = _make_guardrail(backend=explicit_backend, access_token=None)

    assert g._backend_kind == explicit_backend
    _mock_backends[explicit_backend].assert_called_once()


# ---------------------------------------------------------------------------
# Row: backend="hosted" explicit, no token → ValueError
# ---------------------------------------------------------------------------


def test_explicit_hosted_backend_no_token_raises(_mock_backends, monkeypatch) -> None:
    monkeypatch.delenv("ODIN_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("ODIN_API_TOKEN", raising=False)

    with pytest.raises(ValueError, match="access token"):
        _make_guardrail(backend="hosted", access_token=None)


# ---------------------------------------------------------------------------
# Row: backend="hosted" explicit, token resolved → use it
# ---------------------------------------------------------------------------


def test_explicit_hosted_backend_with_token(_mock_backends) -> None:
    g = _make_guardrail(backend="hosted", access_token="secret-token")

    assert g._backend_kind == "hosted"
    _mock_backends["hosted"].assert_called_once()
    call_kwargs = _mock_backends["hosted"].call_args.kwargs
    assert call_kwargs["access_token"] == "secret-token"
    assert call_kwargs["portal_url"] == "https://0din.ai"
    assert call_kwargs["susfactor_url"] == "https://defense.0din.ai"


# ---------------------------------------------------------------------------
# Row: anything else explicit → ValueError (today's "Unknown backend" behavior)
# ---------------------------------------------------------------------------


def test_invalid_backend_name_raises(_mock_backends) -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        _make_guardrail(backend="invalid-backend-name")


# ---------------------------------------------------------------------------
# URL resolution — explicit param → env var → default
# ---------------------------------------------------------------------------


def test_portal_and_susfactor_url_env_fallback(_mock_backends, monkeypatch) -> None:
    monkeypatch.setenv("ODIN_PORTAL_URL", "https://portal.env.example")
    monkeypatch.setenv("ODIN_SUSFACTOR_URL", "https://susfactor.env.example")

    g = _make_guardrail(backend="hosted", access_token="secret-token")

    assert g._backend_kind == "hosted"
    call_kwargs = _mock_backends["hosted"].call_args.kwargs
    assert call_kwargs["portal_url"] == "https://portal.env.example"
    assert call_kwargs["susfactor_url"] == "https://susfactor.env.example"


def test_access_token_falls_back_to_odin_api_token_env(_mock_backends, monkeypatch) -> None:
    monkeypatch.delenv("ODIN_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("ODIN_API_TOKEN", "legacy-token")

    g = _make_guardrail(backend=None, access_token=None)

    assert g._backend_kind == "hosted"
    call_kwargs = _mock_backends["hosted"].call_args.kwargs
    assert call_kwargs["access_token"] == "legacy-token"
