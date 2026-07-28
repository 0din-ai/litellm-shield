"""Phase 0 smoke tests — package is importable and version is exposed."""

import importlib
import importlib.metadata

import litellm_shield


def test_package_importable() -> None:
    assert importlib.import_module("litellm_shield") is not None


def test_version_exposed() -> None:
    assert hasattr(litellm_shield, "__version__")
    assert isinstance(litellm_shield.__version__, str)
    assert litellm_shield.__version__ != ""


def test_version_matches_installed_distribution_metadata() -> None:
    # Version is derived from git tags via hatch-vcs at build time, not
    # hardcoded — it must always match what was actually installed.
    assert litellm_shield.__version__ == importlib.metadata.version("0din-litellm-shield")


def test_guardrail_class_exported() -> None:
    from litellm_shield import SusFactorGuardrail

    assert SusFactorGuardrail is not None
