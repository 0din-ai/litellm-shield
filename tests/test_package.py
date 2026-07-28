"""Phase 0 smoke tests — package is importable and version is exposed."""

import importlib
import importlib.metadata

import susfactor_guardrail


def test_package_importable() -> None:
    assert importlib.import_module("susfactor_guardrail") is not None


def test_version_exposed() -> None:
    assert hasattr(susfactor_guardrail, "__version__")
    assert isinstance(susfactor_guardrail.__version__, str)
    assert susfactor_guardrail.__version__ != ""


def test_version_matches_installed_distribution_metadata() -> None:
    # Version is derived from git tags via hatch-vcs at build time, not
    # hardcoded — it must always match what was actually installed.
    assert susfactor_guardrail.__version__ == importlib.metadata.version("0din-litellm-shield")


def test_guardrail_class_exported() -> None:
    from susfactor_guardrail import SusFactorGuardrail

    assert SusFactorGuardrail is not None
