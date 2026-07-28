"""SusFactor jailbreak/prompt-injection guardrail for LiteLLM.

This package provides a ``CustomGuardrail`` implementation that runs the
0DIN SusFactor classifier to detect jailbreak and prompt-injection attempts
on LLM inputs, outputs, and tool results. The ``sdk``/``sdk-onnx`` backends
run the classifier in-process (no network hop). A ``hosted`` backend is also
available for when self-hosting the model isn't desired — it sends prompts
to 0DIN's hosted SusFactor API over HTTPS instead.

Example config.yaml::

    guardrails:
      - guardrail_name: "susfactor"
        litellm_params:
          guardrail: litellm_shield.SusFactorGuardrail
          mode: "during_call"
          enforcement: "flag"
          threshold: 0.5
          fail_open: true
"""

from __future__ import annotations

import importlib.metadata

try:
    # Version is derived from git tags at build time via hatch-vcs — see
    # [tool.hatch.version] in pyproject.toml. This lookup works both for
    # wheel installs (metadata baked in at build time) and editable/dev
    # installs (uv/hatchling regenerates it from the local git repo).
    __version__ = importlib.metadata.version("0din-litellm-shield")
except importlib.metadata.PackageNotFoundError:
    # Running from a source checkout that was never installed (no dist-info
    # to read). Not expected in normal use, but avoid crashing on import.
    __version__ = "0.0.0"

from litellm_shield.guardrail import SusFactorGuardrail

__all__ = [
    "SusFactorGuardrail",
    "__version__",
]
