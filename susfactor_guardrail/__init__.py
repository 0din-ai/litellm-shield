"""SusFactor jailbreak/prompt-injection guardrail for LiteLLM.

This package provides a ``CustomGuardrail`` implementation that runs the
0DIN SusFactor classifier in-process to detect jailbreak and prompt-injection
attempts on LLM inputs, outputs, and tool results.

Example config.yaml::

    guardrails:
      - guardrail_name: "susfactor"
        litellm_params:
          guardrail: susfactor_guardrail.SusFactorGuardrail
          mode: "during_call"
          action: "flag"
          threshold: 0.5
          fail_open: true
"""

from __future__ import annotations

__version__ = "0.1.0"

# SusFactorGuardrail is imported here so callers can do:
#   from susfactor_guardrail import SusFactorGuardrail
# and LiteLLM can resolve "susfactor_guardrail.SusFactorGuardrail".
#
# The real implementation lives in .guardrail; this stub satisfies P0.
# It will be replaced in P2 with the full CustomGuardrail subclass.
from susfactor_guardrail._stub import SusFactorGuardrail

__all__ = [
    "SusFactorGuardrail",
    "__version__",
]
