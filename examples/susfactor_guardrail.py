# LiteLLM custom guardrail shim.
#
# LiteLLM resolves "susfactor_guardrail.SusFactorGuardrail" by looking for
# susfactor_guardrail.py next to the config file. This shim re-exports from
# the installed package so the file-based loader finds the class.
from susfactor_guardrail import SusFactorGuardrail  # noqa: F401

proxy_handler_instance = SusFactorGuardrail()
