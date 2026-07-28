# LiteLLM custom guardrail shim.
#
# LiteLLM resolves "litellm_shield.SusFactorGuardrail" by looking for
# litellm_shield.py next to the config file. This shim re-exports from
# the installed package so the file-based loader finds the class.
from litellm_shield import SusFactorGuardrail  # noqa: F401

proxy_handler_instance = SusFactorGuardrail()
