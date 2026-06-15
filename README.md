# litellm-shield

SusFactor jailbreak and prompt-injection guardrail for [LiteLLM](https://docs.litellm.ai/).

Runs the 0DIN SusFactor classifier in-process to detect jailbreak and prompt-injection attempts on LLM inputs, outputs, and tool results — with no network hop.

> **Status:** Early development (0DIN-1587). Internal only.

## Quick start

```yaml
# litellm config.yaml
guardrails:
  - guardrail_name: "susfactor"
    litellm_params:
      guardrail: susfactor_guardrail.SusFactorGuardrail
      mode: "during_call"   # runs in parallel with the LLM call
      action: "flag"        # flag | shadow | block
      threshold: 0.5
      fail_open: true
```

## Model setup

SusFactor uses the gated `0dinai/susfactor-e5-large` model from HuggingFace.
Download and cache it before starting the proxy:

```
# Coming in Phase 6 — model provisioning docs
```

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run black --check .
uv run mypy susfactor_guardrail/
```
