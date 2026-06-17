# litellm-shield — LiteLLM proxy with SusFactor guardrail
#
# Model weights are NOT baked in. Mount a pre-provisioned model cache at runtime:
#   -v /path/to/your/model/cache:/model-cache:ro
#
# See scripts/provision_model.py to download the gated 0dinai/susfactor-e5-large model.

FROM python:3.13-slim

# ---- system deps -------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---- uv ----------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# ---- install deps -------------------------------------------------------
# odin-prompt-toolkit is a private repo — the wheel is built locally
# (requires SSH access) and copied in; no GitHub credentials needed here.
COPY pyproject.toml uv.lock ./
COPY wheels/ ./wheels/

# Install all Python deps:
#   1. odin-prompt-toolkit from the pre-built local wheel
#   2. Everything else (litellm, torch, transformers, etc.) from PyPI
RUN uv pip install --system \
        wheels/odin_prompt_toolkit-*.whl \
        "litellm[proxy]>=1.0.0" \
        "torch>=2.0.0" \
        "transformers>=4.0.0" \
        "onnxruntime>=1.16.0"

# ---- install the guardrail package -------------------------------------
COPY susfactor_guardrail/ ./susfactor_guardrail/
COPY pyproject.toml ./
RUN uv pip install --system --no-deps .

# ---- LiteLLM proxy config + guardrail shim -----------------------------
# LiteLLM resolves custom guardrails by looking for a .py file next to the
# config file. The shim re-exports from the installed package.
COPY susfactor_guardrail.py ./susfactor_guardrail.py
COPY examples/config-flag.yaml ./config.yaml

# ---- model cache mount point -------------------------------------------
# The gated 0dinai/susfactor-e5-large model is mounted here at runtime.
# Layout expected:
#   /model-cache/susfactor-v1/encoder/config.json
#   /model-cache/susfactor-v1/encoder/model.safetensors
#   /model-cache/susfactor-v1/encoder/tokenizer.json
#   /model-cache/susfactor-v1/head.pt
ENV SIGNATURE_SDK_MODEL_CACHE=/model-cache

# ---- runtime -----------------------------------------------------------
EXPOSE 4000

# LiteLLM proxy entrypoint. Config can be overridden via LITELLM_CONFIG env var.
ENV LITELLM_CONFIG=/app/config.yaml

CMD ["litellm", "--config", "/app/config.yaml", "--port", "4000"]
