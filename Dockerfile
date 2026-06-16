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

# ---- install deps (lockfile-driven, no editable install) ---------------
# Copy only the dependency manifest + lockfile first for Docker layer caching.
COPY pyproject.toml uv.lock ./

# Install all deps from lockfile. torch brings ~2GB but weights stay outside.
# Use --no-dev to skip test/lint tools. --no-install-project because we
# copy the source in the next step.
RUN uv sync --frozen --no-dev --no-install-project

# ---- install the guardrail package -------------------------------------
COPY susfactor_guardrail/ ./susfactor_guardrail/
RUN uv sync --frozen --no-dev

# ---- LiteLLM proxy config ----------------------------------------------
# The config file is mounted at runtime via docker-compose or -v flag.
# A default config is provided for convenience.
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

CMD ["uv", "run", "litellm", "--config", "/app/config.yaml", "--port", "4000"]
