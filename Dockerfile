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

# This host's sustained throughput to PyPI is slow enough that uv's default
# 30s socket timeout trips spuriously on large wheels (torch, the CUDA
# libraries) well before the transfer would otherwise complete.
ENV UV_HTTP_TIMEOUT=600

# Install straight into the system Python (the "prefix" of python:3.13-slim
# is /usr/local) instead of creating a .venv. `litellm` then lands directly
# on PATH for the CMD below — no `uv run` / venv activation required.
# https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path
ENV UV_PROJECT_ENVIRONMENT=/usr/local

# ---- install deps -------------------------------------------------------
# odin-prompt-toolkit is a private repo — the wheel is built locally
# (requires SSH access) and copied in; no GitHub credentials needed here.
# pyproject.toml pins it via [tool.uv.sources] as a local path dependency,
# resolved through the "self-hosted" extra — see pyproject.toml.
COPY pyproject.toml uv.lock ./
COPY wheels/ ./wheels/

# Install locked dependencies (litellm[proxy], torch, transformers,
# onnxruntime, odin-prompt-toolkit) without the litellm_shield project
# itself. --frozen installs exactly what's pinned in uv.lock — no PyPI
# resolution — and --no-install-project keeps this layer cached across
# litellm_shield source edits, since these deps change far less often.
# --mount=type=cache persists uv's downloaded wheels across builds/retries —
# this step pulls several GB of CUDA/torch wheels, so a retry after a
# transient network timeout shouldn't have to re-download packages already
# fetched by a previous attempt.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra self-hosted

# ---- install the guardrail package -------------------------------------
# README.md is required here (not in the deps layer above) because
# hatchling reads it while building litellm_shield's own metadata —
# pyproject.toml declares readme = "README.md".
COPY litellm_shield/ ./litellm_shield/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra self-hosted --no-editable

# ---- LiteLLM proxy config + guardrail shim -----------------------------
# LiteLLM resolves custom guardrails by looking for a .py file next to the
# config file. The shim re-exports from the installed package.
COPY litellm_shield.py ./litellm_shield.py
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
