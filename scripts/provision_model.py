#!/usr/bin/env python3
"""Download the gated 0dinai/susfactor-e5-large model into the local cache.

The SusFactor classifier requires two components:
  - encoder/   — the e5-large HuggingFace model (config, weights, tokenizer)
  - head.pt    — the trained MLP classification head

Both live under the susfactor-v1 version directory inside the model cache.

Usage
-----
    # Interactive (prompts for HF token if HF_TOKEN not set):
    uv run python scripts/provision_model.py

    # Non-interactive (CI / Docker build):
    HF_TOKEN=hf_... uv run python scripts/provision_model.py

    # Custom cache dir:
    uv run python scripts/provision_model.py --cache-dir /data/models

The script writes files to:
    <cache>/susfactor-v1/encoder/config.json
    <cache>/susfactor-v1/encoder/model.safetensors
    <cache>/susfactor-v1/encoder/tokenizer.json
    <cache>/susfactor-v1/head.pt

After provisioning, verify with:
    uv run python scripts/provision_model.py --check
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "0dinai/susfactor-e5-large"
MODEL_VERSION = "susfactor-v1"
DEFAULT_CACHE_ENV = "SIGNATURE_SDK_MODEL_CACHE"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "signature-sdk" / "models"

REQUIRED_FILES = (
    "encoder/config.json",
    "encoder/model.safetensors",
    "encoder/tokenizer.json",
    "head.pt",
)


def resolve_cache_dir(override: str | None) -> Path:
    if override:
        return Path(override).expanduser()
    env = os.environ.get(DEFAULT_CACHE_ENV)
    if env:
        return Path(env).expanduser()
    return DEFAULT_CACHE_DIR


def check(cache_dir: Path) -> bool:
    """Return True if all required model files are present."""
    model_dir = cache_dir / MODEL_VERSION
    missing = [f for f in REQUIRED_FILES if not (model_dir / f).exists()]
    if missing:
        print(f"[check] Missing files in {model_dir}:")
        for f in missing:
            print(f"  - {f}")
        return False
    print(f"[check] All model files present in {model_dir}")
    return True


def provision(cache_dir: Path, token: str | None) -> None:
    """Download the model from HuggingFace into cache_dir/susfactor-v1/."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("       Run: uv run pip install huggingface_hub")
        sys.exit(1)

    model_dir = cache_dir / MODEL_VERSION
    encoder_dir = model_dir / "encoder"
    encoder_dir.mkdir(parents=True, exist_ok=True)

    if not token:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not token:
        print(
            f"INFO: HF_TOKEN not set. The model {REPO_ID!r} is gated — you need a\n"
            "      HuggingFace token with access. Set HF_TOKEN or pass --token.\n"
            "      Request access at: https://huggingface.co/0dinai/susfactor-e5-large"
        )
        token = input("HuggingFace token (or press Enter to abort): ").strip() or None
        if not token:
            print("Aborted.")
            sys.exit(1)

    print(f"Downloading {REPO_ID} → {model_dir}")

    # Download encoder files (HuggingFace model repo)
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(encoder_dir),
        token=token,
        ignore_patterns=["head.pt", "*.md", ".gitattributes"],
    )

    # head.pt lives at the repo root
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(model_dir),
        token=token,
        allow_patterns=["head.pt"],
    )

    print("\nProvisioning complete. Verifying...")
    ok = check(cache_dir)
    if not ok:
        print("\nWARNING: Some files are missing. The download may have been partial.")
        sys.exit(1)
    print("\nModel is ready.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision the SusFactor model cache.")
    parser.add_argument("--cache-dir", help="Override model cache directory.")
    parser.add_argument("--token", help="HuggingFace API token (or set HF_TOKEN env var).")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check if model files are present; do not download.",
    )
    args = parser.parse_args()

    cache_dir = resolve_cache_dir(args.cache_dir)
    print(f"Model cache: {cache_dir}")

    if args.check:
        ok = check(cache_dir)
        sys.exit(0 if ok else 1)

    if check(cache_dir):
        print("Model already provisioned. Use --force to re-download (not implemented).")
        return

    provision(cache_dir, token=args.token)


if __name__ == "__main__":
    main()
