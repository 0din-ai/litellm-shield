#!/usr/bin/env python3
"""Download SusFactor model artifacts into the local cache.

Two artifacts are available:

  torch (default):
    Repo: 0dinai/susfactor-e5-large  (gated — requires HF access)
    Layout: <cache>/susfactor-v1/encoder/{config,model.safetensors,tokenizer.json}
            <cache>/susfactor-v1/head.pt
    Backend: backend: "sdk"

  onnx (recommended for production / latency-sensitive use):
    Repo: 0dinai/susfactor-e5-large-onnx  (gated — requires HF access)
    Layout: <cache>/susfactor-v1/onnx/model.onnx
            <cache>/susfactor-v1/tokenizer.json
            <cache>/susfactor-v1/tokenizer_config.json
    Backend: backend: "sdk-onnx"

Usage
-----
    # Provision ONNX model (recommended):
    uv run python scripts/provision_model.py --onnx

    # Provision torch model:
    uv run python scripts/provision_model.py

    # Provision both:
    uv run python scripts/provision_model.py --onnx && uv run python scripts/provision_model.py

    # Check what's present:
    uv run python scripts/provision_model.py --check
    uv run python scripts/provision_model.py --onnx --check
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MODEL_VERSION = "susfactor-v1"
DEFAULT_CACHE_ENV = "SIGNATURE_SDK_MODEL_CACHE"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "signature-sdk" / "models"

TORCH_REPO_ID = "0dinai/susfactor-e5-large"
TORCH_REQUIRED_FILES = (
    "encoder/config.json",
    "encoder/model.safetensors",
    "encoder/tokenizer.json",
    "head.pt",
)

ONNX_REPO_ID = "0dinai/susfactor-e5-large-onnx"
ONNX_REQUIRED_FILES = (
    "onnx/model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
)


def resolve_cache_dir(override: str | None) -> Path:
    if override:
        return Path(override).expanduser()
    env = os.environ.get(DEFAULT_CACHE_ENV)
    if env:
        return Path(env).expanduser()
    return DEFAULT_CACHE_DIR


def check(cache_dir: Path, onnx: bool = False) -> bool:
    """Return True if all required model files are present."""
    model_dir = cache_dir / MODEL_VERSION
    required = ONNX_REQUIRED_FILES if onnx else TORCH_REQUIRED_FILES
    label = "ONNX" if onnx else "torch"
    missing = [f for f in required if not (model_dir / f).exists()]
    if missing:
        print(f"[check] {label} — missing files in {model_dir}:")
        for f in missing:
            print(f"  - {f}")
        return False
    print(f"[check] {label} — all model files present in {model_dir}")
    return True


def _resolve_token(repo_id: str, token: str | None) -> str | None:
    if not token:
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        )

    if not token:
        hf_token_file = Path.home() / ".cache" / "huggingface" / "token"
        if hf_token_file.exists():
            print(f"INFO: Using stored HuggingFace token from {hf_token_file}")
        else:
            print(
                f"INFO: No HuggingFace token found. The model {repo_id!r} is gated.\n"
                "      Options:\n"
                "        1. Run: huggingface-cli login\n"
                "        2. Set: export HF_TOKEN=hf_...\n"
                "        3. Pass: --token hf_..."
            )
            if sys.stdin.isatty():
                token = input("HuggingFace token (or press Enter to abort): ").strip() or None
                if not token:
                    print("Aborted.")
                    sys.exit(1)
            else:
                print("Aborted (non-interactive).")
                sys.exit(1)

    return token


def provision(cache_dir: Path, token: str | None, onnx: bool = False) -> None:
    """Download the model from HuggingFace into cache_dir/susfactor-v1/."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("       Run: uv run pip install huggingface_hub")
        sys.exit(1)

    repo_id = ONNX_REPO_ID if onnx else TORCH_REPO_ID
    model_dir = cache_dir / MODEL_VERSION
    model_dir.mkdir(parents=True, exist_ok=True)

    token = _resolve_token(repo_id, token)

    print(f"Downloading {repo_id} → {model_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
        token=token,
        ignore_patterns=["*.md", ".gitattributes"],
    )

    print("\nProvisioning complete. Verifying...")
    ok = check(cache_dir, onnx=onnx)
    if not ok:
        print("\nWARNING: Some files are missing. The download may have been partial.")
        sys.exit(1)
    print("\nModel is ready.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provision the SusFactor model cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  provision ONNX (recommended):  uv run python scripts/provision_model.py --onnx\n"
            "  provision torch:               uv run python scripts/provision_model.py\n"
            "  check both:                    uv run python scripts/provision_model.py --check\n"
            "                                 uv run python scripts/provision_model.py --onnx --check"
        ),
    )
    parser.add_argument("--cache-dir", help="Override model cache directory.")
    parser.add_argument("--token", help="HuggingFace API token (or set HF_TOKEN env var).")
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Provision the ONNX model (0dinai/susfactor-e5-large-onnx) instead of torch.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check if model files are present; do not download.",
    )
    args = parser.parse_args()

    cache_dir = resolve_cache_dir(args.cache_dir)
    print(f"Model cache: {cache_dir}")

    if args.check:
        ok = check(cache_dir, onnx=args.onnx)
        sys.exit(0 if ok else 1)

    if check(cache_dir, onnx=args.onnx):
        print("Model already provisioned.")
        return

    provision(cache_dir, token=args.token, onnx=args.onnx)


if __name__ == "__main__":
    main()
