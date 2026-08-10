#!/usr/bin/env bash
#
# demo_litellm_shield.sh — live-demo the SusFactor guardrail running
# on-prem inside a local LiteLLM proxy.
#
# This script is the presenter's front end. Every unit of actual work lives in
# scripts/demo_step.sh, which demo.tape also drives, one subcommand at a time,
# so the recorded demo and the live demo cannot drift apart. All this script
# owns is the ordering of the beats and the pauses between them.
#
# Narrates, step by step, the flow documented in README.md:
#   1. Bring the proxy up in flag mode (config-flag.yaml, mode: during_call,
#      enforcement: flag) and show that scoring happens in-process via
#      ONNX Runtime against a local model cache — no external classifier
#      API call, only the actual chat completion leaves the box.
#   2. Call POST /v1/chat/completions with a benign prompt, then an
#      adversarial prompt, back to back, so the score contrast is obvious
#      on screen. In flag mode both return HTTP 200; the adversarial one
#      is annotated via the X-SusFactor-Decision response header instead
#      of being blocked, and the guardrail ran in parallel with the live
#      model call so it added ~0 perceived latency.
#   3. Switch the proxy to block mode (config-block.yaml, mode: pre_call,
#      enforcement: block) and re-send the same two prompts. The
#      adversarial prompt now gets rejected with HTTP 400 before it ever
#      reaches OpenRouter — and comes back dramatically faster, since the
#      decision is a pure on-prem inference call instead of a round trip
#      to a hosted LLM. The benign prompt still passes through untouched,
#      proving the block is selective.
#   4. Restore the proxy to flag mode (its resting/default POC state)
#      regardless of which steps ran, and print a summary table of the
#      elapsed-ms numbers observed, next to the published on-prem
#      classifier benchmark for context.
#
# Meant to be run by a presenter live on a call/screenshare — it pauses
# between steps so you can narrate. Runs entirely against the local
# docker-compose stack (http://localhost:4000); nothing here talks to a
# remote cluster.
#
# Usage:
#   scripts/demo_litellm_shield.sh [--no-pause] [--non-interactive] [--skip-block-demo]
#
# Examples:
#   scripts/demo_litellm_shield.sh                        # full demo, narrated
#   scripts/demo_litellm_shield.sh --skip-block-demo       # flag-mode acts only
#   DEMO_NOPAUSE=1 scripts/demo_litellm_shield.sh          # dry run, no pauses
#   scripts/demo_litellm_shield.sh --no-pause              # same, via flag
#
# Env vars:
#   OPENROUTER_API_KEY   (required) routes chat completions through
#                         OpenRouter. See README.md, Quick start step 2.
#   LITELLM_MASTER_KEY    Bearer token clients use against the proxy
#                         (default: sk-1234, matching docker-compose.yml)
#   MODEL_CACHE_DIR /
#   SIGNATURE_SDK_MODEL_CACHE
#                         local model cache dir bind-mounted read-only into
#                         the container (default: ~/.cache/signature-sdk/models)
#   DEMO_STATE_FILE       where scripts/demo_step.sh keeps per-run state
#                         (default: /tmp/litellm-shield-demo/state.env)
#   DEMO_NOPAUSE          if set to any non-empty value, skip the
#                         "Press Enter to continue" pauses (same effect as
#                         passing --no-pause)
#   DEMO_NONINTERACTIVE  if set to any non-empty value, replace the
#                         "Press Enter to continue" pauses with a 5-second
#                         countdown instead of waiting on keyboard input
#                         (same effect as passing --non-interactive).
#                         Retained for scripted use, but no longer has a
#                         caller: demo.tape now drives scripts/demo_step.sh
#                         directly and paces itself with VHS Sleep/Wait, so
#                         this mode is a candidate for removal.
#
# Requires: curl, jq, docker (with the compose plugin) on PATH; a
# provisioned model cache (scripts/provision_model.py); the litellm-shield
# image already built (docker compose build, or let compose build it).
#
# Does NOT stop or tear down the docker compose stack on exit — it's left
# running as a standing dev proxy, in flag mode, matching repo convention.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_STEP="${SCRIPT_DIR}/demo_step.sh"

RED=$'\033[0;31m'
YELLOW=$'\033[0;33m'
RESET=$'\033[0m'

die() {
  echo "${RED}✗ $*${RESET}" >&2
  exit 1
}

info() {
  echo "${YELLOW}→ $*${RESET}"
}

# --- argument parsing -------------------------------------------------

NOPAUSE="${DEMO_NOPAUSE:-}"
NONINTERACTIVE="${DEMO_NONINTERACTIVE:-}"
SKIP_BLOCK_DEMO=""

for arg in "$@"; do
  case "${arg}" in
    --no-pause)
      NOPAUSE="1"
      ;;
    --non-interactive)
      NONINTERACTIVE="1"
      ;;
    --skip-block-demo)
      SKIP_BLOCK_DEMO="1"
      ;;
    *)
      die "Unrecognized argument: ${arg} (expected --no-pause, --non-interactive, and/or --skip-block-demo)"
      ;;
  esac
done

pause() {
  local prompt="$1"
  if [ -n "${NOPAUSE}" ]; then
    return 0
  fi
  if [ -n "${NONINTERACTIVE}" ]; then
    local n
    for n in 5 4 3 2 1; do
      printf '\r%s — continuing in %ds... ' "${prompt}" "${n}"
      sleep 1
    done
    printf '\r%s — continuing.          \n' "${prompt}"
    return 0
  fi
  read -rp "${prompt} — press Enter to continue..." _
}

# --- steps ------------------------------------------------------------
#
# Each beat is a subprocess so that this script and demo.tape run byte-
# identical code. Under set -e a failing beat aborts the demo, leaving the
# proxy on whichever config it had reached — the same behaviour as before, and
# deliberately not wrapped in a restore-on-exit trap, which would also fire on
# Ctrl-C and silently recreate the container out from under the presenter.

step() {
  "${DEMO_STEP}" "$@"
}

# --- main ----------------------------------------------------------------

main() {
  [ -x "${DEMO_STEP}" ] || die "Missing or non-executable step script: ${DEMO_STEP}"

  step preflight

  step up flag
  pause "Proxy is up in flag mode"

  step call flag-benign
  pause "Benign prompt scored (flag mode)"

  step call flag-adversarial
  pause "Adversarial prompt scored (flag mode)"

  if [ -z "${SKIP_BLOCK_DEMO}" ]; then
    step up block
    pause "Proxy is up in block mode"

    step call block-adversarial
    pause "Adversarial prompt rejected (block mode)"

    step call block-benign
    pause "Benign prompt passed through (block mode)"
  else
    info "Skipping block-mode acts (--skip-block-demo)"
  fi

  step restore

  step summary
}

main
