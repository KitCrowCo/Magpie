#!/usr/bin/env bash
# run_mac.sh - native macOS (Apple silicon) launcher for Magpie. Docker on macOS cannot reach Metal, so this runs directly.
# One-time: brew install llama.cpp python3 ; python3 -m venv venv ; source venv/bin/activate ; pip install -r requirements.txt
# Per boot (the GPU wired limit resets on reboot): sudo sysctl iogpu.wired_limit_mb=14336   (16 GB Mac; ~28672 on a 32 GB Mac)
set -euo pipefail
cd "$(dirname "$0")"
[ -d venv ] && source venv/bin/activate
export MODELS_DIR="${MODELS_DIR:-$HOME/llama-models}"
export SERVER_PORT="${SERVER_PORT:-8790}"
export SERVER_HOST="${SERVER_HOST:-127.0.0.1}"          # 0.0.0.0 to reach it from other machines
export KV_MIN_FREE_GB="${KV_MIN_FREE_GB:-5}"
export LLAMA_LOAD_MODE="${LLAMA_LOAD_MODE:-auto}"       # mmap where supported; the Linux default "none" copies weights
export LLAMA_RESERVE_GB="${LLAMA_RESERVE_GB:-3}"
DEFAULT_BOUNDS='{"ctx":[8192,32768]}'                  # speed-first box: cap context at 32K
export LLAMA_BOUNDS="${LLAMA_BOUNDS:-$DEFAULT_BOUNDS}"
export LLAMA_EXPERIMENTS="${LLAMA_EXPERIMENTS:-}"        # e.g. mtp_spec to try MTP self-speculation
mkdir -p "$MODELS_DIR/gguf_local"
echo "GPU wired limit now: $(sysctl -n iogpu.wired_limit_mb) MB (0 = macOS default ~2/3 of RAM)"
exec python3 main.py
