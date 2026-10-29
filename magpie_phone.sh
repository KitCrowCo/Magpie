#!/data/data/com.termux/files/usr/bin/bash
# Magpie phone node - Termux setup + run for a q4 4B model (CPU) on an 8 GB Android phone.
# One-time:  bash magpie_phone.sh setup
# Run it:    bash magpie_phone.sh run      (auto-restarts, holds a wake lock)
# No Docker. The phone serves Magpie's Ollama-compatible API on :8790 for another Portal to point at.
set -e

# ---- Config ----
MAGPIE_GIT="${MAGPIE_GIT:-PUT_YOUR_GIT_URL_HERE}"
MAGPIE_DIR="${MAGPIE_DIR:-$HOME/magpie}"
MODELS_DIR="${MODELS_DIR:-$HOME/magpie-models}"
LLAMA_SRC="${LLAMA_SRC:-$HOME/llama.cpp}"
PORT="${PORT:-8790}"
THREADS="${THREADS:-4}"
RESERVE_GB="${RESERVE_GB:-4}"

MODEL_REPO="${MODEL_REPO:-unsloth/gemma-3-4b-it-GGUF}"
MODEL_FILE="${MODEL_FILE:-gemma-3-4b-it-Q4_K_M.gguf}"

GGUF_DIR="$MODELS_DIR/gguf_local"
MODEL_SUBDIR="$GGUF_DIR/${MODEL_REPO##*/}"
BIN="$LLAMA_SRC/build/bin/llama-server"
VENV_DIR="$MAGPIE_DIR/venv"

setup() {
  echo "== upgrading system packages to fix library symbol link mismatches =="
  pkg update -y
  pkg upgrade -y

  echo "== installing required build and runtime packages =="
  pkg install -y python python-pip clang cmake jsoncpp git wget binutils termux-api python-numpy

  echo "== cloning magpie code =="
  if [ ! -d "$MAGPIE_DIR" ]; then
    [ "$MAGPIE_GIT" = "PUT_YOUR_GIT_URL_HERE" ] && { echo "Set MAGPIE_GIT or put code in $MAGPIE_DIR, then re-run."; exit 1; }
    git clone "$MAGPIE_GIT" "$MAGPIE_DIR"
  fi

  echo "== setting up python virtual environment & deps (Pydantic v1 compatible stack) =="
  cd "$MAGPIE_DIR"
  python3 -m venv venv --system-site-packages --clear
  ./venv/bin/python -m pip install --upgrade pip
  # Pin fastapi to an older version compatible with pydantic v1
  ./venv/bin/pip install "fastapi<=0.95.2" uvicorn httpx "pydantic<2.0" "huggingface_hub<0.31"
  ./venv/bin/pip install --no-deps gguf
  cd "$HOME"

  echo "== building llama-server (CPU, aarch64) =="
  [ -d "$LLAMA_SRC" ] || git clone https://github.com/ggml-org/llama.cpp "$LLAMA_SRC"
  cd "$LLAMA_SRC"
  rm -rf build
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
  cmake --build build --target llama-server -j4
  cd "$HOME"

  echo "== downloading model =="
  mkdir -p "$MODEL_SUBDIR" 
"$MODELS_DIR/cache" "$MODELS_DIR/loras"
  "$VENV_DIR/bin/huggingface-cli" download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$MODEL_SUBDIR"

  echo "== setup complete! run it with: bash $0 run =="
}

run() {
  export MODELS_DIR SERVER_PORT="$PORT" SERVER_HOST=0.0.0.0
  export LLAMA_SERVER_BIN="$BIN"
  export LLAMA_RESERVE_GB="$RESERVE_GB"
  export LLAMA_THREADS="$THREADS" LLAMA_THREADS_BATCH="$THREADS"
  mkdir -p "$GGUF_DIR" "$MODELS_DIR/cache"
  
  [ -x "$BIN" ] || { echo "llama-server not found at $BIN - run 'bash $0 setup' first."; exit 1; }
  [ -d "$VENV_DIR" ] || { echo "Virtual environment not found at $VENV_DIR - run 'bash $0 setup' first."; exit 1; }
  
  cd "$MAGPIE_DIR"
  source "$VENV_DIR/bin/activate"

  termux-wake-lock 2>/dev/null || echo "(no termux-wake-lock - install Termux:API app if Android sleeps the node)"
  trap 'termux-wake-unlock 2>/dev/null || true' EXIT
  
  while true; do
    echo "[run] starting magpie $(date)"
    python main.py 2>&1 | tee -a "$MODELS_DIR/magpie.log"
    echo "[run] magpie exited, restarting in 5s"
    sleep 5
  done
}

case "${1:-}" in
  setup) setup ;;
  run)   run ;;
  *) echo "usage: bash $0 {setup|run}"; exit 1 ;;
esac