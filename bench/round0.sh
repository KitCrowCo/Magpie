#!/usr/bin/env bash
# round0.sh - one benchmark session on this node: machine card, fidelity set, speed by prompt length, wall power while it runs,
# all into results/round0/<node>/<date-time>/, with summary.md made by round0_summary.py at the end.
# Run it on the node itself, from Magpie's bench/ folder, in tmux (a slow model takes hours). Settings come from the environment:
#   MODEL       required: the model name as the server lists it (/api/tags, or the engine's /v1/models)
#   ENGINE      an engine name in engines.json (colibri models): requests go to $URL/engines/$ENGINE over the OpenAI wire; empty = Magpie's own llama.cpp models
#   URL         Magpie on this node (default http://127.0.0.1:8790)
#   NODE_NAME   the name in the results path and the power readings (default: this machine's hostname)
#   NUM_CTX     llama.cpp models: the context the node's other users send (otherwise the model reloads first)
#   LENGTHS     prompt lengths for the speed run (default 128,2048,8192); REPEAT runs per length (default 2: cold, then warm)
#   FID_TOKENS  answer length for the fidelity set (default 64); PERF_TOKENS for the speed run (default 128)
#   STEPS       which parts, in order (default "card fidelity perf")
#   COLD=1      drop the page cache before each part (needs root): the first run reads the model from disk, as after a reboot
#   SENSORS     power meters, space-separated NAME=URL (see power_log.py); logged every 5 s while the session runs
#   NOTE        free text kept with every record: what differs from other runs (mirror on, kv-slots, BIOS GPU memory ...)
#   MAGPIE_CONTAINER  Magpie's container name, for the engine versions in the card (default magpie)
set -euo pipefail
cd "$(dirname "$0")"
: "${MODEL:?set MODEL to the model name the server lists}"
URL="${URL:-http://127.0.0.1:8790}"; NODE_NAME="${NODE_NAME:-$(hostname)}"; STEPS="${STEPS:-card fidelity perf}"
LENGTHS="${LENGTHS:-128,2048,8192}"; REPEAT="${REPEAT:-2}"; FID_TOKENS="${FID_TOKENS:-64}"; PERF_TOKENS="${PERF_TOKENS:-128}"; MAGPIE_CONTAINER="${MAGPIE_CONTAINER:-magpie}"
STAMP="$(date +%Y-%m-%d_%H%M%S)"; DIR="results/round0/$NODE_NAME/$STAMP"; mkdir -p "$DIR"
if [ -n "${ENGINE:-}" ]; then WIRE=openai; TARGET="$URL/engines/$ENGINE"; else WIRE=ollama; TARGET="$URL"; fi
NOTE_ALL="${NOTE:-}"; [ "${COLD:-0}" = "1" ] && NOTE_ALL="$NOTE_ALL | page cache dropped before each part"
echo "== round0: $NODE_NAME, model $MODEL via $TARGET ($WIRE), steps: $STEPS -> $DIR"

if [ -n "${SENSORS:-}" ]; then
  python3 power_log.py --every 5 --out "$DIR/power.jsonl" $SENSORS & PLOG=$!
  trap 'kill $PLOG 2>/dev/null || true' EXIT
fi
cold() { if [ "${COLD:-0}" = "1" ]; then sync && echo 3 > /proc/sys/vm/drop_caches && echo "   page cache dropped"; fi; }
move() { mv "results/$1.jsonl" "$DIR/$1.jsonl"; }
ctx_args=(); [ -n "${NUM_CTX:-}" ] && ctx_args=(--num-ctx "$NUM_CTX")

for step in $STEPS; do
  case "$step" in
    card)
      echo "-- machine card"
      MODEL="$MODEL" ENGINE="${ENGINE:-}" URL="$URL" CONTAINER="$MAGPIE_CONTAINER" NOTE="$NOTE_ALL" python3 - > "$DIR/card.json" <<'PY'
import json, os, platform, subprocess, urllib.request
def sh(*c):
    try: return subprocess.run(c, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired): return None
def get(u):
    try:
        with urllib.request.urlopen(u, timeout=30) as r: return json.loads(r.read())
    except Exception as e: return {"error": f"{type(e).__name__}: {e}"}
u, c = os.environ["URL"], os.environ["CONTAINER"]
print(json.dumps({"node": platform.node(), "model": os.environ["MODEL"], "engine": os.environ["ENGINE"] or None, "note": os.environ["NOTE"], "kernel": platform.release(),
                  "cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), None), "mem_total_kb": next((int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemTotal")), None),
                  "ram_modules": "\n".join(l.strip() for l in (sh("dmidecode", "-t", "memory") or "").splitlines() if l.strip().split(":")[0] in ("Size", "Type", "Speed", "Configured Memory Speed", "Manufacturer", "Part Number", "Locator", "Rank")) or None, "drives": sh("lsblk", "-d", "-o", "NAME,MODEL,SIZE,ROTA,TRAN"),
                  "colibri_ref": sh("docker", "exec", c, "cat", "/opt/colibri/COLIBRI_REF.txt"), "llama_server": sh("docker", "exec", c, "sh", "-c", "llama-server --version 2>&1 | tail -2"),
                  "vulkan": sh("docker", "exec", c, "sh", "-c", "vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverInfo|apiVersion' | head -6"),
                  "magpie_sysinfo": get(f"{u}/api/optimizer/sysinfo"), "magpie_state": get(f"{u}/fleet/state"), "engines": get(f"{u}/api/engines/status")}, indent=1))
PY
      echo "   $DIR/card.json (ram_modules needs root; missing tools leave a field empty)";;
    fidelity)
      cold; L="r0-$NODE_NAME-fid-$STAMP"; echo "-- fidelity set ($FID_TOKENS tokens per answer)"
      python3 bench.py fidelity --wire "$WIRE" --url "$TARGET" --model "$MODEL" ${ctx_args[@]+"${ctx_args[@]}"} --max-tokens "$FID_TOKENS" --label "$L" --note "$NOTE_ALL" --info-url "$URL/fleet/state"; move "$L";;
    perf)
      cold; L="r0-$NODE_NAME-perf-$STAMP"; echo "-- speed: lengths $LENGTHS, $REPEAT runs each, $PERF_TOKENS tokens per answer"
      python3 bench.py perf --wire "$WIRE" --url "$TARGET" --model "$MODEL" ${ctx_args[@]+"${ctx_args[@]}"} --lengths "$LENGTHS" --repeat "$REPEAT" --max-tokens "$PERF_TOKENS" --label "$L" --note "$NOTE_ALL" --info-url "$URL/fleet/state"; move "$L";;
    *) echo "unknown step: $step (card, fidelity, perf)"; exit 1;;
  esac
done
python3 round0_summary.py "$DIR" | tee "$DIR/summary.md"
echo "== done: $DIR"
