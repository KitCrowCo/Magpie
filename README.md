# Magpie

A stateful supervisor around `llama-server` (llama.cpp). It speaks the Ollama wire format (`/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/pull`, `/api/ps`), so anything that talks to Ollama can talk to it. It owns the llama-server process, and on every request it plans the launch flags from what the machine actually has. It does not apply one fixed config.

What it does that plain llama-server or Ollama do not:

- **Plans every launch from the machine.** The supervisor probes CPU, GPU, carve-out, GTT, TTM limit, and RAM. It reads the model's real architecture from the GGUF header, including attention-only layer counts for hybrid models and MTP heads. It then searches context size, KV precision, GPU layers and ubatch in the goal order you pick (default: context, then quality, then speed).
- **Keeps the model's state between conversations.** The KV cache is saved to disk after idle and restored on reload (a "ring" of the newest N contexts).
- **Learns from failures without poisoning itself.**
  - Memory-class crashes lower the GPU budget for that model. GPU kernel faults are only counted.
  - Everything learned resets automatically when the hardware or the rules file changes.
  - If llama-server dies before the first token, the supervisor replans and retries up to three times.
- **Measures and self-corrects.** After each load it compares the real VRAM+GTT taken with its own estimate and keeps a correction factor.

Files:

| File | Role |
|---|---|
| `main.py` | the supervisor (FastAPI). Ollama-compatible API, process control, KV ring, retries, diagnostic page at `/chat` |
| `optimizer.py` | the planner. Probes the node, resolves constraints, searches launch params, keeps learned state |
| `optimizer_rules.json` | all hardware knowledge, fit constants and experiments. **Edit this, not the code.** |
| `Dockerfile`, `docker-compose.yml`, `requirements.txt` | Linux/ROCm container |
| `run_mac.sh` | native macOS launcher (Docker on macOS cannot reach Metal) |

## Linux / AMD APU (Docker)

1. Put GGUF models in `${TEXT_MODELS}/gguf_local/` (any depth), or keep using an Ollama store at `${TEXT_MODELS}` (`manifests/` + `blobs/`). Both appear in one list.
2. `docker compose up -d --build`, then `docker logs -f llama-proxy`. Startup prints the `[OPT] node:`, `tags:`, `rules matched:`, `experiments enabled:` and `models indexed:` lines.
3. Open `http://<node>:8790/chat`, the diagnostic chat page with plan, sysinfo, crash, log and cache buttons.

ROCm environment variables come from the rules (`HSA_OVERRIDE_GFX_VERSION`, `HSA_ENABLE_SDMA`, `GGML_CUDA_ENABLE_UNIFIED_MEMORY`, `HIP/ROCR_VISIBLE_DEVICES`). **A value set in `docker-compose.yml` wins over the rules**, so leave them out of compose unless you are overriding on purpose.

## macOS / Apple silicon (native)

1. Install the tools:
   - `brew install llama.cpp`
   - Check the build: `llama-server --help | grep -E "load-mode|spec-type"`. If `load-mode` is missing, run `brew upgrade llama.cpp`. If `spec-type` is missing, don't enable `mtp_spec`.
2. Set up the folder:
   - Put this folder anywhere.
   - `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
3. Put models in `~/llama-models/gguf_local/`.
4. Raise the GPU wired limit after every reboot:
   - 16 GB Mac: `sudo sysctl iogpu.wired_limit_mb=14336`.
   - 32 GB Mac: about `28672`.
   - If macOS starts stuttering, go 1024 lower.
5. `./run_mac.sh`, then open `http://localhost:8790/chat`.

Checks:

- `curl -s localhost:8790/api/optimizer/sysinfo`: `vram` should equal the wired limit (14336 MB is about 15.0e9).
- `curl -s localhost:8790/api/optimizer/plan -d '{"model":"<name>"}'`: on a model that fits, `ngl` should be 99 (all layers on the GPU).

`run_mac.sh` caps context at 32K (`LLAMA_BOUNDS`) because that box is speed-first. Inside that bound the planner picks the largest context that fits, then the best KV precision.

## Goals, modes, bounds (how the planner decides)

- **Goals** (`LLAMA_GOALS`, or request `constraints.goals`) set the strict priority order. `ctx,quality,speed` means: largest context first, then best KV precision at that context, then the fastest layer split. Use `speed,quality,ctx` for a speed-first box.
- **Modes** (`LLAMA_OPTIMIZE`, or request `options.optimize`):

  | Mode | Behavior |
  |---|---|
  | `UPPER` (default) | given values are maxima |
  | `LOWER` | given values are minima |
  | `SET` | given request/env values are pinned |
  | `ON` | ignore given values entirely |
  | `OFF` | no planning; given values passed as-is |

- **Bounds** (`LLAMA_BOUNDS`, or request `constraints.bounds`): explicit `[lo, hi]` per param, e.g. `{"ctx":[8192,32768],"kv_cache_type":["q8_0","f16"]}`.
- **Budget:**
  - `LLAMA_RESERVE_GB` is RAM kept back for the OS.
  - `LLAMA_MODELS_EXPECTED` splits memory for several resident models.
  - `LLAMA_BUDGET_GB` hard-caps this model's share.
  - The sDAG/portal can send any of these per request under `constraints`.
- **Reuse:** a running server is reused whenever its config sits inside the new request's bounds. Reloads happen only when something actually conflicts.

## Experiments (switchable without a rebuild)

The registry is the `experiments` section of `optimizer_rules.json`. Each entry records status, what it does and what is still open. The current state is at `GET /api/optimizer/experiments` and on the `/chat` page.

- **Node-wide:** `LLAMA_EXPERIMENTS=mtp_spec,-tool_calls` in compose or the environment, then restart. A leading `-` disables a default-on experiment.
- **Per request:** `"constraints": {"experiments": ["no_reuse"]}`, or the experiments box on `/chat`. llama-server reloads with that config, and nothing is rebuilt.

| Experiment | Default | Purpose |
|---|---|---|
| `tool_calls` | on | `--jinja`, so OpenAI-style `tools` / `tool_calls` work (needed by the code agent) |
| `mtp_spec` | off | MTP self-speculative decoding for models tagged `mtp` (tested: 2-7 t/s on 27B / 780M) |
| `no_reuse` | off | tests whether GPU hangs on hybrid models come from reusing saved/cached state |
| `llama_fit` | off | lets llama.cpp choose GPU layers itself, to compare with the planner |
| `unified_unset` | off | removes `GGML_CUDA_ENABLE_UNIFIED_MEMORY` entirely (tests whether `=0` still enables it) |
| `stall_watchdog` | off | kills and retries a server that produces no output for 900 s |
| `verbose_load` | off | extra load logging for calibration |

To add an experiment, add a registry entry plus a rule with `"experiment": "<name>"`. Rules can set `args`, `env` (null = force unset), `fit`, `drop`, `payload` and `features`.

## Extension points already wired

- **Tool calling (code agent):**
  - `/api/chat` accepts Ollama-style `tools` and returns `message.tool_calls` (streamed calls are assembled and emitted once, before the final chunk).
  - Assistant `tool_calls` and `role: tool` messages in the history are converted to llama-server's OpenAI shape and are never merged together.
  - `format: "json"` or a JSON schema maps to `response_format`.
- **LoRA:**
  - `options.lora_paths: ["file.gguf"]` (files in `${MODELS_DIR}/loras/`) loads adapters at launch.
  - `POST /api/lora/set {"adapters":[{"id":0,"scale":0.7}]}` changes scales live (0 = off) without a reload.
  - Training adapters is outside this service; it only serves them.
- **Portal manager:**
  - Set `PORTAL_SERVER`, `PORTAL_TOKEN` and `NODE_ID`. On startup the node posts its hardware profile and model index to `<portal>/module/ai_tools/node_hello`.
  - Whatever constraints the portal answers with become defaults below request constraints. `POST /api/optimizer/refresh` re-polls.
- **Model facts for AI Calc:** `/api/show` returns `params_b`, `bpw` and `file_type` from the tensor table, so effective bits-per-weight is right even when the quant label is wrong or unknown (e.g. NVFP4 files labeled Q8_0).

## Diagnostics

| Endpoint | Shows |
|---|---|
| `/chat` | test chat with request/response inspection and all buttons below |
| `/api/optimizer/plan` (POST, dry run) | what would launch for a model + options, without starting anything |
| `/api/optimizer/last_crash` | exit code, classification (`memory` / `gpu_fault` / `stall` / `unknown`), params, env, last 40 log lines |
| `/api/optimizer/log?n=150` | live tail of llama-server output |
| `/api/hang_log` | spawn/crash journal (trimmed automatically) |
| `/api/optimizer/forget` | clear learned state (all, or `?model=<name>`) |
| `/api/optimizer/clear_ring` | delete saved KV contexts |
| `/api/optimizer/sysinfo?refresh=true` | re-probe the node |

## Control channel

Requests to `/api/chat` and `/api/generate` are tracked under their `X-Request-Id` header (the portal sends `<job id>.<random>`; a random id otherwise) from arrival to their last byte. These endpoints answer at once, whatever inference is running:

| Endpoint | Does |
|---|---|
| `GET /control/status` | node id, uptime; requests in progress (id, kind, model, session, phase `loading` / `running` / `cancelling`, age); the latest 20 finished (outcome `served` / `cancelled` / `failed`, seconds); totals; llama-server (alive, pid, model, ctx, kv, in flight, RSS); memory (RAM, available, swap used, GPU VRAM+GTT used); engines; load average |
| `POST /control/cancel {"request_id"}` | drops one request, or every request of one job (`job_x` cancels `job_x.*`). A stream ends with a final `{"done": true, "done_reason": "cancelled"}` line; a non-streaming request answers 499. The connection to llama-server is closed, which makes it stop generating. |
| `POST /control/kill {"scope"}` | `requests` (default): drops every request. `server`: also kills llama-server at once (SIGKILL, no context save); the next request starts it again. |

The portal (AI Manager, `magpie` profile) uses status while a stream is silent - a request the node no longer has ends the portal's call at once instead of after the read timeout - and cancel when a job is killed. Not covered yet: requests to other engines through `/engines/<name>/...` (colibri).

## Known open items (tracked, with the current workaround)

- **Intermittent GPU hangs on gfx1103.**
  - Hangs happen mostly right after a KV restore or a partial-prefix reuse.
  - Workaround: the automatic retry, which usually succeeds on the next launch.
  - To narrow it down: `no_reuse` experiment.
- **`GGML_CUDA_ENABLE_UNIFIED_MEMORY=0` semantics are unverified.** It may still enable managed memory. The tested rules use `"0"`. Test with `unified_unset`.
- **The planner's fit constants are estimates.** `measure()` corrects GPU-side estimates per model and KV/FA combination after each load. Host-side estimates are not measured yet.
- **The hybrid KV layer count relies on the GGUF key `full_attention_interval`, or per-layer `head_count_kv` arrays.** Check `/api/show` `kv_layers` for new architectures.
- **Quantized V cache requires flash attention in llama.cpp.** The planner forces flash attention on for quantized KV.
