# llama_direct main.py
import os
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")  # huggingface_hub reads this at import time - keep above the import
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")        # classic HTTP downloads: resumable .incomplete files the pull progress can measure (Xet writes the file only when done)
import sys, json, re, glob, time, uuid, shutil, platform, subprocess, asyncio, collections, threading, contextvars, uvicorn
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, Any, List, Optional
from contextlib import asynccontextmanager, contextmanager
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse
import httpx
from huggingface_hub import hf_hub_download, HfApi
import optimizer as optim  # NOTE: aliased - main.py already defines opt() (the options reader)
from passive_worker import start_passive_learning_worker
import aux_servers as aux
import engines
import fleet, fleet_tests   # Fleet, the node side: journal, state, actions, tests (the portal's Fleet tool reads and steers them)

# --- Config ---

MODELS_DIR = os.getenv("MODELS_DIR", "/root/.ollama/models")
LOCAL_DIR = os.path.join(MODELS_DIR, "gguf_local")   # real files (pulled) - nothing here is ever auto-deleted
LORA_DIR = os.path.join(MODELS_DIR, "loras")
CACHE_DIR = os.path.join(MODELS_DIR, "cache")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8790)) # FastAPI supervisor port (exposed to Portal Server)
DRAFT_ENDPOINTS = [e.strip().rstrip("/") for e in os.getenv("DRAFT_ENDPOINTS", f"""http://127.0.0.1:{SERVER_PORT}""").split(",") if e.strip()]   # supervisors whose draft slots receive this node's passive samples at /api/draft/adapt - this node by default; more nodes comma-separated
INTERNAL_LLAMA_PORT = int(os.getenv("INTERNAL_LLAMA_PORT", 8080))  # Primary llama-server internal port
LLAMA_SERVER_URL = f"http://127.0.0.1:{INTERNAL_LLAMA_PORT}"
LLAMA_SERVER_BIN = os.getenv("LLAMA_SERVER_BIN") or shutil.which("llama-server") or "/app/llama-server"
KV_RING_SIZE = int(os.getenv("KV_RING_SIZE", 3))              # saved contexts kept on disk
KV_AUTOSAVE_IDLE_S = int(os.getenv("KV_AUTOSAVE_IDLE_S", 45)) # save after this long with no requests
KV_MIN_FREE_GB = float(os.getenv("KV_MIN_FREE_GB", 20))       # never save below this much free disk
DRAIN_S = float(os.getenv("LLAMA_DRAIN_S", 600))               # max wait for in-flight requests before a restart
INFER_TIMEOUT_S = float(os.getenv("LLAMA_INFER_TIMEOUT_S", 1200))  # a non-streaming request with no reply in this long means llama-server is wedged: kill it so the next request starts fresh, instead of hanging until the caller's own timeout (the overnight-ingest wedge)
FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4"}  # llama.h LLAMA_FTYPE ids as recalled; unknown ids show as the number, and bpw (measured) is authoritative either way
CRASH_LOG = os.path.join(MODELS_DIR, "gpu_hang_log.jsonl")
CRASH_LOG_MAX_LINES = 2000
TRAJECTORY_LOG = os.path.join(MODELS_DIR, "draft_trajectories.jsonl")

CURRENT_MODEL: Optional[dict] = None
CURRENT_CTX_SIZE, CURRENT_KV_CACHE, CURRENT_LORAS, CURRENT_SESSION = 0, None, [], "default"
CURRENT_BATCH, CURRENT_UBATCH = 0, 0
CURRENT_PLAN: dict = {}
LLAMA_PROCESS = None
IN_FLIGHT = 0
_AUTOSAVE_TASK = None
_KILLED: set = set()           # pids this supervisor terminated on purpose (restarts) - their exits are not failures
LAST_RESTORE: dict = {}
LAST_CRASH: dict = {}
_OOM_BEFORE = -1    # cgroup oom_kill counter at the last spawn (-1 = not available)
LOG_TAIL = collections.deque(maxlen=200)
NODE_ID, STARTED = os.getenv("NODE_ID") or platform.node(), time.time()
_REQ: dict = {}                            # request id -> the inference request in progress (control channel: /control/status, /control/cancel)
_DONE = collections.deque(maxlen=50)       # the latest finished requests: id, kind, model, outcome, seconds
_COUNTS = {"served": 0, "cancelled": 0, "failed": 0}
RID = contextvars.ContextVar("magpie_request_id", default="")
process_lock = asyncio.Lock()
learning_queue = asyncio.Queue(maxsize=200)

# --- Helpers ---

class StreamThinkParser:
    """Statefully parses split <think> tags from the stream just like native Ollama."""
    def __init__(self):
        self.is_thinking = False
        self.buffer = ""

    def process(self, chunk: str):
        self.buffer += chunk
        reasoning_out = ""
        content_out = ""
        while self.buffer:
            if not self.is_thinking:
                idx = self.buffer.find("<think>")
                if idx != -1:
                    content_out += self.buffer[:idx]
                    self.buffer = self.buffer[idx+7:]
                    self.is_thinking = True
                else:
                    last_lt = self.buffer.rfind("<")
                    if last_lt != -1 and "<think>".startswith(self.buffer[last_lt:]):
                        content_out += self.buffer[:last_lt]
                        self.buffer = self.buffer[last_lt:]
                        break
                    else:
                        content_out += self.buffer
                        self.buffer = ""
                        break
            else:
                idx = self.buffer.find("</think>")
                if idx != -1:
                    reasoning_out += self.buffer[:idx]
                    self.buffer = self.buffer[idx+8:]
                    self.is_thinking = False
                else:
                    last_lt = self.buffer.rfind("<")
                    if last_lt != -1 and "</think>".startswith(self.buffer[last_lt:]):
                        reasoning_out += self.buffer[:last_lt]
                        self.buffer = self.buffer[last_lt:]
                        break
                    else:
                        reasoning_out += self.buffer
                        self.buffer = ""
                        break
        return reasoning_out, content_out

    def flush(self):
        r, c = ("", "")
        if self.is_thinking: r = self.buffer
        else: c = self.buffer
        self.buffer = ""
        return r, c

def split_think(text: str):
    p = StreamThinkParser()
    r, c = p.process(text)
    r2, c2 = p.flush()
    return r + r2, c + c2

def _to_oai_msg(m: dict) -> dict:
    """Ollama-shaped message -> OpenAI-shaped: tool_calls arguments as JSON strings with ids; images (Ollama's base64 list) as image_url content parts, which llama-server reads when the model was started with its --mmproj projector."""
    out = {k: v for k, v in m.items() if k != "images"}
    if m.get("images"): out["content"] = [{"type": "text", "text": m.get("content") or ""}, *({"type": "image_url", "image_url": {"url": i if str(i).startswith("data:") else f"data:image/png;base64,{i}"}} for i in m["images"])]
    if m.get("tool_calls"): out["tool_calls"] = [{"id": c.get("id") or f"call_{i}", "type": "function", "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"] if isinstance(c["function"].get("arguments"), str) else json.dumps(c["function"].get("arguments") or {})}} for i, c in enumerate(m["tool_calls"])]
    return out

def coalesce_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merges sequential plain messages of the same role to prevent template duplication. Tool traffic (role tool, or assistant tool_calls) is never merged."""
    merged = []
    for msg in messages:
        plain = msg.get("role") != "tool" and not msg.get("tool_calls") and not msg.get("images") and not (merged and not isinstance(merged[-1].get("content"), str))   # image messages carry content parts, never merged
        if merged and plain and merged[-1].get("role") == msg.get("role") and not merged[-1].get("tool_calls") and merged[-1].get("role") != "tool": merged[-1]["content"] += "\n\n" + (msg.get("content") or "")
        else: merged.append(_to_oai_msg(msg))
    return merged

def _from_oai_calls(calls: list) -> list:
    """OpenAI tool_calls -> Ollama shape (arguments parsed back to a dict when they are valid JSON)."""
    out = []
    for c in calls or []:
        fn, args = c.get("function", {}), c.get("function", {}).get("arguments", "")
        try: args = json.loads(args) if isinstance(args, str) and args else (args or {})
        except json.JSONDecodeError: pass
        out.append({"id": c.get("id"), "function": {"name": fn.get("name", ""), "arguments": args}})
    return out

def _now(): return datetime.now(timezone.utc).isoformat()
def _fsize(p):
    try: return os.path.getsize(p)
    except OSError: return 0

def _mem_gb(pid: int, field: str = "VmRSS") -> float:
    """A process's resident memory from /proc (VmRSS = now, VmHWM = peak since start). On unified-memory nodes this includes llama-server's GPU buffers. 0 when the process is gone."""
    st = Path(f"/proc/{pid}/status")
    return next((int(l.split()[1]) * 1024 / 1e9 for l in st.read_text().splitlines() if l.startswith(f"{field}:")), 0.0) if st.exists() else 0.0

def opt(body: Dict[str, Any], key: str, default=None):
    """Reads a setting from options first, then the request root."""
    o = body.get("options") if isinstance(body.get("options"), dict) else {}
    return o.get(key, body.get(key, default))

def extract_session(body): return re.sub(r"[^A-Za-z0-9_-]", "_", str(opt(body, "cache_session") or "default"))[:64]
def extract_loras(body): return [os.path.basename(str(p)) for p in (opt(body, "lora_paths") or [])]
def think_on(v): return bool(v) and str(v).lower() not in ("false", "off", "none", "0")

def map_options(options: Dict[str, Any]) -> Dict[str, Any]:
    if not options: return {}
    mapping = {"num_predict": "n_predict", "temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p", "repeat_penalty": "repeat_penalty", "presence_penalty": "presence_penalty", "frequency_penalty": "frequency_penalty", "seed": "seed", "stop": "stop"}
    return {mapping[k]: v for k, v in options.items() if k in mapping and v is not None}

def _log_attempt(event: str, **extra):
    """Append-only spawn/crash journal (gpu_hang_log.jsonl), trimmed to the newest CRASH_LOG_MAX_LINES lines."""
    fleet.event(f"llama.{event}", model=CURRENT_MODEL["name"] if CURRENT_MODEL else None, params=CURRENT_PLAN.get("params", {}), est=CURRENT_PLAN.get("est"), experiments=CURRENT_PLAN.get("experiments"), **extra)
    try:
        p = CURRENT_PLAN.get("params", {})
        with open(CRASH_LOG, "a") as f: f.write(json.dumps({"t": _now(), "event": event, "model": CURRENT_MODEL["name"] if CURRENT_MODEL else None, "params": p, "experiments": CURRENT_PLAN.get("experiments"), "env": CURRENT_PLAN.get("env"), **extra}) + "\n")
        if _fsize(CRASH_LOG) > 2_000_000:
            lines = open(CRASH_LOG).readlines()[-CRASH_LOG_MAX_LINES:]
            open(CRASH_LOG, "w").writelines(lines)
    except OSError as e: print(f"[PROXY] crash log write failed: {e}", flush=True)

# --- Model discovery (Ollama store + local GGUF files, one list) ---

def scan_models() -> List[dict]:
    out, mdir, bdir = [], os.path.join(MODELS_DIR, "manifests"), os.path.join(MODELS_DIR, "blobs")
    for root, _, files in os.walk(mdir):
        for file in files:
            mp = os.path.join(root, file)
            parts = [p for p in os.path.relpath(mp, mdir).split(os.sep) if p not in ("registry.ollama.ai", "library")]
            if len(parts) < 2: continue
            try:
                layer = next(l for l in json.loads(Path(mp).read_text()).get("layers", []) if l.get("mediaType") == "application/vnd.ollama.image.model")
                blob = os.path.join(bdir, layer["digest"].replace(":", "-"))
                if os.path.exists(blob): out.append({"name": parts[-2] if parts[-1] == "latest" else f"{parts[-2]}:{parts[-1]}", "path": blob, "files": [blob], "size": _fsize(blob), "mtime": os.path.getmtime(mp), "source": "ollama_store", "manifest": mp})
            except Exception: continue
    for f in sorted(glob.glob(os.path.join(LOCAL_DIR, "**", "*.gguf"), recursive=True)):
        if "mmproj" in os.path.basename(f).lower(): continue
        shard = re.search(r"-(\d+)-of-\d+\.gguf$", f)
        if shard and int(shard.group(1)) != 1: continue
        files = sorted(glob.glob(re.sub(r"-\d+-of-(\d+)\.gguf$", r"-*-of-\1.gguf", f))) if shard else [f]
        out.append({"name": re.sub(r"-\d+-of-\d+$", "", os.path.splitext(os.path.basename(f))[0]), "path": f, "files": files, "size": sum(_fsize(x) for x in files), "mtime": os.path.getmtime(f), "source": "local", "manifest": None})
    return out

def _tag(n: str) -> str: return n if ":" in n else f"{n}:latest"
def find_model(requested: str) -> Optional[dict]: return next((m for m in scan_models() if _tag(m["name"]) == _tag(requested.strip())), None)

def require_model(requested: str) -> dict:
    m = find_model(requested)
    if not m: raise HTTPException(status_code=404, detail=f"Model not found: {requested}")
    return m

def _tags_entry(m: dict) -> dict:
    up = m["name"].upper()
    ql, ps = re.search(r"(I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32)", up), re.search(r"(\d+(?:\.\d+)?)B", up)
    bpw = round(m["size"] * 8 / (float(ps.group(1)) * 1e9), 2) if ps else None
    return {"name": m["name"], "model": m["name"], "modified_at": datetime.fromtimestamp(m["mtime"], tz=timezone.utc).isoformat(), "size": m["size"], "source": m["source"], "loaded": bool(CURRENT_MODEL and CURRENT_MODEL["name"] == m["name"]), "details": {"format": "gguf", "quantization_level": ql.group(1) if ql else "", "parameter_size": f"{ps.group(1)}B" if ps else "", "bpw_est": bpw}}

def _gv(r, key):
    f = r.fields.get(key)
    if f is None: return None
    if f.types and f.types[0].name == "STRING": return bytes(f.parts[f.data[0]]).decode("utf-8", "ignore")
    v = f.parts[f.data[0]].tolist() if len(f.data) == 1 else [f.parts[i].tolist()[0] for i in f.data]
    return v[0] if isinstance(v, list) and len(v) == 1 else v

@lru_cache(maxsize=32)
def gguf_info(path: str) -> dict:
    """Real architecture + quant read from the GGUF header (needs: pip install gguf) - authoritative, unlike a tag name.
    kv_layers counts only attention layers (hybrid recurrent+attention models such as Qwen3.5+ have far fewer); params/bpw come from the tensor table, so they are right even when file_type is unknown (e.g. NVFP4)."""
    from gguf import GGUFReader
    r = GGUFReader(path)
    arch = _gv(r, "general.architecture")
    g = lambda k: _gv(r, f"{arch}.{k}")
    layers, heads, kvh, emb, hd = g("block_count"), g("attention.head_count"), g("attention.head_count_kv"), g("embedding_length"), g("attention.key_length")
    hd = hd or ((emb // (max(heads) if isinstance(heads, list) else heads)) if emb and heads else None)
    ival = g("full_attention_interval")  # key name as recalled for Qwen3-Next-style hybrids; absent on plain transformers
    kv_layers = sum(1 for x in kvh if x) if isinstance(kvh, list) else (layers // ival if ival and layers else layers)
    kv_sum = (sum(kvh) if isinstance(kvh, list) else kvh * kv_layers) if kvh and layers else None
    n_params = sum(int(t.n_elements) for t in r.tensors)
    size = sum(_fsize(f) for f in glob.glob(re.sub(r"-\d+-of-(\d+)\.gguf$", r"-*-of-\1.gguf", path))) or _fsize(path)
    return {"general.architecture": arch, "file_type": FILE_TYPES.get(_gv(r, "general.file_type"), _gv(r, "general.file_type")), "block_count": layers, "kv_layers": kv_layers, "mtp": any(".nextn." in t.name for t in r.tensors), "nextn_layers": g("nextn_predict_layers"),
            "head_count": heads, "head_count_kv": kvh, "embedding_length": emb, "context_length": g("context_length"), "kv_bytes_per_token_f16": 2 * kv_sum * hd * 2 if kv_sum and hd else None, "params_b": round(n_params / 1e9, 2), "bpw": round(size * 8 / n_params, 2) if n_params else None}

async def model_meta(m: dict) -> dict:
    try: return await asyncio.to_thread(gguf_info, m["path"])
    except Exception as e: return {"error": str(e)}

# --- llama-server output pump + crash classification ---

LAUNCH_MARK = "=== llama-server launch ==="   # written into LOG_TAIL at every spawn, so a crash is judged only by its own launch's output

def _pump(proc):
    """Echoes llama-server output to the container log while keeping a tail for crash reports."""
    for line in proc.stdout:
        sys.stdout.write(line); sys.stdout.flush(); LOG_TAIL.append(line.rstrip())

def _launch_tail() -> list:
    """LOG_TAIL lines since the newest launch marker (all of it if the marker has rotated out - then every line is from this launch anyway)."""
    t = list(LOG_TAIL)
    return t[max((i for i, l in enumerate(t) if l == LAUNCH_MARK), default=-1) + 1:]

def _oom_kills() -> int:
    """This container's OOM-kill counter from cgroup v2 memory.events (counts kills by any OOM killer, kernel-wide or cgroup limit). -1 where unavailable (cgroup v1, macOS)."""
    ev = Path("/sys/fs/cgroup/memory.events")
    return int(next((l.split()[1] for l in ev.read_text().splitlines() if l.startswith("oom_kill ")), 0)) if ev.exists() else -1

def classify_exit(tail: list, returncode: int = 0, oom_killed: bool = False) -> str:
    """tail must be the current launch only. An OOM kill is system-RAM exhaustion (memory_host); a SIGKILL with no OOM record is reported as killed, cause unknown."""
    if oom_killed: return "memory_host"
    t = "\n".join(tail).lower()
    if any(k in t for k in ("out of memory", "failed to allocate", "unable to allocate", "std::bad_alloc", "alloc_buffer")): return "memory"
    if any(k in t for k in ("rocm error", "invalid device function", "hiperror", "illegal memory access", "hsa_status", "gpu hang", "hw exception")): return "gpu_fault"
    return "killed" if returncode == -9 else "unknown"

def note_exit(proc, name: str, gb: float, kind: str = "") -> str:
    """Records why llama-server died. memory lowers the planner's GPU budget, memory_host its system-RAM budget; a kernel fault is counted per config and per boot (fault_ladder); killed/unknown are only logged."""
    tail = _launch_tail()
    kind = kind or classify_exit(tail, proc.returncode, _OOM_BEFORE >= 0 and _oom_kills() > _OOM_BEFORE)
    LAST_CRASH.clear()
    LAST_CRASH.update({"t": _now(), "model": name, "returncode": proc.returncode, "kind": kind, "params": CURRENT_PLAN.get("params"), "est": CURRENT_PLAN.get("est"), "experiments": CURRENT_PLAN.get("experiments"), "env": CURRENT_PLAN.get("env"), "tail": tail[-40:]})
    print(f"[OPT] llama-server exited rc={proc.returncode} classified={kind} - details at /api/optimizer/last_crash", flush=True)
    _log_attempt("crash", kind=kind, returncode=proc.returncode)
    if kind == "memory": optim.mark(name, "bad", gb)
    elif kind == "memory_host": optim.mark(name, "bad_host", gb, ram_gb=CURRENT_PLAN.get("est", {}).get("ram_gb", 0))
    elif kind in ("gpu_fault", "stall"): optim.mark(name, "fault", gb, optim.fault_key(CURRENT_PLAN))
    return kind

# --- KV ring (automatic save/restore of the slot's attention state, newest N kept) ---

def _ring_file(session: str) -> str: return re.sub(r"[^A-Za-z0-9._-]", "_", f"""ring__{CURRENT_MODEL["name"]}__{CURRENT_KV_CACHE}__fa{CURRENT_PLAN.get("params", {}).get("flash_attn")}__{session}""") + ".bin"

def _ring_entries() -> List[dict]:
    es = [{"file": f, "size": _fsize(os.path.join(CACHE_DIR, f)), "mtime": os.path.getmtime(os.path.join(CACHE_DIR, f))} for f in os.listdir(CACHE_DIR) if f.startswith("ring__") and f.endswith(".bin")]
    return sorted(es, key=lambda e: e["mtime"], reverse=True)

def _ring_prune(keep: str = ""):
    for e in _ring_entries()[KV_RING_SIZE:]:
        if e["file"] != keep and not fleet.pinned(e["file"]): Path(CACHE_DIR, e["file"]).unlink(missing_ok=True)   # a pinned context stays (the ring then holds more than KV_RING_SIZE)

@fleet.observe("kv.save", "session")
async def ring_save(session: str) -> dict:
    if not (CURRENT_MODEL and LLAMA_PROCESS and LLAMA_PROCESS.poll() is None): return {"saved": False, "reason": "no model loaded"}
    free_gb = shutil.disk_usage(CACHE_DIR).free / 1e9
    if free_gb < KV_MIN_FREE_GB: return {"saved": False, "reason": f"free disk {free_gb:.1f} GB below KV_MIN_FREE_GB={KV_MIN_FREE_GB}"}
    fn = _ring_file(session)
    r = await client.post("/slots/0", params={"action": "save"}, json={"filename": f"tmp__{fn}"})  # write beside the good copy, swap only if the slot actually had content
    d = r.json() if r.status_code == 200 else {}
    if not d.get("n_saved"):
        Path(CACHE_DIR, f"tmp__{fn}").unlink(missing_ok=True)
        return {"saved": False, "reason": f"HTTP {r.status_code} or empty slot", "detail": r.text[:200]}
    os.replace(os.path.join(CACHE_DIR, f"tmp__{fn}"), os.path.join(CACHE_DIR, fn))
    _ring_prune(keep=fn)
    return {"saved": True, "file": fn, "n_saved": d["n_saved"], "bytes": d.get("n_written")}

@fleet.observe("kv.restore", "session")
async def ring_restore(session: str) -> dict:
    fn = _ring_file(session)
    if not os.path.exists(os.path.join(CACHE_DIR, fn)): return {"restored": False, "reason": "no saved context for this model/kv/fa/session"}
    r = await client.post("/slots/0", params={"action": "restore"}, json={"filename": fn})
    if r.status_code == 200: os.utime(os.path.join(CACHE_DIR, fn))  # touch = most recently used
    return {"restored": r.status_code == 200, "file": fn, "detail": r.text[:200] if r.status_code != 200 else ""}

async def _autosave_after_idle(session: str):
    try:
        await asyncio.sleep(KV_AUTOSAVE_IDLE_S)
        if IN_FLIGHT == 0 and CURRENT_PLAN.get("features", {}).get("ring_autosave", True): print(f"[KV] autosave: {await ring_save(session)}", flush=True)
    except asyncio.CancelledError: pass

@asynccontextmanager
async def busy(session: str):
    """Marks a request in flight (blocks autosave and restarts), records the outcome for the optimizer, and schedules one debounced save when the last request ends."""
    global IN_FLIGHT, _AUTOSAVE_TASK, CURRENT_SESSION
    IN_FLIGHT += 1
    CURRENT_SESSION = session
    if RID.get() in _REQ: _REQ[RID.get()].update(phase="running", task=asyncio.current_task())   # the task actually talking to llama-server: /control/cancel cancels this one
    fleet.request_running(RID.get())
    if _AUTOSAVE_TASK and not _AUTOSAVE_TASK.done(): _AUTOSAVE_TASK.cancel()
    proc, name, gb, ram = LLAMA_PROCESS, CURRENT_PLAN.get("model"), CURRENT_PLAN.get("est", {}).get("gpu_gb", 0), CURRENT_PLAN.get("est", {}).get("ram_gb", 0)
    optim.mark(name, "pending", gb)
    try:
        yield
        optim.mark(name, "ok" if proc.poll() is None else "clear", gb, ram_gb=ram)
    except BaseException as e:
        if isinstance(e, httpx.HTTPError): await asyncio.sleep(1.5)
        if proc.poll() is not None and proc.pid not in _KILLED: note_exit(proc, name, gb)
        optim.mark(name, "clear", gb, ram_gb=ram)
        raise
    finally:
        IN_FLIGHT -= 1
        if IN_FLIGHT == 0: _AUTOSAVE_TASK = asyncio.create_task(_autosave_after_idle(session))

async def stream_lines(path: str, payload: dict, model_name: str, body: dict, session: str):
    """Streams raw lines from llama-server. If it dies (or stalls past features.stall_s, when that experiment is on) before the first line,
    the failure is recorded, the plan is recomputed under what was learned, and the request is replayed - up to 3 launches."""
    for attempt in range(3):
        got, stall = False, CURRENT_PLAN.get("features", {}).get("stall_s")
        try:
            async with busy(session):
                async with client.stream("POST", path, json=payload) as response:
                    it = response.aiter_lines().__aiter__()
                    while True:
                        try: line = await (asyncio.wait_for(it.__anext__(), stall) if stall else it.__anext__())
                        except StopAsyncIteration: break
                        except asyncio.TimeoutError:
                            _KILLED.add(LLAMA_PROCESS.pid); LLAMA_PROCESS.kill()
                            note_exit(LLAMA_PROCESS, model_name, CURRENT_PLAN.get("est", {}).get("gpu_gb", 0), kind="stall")
                            raise httpx.ReadTimeout(f"no output for {stall}s - llama-server killed as stalled")
                        got = True
                        yield line
            return
        except httpx.HTTPError:
            if got or attempt == 2: raise
            print(f"[OPT] llama-server died before the first token (attempt {attempt + 1}) - replanning and retrying", flush=True)
            await ensure_llama_server_running(require_model(model_name), body, session)

async def infer_post(path: str, payload: dict, model_name: str):
    """A non-streaming llama-server call, bounded by INFER_TIMEOUT_S. On a timeout the server is wedged (not just slow - these prompts are small), so it is killed and recorded as a stall; the caller gets 503 and retries into a fresh server. All four httpx timeout fields are set so a hung read cannot wait forever."""
    try:
        resp = await client.post(path, json=payload, timeout=httpx.Timeout(connect=10.0, read=INFER_TIMEOUT_S, write=30.0, pool=10.0))
        fleet.response_timings(RID.get(), resp)
        return resp
    except (httpx.ReadTimeout, httpx.PoolTimeout, httpx.WriteTimeout):
        if LLAMA_PROCESS and LLAMA_PROCESS.poll() is None:
            _KILLED.add(LLAMA_PROCESS.pid); LLAMA_PROCESS.kill()
            note_exit(LLAMA_PROCESS, model_name, CURRENT_PLAN.get("est", {}).get("gpu_gb", 0), kind="stall")
        raise HTTPException(status_code=503, detail=f"llama-server gave no response in {INFER_TIMEOUT_S}s - killed as wedged; retry")
    except httpx.TransportError as e:   # llama-server went away mid-request (crash, OOM killer, GPU reset): a clean 503 the caller can retry, instead of an unhandled 500 with no reason
        dead = LLAMA_PROCESS is not None and LLAMA_PROCESS.poll() is not None
        print(f"[PROXY] llama-server dropped a request ({type(e).__name__}){f' - exited with code {LLAMA_PROCESS.returncode}' if dead else ''}", flush=True)
        raise HTTPException(status_code=503, detail=f"""llama-server dropped the request ({type(e).__name__}){f" - it exited with code {LLAMA_PROCESS.returncode}, see /api/optimizer/last_crash" if dead else ""}; retry""")

def _json_or_error(resp) -> dict: return resp.json() if "json" in resp.headers.get("content-type", "") else {"error": resp.text[:500]}   # llama-server's error pages are not always JSON

# --- Passive Model Distilliation Mode ---

async def enqueue_passive_learning_sample(prompt, target_output): await learning_queue.put({"prompt": prompt, "target_output": target_output})

# --- llama-server supervision ---

async def smoke_test(c: httpx.AsyncClient) -> dict:
    """Generates a few tokens right after load; a degenerate answer (almost no distinct characters) flags a bad flag combination before a real request hits it."""
    try:
        r = await c.post("/completion", json={"prompt": "The capital of France is", "n_predict": 12, "temperature": 0, "cache_prompt": False})
        text = (r.json().get("content") or "").strip()
    except Exception as e: return {"ok": None, "detail": str(e)}
    return {"ok": not (len(text) < 2 or (len(text) >= 6 and len(set(text.replace(" ", ""))) <= 2)), "text": text[:60]}

async def stop_llama_server():
    """Stops the main llama-server (its context saved first, running requests finished first, up to DRAIN_S) and returns once it has exited - for an exclusive engine about to take the node's memory. The next llama request starts it again."""
    async with process_lock:
        if not (LLAMA_PROCESS and LLAMA_PROCESS.poll() is None): return
        t0 = time.time()
        while IN_FLIGHT > 0 and time.time() - t0 < DRAIN_S: await asyncio.sleep(0.5)
        try: print(f"[KV] {await ring_save(CURRENT_SESSION)}", flush=True)
        except Exception as e: print(f"[KV] save before stopping failed: {e}", flush=True)
        _KILLED.add(LLAMA_PROCESS.pid)
        LLAMA_PROCESS.terminate()
        try: await asyncio.to_thread(LLAMA_PROCESS.wait, 15)
        except subprocess.TimeoutExpired:
            LLAMA_PROCESS.kill()
            await asyncio.to_thread(LLAMA_PROCESS.wait)
        print("[PROXY] llama-server stopped - an exclusive engine needs the memory", flush=True)
        fleet.event("llama.stop", model=CURRENT_MODEL["name"] if CURRENT_MODEL else None)

async def ensure_llama_server_running(model: dict, body: dict, session: str, _replanned: bool = False):
    global CURRENT_MODEL, CURRENT_CTX_SIZE, CURRENT_KV_CACHE, CURRENT_LORAS, LLAMA_PROCESS, CURRENT_BATCH, CURRENT_UBATCH, CURRENT_PLAN, _OOM_BEFORE
    request, loras, meta = {**body, **(body.get("options") if isinstance(body.get("options"), dict) else {})}, extract_loras(body), await model_meta(model)
    s = optim.setup(meta, request, model["name"])
    replan = False
    if engines.any_exclusive_running(): await engines.stop_exclusive()   # before process_lock: an engine that is starting waits on process_lock in stop_llama_server
    async with process_lock:
        alive = LLAMA_PROCESS is not None and LLAMA_PROCESS.poll() is None
        if alive and CURRENT_MODEL["path"] == model["path"] and set(CURRENT_LORAS) == set(loras) and optim.satisfies(CURRENT_PLAN, s): return
        plan = optim.plan(model["size"], meta, s, model["name"])
        plan["sigc"], plan["want_ctx"] = s["sigc"], s["given"]["ctx"][0]
        p = plan["params"]
        print("[OPT] plan:", json.dumps({k: plan[k] for k in ("mode", "goals", "experiments", "rules", "params", "bounds", "est", "fits", "notes") if k in plan}), flush=True)
        if alive:
            t0 = time.time()
            while IN_FLIGHT > 0 and time.time() - t0 < DRAIN_S: await asyncio.sleep(0.5)  # let other requests finish before pulling the server out from under them
            print("[PROXY] Restarting - saving context first", flush=True)
            try: print(f"[KV] {await ring_save(CURRENT_SESSION)}", flush=True)
            except Exception as e: print(f"[KV] pre-restart save failed: {e}", flush=True)
            _KILLED.add(LLAMA_PROCESS.pid)
            LLAMA_PROCESS.terminate()
            try: LLAMA_PROCESS.wait(timeout=5)
            except subprocess.TimeoutExpired: LLAMA_PROCESS.kill()
            await asyncio.sleep(2)
        lora_files = [os.path.join(LORA_DIR, l) for l in loras]
        missing = [l for l in lora_files if not os.path.exists(l)]
        if missing: raise HTTPException(status_code=404, detail=f"LoRA file(s) not found: {missing}")
        opts_dict = body.get("options") if isinstance(body.get("options"), dict) else {}
        if "draft_model" in opts_dict:
            dm = opts_dict["draft_model"]
            if dm and not os.path.isabs(dm):
                # Automatically map relative GGUF model names to LOCAL_DIR
                resolved_dm = os.path.join(LOCAL_DIR, dm)
                if os.path.exists(resolved_dm): opts_dict["draft_model"] = resolved_dm
        cmd = [LLAMA_SERVER_BIN, "--host", "127.0.0.1", "--port", str(INTERNAL_LLAMA_PORT), "-m", model["path"], "-np", "1", "--load-mode", os.getenv("LLAMA_LOAD_MODE", "none"), "--slot-save-path", CACHE_DIR, *optim.cli_args(plan)]
        for lf in lora_files: cmd += ["--lora", lf]
        mmproj = sorted(glob.glob(os.path.join(os.path.dirname(model["path"]), "*mmproj*.gguf"))) if os.path.dirname(model["path"]) != LOCAL_DIR.rstrip("/") else []   # a vision projector in the model's own folder enables image input; its memory is not in the planner's estimate
        if mmproj and os.getenv("LLAMA_MMPROJ", "auto") != "off": cmd += ["--mmproj", mmproj[0]]
        CURRENT_MODEL, CURRENT_CTX_SIZE, CURRENT_KV_CACHE, CURRENT_LORAS, CURRENT_BATCH, CURRENT_UBATCH, CURRENT_PLAN = model, p.get("ctx") or 0, p.get("kv_cache_type") or "default", list(loras), p.get("batch_size") or 0, p.get("ubatch_size") or 0, plan
        print("[PROXY] cmd:", cmd, "env overrides:", plan["env"], flush=True)
        used_before = optim.gpu_used_bytes()
        LOG_TAIL.append(LAUNCH_MARK)
        _OOM_BEFORE = _oom_kills()
        LLAMA_PROCESS = subprocess.Popen(cmd, env={k: v for k, v in {**os.environ, **plan["env"]}.items() if v is not None}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
        threading.Thread(target=_pump, args=(LLAMA_PROCESS,), daemon=True).start()
        _log_attempt("spawn")
        async with httpx.AsyncClient(base_url=LLAMA_SERVER_URL, timeout=None) as c:
            for attempt in range(600):
                if LLAMA_PROCESS.poll() is not None:
                    await asyncio.sleep(0.5)
                    kind = note_exit(LLAMA_PROCESS, model["name"], plan.get("est", {}).get("gpu_gb", 0))
                    if kind in ("memory", "memory_host") and not _replanned: replan = True; break   # the budget just learned from this death changes the next plan - one automatic retry
                    raise HTTPException(status_code=500, detail=f"llama-server exited during startup ({kind}) - see /api/optimizer/last_crash")
                try:
                    if (await c.get("/health")).status_code == 200:
                        smoke = await smoke_test(c)
                        print("[OPT] smoke test:", smoke, flush=True)
                        if smoke["ok"] is False: print("[OPT] WARNING degenerate output right after load - suspect flash_attn / kv type / unified memory:", p, plan["env"], flush=True)
                        if plan["mode"] != "OFF": optim.measure(plan, model["name"], (optim.gpu_used_bytes() - used_before) / 1e9, _mem_gb(LLAMA_PROCESS.pid))
                        if plan.get("features", {}).get("ring_restore", True):
                            try:
                                res = await ring_restore(session)
                                LAST_RESTORE[session] = res
                                print(f"[KV] {res}", flush=True)
                            except Exception as e:
                                LAST_RESTORE[session] = {"restored": False, "reason": str(e)}
                                print(f"[KV] restore failed: {e}", flush=True)
                        fleet.event("llama.ready", model=model["name"], params=p, est=plan.get("est"), smoke=smoke, session=session)
                        return
                except httpx.RequestError: pass
                await asyncio.sleep(1)
        if not replan: raise HTTPException(status_code=500, detail="llama-server timed out during startup (600 s).")
    print("[OPT] startup died from memory exhaustion - replanning once under the budget just learned", flush=True)
    await ensure_llama_server_running(model, body, session, _replanned=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    for d in (LOCAL_DIR, LORA_DIR, CACHE_DIR): os.makedirs(d, exist_ok=True)
    await fleet.start(**_fleet_hooks())   # first, so everything after it is journaled
    await optim.startup([{"name": m["name"], "size": m["size"], "meta": await model_meta(m)} for m in scan_models()])
    await aux.start(find_model, LLAMA_SERVER_BIN, optim, main_busy=lambda: IN_FLIGHT > 0)   # helper servers (draft and embedding slots) start before the main model is planned; what they hold is reserved in the planner. main_busy lets them wait while a user's request runs
    await engines.start(stop_main=stop_llama_server)   # other engines (engines.json): registered here, started by their first request at /engines/<name>/...
    worker_task = asyncio.create_task(start_passive_learning_worker(learning_queue, draft_endpoints=DRAFT_ENDPOINTS))
    yield
    worker_task.cancel()
    await aux.stop()
    await engines.stop()
    await fleet.stop()
    if LLAMA_PROCESS and LLAMA_PROCESS.poll() is None:
        try: await ring_save(CURRENT_SESSION)
        except Exception: pass
        LLAMA_PROCESS.terminate()

app = FastAPI(title="Stateful llama.cpp Supervisor (Ollama wire-compatible)", lifespan=lifespan)
client = httpx.AsyncClient(base_url=LLAMA_SERVER_URL, timeout=None)
app.include_router(aux.router)   # helper servers: /api/draft/adapt, /api/aux/status, /api/aux/log, /api/aux/train_now, /api/embed, /api/embeddings
app.include_router(engines.router)   # other engines: /engines/<name>/... (their own API), /api/engines/status, /api/engines/<name>/start|stop
app.include_router(fleet.router)   # Fleet: /fleet/events, /fleet/state, /fleet/apply
app.include_router(fleet_tests.router)   # Fleet tests: /fleet/test, /fleet/test/cancel

def _final_fields(t: dict, elapsed: float, chunks: int) -> dict:
    """Ollama-shaped final stats, using llama-server's own timings when present. A streamed request's timings go to the Fleet journal from here (a non-streamed one's from infer_post, before it is settled)."""
    fleet.timings(RID.get(), t)
    pn, pms, gn, gms = t.get("prompt_n"), t.get("prompt_ms"), t.get("predicted_n"), t.get("predicted_ms")
    out = {"total_duration": int(elapsed * 1e9), "eval_count": gn if gn is not None else chunks, "eval_duration": int(gms * 1e6) if gms else int(elapsed * 1e9)}
    if pn is not None: out.update({"prompt_eval_count": pn, "prompt_eval_duration": int((pms or 0) * 1e6), "cache_n": t.get("cache_n")})
    return out

def _first(t: dict, *keys): return next((t[k] for k in keys if t.get(k) is not None), None)

def _log_stats(kind: str, t: dict, elapsed: float, chunks: int):
    pn, pms, gn, gms = t.get("prompt_n"), t.get("prompt_ms"), t.get("predicted_n"), t.get("predicted_ms")
    rd = f"read {pn} ({pn / pms * 1000:.2f} t/s) cached {t.get('cache_n', 0)}" if pn and pms else "read n/a"
    wr = f"write {gn} ({gn / gms * 1000:.2f} t/s)" if gn and gms else f"write ~{chunks} chunks"
    print(f"[STATS] {kind}: {rd} | {wr} | {elapsed:.1f}s", flush=True)
    if gn and gms and CURRENT_MODEL: aux.note_main_speed(CURRENT_MODEL["name"], gms / gn)   # the main model's ms per token, for the drafts' measured cost ratio
    # Log speculative drafting divergence stats if reported by llama-server timings
    draft_n, draft_accept = _first(t, "draft_n", "spec_n"), _first(t, "draft_n_accepted", "draft_accept", "spec_accept") or 0
    if draft_n is not None and draft_accept is not None:
        acc_rate = (draft_accept / draft_n * 100.0) if draft_n > 0 else 0.0
        print(f"[DRAFT TELEMETRY] Drafted: {draft_n} | Accepted: {draft_accept} ({acc_rate:.1f}%)", flush=True)
        try:
            with open(TRAJECTORY_LOG, "a") as f:
                f.write(json.dumps({"t": _now(), "model": CURRENT_MODEL["name"] if CURRENT_MODEL else None, "draft_n": draft_n, "draft_accept": draft_accept, "accept_rate": round(acc_rate, 2), "eval_count": gn}) + "\n")
        except OSError as e:
            print(f"[DRAFT TELEMETRY] Write error: {e}", flush=True)

def _crash_line(model_name: str, chunks: int, key: str) -> str:
    _log_attempt("crash_mid_stream", returncode=LLAMA_PROCESS.returncode, progress_tokens=chunks)
    return json.dumps({"model": model_name, "created_at": _now(), key: {"role": "assistant", "content": ""} if key == "message" else "", "done": True, "done_reason": "error", "error": f"llama-server exited (code {LLAMA_PROCESS.returncode}) mid-generation - see /api/optimizer/last_crash"}) + "\n"

# --- Control channel (a trusted local setup: the portal asks about and steers requests separately from the requests themselves) ---
# Every /api/chat and /api/generate request is tracked under its X-Request-Id (the portal sends <job id>.<random>; a random id otherwise) from arrival to its last byte.
# /control/status: what this node is doing (requests in progress and their phase, the latest finished ones, llama-server, memory) - the portal checks it while a long request is silent, so a request the node no longer has ends at once instead of after a read timeout.
# /control/cancel: drops one request (or every request of one job: an id prefix up to the "."); /control/kill: drops every request, scope "server" also kills llama-server (the next request starts it again).

def _track(request: Request, kind: str, model: str, session: str) -> str:
    rid = re.sub(r"[^A-Za-z0-9._:-]", "_", request.headers.get("x-request-id") or uuid.uuid4().hex[:12])[:96]
    RID.set(rid)
    _REQ[rid] = {"id": rid, "kind": kind, "model": model, "session": session, "phase": "loading", "started": time.time(), "task": asyncio.current_task()}
    fleet.request_start(rid, kind, model, session)
    return rid

def _untrack(rid: str, outcome: str):
    r = _REQ.pop(rid, None)
    if not r: return
    outcome = "cancelled" if r.get("cancelled") else outcome
    _COUNTS[outcome] += 1
    fleet.request_end(rid, outcome)
    _DONE.append({"id": rid, "kind": r["kind"], "model": r["model"], "outcome": outcome, "seconds": round(time.time() - r["started"], 1), "ended": _now()})

@contextmanager
def _settles(rid: str, on_error_only: bool = False):
    """Ends a tracked request with its outcome: served, cancelled (a cancel, or the client went away), failed. on_error_only: a step before the reply (loading the model) ends it only when it fails."""
    outcome = "served"
    try: yield
    except asyncio.CancelledError:
        outcome = "cancelled"
        if not _REQ.get(rid, {}).get("cancelled"): raise
        getattr(asyncio.current_task(), "uncancel", lambda: 0)()   # cancelled on purpose through the control channel: answered, not torn down
        raise HTTPException(status_code=499, detail="cancelled through the control channel")
    except GeneratorExit: outcome = "cancelled"; raise
    except BaseException: outcome = "failed"; raise
    finally:
        if outcome != "served" or not on_error_only: _untrack(rid, outcome)

async def _settled_stream(rid: str, gen):
    """A tracked reply stream; a cancel through the control channel ends it with a final done line instead of a broken connection."""
    with _settles(rid):
        try:
            async for chunk in gen: yield chunk
        except asyncio.CancelledError:
            if not _REQ.get(rid, {}).get("cancelled"): raise
            getattr(asyncio.current_task(), "uncancel", lambda: 0)()
            yield json.dumps({"created_at": _now(), "done": True, "done_reason": "cancelled", "error": "cancelled through the control channel"}) + "\n"

def _requests() -> list:
    for rid in [k for k, r in _REQ.items() if r["task"].done()]: _untrack(rid, "failed")   # a request whose task ended without settling (cut off before its stream started)
    return [{**{k: v for k, v in r.items() if k not in ("task", "started", "cancelled")}, "age_s": round(time.time() - r["started"], 1)} for r in _REQ.values()]

def _cancel(prefix: str) -> list:
    ids = [k for k in _REQ if k == prefix or k.startswith(prefix + ".")]
    for k in ids: _REQ[k].update(phase="cancelling", cancelled=True); _REQ[k]["task"].cancel()
    return ids

def _meminfo() -> dict:
    if not os.path.exists("/proc/meminfo"): return {}
    m = {l.split(":")[0]: int(l.split()[1]) / 1e6 for l in open("/proc/meminfo") if l.split(":")[0] in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")}
    return {"ram_total_gb": round(m.get("MemTotal", 0), 1), "ram_avail_gb": round(aux._mem_avail_gb(), 1), "swap_used_gb": round(m.get("SwapTotal", 0) - m.get("SwapFree", 0), 1)}

@app.get("/control/status")
async def control_status():
    alive = LLAMA_PROCESS is not None and LLAMA_PROCESS.poll() is None
    return {"node": NODE_ID, "uptime_s": round(time.time() - STARTED), "requests": _requests(), "recent": list(_DONE)[-20:], "counts": _COUNTS,
            "llama": {"alive": alive, "pid": LLAMA_PROCESS.pid if alive else None, "model": CURRENT_MODEL["name"] if CURRENT_MODEL else None, "ctx": CURRENT_CTX_SIZE, "kv": CURRENT_KV_CACHE, "in_flight": IN_FLIGHT, "rss_gb": round(_mem_gb(LLAMA_PROCESS.pid), 2) if alive else 0},
            "memory": {**_meminfo(), "gpu_used_gb": round(optim.gpu_used_bytes() / 1e9, 2)}, "engines": [{"name": n, "alive": e.alive()} for n, e in engines.ENGINES.items()], "load": os.getloadavg() if hasattr(os, "getloadavg") else None}

@app.post("/control/cancel")
async def control_cancel(request: Request):
    rid = str((await request.json()).get("request_id") or "")
    if not rid: raise HTTPException(status_code=400, detail="request_id needed")
    ids = _cancel(rid)
    print(f"[CONTROL] cancel {rid}: {ids or 'no such request'}", flush=True)
    return {"cancelled": bool(ids), "ids": ids}

@app.post("/control/kill")
async def control_kill(request: Request):
    """scope "requests": drops every request; "server": also kills llama-server at once (SIGKILL, no context save) - for a wedged GPU or a request that will not let go."""
    scope = str((await request.json()).get("scope") or "requests")
    ids = [k for k in list(_REQ) if _cancel(k)]
    killed = scope == "server" and LLAMA_PROCESS is not None and LLAMA_PROCESS.poll() is None
    if killed: _KILLED.add(LLAMA_PROCESS.pid); LLAMA_PROCESS.kill()
    print(f"[CONTROL] kill ({scope}): {len(ids)} request(s) dropped{', llama-server killed' if killed else ''}", flush=True)
    return {"cancelled": len(ids), "server_killed": killed}

# --- Fleet (the node side of the portal's Fleet tool: journal, state, actions - fleet.py; tests - fleet_tests.py) ---

def _fleet_sample() -> dict:
    """One memory and load sample (journal event mem every fleet.json sample_s, and part of /fleet/state)."""
    alive = LLAMA_PROCESS is not None and LLAMA_PROCESS.poll() is None
    return {**_meminfo(), "gpu_used_gb": round(optim.gpu_used_bytes() / 1e9, 2), "load1": round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None, "model": CURRENT_MODEL["name"] if alive and CURRENT_MODEL else None, "in_flight": IN_FLIGHT, "rss_gb": round(_mem_gb(LLAMA_PROCESS.pid), 2) if alive else 0, "engines": [n for n, e in engines.ENGINES.items() if e.alive()]}

def _fleet_resident() -> dict:
    alive = LLAMA_PROCESS is not None and LLAMA_PROCESS.poll() is None
    return {"alive": alive, "model": CURRENT_MODEL["name"] if CURRENT_MODEL else None, "ctx": CURRENT_CTX_SIZE, "kv": CURRENT_KV_CACHE, "loras": CURRENT_LORAS, "session": CURRENT_SESSION, "in_flight": IN_FLIGHT, **{k: CURRENT_PLAN.get(k) for k in ("params", "est", "experiments")}}

async def _fleet_warm(model: str, options: dict, session: str): await ensure_llama_server_running(require_model(model), {"model": model, "options": options}, session)

async def _fleet_kv_save(session: str) -> dict:
    async with process_lock: return await ring_save(session)   # a load in progress holds the lock until its server answers, so a save sent meanwhile waits instead of failing to connect

def _fleet_hooks() -> dict: return {"node": NODE_ID, "port": SERVER_PORT, "busy": lambda: IN_FLIGHT > 0 or bool(_REQ), "sample": _fleet_sample, "resident": _fleet_resident, "find": find_model, "ring": _ring_entries, "cache_dir": CACHE_DIR, "engines": engines, "warm": _fleet_warm, "unload": stop_llama_server, "kv_save": _fleet_kv_save, "cancel": _cancel, "models": lambda: [{"name": m["name"], "size_gb": round(m["size"] / 1e9, 2), "source": m["source"], "mtime": m["mtime"]} for m in scan_models()]}

# --- Model listing / management ---

@app.get("/health")
async def health(): return {"status": "ok", "loaded": CURRENT_MODEL["name"] if CURRENT_MODEL else None, "experiments": CURRENT_PLAN.get("experiments")}

@app.get("/api/tags")
async def list_models(): return {"models": [_tags_entry(m) for m in scan_models()]}

@app.get("/api/ps")
async def running():
    if not (CURRENT_MODEL and LLAMA_PROCESS and LLAMA_PROCESS.poll() is None): return {"models": []}
    return {"models": [{"name": CURRENT_MODEL["name"], "model": CURRENT_MODEL["name"], "size": CURRENT_MODEL["size"], "size_vram": CURRENT_MODEL["size"], "context_length": CURRENT_CTX_SIZE, "kv_cache_type": CURRENT_KV_CACHE, "loras": CURRENT_LORAS, "rss_gb": round(_mem_gb(LLAMA_PROCESS.pid), 2), "peak_rss_gb": round(_mem_gb(LLAMA_PROCESS.pid, "VmHWM"), 2), "plan": {k: CURRENT_PLAN.get(k) for k in ("mode", "params", "est", "experiments", "notes")}}]}

@app.post("/api/show")
async def model_info(request: Request):
    body = await request.json()
    m = require_model(body.get("model") or body.get("name") or "")
    info = await model_meta(m)
    return {"modelfile": f"""FROM {m["name"]}""", "details": {**_tags_entry(m)["details"], "bpw": info.get("bpw"), "params_b": info.get("params_b"), "file_type": info.get("file_type")}, "model_info": info}

@app.delete("/api/delete")
async def delete_model(request: Request):
    body = await request.json()
    m = require_model(body.get("name") or body.get("model") or "")
    if not body.get("confirm"): raise HTTPException(status_code=400, detail="destructive - send confirm=true")
    if CURRENT_MODEL and CURRENT_MODEL["path"] == m["path"] and LLAMA_PROCESS and LLAMA_PROCESS.poll() is None: raise HTTPException(status_code=409, detail="model is currently loaded - load a different one first")
    shared = m["source"] == "ollama_store" and any(o["path"] == m["path"] and o["manifest"] != m["manifest"] for o in scan_models())  # another tag points at the same blob
    freed = 0
    if not shared:
        for f in m["files"]: freed += _fsize(f); Path(f).unlink(missing_ok=True)
    if m["manifest"]: Path(m["manifest"]).unlink(missing_ok=True)
    return {"status": "success", "freed_bytes": freed, "shared_blob_kept": shared}

def _quant_match(fn: str, quant: str) -> bool: return any(t == quant or t.endswith("_" + quant) for t in re.split(r"[-.]", os.path.basename(fn).upper()))

def _resolve_pull_files(repo: str, quant: str, filenames: List[str]) -> List[str]:
    if filenames: return filenames
    ggufs = [f for f in HfApi().list_repo_files(repo) if f.lower().endswith(".gguf") and "mmproj" not in f.lower() and _quant_match(f, quant.upper())]
    series = {re.sub(r"-\d+-of-\d+\.gguf$", "", f) for f in ggufs}
    if len(series) != 1: raise ValueError(f"""{len(series)} GGUF series match quant '{quant}' in {repo}: {sorted(series)} - pass 'filenames' explicitly""")
    return sorted(ggufs)

@app.post("/api/pull")
async def pull_model(request: Request):
    body = await request.json()
    repo, quant = body.get("repo") or body.get("name"), body.get("quant", "")
    filenames = body.get("filenames") or ([body["filename"]] if body.get("filename") else [])
    if not repo or not (filenames or quant): raise HTTPException(status_code=400, detail="need repo plus quant or filename(s)")

    async def download_stream():
        yield json.dumps({"status": f"resolving files in {repo}...", "done": False}) + "\n"
        try:
            files = await asyncio.to_thread(_resolve_pull_files, repo, quant, filenames)
            total = sum(i.size or 0 for i in await asyncio.to_thread(HfApi().get_paths_info, repo, files))
            dest, finished, logged = os.path.join(LOCAL_DIR, repo.replace("/", "__")), 0, -1
            print(f"[PULL] {repo}: {len(files)} file(s), {total / 1e9:.1f} GB -> {dest}", flush=True)
            for fn in files:
                task = asyncio.create_task(asyncio.to_thread(hf_hub_download, repo_id=repo, filename=fn, local_dir=dest))
                while not task.done():
                    await asyncio.sleep(2)
                    part = sum(_fsize(os.path.join(r, f)) for r, _, fs in os.walk(dest) for f in fs if f.endswith(".incomplete"))   # os.walk enters hidden folders (.cache/huggingface/download), glob's ** does not
                    pct = int((finished + part) * 100 / total) if total else 0
                    if pct // 5 != logged: logged = pct // 5; print(f"[PULL] {repo}: {os.path.basename(fn)} {pct}% ({(finished + part) / 1e9:.1f}/{total / 1e9:.1f} GB)", flush=True)
                    yield json.dumps({"status": f"downloading {os.path.basename(fn)}", "total": total, "completed": finished + part, "done": False}) + "\n"
                finished += _fsize(await task)
            gguf_info.cache_clear()
            print(f"[PULL] {repo}: done, {finished / 1e9:.1f} GB", flush=True)
            yield json.dumps({"status": "success", "total": total, "completed": finished, "done": True}) + "\n"
        except Exception as e:
            print(f"[PULL] {repo}: failed - {e}", flush=True)
            yield json.dumps({"status": f"error: {e}", "done": True}) + "\n"

    return StreamingResponse(download_stream(), media_type="application/x-ndjson")

# --- Inference ---

@app.post("/api/generate")
async def generate(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    # 1. Flexible configuration: Request header overrides ENV default
    passive_enabled = request.headers.get("x-passive-learning", os.getenv("PASSIVE_LEARNING_DEFAULT", "false")).lower() == "true"
    model_name, session = body.get("model", ""), extract_session(body)
    rid = _track(request, "generate", model_name, session)
    with _settles(rid, on_error_only=True): await ensure_llama_server_running(require_model(model_name), body, session)
    payload = {"prompt": body.get("prompt", ""), "stream": body.get("stream", True), "cache_prompt": bool(opt(body, "cache_prompt", True)), **map_options(body.get("options", {})), **CURRENT_PLAN.get("payload", {})}

    if not payload["stream"]:
        with _settles(rid):
            async with busy(session): resp = await infer_post("/completion", payload, model_name)
        d = _json_or_error(resp)
        # Capture for passive learning (Non-stream)
        if passive_enabled and not learning_queue.full(): background_tasks.add_task(enqueue_passive_learning_sample, body.get("prompt", ""), d)
        r, c = split_think(d.get("content", ""))
        return {"model": model_name, "created_at": _now(), "response": c, **({"thinking": r} if r else {}), "done": True, "done_reason": "stop", **_final_fields(d.get("timings") or {}, 0, d.get("tokens_predicted", 0))}

    async def stream_gen() -> AsyncGenerator[str, None]:
        start, chunks, timings, parser = time.time(), 0, {}, StreamThinkParser()
        full_content = ""
        async for line in stream_lines("/completion", payload, model_name, body, session):
            if not line.startswith("data: "): continue
            try: data = json.loads(line[6:].strip())
            except json.JSONDecodeError: continue
            raw_c = data.get("content", "")
            full_content += raw_c
            timings = data.get("timings") or timings
            r_out, c_out = parser.process(raw_c)
            if not (r_out or c_out): continue
            chunks += 1
            yield json.dumps({"model": model_name, "created_at": _now(), "response": c_out, **({"thinking": r_out} if r_out else {}), "done": False}) + "\n"
        if LLAMA_PROCESS.poll() is not None: yield _crash_line(model_name, chunks, "response"); return
        r_out, c_out = parser.flush()
        elapsed = time.time() - start
        yield json.dumps({"model": model_name, "created_at": _now(), "response": c_out, **({"thinking": r_out} if r_out else {}), "done": True, "done_reason": "stop", **_final_fields(timings, elapsed, chunks)}) + "\n"
        _log_stats("generate", timings, elapsed, chunks)

        # Capture for passive learning (Stream) - Reconstruct target output
        if passive_enabled and not learning_queue.full():
            target_output = {"content": full_content, "timings": timings}
            await enqueue_passive_learning_sample(body.get("prompt", ""), target_output)

    return StreamingResponse(_settled_stream(rid, stream_gen()), media_type="application/x-ndjson", background=background_tasks)

@app.post("/api/chat")
async def chat(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    # 1. Flexible configuration: Request header overrides ENV default
    passive_enabled = request.headers.get("x-passive-learning", os.getenv("PASSIVE_LEARNING_DEFAULT", "false")).lower() == "true"
    model_name, session = body.get("model", ""), extract_session(body)
    rid = _track(request, "chat", model_name, session)
    with _settles(rid, on_error_only=True): await ensure_llama_server_running(require_model(model_name), body, session)
    payload = {"messages": coalesce_messages(body.get("messages", [])), "stream": body.get("stream", True), "cache_prompt": bool(opt(body, "cache_prompt", True)), **map_options(body.get("options", {})), **CURRENT_PLAN.get("payload", {})}
    if "think" in body: payload["chat_template_kwargs"] = {"enable_thinking": think_on(body["think"])}  # Ollama semantics: present-and-falsy = thinking off
    if body.get("tools"): payload.update({"tools": body["tools"], **({"tool_choice": body["tool_choice"]} if body.get("tool_choice") else {})})  # needs experiment tool_calls (--jinja) on builds that do not default to it
    if body.get("format"): payload["response_format"] = {"type": "json_object"} if body["format"] == "json" else {"type": "json_schema", "json_schema": {"name": "response", "schema": body["format"]}}

    if not payload["stream"]:
        with _settles(rid):
            async with busy(session): resp = await infer_post("/v1/chat/completions", payload, model_name)
        d = _json_or_error(resp)
        if resp.status_code != 200: raise HTTPException(status_code=resp.status_code, detail=d)
        # Capture for passive learning (Non-stream)
        if passive_enabled and not learning_queue.full(): background_tasks.add_task(enqueue_passive_learning_sample, body.get("messages", []), d)
        msg = (d.get("choices") or [{}])[0].get("message", {})
        r, c = msg.get("reasoning_content") or "", msg.get("content") or ""
        if not r: r, c = split_think(c)
        if "think" in body and not think_on(body["think"]): r, c = (r if r.strip() else ""), c.lstrip()   # thinking off: the template's empty think block leaves only whitespace before the reply
        calls = _from_oai_calls(msg.get("tool_calls"))
        return {"model": model_name, "created_at": _now(), "message": {"role": "assistant", "content": c, **({"thinking": r} if r else {}), **({"tool_calls": calls} if calls else {})}, "done": True, "done_reason": (d.get("choices") or [{}])[0].get("finish_reason") or "stop", **_final_fields(d.get("timings") or {}, 0, 0)}

    async def chat_stream_gen() -> AsyncGenerator[str, None]:
        start, chunks, finish, timings, parser, calls = time.time(), 0, None, {}, StreamThinkParser(), {}
        strip_lead = "think" in body and not think_on(body["think"])   # thinking off: the template's empty think block leaves only whitespace before the reply - dropped until the first visible character
        msg = lambda c, r: {"role": "assistant", "content": c, **({"thinking": r} if r else {})}
        full_c, full_r = "", ""
        async for line in stream_lines("/v1/chat/completions", payload, model_name, body, session):
            if not line.startswith("data: "): continue
            raw = line[6:].strip()
            if not raw or raw == "[DONE]": continue  # never break out of stream_lines early - its cleanup (busy) must run
            try: data = json.loads(raw)
            except json.JSONDecodeError: continue
            timings = data.get("timings") or timings  # may arrive on the finish chunk or a trailing one - keep the last seen
            choices = data.get("choices", [])
            if not choices: continue
            delta = choices[0].get("delta", {})
            finish = choices[0].get("finish_reason") or finish
            for tc in delta.get("tool_calls") or []:  # streamed tool calls arrive as fragments keyed by index - assemble, emit once at the end
                slot = calls.setdefault(tc.get("index", 0), {"id": tc.get("id"), "function": {"name": "", "arguments": ""}})
                slot["id"] = slot["id"] or tc.get("id")
                slot["function"]["name"] += (tc.get("function") or {}).get("name") or ""
                slot["function"]["arguments"] += (tc.get("function") or {}).get("arguments") or ""
            raw_r, raw_c = delta.get("reasoning_content") or "", delta.get("content") or ""
            full_c += raw_c
            full_r += raw_r

            r_out, c_out = (raw_r, raw_c) if raw_r else parser.process(raw_c)
            if strip_lead:
                r_out, c_out = (r_out if r_out.strip() else ""), c_out.lstrip()
                strip_lead = not c_out
            if not (r_out or c_out): continue
            chunks += 1
            yield json.dumps({"model": model_name, "created_at": _now(), "message": msg(c_out, r_out), "done": False}) + "\n"
        if LLAMA_PROCESS.poll() is not None: yield _crash_line(model_name, chunks, "message"); return
        r_out, c_out = parser.flush()
        if strip_lead: r_out, c_out = (r_out if r_out.strip() else ""), c_out.lstrip()
        if calls: yield json.dumps({"model": model_name, "created_at": _now(), "message": {**msg("", ""), "tool_calls": _from_oai_calls([calls[i] for i in sorted(calls)])}, "done": False}) + "\n"
        elapsed = time.time() - start
        yield json.dumps({"model": model_name, "created_at": _now(), "message": msg(c_out, r_out), "done": True, "done_reason": finish or "stop", **_final_fields(timings, elapsed, chunks)}) + "\n"
        _log_stats("chat", timings, elapsed, chunks)

        # Capture for passive learning (Stream) - Reconstruct target output
        if passive_enabled and not learning_queue.full():
            compiled_msg = {"role": "assistant", "content": full_c}
            if full_r: compiled_msg["reasoning_content"] = full_r
            if calls:
                compiled_msg["tool_calls"] = [{"id": calls[i]["id"], "type": "function", "function": {"name": calls[i]["function"]["name"], "arguments": calls[i]["function"]["arguments"]}} for i in sorted(calls)]
            target_output = {"choices": [{"message": compiled_msg, "finish_reason": finish or "stop"}]}
            # Non-blocking async queue insertion at stream finish
            await enqueue_passive_learning_sample(body.get("messages", []), target_output)

    return StreamingResponse(_settled_stream(rid, chat_stream_gen()), media_type="application/x-ndjson", background=background_tasks)

# --- Context cache + LoRA ---

@app.get("/api/hang_log")
async def hang_log(): return {"entries": [json.loads(l) for l in open(CRASH_LOG).readlines()[-30:]]} if os.path.exists(CRASH_LOG) else {"entries": []}

@app.get("/api/cache/last_restore")
async def last_restore(request: Request): return LAST_RESTORE.pop(request.query_params.get("session", "default"), {"restored": None, "reason": "no restore attempted since the last report"})

@app.post("/api/cache/save")
async def cache_save(request: Request):
    body = await request.json()
    if body.get("filename"): return (await client.post("/slots/0", params={"action": "save"}, json={"filename": os.path.basename(body["filename"])})).json()  # explicit named save, outside the ring
    return await ring_save(extract_session(body))

@app.post("/api/cache/restore")
async def cache_restore(request: Request):
    body = await request.json()
    if body.get("filename"): return (await client.post("/slots/0", params={"action": "restore"}, json={"filename": os.path.basename(body["filename"])})).json()
    return await ring_restore(extract_session(body))

@app.get("/api/cache/list")
async def cache_list(): return {"ring_size": KV_RING_SIZE, "free_gb": round(shutil.disk_usage(CACHE_DIR).free / 1e9, 1), "entries": _ring_entries()}

@app.delete("/api/cache")
async def cache_purge(request: Request):
    body = await request.json()
    fn = os.path.basename(body.get("file", ""))
    if not (fn.startswith("ring__") and fn.endswith(".bin")): raise HTTPException(status_code=400, detail="not a ring file")
    Path(CACHE_DIR, fn).unlink(missing_ok=True)
    return {"status": "success"}

@app.get("/api/lora/available")
async def lora_available(): return {"loras": sorted(f for f in os.listdir(LORA_DIR) if f.endswith(".gguf"))}

@app.get("/api/lora/list")
async def lora_list(): return (await client.get("/lora-adapters")).json()

@app.post("/api/lora/set")
async def lora_set(request: Request):
    """Adjusts scale (0 = off) on adapters passed at spawn via options.lora_paths - a new adapter file needs a restart, same as a model swap. body: {"adapters":[{"id":0,"scale":0.7}]}"""
    body = await request.json()
    return (await client.post("/lora-adapters", json=body.get("adapters", []))).json()

# --- Optimizer ---

@app.get("/api/optimizer/sysinfo")
async def optimizer_sysinfo(refresh: bool = False): return optim.info(refresh)

@app.post("/api/optimizer/plan")
async def optimizer_plan(request: Request):
    """Dry run: plans a launch for body.model with body.options and body.constraints and starts nothing."""
    body = await request.json()
    m = require_model(body.get("model", ""))
    meta = await model_meta(m)
    return optim.plan(m["size"], meta, optim.setup(meta, {**body, **(body.get("options") or {})}, m["name"]), m["name"])

@app.get("/api/optimizer/experiments")
async def optimizer_experiments(): return optim.experiments_status()

@app.post("/api/optimizer/refresh")
async def optimizer_refresh():
    await optim.manager_hello()
    return optim.MANAGER

@app.api_route("/api/optimizer/forget", methods=["GET", "POST"])
async def optimizer_forget(model: str = ""):
    optim.forget(model)
    return {"status": "success", "cleared": model or "all"}

@app.api_route("/api/optimizer/clear_ring", methods=["GET", "POST"])
async def optimizer_clear_ring():
    n = 0
    for e in _ring_entries(): Path(CACHE_DIR, e["file"]).unlink(missing_ok=True); n += 1
    return {"status": "success", "removed": n}

@app.get("/api/optimizer/last_crash")
async def optimizer_last_crash(): return LAST_CRASH or {"note": "no crash recorded since this supervisor started"}

@app.get("/api/optimizer/log")
async def optimizer_log(n: int = 80): return {"tail": list(LOG_TAIL)[-n:]}

# --- Diagnostic Chat Page (standalone test surface; the portal's Athena/Tessa are the real UIs) ---

CHAT_PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>llama_direct diagnostics</title>
<style>body{font:15px system-ui,sans-serif;max-width:58rem;margin:.6rem auto;padding:0 .8rem;color:#222;background:#fbfbf8}#log div{margin:.4rem 0;white-space:pre-wrap;line-height:1.45}.u{color:#1a4f8a}.t{color:#777;font-size:.85em}textarea,input[type=text]{width:100%;box-sizing:border-box;font:inherit}#dbg{background:#1d2021;color:#b8e0a8;padding:.6rem;max-height:22rem;overflow:auto;white-space:pre-wrap;font-size:.8rem}label{margin-right:.7rem;white-space:nowrap}button{margin:.15rem .1rem;font:inherit}fieldset{border:1px solid #ccc;margin:.4rem 0;padding:.3rem .6rem}legend{font-size:.85rem;color:#555}</style>
<select id="m"></select> <button onclick="clr()">New chat</button> <span id="st"></span>
<fieldset><legend>Request options (merged into the request; constraints go under "constraints")</legend>
<textarea id="o" rows="2">{}</textarea>
<label><input type="checkbox" id="cs" checked> stream</label><label><input type="checkbox" id="cq"> show request</label><label><input type="checkbox" id="cr"> show raw response</label><label><input type="checkbox" id="ct" checked> show thinking</label>
<label>experiments for this request: <input type="text" id="ex" placeholder="e.g. mtp_spec,-tool_calls (blank = node default)" style="width:18rem"></label>
</fieldset>
<div id="log"></div>
<textarea id="q" rows="3" placeholder="Message (Ctrl+Enter to send)" onkeydown="if(event.ctrlKey&&event.key=='Enter')send()"></textarea>
<button onclick="send()">Send</button> <button onclick="ac&&ac.abort()">Stop</button>
<fieldset><legend>Diagnostics</legend>
<button onclick="plan()">Plan (dry run)</button><button onclick="g('/api/optimizer/sysinfo')">Sysinfo</button><button onclick="g('/api/optimizer/experiments')">Experiments</button><button onclick="g('/api/ps')">Running</button><button onclick="g('/api/optimizer/last_crash')">Last crash</button><button onclick="g('/api/optimizer/log?n=150')">Log tail</button><button onclick="g('/api/hang_log')">Crash journal</button><button onclick="g('/api/cache/list')">Ring list</button><button onclick="g('/api/optimizer/clear_ring')">Clear KV ring</button><button onclick="g('/api/optimizer/forget')">Forget learned</button><button onclick="dbg.textContent=''">Clear panel</button>
</fieldset>
<pre id="dbg"></pre>
<script>
var H=[],ac=null;
fetch('/api/tags').then(r=>r.json()).then(d=>d.models.forEach(x=>m.add(new Option(x.name))));
function show(l,o){dbg.textContent+='['+new Date().toLocaleTimeString()+'] '+l+': '+(typeof o=='string'?o:JSON.stringify(o,null,1))+'\n';dbg.scrollTop=dbg.scrollHeight}
function add(c,t){var d=document.createElement('div');d.className=c;d.textContent=t;log.appendChild(d);return d}
function clr(){H=[];log.innerHTML=''}
function opts(){try{var op=JSON.parse(o.value||'{}');if(ex.value.trim()){op.constraints=op.constraints||{};op.constraints.experiments=ex.value.split(',').map(s=>s.trim()).filter(Boolean)}return op}catch(e){show('options JSON error',String(e));return null}}
async function g(u){try{var r=await fetch(u);show(u+' -> '+r.status,await r.json())}catch(e){show(u,String(e))}}
async function plan(){var op=opts();if(!op)return;var r=await fetch('/api/optimizer/plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:m.value,options:op})});show('plan -> '+r.status,await r.json())}
async function send(){
 var t=q.value.trim(),op=opts(); if(!t||ac||!op) return; q.value=''; H.push({role:'user',content:t});
 add('u',t); var box=add('a',''),think=add('t',''),text='',th='',body={model:m.value,messages:H,stream:cs.checked,options:op};
 if(cq.checked) show('REQUEST',body);
 ac=new AbortController(); st.textContent='working...'; var t0=Date.now();
 try{
  var r=await fetch('/api/chat',{method:'POST',signal:ac.signal,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(cr.checked) show('HTTP',r.status);
  if(!r.ok) throw new Error(await r.text());
  if(!cs.checked){var j=await r.json(); if(cr.checked) show('RAW',j); text=(j.message||{}).content||''; th=(j.message||{}).thinking||''; box.textContent=text; if(ct.checked&&th) think.textContent='thinking: '+th; if(j.error) box.textContent+=' [error] '+j.error}
  else{
   var rd=r.body.getReader(),dec=new TextDecoder(),buf='';
   for(;;){
    var x=await rd.read(); if(x.done) break;
    buf+=dec.decode(x.value,{stream:true}); var ls=buf.split('\n'); buf=ls.pop();
    for(var l of ls){
     if(!l.trim()) continue; if(cr.checked) show('RAW',l); var j=JSON.parse(l);
     if(j.error){box.textContent+=' [error] '+j.error; continue}
     var mm=j.message||{}; text+=mm.content||''; th+=mm.thinking||'';
     if(mm.tool_calls) show('TOOL CALLS',mm.tool_calls);
     box.textContent=text; think.textContent=(ct.checked&&th)?'thinking: '+th:'';
     if(j.done&&j.eval_count&&j.eval_duration) st.textContent=(j.eval_count/(j.eval_duration/1e9)).toFixed(1)+' t/s';
    }
   }
  }
 }catch(e){ if(e.name!='AbortError'){box.textContent+=' [error] '+e; show('ERROR',String(e))} }
 show('done',((Date.now()-t0)/1000).toFixed(1)+'s, '+text.length+' chars');
 if(text) H.push({role:'assistant',content:text}); else H.pop();
 ac=null; if(st.textContent=='working...') st.textContent='';
}
</script>"""

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(): return CHAT_PAGE

if __name__ == "__main__": uvicorn.run(app, host=os.getenv("SERVER_HOST", "0.0.0.0"), port=SERVER_PORT)

