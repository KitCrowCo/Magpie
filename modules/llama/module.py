"""modules/llama/module.py - llama.cpp as one adapter: chat (vision through a projector file), completion, embeddings and decision models, each on a llama-server of its own that the node's scheduler starts, stops and accounts for.
The main server is planned per request from the machine and the model (planner.py, optimizer_rules.json) and learns from failures. Its KV state is saved to a ring on disk, its LoRA scales change live. All of its paths under /api/cache, /api/lora and /api/optimizer are kept as they were."""
import os, re, glob, json, time, shutil, asyncio
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone
import httpx
from core import settings, budget, services, jobs, journal, hardware, drivers
from core.adapters import Adapter
from core.http import Router, HTTPException, json_body
from core.services import Service
from modules.llm import oai
from modules.llama import planner as optim
from modules.llama.gguf_lite import Reader

router = Router()
def models_dir() -> str: return str(settings.location("models"))
def local_dir() -> str: return os.path.join(models_dir(), "gguf_local")
def lora_dir() -> str: return str(settings.location("loras"))
def cache_dir() -> str: return str(settings.location("cache"))
def server_bin() -> str: return drivers.state("llama")["provides"]["server"]
KV_RING_SIZE, KV_AUTOSAVE_IDLE_S, KV_MIN_FREE_GB = int(os.getenv("KV_RING_SIZE", 3)), int(os.getenv("KV_AUTOSAVE_IDLE_S", 45)), float(os.getenv("KV_MIN_FREE_GB", 20))
INFER_TIMEOUT_S = float(os.getenv("LLAMA_INFER_TIMEOUT_S", 1200))   # a client that does not stream: this long with no output from llama-server means it is wedged
CRASH_LOG_MAX_LINES = 2000
def crash_log() -> str: return str(settings.location("logs") / "llama_launches.jsonl")
def trajectory_log() -> str: return str(settings.location("logs") / "draft_trajectories.jsonl")
FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4"}   # llama.h LLAMA_FTYPE ids as recalled; unknown ids show as the number, and bpw (measured) is authoritative either way
LAST_CRASH, LAST_RESTORE, SPEED = {}, {}, {}   # SPEED: {"model", "ms"}: the main model's last milliseconds per generated token
A: dict = {"adapter": None}

def _now(): return datetime.now(timezone.utc).isoformat()
def _fsize(p):
    try: return os.path.getsize(p)
    except OSError: return 0
def opt(body: dict, key: str, default=None):
    o = body.get("options") if isinstance(body.get("options"), dict) else {}
    return o.get(key, body.get(key, default))
def extract_session(body): return re.sub(r"[^A-Za-z0-9_-]", "_", str(opt(body, "cache_session") or "default"))[:64]
def extract_loras(body): return [os.path.basename(str(p)) for p in (opt(body, "lora_paths") or [])]

# --- Models (Ollama store and local GGUF files, one list; reads files only) ---

def scan_models() -> list:
    out, mdir, bdir = [], os.path.join(models_dir(), "manifests"), os.path.join(models_dir(), "blobs")
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
    for f in sorted(glob.glob(os.path.join(local_dir(), "**", "*.gguf"), recursive=True)):
        if "mmproj" in os.path.basename(f).lower(): continue
        shard = re.search(r"-(\d+)-of-\d+\.gguf$", f)
        if shard and int(shard.group(1)) != 1: continue
        files = sorted(glob.glob(re.sub(r"-\d+-of-(\d+)\.gguf$", r"-*-of-\1.gguf", f))) if shard else [f]
        out.append({"name": re.sub(r"-\d+-of-\d+$", "", os.path.splitext(os.path.basename(f))[0]), "path": f, "files": files, "size": sum(_fsize(x) for x in files), "mtime": os.path.getmtime(f), "source": "local", "manifest": None})
    return out

def _tag(n: str) -> str: return n if ":" in n else f"{n}:latest"
def find_model(requested: str): return next((m for m in scan_models() if _tag(m["name"]) == _tag(requested.strip())), None)

def require_model(requested: str) -> dict:
    m = find_model(requested)
    if not m: raise HTTPException(status_code=404, detail=f"Model not found: {requested}")
    return m

@lru_cache(maxsize=64)
def gguf_info(path: str) -> dict:
    """Real architecture and quantization from the GGUF header - authoritative, unlike a tag name. kv_layers counts only attention layers (hybrid recurrent+attention models have far fewer); params and bpw come from the tensor table, so they are right even when file_type is unknown."""
    r = Reader(path)
    arch = r.fields.get("general.architecture")
    g = lambda k: r.fields.get(f"{arch}.{k}")
    layers, heads, kvh, emb, hd = g("block_count"), g("attention.head_count"), g("attention.head_count_kv"), g("embedding_length"), g("attention.key_length")
    hd = hd or ((emb // (max(heads) if isinstance(heads, list) else heads)) if emb and heads else None)
    ival = g("full_attention_interval")
    kv_layers = sum(1 for x in kvh if x) if isinstance(kvh, list) else (layers // ival if ival and layers else layers)
    kv_sum = (sum(kvh) if isinstance(kvh, list) else kvh * kv_layers) if kvh and layers else None
    n_params = sum(t["n_elements"] for t in r.tensors)
    size = sum(_fsize(f) for f in glob.glob(re.sub(r"-\d+-of-(\d+)\.gguf$", r"-*-of-\1.gguf", path))) or _fsize(path)
    return {"general.architecture": arch, "file_type": FILE_TYPES.get(r.fields.get("general.file_type"), r.fields.get("general.file_type")), "block_count": layers, "kv_layers": kv_layers, "mtp": any(".nextn." in t["name"] for t in r.tensors), "nextn_layers": g("nextn_predict_layers"), "head_count": heads, "head_count_kv": kvh, "embedding_length": emb, "context_length": g("context_length"), "pooling_type": g("pooling_type"), "kv_bytes_per_token_f16": 2 * kv_sum * hd * 2 if kv_sum and hd else None, "params_b": round(n_params / 1e9, 2), "bpw": round(size * 8 / n_params, 2) if n_params else None}

def model_meta_sync(m: dict) -> dict:
    try: return gguf_info(m["path"])
    except Exception as e: return {"error": str(e)}

async def model_meta(m: dict) -> dict: return await asyncio.to_thread(model_meta_sync, m)

def _mmproj(path: str) -> list: return sorted(glob.glob(os.path.join(os.path.dirname(path), "*mmproj*.gguf"))) if os.path.dirname(path) != local_dir().rstrip("/") else []   # a vision projector in the model's own folder enables image input

def _caps(m: dict) -> list:
    """What a model file can be asked to do, from its header: an embedding architecture embeds, everything else chats; a projector beside it adds vision."""
    meta, embed = model_meta_sync(m), settings.get("llama.embed_archs", [])
    if meta.get("general.architecture") in embed or meta.get("pooling_type"): return ["embed"]
    return ["chat", "generate", "decide", *(["vision"] if _mmproj(m["path"]) else [])]

def tags_entry(m: dict) -> dict:
    up = m["name"].upper()
    ql, ps = re.search(r"(I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32)", up), re.search(r"(\d+(?:\.\d+)?)B", up)
    a = A["adapter"]
    return {"name": m["name"], "model": m["name"], "modified_at": datetime.fromtimestamp(m["mtime"], tz=timezone.utc).isoformat(), "size": m["size"], "source": m["source"], "loaded": bool(a and a.main.model and a.main.model["name"] == m["name"] and a.main.ready()), "details": {"format": "gguf", "quantization_level": ql.group(1) if ql else "", "parameter_size": f"{ps.group(1)}B" if ps else "", "bpw_est": round(m["size"] * 8 / (float(ps.group(1)) * 1e9), 2) if ps else None}}

# --- Crash classification ---

def _oom_kills() -> int:
    """This container's OOM-kill counter (cgroup v2), by any OOM killer. -1 where unavailable."""
    ev = Path("/sys/fs/cgroup/memory.events")
    return int(next((l.split()[1] for l in ev.read_text().splitlines() if l.startswith("oom_kill ")), 0)) if ev.exists() else -1

def classify_exit(tail: list, returncode: int = 0, oom_killed: bool = False) -> str:
    """tail must be the current launch only. An OOM kill is system-RAM exhaustion (memory_host); a SIGKILL with no OOM record is reported as killed, cause unknown."""
    if oom_killed: return "memory_host"
    t = "\n".join(tail).lower()
    if any(k in t for k in ("out of memory", "failed to allocate", "unable to allocate", "std::bad_alloc", "alloc_buffer")): return "memory"
    if any(k in t for k in ("rocm error", "invalid device function", "hiperror", "illegal memory access", "hsa_status", "gpu hang", "hw exception")): return "gpu_fault"
    return "killed" if returncode == -9 else "unknown"

async def smoke_test(c: httpx.AsyncClient) -> dict:
    """Generates a few tokens right after load; a degenerate answer (almost no distinct characters) flags a bad flag combination before a real request hits it."""
    try:
        r = await c.post("/completion", json={"prompt": "The capital of France is", "n_predict": 12, "temperature": 0, "cache_prompt": False})
        text = (r.json().get("content") or "").strip()
    except Exception as e: return {"ok": None, "detail": str(e)}
    return {"ok": not (len(text) < 2 or (len(text) >= 6 and len(set(text.replace(" ", ""))) <= 2)), "text": text[:60]}

# --- The main server ---

class MainServer(Service):
    """llama-server with the planner, the KV ring and crash learning around it. The scheduler decides when; this knows how."""
    def __init__(self):
        super().__init__("llama:main", {"health": "/health", "slots": 1, "stop_timeout_s": 15}, events="llama", owner="llama")
        self.model, self.plan, self.loras, self.session, self.oom_before, self.killed, self.autosave = None, {}, [], "default", -1, set(), None
        self.spec["idle_stop_s"] = 0

    def _log_attempt(self, event: str, **extra):
        """Append-only spawn and crash journal (gpu_hang_log.jsonl), trimmed to the newest lines."""
        name = self.model["name"] if self.model else None
        journal.event(f"llama.{event}", model=name, params=self.plan.get("params", {}), est=self.plan.get("est"), experiments=self.plan.get("experiments"), **extra)
        try:
            with open(crash_log(), "a") as f: f.write(json.dumps({"t": _now(), "event": event, "model": name, "params": self.plan.get("params", {}), "experiments": self.plan.get("experiments"), "env": self.plan.get("env"), **extra}) + "\n")
            if _fsize(crash_log()) > 2_000_000:
                keep = open(crash_log()).readlines()[-CRASH_LOG_MAX_LINES:]
                open(crash_log(), "w").writelines(keep)
        except OSError as e: print(f"[LLAMA] crash log write failed: {e}", flush=True)

    def note_exit(self, name: str, gb: float, kind: str = "") -> str:
        """Records why llama-server died. memory lowers the planner's GPU budget, memory_host its system-RAM budget; a kernel fault is counted per config and per boot; killed and unknown are only logged."""
        tail, proc = self.tail(40), self.proc
        kind = kind or classify_exit(tail, proc.returncode, self.oom_before >= 0 and _oom_kills() > self.oom_before)
        LAST_CRASH.clear()
        LAST_CRASH.update({"t": _now(), "model": name, "returncode": proc.returncode, "kind": kind, "params": self.plan.get("params"), "est": self.plan.get("est"), "experiments": self.plan.get("experiments"), "env": self.plan.get("env"), "tail": tail})
        print(f"[OPT] llama-server exited rc={proc.returncode} classified={kind} - details at /api/optimizer/last_crash", flush=True)
        self._log_attempt("crash", kind_=kind, returncode=proc.returncode)
        if kind == "memory": optim.mark(name, "bad", gb)
        elif kind == "memory_host": optim.mark(name, "bad_host", gb, ram_gb=self.plan.get("est", {}).get("ram_gb", 0))
        elif kind in ("gpu_fault", "stall"): optim.mark(name, "fault", gb, optim.fault_key(self.plan))
        self.status = f"failed ({kind})"
        return kind

    def budget_plan(self, model: dict, meta: dict, body: dict) -> tuple:
        """(setup, plan) for this request, planned around what the other services on this node hold."""
        ram, gpu = budget.held_by_others(self.name)
        optim.reserve_external(gpu, ram, "other services on this node")
        s = optim.setup(meta, {**body, **(body.get("options") if isinstance(body.get("options"), dict) else {})}, model["name"])
        return s, optim.plan(model["size"], meta, s, model["name"])

    def satisfied(self, model: dict, body: dict) -> bool:
        """A running server is reused whenever its configuration sits inside this request's bounds; only a real conflict reloads."""
        if not (self.model and self.model["path"] == model["path"] and set(self.loras) == set(extract_loras(body))): return False
        return optim.satisfies(self.plan, optim.setup(model_meta_sync(model), {**body, **(body.get("options") if isinstance(body.get("options"), dict) else {})}, model["name"]))

    async def before_stop(self):
        try: print(f"[KV] {await ring_save(self, self.session)}", flush=True)
        except Exception as e: print(f"[KV] save before stopping failed: {e}", flush=True)

    async def load(self, model: dict, body: dict, session: str, _replanned: bool = False):
        loras, meta = extract_loras(body), await model_meta(model)
        async with self.lock:
            s, plan = self.budget_plan(model, meta, body)
            plan["sigc"], plan["want_ctx"] = s["sigc"], s["given"]["ctx"][0]
            p = plan["params"]
            print("[OPT] plan:", json.dumps({k: plan[k] for k in ("mode", "goals", "experiments", "rules", "params", "bounds", "est", "fits", "notes") if k in plan}), flush=True)
            lora_files = [os.path.join(lora_dir(), l) for l in loras]
            if missing := [l for l in lora_files if not os.path.exists(l)]: raise HTTPException(status_code=404, detail=f"LoRA file(s) not found: {missing}")
            o = body.get("options") if isinstance(body.get("options"), dict) else {}
            if o.get("draft_model") and not os.path.isabs(o["draft_model"]) and os.path.exists(os.path.join(local_dir(), o["draft_model"])): o["draft_model"] = os.path.join(local_dir(), o["draft_model"])
            self.port = int(self.spec.get("port") or os.getenv("INTERNAL_LLAMA_PORT") or services.free_port())
            cmd = [server_bin(), "--host", "127.0.0.1", "--port", str(self.port), "-m", model["path"], "-np", "1", "--load-mode", os.getenv("LLAMA_LOAD_MODE", "none"), "--slot-save-path", cache_dir(), *optim.cli_args(plan)]
            for lf in lora_files: cmd += ["--lora", lf]
            if (mm := _mmproj(model["path"])) and os.getenv("LLAMA_MMPROJ", "auto") != "off": cmd += ["--mmproj", mm[0]]
            self.model, self.loras, self.plan, self.session, self.status, self.closing = model, list(loras), plan, session, "starting", False
            self.meta = {"model": model["name"], "ctx": p.get("ctx") or 0, "kv": p.get("kv_cache_type") or "default"}
            print("[LLAMA] cmd:", cmd, "env overrides:", plan["env"], flush=True)
            used_before, self.oom_before = optim.gpu_used_bytes(), _oom_kills()
            self.spawn(cmd, plan["env"])
            self._log_attempt("spawn")
            replan = False
            async with httpx.AsyncClient(base_url=self.base(), timeout=None, **self.connector()) as c:
                for _ in range(int(os.getenv("LLAMA_START_TIMEOUT_S", 600))):
                    if self.proc.poll() is not None:
                        await asyncio.sleep(0.5)
                        kind = self.note_exit(model["name"], plan.get("est", {}).get("gpu_gb", 0))
                        if kind in ("memory", "memory_host") and not _replanned: replan = True; break   # the budget just learned from this death changes the next plan: one automatic retry
                        raise HTTPException(status_code=500, detail=f"llama-server exited during startup ({kind}) - see /api/optimizer/last_crash")
                    try:
                        if (await c.get("/health")).status_code == 200:
                            smoke = await smoke_test(c)
                            print("[OPT] smoke test:", smoke, flush=True)
                            if smoke["ok"] is False: print("[OPT] WARNING degenerate output right after load - suspect flash_attn / kv type / unified memory:", p, plan["env"], flush=True)
                            if plan["mode"] != "OFF": optim.measure(plan, model["name"], (optim.gpu_used_bytes() - used_before) / 1e9, hardware.rss_gb(self.proc.pid))
                            self.status, self.last_use, self.t_up = "up", time.time(), time.time()
                            self.measure((optim.gpu_used_bytes() - used_before) / 1e9)
                            if plan.get("features", {}).get("ring_restore", True):
                                try: LAST_RESTORE[session] = await ring_restore(self, session)
                                except Exception as e: LAST_RESTORE[session] = {"restored": False, "reason": str(e)}
                                print(f"[KV] {LAST_RESTORE[session]}", flush=True)
                            journal.event("llama.ready", model=model["name"], params=p, est=plan.get("est"), smoke=smoke, session=session)
                            return
                    except httpx.RequestError: pass
                    await asyncio.sleep(1)
            if not replan:
                await self.kill("no answer during startup")
                raise HTTPException(status_code=500, detail="llama-server timed out during startup")
        print("[OPT] startup died from memory exhaustion - replanning once under the budget just learned", flush=True)
        await self.load(model, body, session, _replanned=True)

    def on_exit(self):
        """Died by itself: classified and learned from, as a death during a request is."""
        self.note_exit(self.model["name"] if self.model else "", self.plan.get("est", {}).get("gpu_gb", 0))
        budget.release(self.name)

    def info(self) -> dict: return {**super().info(), **({"model": self.model["name"], "plan": {k: self.plan.get(k) for k in ("mode", "params", "est", "experiments", "notes")}} if self.model and self.ready() else {})}

# --- KV ring (the slot's attention state saved to disk, newest N kept) ---

def _ring_file(sv: MainServer, session: str) -> str: return re.sub(r"[^A-Za-z0-9._-]", "_", f"""ring__{sv.model["name"]}__{sv.meta.get("kv")}__fa{sv.plan.get("params", {}).get("flash_attn")}__{session}""") + ".bin"

def ring_entries() -> list:
    es = [{"file": f, "size": _fsize(os.path.join(cache_dir(), f)), "mtime": os.path.getmtime(os.path.join(cache_dir(), f))} for f in os.listdir(cache_dir()) if f.startswith("ring__") and f.endswith(".bin")]
    return sorted(es, key=lambda e: e["mtime"], reverse=True)

PINNED: dict = {"fn": lambda f: False}   # modules/fleet replaces this: a pinned context survives pruning

def _ring_prune(keep: str = ""):
    for e in ring_entries()[KV_RING_SIZE:]:
        if e["file"] != keep and not PINNED["fn"](e["file"]): Path(cache_dir(), e["file"]).unlink(missing_ok=True)

def _observed(kind: str, t0: float, session: str, res: dict) -> dict:
    journal.event(kind, session=session, seconds=round(time.time() - t0, 3), **{f: v for f, v in res.items() if v is None or isinstance(v, (str, int, float, bool))})
    return res

async def ring_save(sv: MainServer, session: str) -> dict:
    t0 = time.time()
    if not (sv.model and sv.alive()): return _observed("kv.save", t0, session, {"saved": False, "reason": "no model loaded"})
    free_gb = shutil.disk_usage(cache_dir()).free / 1e9
    if free_gb < KV_MIN_FREE_GB: return _observed("kv.save", t0, session, {"saved": False, "reason": f"free disk {free_gb:.1f} GB below KV_MIN_FREE_GB={KV_MIN_FREE_GB}"})
    fn = _ring_file(sv, session)
    r = await sv.client().post("/slots/0", params={"action": "save"}, json={"filename": f"tmp__{fn}"})   # written beside the good copy, swapped only if the slot actually had content
    d = r.json() if r.status_code == 200 else {}
    if not d.get("n_saved"):
        Path(cache_dir(), f"tmp__{fn}").unlink(missing_ok=True)
        return _observed("kv.save", t0, session, {"saved": False, "reason": f"HTTP {r.status_code} or empty slot", "detail": r.text[:200]})
    os.replace(os.path.join(cache_dir(), f"tmp__{fn}"), os.path.join(cache_dir(), fn))
    _ring_prune(keep=fn)
    return _observed("kv.save", t0, session, {"saved": True, "file": fn, "n_saved": d["n_saved"], "bytes": d.get("n_written")})

async def ring_restore(sv: MainServer, session: str) -> dict:
    t0, fn = time.time(), _ring_file(sv, session)
    if not os.path.exists(os.path.join(cache_dir(), fn)): return _observed("kv.restore", t0, session, {"restored": False, "reason": "no saved context for this model/kv/fa/session"})
    r = await sv.client().post("/slots/0", params={"action": "restore"}, json={"filename": fn})
    if r.status_code == 200: os.utime(os.path.join(cache_dir(), fn))
    return _observed("kv.restore", t0, session, {"restored": r.status_code == 200, "file": fn, "detail": r.text[:200] if r.status_code != 200 else ""})

async def _autosave_after_idle(sv: MainServer, session: str):
    try:
        await asyncio.sleep(KV_AUTOSAVE_IDLE_S)
        if sv.active == 0 and sv.plan.get("features", {}).get("ring_autosave", True) and sv.ready(): print(f"[KV] autosave: {await ring_save(sv, session)}", flush=True)
    except asyncio.CancelledError: pass

# --- The adapter ---

CHAT = {"options": {"num_predict": "n_predict", "temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p", "repeat_penalty": "repeat_penalty", "presence_penalty": "presence_penalty", "frequency_penalty": "frequency_penalty", "seed": "seed", "stop": "stop"}, "think": {"field": "chat_template_kwargs.enable_thinking", "map": {"false": False, "off": False, "none": False, "": False, "0": False}, "else": True}, "tools": "native"}

class LlamaAdapter(Adapter):
    journal_family = "llama"
    capabilities = {c: {"streams": c in ("chat", "generate")} for c in ("chat", "generate", "embed", "decide", "kv_cache", "lora")}

    def __init__(self):
        super().__init__("llama")
        self.main, self.roles = MainServer(), {}

    def models(self) -> list: return [{**tags_entry(m), "capabilities": _caps(m), "format": "gguf", "mtime": m["mtime"]} for m in scan_models()]

    def status(self) -> dict: return {"server_bin": server_bin(), "main_speed": SPEED, "last_crash": {k: LAST_CRASH.get(k) for k in ("t", "model", "kind")} if LAST_CRASH else None}

    # --- Which server a job needs ---

    def _role(self, job) -> str: return {"embed": "embed", "decide": "decide"}.get(job.capability, "main")

    def _role_server(self, role: str) -> Service:
        """A small server beside the main one (embeddings, decisions), with the arguments node settings give that role. Kept once started: the main model plans around it, instead of the two evicting each other during an ingest that alternates between them."""
        if role not in self.roles: self.roles[role] = Service(f"llama:{role}", {"health": "/health", "slots": 1, "idle_stop_s": settings.get(f"llama.roles.{role}.idle_stop_s", 0), "stop_timeout_s": 15}, events="engine", owner="llama")
        return self.roles[role]

    def need(self, job) -> dict:
        m = require_model(self._model_for(job))
        role = self._role(job)
        if role == "main":
            s, plan = self.main.budget_plan(m, model_meta_sync(m), job.body)
            est = plan.get("est") or {}
            ram = min(float(est.get("ram_gb") or m["size"] / 1e9), budget.can_free(services.REG, self.main.name))
            return {"service": self.main, "est_gb": float(est.get("ram_gb") or m["size"] / 1e9), "ram_gb": ram, "gpu_gb": float(est.get("gpu_gb") or 0), "label": m["name"], "load": lambda j: self.main.load(m, j.body, extract_session(j.body))}
        sv = self._role_server(role)
        async def load(j):
            sv.meta = {"model": m["name"]}
            args = [server_bin(), "--host", "127.0.0.1", "--port", "{port}", "-m", m["path"], "-np", "1", *settings.get(f"llama.roles.{role}.args", [])]
            if not await sv.start(args, optim.node_env(), sig=m["name"]): raise RuntimeError(f"""llama:{role} did not start ({sv.status}): {" | ".join(sv.tail(3))}""")
            sv.measure()
        return {"service": sv, "est_gb": m["size"] / 1e9 * float(settings.get("llama.role_ram_factor", 1.2)), "ram_gb": min(m["size"] / 1e9 * float(settings.get("llama.role_ram_factor", 1.2)), budget.can_free(services.REG, sv.name)), "keep": bool(settings.get(f"llama.roles.{role}.keep", True)), "label": m["name"], "load": load}

    def _model_for(self, job) -> str: return job.model or settings.get(f"defaults.{job.capability}_model") or settings.get(f"llama.roles.{job.capability}.model") or ""

    def fits(self, service, job) -> bool:
        m = find_model(self._model_for(job))
        if not m: return True   # the job fails with its own message when it runs
        return self.main.satisfied(m, job.body) if service is self.main else service.sig == m["name"]

    def on_job_end(self, job):
        if job.service is self.main and self.main.active == 0: self.main.autosave = jobs.spawn(_autosave_after_idle(self.main, self.main.session))

    # --- Running ---

    async def run(self, job, sv):
        if job.capability == "warm":
            yield {"type": "done", "reason": "stop"}
            return
        if job.capability == "embed":
            yield {"type": "result", "data": await self._embed(job, sv)}
        elif job.capability == "decide":
            yield {"type": "result", "data": await self._decide(job, sv)}
        else:
            async for ev in self._main_stream(job, sv): yield ev
        if job.capability in ("embed", "decide"): yield {"type": "done", "reason": "stop"}

    async def _embed(self, job, sv) -> dict:
        texts = job.body["input"] if isinstance(job.body.get("input"), list) else [job.body.get("input") or job.body.get("prompt") or ""]
        r = await sv.client().post("/v1/embeddings", json={"input": texts, "model": sv.sig})
        if r.status_code != 200: raise HTTPException(status_code=r.status_code, detail=r.text[:400])
        d = r.json()
        return {"model": job.model, "embeddings": [e["embedding"] for e in sorted(d["data"], key=lambda e: e["index"])], "prompt_eval_count": (d.get("usage") or {}).get("prompt_tokens")}

    async def _decide(self, job, sv) -> dict:
        """A short choice from a small model kept up for this: {input or messages, choices, system?}. A grammar holds the answer to exactly one of the choices, so a small model cannot wander and no JSON has to be generated or parsed; ms is the whole call."""
        b, t0 = job.body, time.time()
        if not b.get("choices"): raise HTTPException(status_code=400, detail="decide needs choices")
        msgs = b.get("messages") or [*([{"role": "system", "content": b["system"]}] if b.get("system") else []), {"role": "user", "content": str(b.get("input") or "")}]
        r = await sv.client().post("/v1/chat/completions", json={"messages": oai.coalesce(msgs), "stream": False, "temperature": 0, "n_predict": int(b.get("max_tokens", 32)), "grammar": "root ::= " + " | ".join(json.dumps(str(c)) for c in b["choices"]), "chat_template_kwargs": {"enable_thinking": False}})
        if r.status_code != 200: raise HTTPException(status_code=r.status_code, detail=r.text[:400])
        return {"choice": ((r.json().get("choices") or [{}])[0].get("message", {}).get("content") or "").strip(), "ms": round((time.time() - t0) * 1000), "model": job.model}

    async def _main_stream(self, job, sv):
        """chat or generate on the main server. If it dies before the first token, the failure is recorded, the plan recomputed under what was learned, and the request replayed - up to 3 launches."""
        body, name = job.body, job.model
        session = extract_session(body)
        sv.session = session
        if sv.autosave and not sv.autosave.done(): sv.autosave.cancel()
        extra = {"cache_prompt": bool(opt(body, "cache_prompt", True)), **sv.plan.get("payload", {})}
        gb, ram = sv.plan.get("est", {}).get("gpu_gb", 0), sv.plan.get("est", {}).get("ram_gb", 0)
        features = sv.plan.get("features", {})
        stall = features.get("stall_s") or (INFER_TIMEOUT_S if body.get("stream") is False else None)
        optim.mark(sv.plan.get("model"), "pending", gb)
        for attempt in range(3):
            got = False
            try:
                gen = oai.chat_events(sv.client(), body, CHAT, extra, stall) if job.capability == "chat" else _generate_events(sv, body, extra, stall)
                async for ev in gen:
                    if ev["type"] in ("text", "thinking", "tool_calls"): got = True
                    if ev["type"] == "usage": _log_stats(job.capability, name, ev)
                    if ev["type"] == "done":
                        job.record["full"], job.record["calls"] = ev.pop("_full", None), ev.pop("_calls", None)
                    yield ev
                optim.mark(sv.plan.get("model"), "ok" if sv.alive() else "clear", gb, ram_gb=ram)
                return
            except asyncio.TimeoutError:
                optim.mark(sv.plan.get("model"), "clear", gb, ram_gb=ram)
                await sv.kill("no output: wedged")
                sv.note_exit(name, gb, kind="stall")
                raise HTTPException(status_code=503, detail=f"llama-server gave no output in {stall}s - killed as wedged; retry")
            except httpx.TransportError as e:
                await asyncio.sleep(1.5)
                dead = not sv.alive()
                if dead and sv.status == "up": sv.note_exit(name, gb)
                optim.mark(sv.plan.get("model"), "clear", gb, ram_gb=ram)
                print(f"""[LLAMA] llama-server dropped a request ({type(e).__name__}){f" - exited with code {sv.proc.returncode}" if dead else ""}""", flush=True)
                if got or attempt == 2: raise HTTPException(status_code=503, detail=f"""llama-server dropped the request ({type(e).__name__}){f" - it exited with code {sv.proc.returncode}, see /api/optimizer/last_crash" if dead else ""}; retry""")
                print(f"[OPT] llama-server died before the first token (attempt {attempt + 1}) - replanning and retrying", flush=True)
                await sv.load(require_model(name), body, session)
                optim.mark(sv.plan.get("model"), "pending", gb)
            except BaseException:
                optim.mark(sv.plan.get("model"), "clear", gb, ram_gb=ram)
                raise

async def _generate_events(sv, body: dict, extra: dict, stall):
    t0, chunks, timings, parser = time.time(), 0, {}, oai.StreamThinkParser()
    o = body.get("options") if isinstance(body.get("options"), dict) else {}
    t = CHAT["options"]
    full = ""
    async for line in oai.lines(sv.client(), "/completion", {"prompt": body.get("prompt", ""), "stream": True, **{t[k]: v for k, v in o.items() if k in t and v is not None}, **extra}, stall):
        if not line.startswith("data: "): continue
        try: d = json.loads(line[6:].strip())
        except json.JSONDecodeError: continue
        timings, full = d.get("timings") or timings, full + d.get("content", "")
        r, c = parser.process(d.get("content", ""))
        if r: yield {"type": "thinking", "text": r}
        if c:
            chunks += 1
            yield {"type": "text", "text": c}
    r, c = parser.flush()
    if r: yield {"type": "thinking", "text": r}
    if c: yield {"type": "text", "text": c}
    yield oai.usage_event(timings, {}, time.time() - t0, chunks)
    yield {"type": "done", "reason": "stop", "_full": {"content": full, "reasoning": ""}, "_calls": []}

def _log_stats(kind: str, name: str, u: dict):
    pn, pms, gn, gms = u.get("prompt_n"), u.get("prompt_ms"), u.get("gen_n"), u.get("gen_ms")
    rd = f"""read {pn} ({pn / pms * 1000:.2f} t/s) cached {u.get("cache_n", 0)}""" if pn and pms else "read n/a"
    print(f"""[STATS] {kind}: {rd} | {f"write {gn} ({gn / gms * 1000:.2f} t/s)" if gn and gms else f"write ~{gn} chunks"}""", flush=True)
    if gn and gms: SPEED.update(model=name, ms=gms / gn)   # the main model's ms per token, for the drafts' measured cost ratio
    if u.get("draft_n") is not None:
        ok = u.get("draft_ok") or 0
        try:
            with open(trajectory_log(), "a") as f: f.write(json.dumps({"t": _now(), "model": name, "draft_n": u["draft_n"], "draft_accept": ok, "accept_rate": round(ok / u["draft_n"] * 100, 2) if u["draft_n"] else 0.0, "eval_count": gn}) + "\n")
        except OSError as e: print(f"[DRAFT TELEMETRY] Write error: {e}", flush=True)

# --- Routes kept from the llama supervisor (their paths are part of what the portal's profile and bench tools know) ---

def _main() -> MainServer: return A["adapter"].main

@router.post("/api/cache/save", scope="infer")
async def cache_save(request):
    body, sv = await json_body(request), _main()
    if body.get("filename"): return (await sv.client().post("/slots/0", params={"action": "save"}, json={"filename": os.path.basename(body["filename"])})).json()   # explicit named save, outside the ring
    return await ring_save(sv, extract_session(body))

@router.post("/api/cache/restore", scope="infer")
async def cache_restore(request):
    body, sv = await json_body(request), _main()
    if body.get("filename"): return (await sv.client().post("/slots/0", params={"action": "restore"}, json={"filename": os.path.basename(body["filename"])})).json()
    return await ring_restore(sv, extract_session(body))

@router.get("/api/cache/list")
async def cache_list(request): return {"ring_size": KV_RING_SIZE, "free_gb": round(shutil.disk_usage(cache_dir()).free / 1e9, 1), "entries": ring_entries()}

@router.get("/api/cache/last_restore")
async def last_restore(request): return LAST_RESTORE.pop(request.query_params.get("session", "default"), {"restored": None, "reason": "no restore attempted since the last report"})

@router.delete("/api/cache", scope="infer")
async def cache_purge(request):
    fn = os.path.basename((await json_body(request)).get("file", ""))
    if not (fn.startswith("ring__") and fn.endswith(".bin")): raise HTTPException(status_code=400, detail="not a ring file")
    Path(cache_dir(), fn).unlink(missing_ok=True)
    return {"status": "success"}

@router.get("/api/lora/available")
async def lora_available(request): return {"loras": sorted(f for f in os.listdir(lora_dir()) if f.endswith(".gguf"))}

@router.get("/api/lora/list")
async def lora_list(request):
    sv = _main()
    return (await sv.client().get("/lora-adapters")).json() if sv.ready() else {"adapters": [], "note": "no model loaded"}

@router.post("/api/lora/set", scope="infer")
async def lora_set(request):
    """Adjusts scale (0 = off) on adapters passed at launch through options.lora_paths; a new adapter file needs a reload, same as a model swap. body: {"adapters": [{"id": 0, "scale": 0.7}]}"""
    return (await _main().client().post("/lora-adapters", json=(await json_body(request)).get("adapters", []))).json()

@router.get("/api/optimizer/sysinfo")
async def optimizer_sysinfo(request): return optim.info(bool(request.query_params.get("refresh")))

@router.post("/api/optimizer/plan", scope="read")
async def optimizer_plan(request):
    """Dry run: plans a launch for body.model with body.options and body.constraints and starts nothing."""
    body = await json_body(request)
    m = require_model(body.get("model", ""))
    return _main().budget_plan(m, await model_meta(m), body)[1]

@router.get("/api/optimizer/experiments")
async def optimizer_experiments(request): return optim.experiments_status()

@router.post("/api/optimizer/refresh")
async def optimizer_refresh(request):
    await optim.manager_hello()
    return optim.MANAGER

@router.route("/api/optimizer/forget", ("GET", "POST"))
async def optimizer_forget(request):
    optim.forget(request.query_params.get("model", ""))
    return {"status": "success", "cleared": request.query_params.get("model") or "all"}

@router.route("/api/optimizer/clear_ring", ("GET", "POST"))
async def optimizer_clear_ring(request):
    n = 0
    for e in ring_entries(): Path(cache_dir(), e["file"]).unlink(missing_ok=True); n += 1
    return {"status": "success", "removed": n}

@router.get("/api/optimizer/last_crash")
async def optimizer_last_crash(request): return LAST_CRASH or {"note": "no crash recorded since this node started"}

@router.get("/api/optimizer/log")
async def optimizer_log(request): return {"tail": _main().tail(int(request.query_params.get("n", 80)))}

@router.get("/api/hang_log")
async def hang_log(request): return {"entries": [json.loads(l) for l in open(crash_log()).readlines()[-30:]]} if os.path.exists(crash_log()) else {"entries": []}

# --- Setup ---

def setup(ctx):
    os.environ.update(drivers.state("llama").get("env") or {})   # which backend the driver built: the planner's rules read it
    for d in (local_dir(), lora_dir(), cache_dir()): os.makedirs(d, exist_ok=True)
    A["adapter"] = LlamaAdapter()
    ctx.add_adapter(A["adapter"])
    ctx.add_router(router)
    async def startup(): await optim.startup([{"name": m["name"], "size": m["size"], "meta": await model_meta(m)} for m in scan_models()])
    return startup()
