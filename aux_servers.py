"""aux_servers.py - helper llama-server processes beside the main one on this node, configured in aux_servers.json (path: AUX_SERVERS_CONFIG).
Roles:
  draft - a smaller model whose LoRA adapter is trained live, while the node serves, from the main model's finished replies (llama-finetune-lora: llama.cpp, no PyTorch).
          Its agreement with the main model is measured before and after every round. The main model is never touched: users get the model they asked for, and the drafts grow beside it.
  embed - an embedding model served Ollama-style at /api/embed and /api/embeddings.
Every slot's memory is measured once it is up and reserved in the planner (optimizer.reserve_external), so the main model is planned around the slots.
Experiment record, under MODELS_DIR/aux: samples.jsonl (every sample received), lab_log.jsonl (one line per evaluation, training round and slot event), <slot>/ (server.log, round data and trainer output, adapters, state.json)."""
import os, re, json, time, asyncio, subprocess
from pathlib import Path
import httpx
from fastapi import APIRouter, Request, HTTPException

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/root/.ollama/models"))
AUX_DIR = MODELS_DIR / "aux"
CONFIG_PATH = Path(os.getenv("AUX_SERVERS_CONFIG", Path(__file__).parent / "aux_servers.json"))
LAB_LOG, SAMPLES_LOG = AUX_DIR / "lab_log.jsonl", AUX_DIR / "samples.jsonl"
router = APIRouter()
CFG: dict = {}
SLOTS: dict = {}
MAIN_SPEED: dict = {}      # {"model", "ms"}: the main model's last measured milliseconds per generated token (from main.py's stats line)
RESERVED: dict = {}
GPU_CHECK: dict = {}   # {"result": optimizer.gpu_check of the slots' environment} - "" when the build has kernels for the GPU
_TRAIN_SEM = None
_SAMPLE_N = 0
_MAIN_BUSY = lambda: False   # set by start(): True while the main llama-server is answering a request
_TRAINING_GB: dict = {}    # slot -> GB set aside for its running trainer, so trainers started side by side do not each count the same free memory

# --- Helpers ---

def _now() -> str: return time.strftime("%Y-%m-%dT%H:%M:%S")
def _append(path: Path, rec: dict):
    with open(path, "a") as f: f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
def log_event(event: str, **rec): _append(LAB_LOG, {"t": _now(), "event": event, **rec})
def _fill(args: list, values: dict) -> list: return [re.sub(r"\{(\w+)\}", lambda m: str(values[m.group(1)]), a) for a in args]
def _mem_avail_gb() -> float:
    """Memory a new process can take: the smaller of MemAvailable and, inside a container with a memory limit (cgroup v2), the limit minus what the container uses (inactive file cache not counted, as docker stats does)."""
    avail = next(int(l.split()[1]) / 1e6 for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:"))
    cg = Path("/sys/fs/cgroup")
    if not (cg / "memory.max").exists() or (cg / "memory.max").read_text().strip() == "max": return avail
    inactive = next((int(l.split()[1]) for l in (cg / "memory.stat").read_text().splitlines() if l.startswith("inactive_file ")), 0)
    return min(avail, (int((cg / "memory.max").read_text()) - int((cg / "memory.current").read_text()) + inactive) / 1e9)
def note_main_speed(model: str, ms_per_token: float): MAIN_SPEED.update(model=model, ms=ms_per_token)

async def _main_idle():
    """With yield_to_main (default on), waits while the main model is answering, so draft evaluation and trainer starts do not run on the GPU / memory bus at the same time as a user's request. A request that arrives mid-call still overlaps that one call."""
    while CFG.get("yield_to_main", True) and _MAIN_BUSY(): await asyncio.sleep(1)

def speedup(p: float, k: int, c: float) -> float:
    """Expected lossless speculative speedup: tokens gained per verification, (1 - p^(k+1)) / (1 - p) (k + 1 when every draft token is accepted), over its cost: k draft tokens at c of a main token each plus one main pass."""
    return ((k + 1) if p >= 1 else (1 - p ** (k + 1)) / (1 - p)) / (k * c + 1)

def _pool(ms: list) -> dict:
    """Pooled agreement over several measurements. p is the per-token acceptance estimate: matched / (matched + misses), each draft run stopping at its first miss."""
    ms = [m for m in ms if m]
    if not ms: return {}
    a, f, pos = sum(m["matched"] for m in ms), sum(m["misses"] for m in ms), sum(m["positions"] for m in ms)
    p = a / (a + f) if a + f else 0.0
    return {"samples": len(ms), "positions": pos, "matched": a, "misses": f, "p": round(p, 4), "mean_accept": round(a / pos, 3) if pos else 0.0, "speedup": round(speedup(p, CFG["spec_k"], CFG["cost_ratio"]), 3)}

# --- Slots ---

class Slot:
    def __init__(self, spec: dict, model: dict, server_bin: str, env: dict):
        self.spec, self.name, self.role, self.model, self.bin, self.env = spec, spec["name"], spec.get("role", "draft"), model, server_bin, env
        self.port, self.dir, self.proc, self.task = int(spec["port"]), AUX_DIR / spec["name"], None, None
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state = json.loads((self.dir / "state.json").read_text()) if (self.dir / "state.json").exists() else {"version": 0, "adapter": None, "rounds": 0, "failures": 0, "training_off": ""}
        args = spec.get("args", [])
        self.ctx = int(next((args[i + 1] for i, a in enumerate(args[:-1]) if a in ("-c", "--ctx-size")), 4096))
        self.status, self.queue, self.last_error = "stopped", asyncio.Queue(maxsize=int(spec.get("queue_max", 100))), {}
        self.buffer, self.buffer_tokens, self.holdout, self.dropped, self.last = [], 0, [], 0, {}
        self.start_lock = asyncio.Lock()   # one start at a time: two requests finding the server down would otherwise both start it on the same port
        self.http = httpx.AsyncClient(base_url=f"http://127.0.0.1:{self.port}", timeout=httpx.Timeout(connect=10.0, read=900.0, write=60.0, pool=10.0))

    def _save_state(self): (self.dir / "state.json").write_text(json.dumps(self.state, indent=2))
    def rss_gb(self) -> float:
        st = Path(f"/proc/{self.proc.pid}/status") if self.proc else None
        return next((int(l.split()[1]) / 1e6 for l in st.read_text().splitlines() if l.startswith("VmRSS:")), 0.0) if st and st.exists() else 0.0
    def _log_tail(self, n: int = 15) -> list: return (self.dir / "server.log").read_text(errors="replace").splitlines()[-n:] if (self.dir / "server.log").exists() else []

    async def start(self) -> bool:
        """Starts the server (a draft with its current adapter) and waits for /health. False, with the log tail recorded, when it exits or does not answer within 300 s."""
        adapter = self.state.get("adapter") if self.role == "draft" else None
        cmd = [self.bin, "--host", "127.0.0.1", "--port", str(self.port), "-m", self.model["path"], "-np", "1", *self.spec.get("args", []), *(["--lora", adapter] if adapter else [])]
        with open(self.dir / "server.log", "a") as f: f.write(f"""\n=== {_now()} {" ".join(cmd)}\n""")
        self.proc = subprocess.Popen(cmd, env={**os.environ, **self.env}, stdout=open(self.dir / "server.log", "a"), stderr=subprocess.STDOUT)
        self.status = "starting"
        for _ in range(300):
            if self.proc.poll() is not None: break
            try:
                if (await self.http.get("/health")).status_code == 200: self.status = "up"; return True
            except httpx.HTTPError: pass
            await asyncio.sleep(1)
        self.status = "failed"
        log_event("slot_failed", slot=self.name, returncode=self.proc.poll(), tail=self._log_tail())
        print(f"""[AUX] {self.name} did not come up (exit {self.proc.poll()}) - see {self.dir / "server.log"}""", flush=True)
        return False

    async def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try: await asyncio.to_thread(self.proc.wait, 15)
            except subprocess.TimeoutExpired: self.proc.kill()
        self.proc, self.status = None, "stopped"   # a stop on purpose, so ensure() does not record it as a death

    async def ensure(self) -> bool:
        """Running, or restarted (a death is recorded first). One start at a time; a server that started but never answered is stopped before the next start."""
        async with self.start_lock:
            if self.proc and self.proc.poll() is None and self.status == "up": return True
            if self.proc and self.proc.poll() is not None: log_event("slot_exited", slot=self.name, returncode=self.proc.returncode, tail=self._log_tail())
            if self.proc and self.proc.poll() is None: await self.stop()
            return await self.start()

    async def _post(self, path: str, body: dict) -> dict:
        r = await self.http.post(path, json=body)
        if r.status_code != 200: raise RuntimeError(f"""{self.name} {path}: HTTP {r.status_code} {r.text[:200]}""")
        return r.json()

    async def tokens(self, text: str, special: bool) -> list: return (await self._post("/tokenize", {"content": text, "add_special": special}))["tokens"]

    # --- Draft: measuring agreement with the main model ---

    async def encode(self, sample: dict) -> dict:
        """The sample in this slot's own tokens: the prompt rendered with the slot's chat template (as /chat/completions would), and the main model's reply text."""
        inp, out = sample["prompt"], sample["target_output"]
        reply = (((out.get("choices") or [{}])[0].get("message") or {}).get("content") if "choices" in out else out.get("content")) or ""
        msgs = [{"role": m.get("role"), "content": m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))} for m in inp] if isinstance(inp, list) else None
        prompt = (await self._post("/apply-template", {"messages": msgs}))["prompt"] if msgs else str(inp)
        return {"prompt_text": prompt, "reply_text": reply, "prompt_ids": await self.tokens(prompt, True), "reply_ids": await self.tokens(reply, False)}

    async def accept(self, enc: dict) -> dict:
        """Greedy draft runs of spec_k tokens from eval_positions points spread over the main model's reply; each run counts its tokens up to the first one that differs from the main model's."""
        k, ids, n = CFG["spec_k"], enc["reply_ids"], len(enc["reply_ids"])
        if n < 2: return {}
        pos_n = max(1, min(CFG["eval_positions"], n - 1))
        idx = sorted({int(i * (n - 1) / max(pos_n - 1, 1)) for i in range(pos_n)})
        matched = misses = 0; runs, ms = [], []
        for i in idx:
            await _main_idle()
            want = ids[i:i + k]
            r = await self._post("/completion", {"prompt": (enc["prompt_ids"] + ids[:i])[-(self.ctx - k - 8):], "n_predict": len(want), "temperature": 0, "top_k": 1, "cache_prompt": True, "return_tokens": True})
            got = r.get("tokens") or await self.tokens(r.get("content", ""), False)
            m = next((j for j, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            runs.append(m); matched += m; misses += 1 if m < len(want) else 0
            if (r.get("timings") or {}).get("predicted_per_token_ms"): ms.append(r["timings"]["predicted_per_token_ms"])
        p, d_ms = (matched / (matched + misses) if matched + misses else 0.0), (sum(ms) / len(ms) if ms else None)
        c = d_ms / MAIN_SPEED["ms"] if d_ms and MAIN_SPEED.get("ms") else None
        return {"positions": len(idx), "matched": matched, "misses": misses, "runs": runs, "p": round(p, 4), "mean_accept": round(matched / len(idx), 3), "speedup": round(speedup(p, k, CFG["cost_ratio"]), 3),
                "draft_ms_per_token": round(d_ms, 2) if d_ms else None, "c_measured": round(c, 3) if c else None, "speedup_measured": round(speedup(p, k, c), 3) if c else None}

    async def accept_many(self, encs: list) -> dict: return _pool([await self.accept(e) for e in encs])

    # --- Draft: live adaptation ---

    async def run(self):
        """One sample at a time per slot, so evaluation and training never overlap within a slot; slots run side by side."""
        while True:
            sample = await self.queue.get()
            try: await self._handle(sample)
            except Exception as e:
                self.last_error = {"t": _now(), "sample": sample.get("n"), "error": f"""{type(e).__name__}: {str(e)[:400]}"""}
                log_event("error", slot=self.name, **self.last_error); print(f"""[AUX] {self.name} sample {sample.get("n")}: {self.last_error["error"]}""", flush=True)
            finally: self.queue.task_done()

    async def _handle(self, sample: dict):
        if not await self.ensure(): return
        enc = await self.encode(sample)
        if len(enc["reply_ids"]) < 2: return
        if sample["n"] % CFG["holdout_every"] == 0: self.holdout = (self.holdout + [enc])[-CFG["holdout_max"]:]; log_event("holdout", slot=self.name, sample=sample["n"], held=len(self.holdout)); return   # never trained on - the generalization check
        m = await self.accept(enc)
        self.last = m
        log_event("eval", slot=self.name, version=self.state["version"], sample=sample["n"], main_model=sample.get("main_model"), **m)
        if self.state.get("training_off") or CFG["train"].get("off"): return
        self.buffer.append({"enc": enc, "before": m, "sample": sample["n"]})
        self.buffer_tokens += len(enc["reply_ids"]) + (len(enc["prompt_ids"]) if CFG["train"].get("text") == "prompt+reply" else 0)
        if self.buffer_tokens >= CFG["train"]["min_tokens"]: await self.train_round()

    async def train_round(self):
        """One adaptation round: holdout measured with the current adapter -> trainer continues the adapter on this round's samples -> server restarted with the new adapter -> holdout and this round's samples measured again.
        A trainer failure keeps the previous adapter; three failures in a row turn training off for this slot (recorded in state.json; clear training_off there to resume)."""
        t, batch = CFG["train"], self.buffer
        self.buffer, self.buffer_tokens = [], 0
        if not batch: return
        async with _TRAIN_SEM:
            hold_before = await self.accept_many(self.holdout)
            v_out = self.state["version"] + 1
            data, out = self.dir / f"round_{v_out:04d}.txt", self.dir / f"adapter_v{v_out:04d}.gguf"
            data.write_text("\n\n".join((b["enc"]["prompt_text"] if t.get("text") == "prompt+reply" else "") + b["enc"]["reply_text"] for b in batch))
            await _main_idle()
            if t.get("stop_server", True): await self.stop()
            need, free = self.model["size"] / 1e9 * t["mem_factor"], _mem_avail_gb() - sum(_TRAINING_GB.values())
            if free < need + t["min_free_gb"]:
                log_event("round_skipped", slot=self.name, reason=f"""{free:.1f} GB free (after running trainers), trainer needs about {need:.1f} GB + min_free_gb {t["min_free_gb"]}""", samples=len(batch))
                self.buffer = batch; self.buffer_tokens = sum(len(b["enc"]["reply_ids"]) for b in batch)
                await self.ensure(); return
            _TRAINING_GB[self.name] = need
            args = _fill(t["args"], {"model": self.model["path"], "data": data, "ctx": t["ctx"], "threads": t["threads"], "lora_out": out}) + (_fill(t["args_continue"], {"lora_in": self.state["adapter"]}) if self.state.get("adapter") else [])
            t0 = time.time()
            proc = await asyncio.create_subprocess_exec(t["bin"], *args, env={**os.environ, **self.env, **t.get("env", {})}, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            try: text = (await proc.communicate())[0].decode(errors="replace")
            finally: _TRAINING_GB.pop(self.name, None)
            (self.dir / f"round_{v_out:04d}.log").write_text(f"""{t["bin"]} {" ".join(args)}\n\n{text}""")
            ok, secs = proc.returncode == 0 and out.exists(), round(time.time() - t0, 1)
            if ok: self.state.update(version=v_out, adapter=str(out), rounds=self.state["rounds"] + 1, failures=0)
            else: self.state["failures"] += 1; self.state["training_off"] = "3 trainer failures in a row" if self.state["failures"] >= 3 else ""
            self._save_state()
            await self.ensure()
            rec = {"slot": self.name, "ok": ok, "version": self.state["version"], "samples": [b["sample"] for b in batch], "tokens": sum(len(b["enc"]["reply_ids"]) for b in batch), "seconds": secs, "returncode": proc.returncode,
                   "loss_lines": [l for l in text.splitlines() if "loss" in l.lower()][-3:], "holdout_before": hold_before}
            if ok: rec.update(holdout_after=await self.accept_many(self.holdout), fit_before=_pool([b["before"] for b in batch]), fit_after=await self.accept_many([b["enc"] for b in batch]))
            else: rec["tail"] = text.splitlines()[-15:]
            log_event("round", **rec)
            print(f"""[AUX] {self.name} round {"v" + str(v_out) if ok else "FAILED"} in {secs}s - holdout p {hold_before.get("p")} -> {rec.get("holdout_after", {}).get("p")}, this round's samples p {rec.get("fit_before", {}).get("p")} -> {rec.get("fit_after", {}).get("p")}""", flush=True)

    def info(self) -> dict:
        """Status as of now: a server that died since it was last used shows as exited with its return code and the end of its log, not as up."""
        dead = self.proc is not None and self.proc.poll() is not None
        return {"name": self.name, "role": self.role, "model": self.model["name"], "port": self.port, "status": f"exited ({self.proc.returncode})" if dead else self.status, "pid": self.proc.pid if self.proc else None, "rss_gb": round(self.rss_gb(), 2), "last_error": self.last_error, **({"log_tail": self._log_tail(8)} if dead or self.status == "failed" else {}),
                **({"adapter": self.state, "queued": self.queue.qsize(), "dropped": self.dropped, "buffer_tokens": self.buffer_tokens, "holdout": len(self.holdout), "last_eval": {k: v for k, v in self.last.items() if k != "runs"}} if self.role == "draft" else {"aliases": self.spec.get("aliases", [])})}

# --- Lifecycle (called from main.py's lifespan) ---

def _check_trainer():
    """The trainer is a separate binary whose flags vary by build: report flags from aux_servers.json that its --help does not mention, and turn training off (evaluation still runs) when it is missing."""
    t = CFG["train"]
    if not Path(t["bin"]).exists(): t["off"] = f"""trainer not found: {t["bin"]}"""; print(f"""[AUX] {t["off"]} - drafts are measured but not trained""", flush=True); log_event("trainer_check", off=t["off"]); return
    r = subprocess.run([t["bin"], "--help"], capture_output=True, text=True, timeout=30, env={**os.environ, **t.get("env", {})})
    missing = sorted({a for a in t["args"] + t["args_continue"] if a.startswith("-") and a not in r.stdout + r.stderr})
    log_event("trainer_check", bin=t["bin"], returncode=r.returncode, flags_not_in_help=missing)
    if missing: print(f"[AUX] WARNING trainer --help does not mention {missing} - check train.args in {CONFIG_PATH}", flush=True)

async def start(find_model, server_bin: str, optim, main_busy=lambda: False):
    """Starts every enabled slot one after another, measures what they took, and reserves it in the planner. A slot that fails is recorded and skipped; the main model still starts.
    Slot values may name environment variables as ${NAME} (e.g. a model set in the compose .env); a model name may end in .gguf. main_busy() tells whether the main model is answering (see yield_to_main)."""
    global CFG, _TRAIN_SEM, _SAMPLE_N, _MAIN_BUSY
    if not CONFIG_PATH.exists(): print(f"[AUX] {CONFIG_PATH} not found - no helper servers", flush=True); return
    CFG = json.loads(CONFIG_PATH.read_text())
    AUX_DIR.mkdir(parents=True, exist_ok=True)
    _TRAIN_SEM, _MAIN_BUSY = asyncio.Semaphore(int(CFG["train"]["parallel"])), main_busy
    _SAMPLE_N = sum(1 for _ in open(SAMPLES_LOG)) if SAMPLES_LOG.exists() else 0
    env, before = optim.node_env(), optim.gpu_used_bytes()
    GPU_CHECK["result"] = optim.gpu_check({**os.environ, **env})   # the slots start with this env: a missing kernel target shows here before they fail on it
    if GPU_CHECK["result"]: print(f"""[AUX] GPU check: {GPU_CHECK["result"]}""", flush=True); log_event("gpu_check", result=GPU_CHECK["result"])
    for spec in (json.loads(os.path.expandvars(json.dumps(s))) for s in CFG["slots"] if s.get("enabled", True)):
        if "${" in json.dumps(spec): print(f"""[AUX] {spec["name"]}: unset environment variable in {spec} - slot skipped""", flush=True); log_event("slot_missing", slot=spec["name"], unresolved=spec); continue
        m = find_model(re.sub(r"\.gguf$", "", spec["model"]))
        if not m: print(f"""[AUX] {spec["name"]}: model {spec["model"]!r} is not in /api/tags - slot skipped""", flush=True); log_event("slot_missing", slot=spec["name"], model=spec["model"]); continue
        SLOTS[spec["name"]] = Slot(spec, m, server_bin, env)
        print(f"""[AUX] starting {spec["name"]} ({spec.get("role", "draft")}, {m["name"]}) on port {spec["port"]}""", flush=True)
        await SLOTS[spec["name"]].start()
    RESERVED.update(gpu_gb=round(max(0, optim.gpu_used_bytes() - before) / 1e9, 2), ram_gb=round(sum(s.rss_gb() for s in SLOTS.values()), 2))
    optim.reserve_external(RESERVED["gpu_gb"], RESERVED["ram_gb"], f"""aux slots {", ".join(SLOTS) or "(none)"}""")
    print(f"""[AUX] up: {[s.name for s in SLOTS.values() if s.status == "up"]} | reserved from the planner: {RESERVED["gpu_gb"]} GB GPU, {RESERVED["ram_gb"]} GB RAM""", flush=True)
    log_event("start", slots={n: s.status for n, s in SLOTS.items()}, reserved=RESERVED)
    if any(s.role == "draft" for s in SLOTS.values()): _check_trainer()
    for s in SLOTS.values():
        if s.role == "draft": s.task = asyncio.create_task(s.run())

async def stop():
    for s in SLOTS.values():
        if s.task: s.task.cancel()
        await s.stop()

# --- Routes ---

@router.post("/api/draft/adapt")
async def draft_adapt(request: Request):
    """Receives one finished main-model request {prompt (messages or text), target_output} - from this node's passive worker or another node's - records it, and queues it for every draft slot."""
    global _SAMPLE_N
    _SAMPLE_N += 1
    sample = {"n": _SAMPLE_N, "t": _now(), "main_model": MAIN_SPEED.get("model"), "main_ms_per_token": MAIN_SPEED.get("ms"), **(await request.json())}
    _append(SAMPLES_LOG, sample)
    queued = []
    for s in (s for s in SLOTS.values() if s.role == "draft"):
        if s.queue.full(): s.dropped += 1; continue
        s.queue.put_nowait(sample); queued.append(s.name)
    return {"sample": _SAMPLE_N, "queued": queued}

@router.get("/api/aux/status")
async def aux_status(): return {"config": str(CONFIG_PATH), "reserved": RESERVED, "gpu_check": GPU_CHECK.get("result") or "ok", "trainer": CFG.get("train", {}).get("off") or CFG.get("train", {}).get("bin"), "spec_k": CFG.get("spec_k"), "cost_ratio": CFG.get("cost_ratio"), "main_speed": MAIN_SPEED, "slots": [s.info() for s in SLOTS.values()]}

@router.get("/api/aux/log")
async def aux_log(n: int = 50, event: str = ""):
    if not LAB_LOG.exists(): return {"entries": []}
    rows = [json.loads(l) for l in LAB_LOG.read_text().splitlines()]
    return {"entries": [r for r in rows if not event or r["event"] == event][-n:]}

@router.post("/api/aux/train_now")
async def aux_train_now(slot: str):
    """Runs a round now on whatever the slot has buffered (below min_tokens too)."""
    s = SLOTS.get(slot)
    if not s or s.role != "draft": raise HTTPException(status_code=404, detail=f"no draft slot {slot}")
    if not s.buffer: return {"started": False, "reason": "nothing buffered"}
    asyncio.create_task(s.train_round())
    return {"started": True, "samples": len(s.buffer)}

def _embed_slot(model: str) -> Slot:
    slot = next((s for s in SLOTS.values() if s.role == "embed" and (not model or model in s.spec.get("aliases", []) or model == s.model["name"])), None)
    if not slot: raise HTTPException(status_code=404, detail=f"""no embedding slot serves {model!r} - add one to {CONFIG_PATH.name} (role "embed", with this name in aliases)""")
    return slot

async def _embed(model: str, texts: list) -> list:
    slot = _embed_slot(model)
    if not await slot.ensure(): raise HTTPException(status_code=503, detail=f"""embedding slot {slot.name} is not running ({slot.status}) - {slot.dir / "server.log"}: {" | ".join(slot._log_tail(3))}""")
    return [e["embedding"] for e in sorted((await slot._post("/v1/embeddings", {"input": texts, "model": slot.model["name"]}))["data"], key=lambda e: e["index"])]

@router.post("/api/embed")
async def ollama_embed(request: Request):
    """Ollama /api/embed: {model, input: text or list} -> {model, embeddings}."""
    b = await request.json()
    return {"model": b.get("model", ""), "embeddings": await _embed(b.get("model", ""), b["input"] if isinstance(b.get("input"), list) else [b.get("input", "")])}

@router.post("/api/embeddings")
async def ollama_embeddings(request: Request):
    """Older Ollama /api/embeddings: {model, prompt} -> {embedding}."""
    b = await request.json()
    return {"embedding": (await _embed(b.get("model", ""), [b.get("prompt", "")]))[0]}


