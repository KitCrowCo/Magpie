"""fleet.py - the node side of Fleet (the portal's Fleet tool is the other side): a journal of what this node does, what it holds now, and the actions a manager may ask for.
Journal - MODELS_DIR/fleet/journal.<first id>.jsonl, one event per line: {"id", "t", "kind", ...}. Ids rise by one per event within one epoch (MODELS_DIR/fleet/epoch: a random id made when the journal starts empty),
          so a reader tells a continued journal from a reset one. A file rotates at journal.max_mb; files beyond journal.keep_files are deleted, oldest first.
Readers pull - POST /fleet/events {"after": id, "limit": n}. A field named id, t or kind is stored as id_, t_, kind_ (they are the journal's own). Nothing is pushed: the node never needs to know who reads it, and any number of managers can read it.
Hooks - main.py and engines.py call event(), request_start/running/end() and timings() where things happen. A hook never raises into its caller: a failure is counted and shown in /fleet/state (hook_errors).
        This is the one place errors are caught on purpose - the journal must never fail a user's request.
State - GET /fleet/state: what the node holds now (the loaded model and its plan, models on disk, KV contexts and pins, engines, memory, prefetch, tests).
Actions - POST /fleet/apply: warm, unload, kv_save, kv_drop, pin, unpin, prefetch, engine_start, engine_stop. fleet.json decides which ones this node accepts and whether they only report what they would do -
          the node owner's choice, not the manager's. Every request for an action is journaled (directive), and so is its outcome (directive.done).
Collectors - fleet.json "collectors": Prometheus endpoints (colibri's /metrics), log lines matched by a regex (an engine's log), file sizes (a usage file growing); each reading becomes an event.
Tests - fleet_tests.py (POST /fleet/test).
Settings: fleet.json (path FLEET_CONFIG). FLEET=0 turns all of it off: hooks do nothing, routes answer 503."""
import os, re, json, time, uuid, asyncio, threading, collections, functools
from pathlib import Path
import httpx
from fastapi import APIRouter, Request, HTTPException

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/root/.ollama/models"))
DIR = MODELS_DIR / "fleet"
PINS_PATH = DIR / "pins.json"
CONFIG_PATH = Path(os.getenv("FLEET_CONFIG", Path(__file__).parent / "fleet.json"))
ON = os.getenv("FLEET", "1") != "0"
DEFAULTS = {"journal": {"max_mb": 20, "keep_files": 5, "memory_events": 5000}, "sample_s": 30, "apply": {"mode": "on", "allow": ["warm", "unload", "kv_save", "kv_drop", "pin", "unpin", "prefetch", "engine_start", "engine_stop"], "idle_only": True},
            "tests": {"allow": True, "idle_wait_s": 600}, "prefetch": {"chunk_mb": 16}, "collectors": []}
DISRUPTIVE = {"warm", "unload", "prefetch", "engine_start", "engine_stop"}   # actions that take the model, the memory or the disk away from a request in progress: refused while busy under apply.idle_only (args.force overrides)
router = APIRouter()
CFG: dict = {}
HOOKS: dict = {}                 # from main.py at start(): node, port, busy, sample, resident, models, find, ring, cache_dir, engines, warm, unload, kv_save, cancel
EXTRA_STATE: dict = {}           # name -> function returning a part of /fleet/state, for other files (fleet_tests.py adds "tests")
ACTIONS: dict = {}               # action name -> async function(args) -> result dict
HOOK_ERRORS = {"n": 0, "last": ""}
PINS: set = set()
PREFETCH = {"running": None, "last": None}
_J = {"id": 0, "epoch": "", "file": None, "fh": None}
_RECENT = collections.deque(maxlen=5000)   # the newest events, so a reader that keeps up never touches the files
_REQ: dict = {}                  # request id -> what the hooks learned about it so far (start, running, timings)
_COLLECTORS: dict = {}           # collector name -> its status (runs, events, last, error, file offset)
_LOCK = threading.Lock()         # events come from the event loop and from threads (prefetch)
_TASKS: set = set()              # background tasks, held because the event loop keeps only weak references

def _now() -> float: return round(time.time(), 3)
def _job(rid: str) -> str: return rid.split(".", 1)[0]   # the portal sends <job id>.<random>: the job is everything before the first dot
def _num(v):
    try: return int(v)
    except (TypeError, ValueError):
        try: return float(v)
        except (TypeError, ValueError): return v

def spawn(coro) -> asyncio.Task:
    t = asyncio.create_task(coro)
    _TASKS.add(t)
    t.add_done_callback(_TASKS.discard)
    return t

def require_on():
    if not ON: raise HTTPException(status_code=503, detail="fleet is off on this node (FLEET=0)")

def _hook_error(e: BaseException):
    HOOK_ERRORS.update(n=HOOK_ERRORS["n"] + 1, last=f"""{time.strftime("%Y-%m-%d %H:%M:%S")} {type(e).__name__}: {e}"""[:300])
    print(f"""[FLEET] hook failed ({HOOK_ERRORS["n"]} so far): {HOOK_ERRORS["last"]}""", flush=True)

def _hook(fn):
    """For functions the serving path calls: nothing when FLEET=0, and a failure is counted and printed, never raised into the caller."""
    @functools.wraps(fn)
    def inner(*a, **k):
        if not ON: return None
        try: return fn(*a, **k)
        except Exception as e: _hook_error(e)
    return inner

# --- Journal ---

def _files() -> list: return sorted(DIR.glob("journal.*.jsonl"))   # zero-padded first ids, so name order is id order
def _first_id(f: Path) -> int: return int(f.name.split(".")[1])

def _last_id(f: Path) -> int:
    """The id of the last complete event in f; a line cut off by a crash is skipped."""
    with open(f, "rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 65536))
        lines = fh.read().decode("utf-8", "replace").splitlines()
    for line in reversed(lines):
        try: return int(json.loads(line)["id"])
        except (ValueError, KeyError, TypeError): continue
    return _first_id(f) - 1

def _open():
    DIR.mkdir(parents=True, exist_ok=True)
    files, ep = _files(), DIR / "epoch"
    if not files or not ep.exists(): ep.write_text(uuid.uuid4().hex[:12])   # an empty journal starts a new epoch: readers start over instead of waiting for ids they already have
    _J["epoch"], _J["id"] = ep.read_text().strip(), _last_id(files[-1]) if files else 0
    _J["file"] = files[-1] if files else DIR / f"journal.{1:012d}.jsonl"
    _J["fh"] = open(_J["file"], "a", buffering=1, encoding="utf-8")
    _RECENT.extend(_scan(max(0, _J["id"] - _RECENT.maxlen)))

def _rotate():
    _J["fh"].close()
    _J["file"] = DIR / f"""journal.{_J["id"]:012d}.jsonl"""   # named after the first id it holds (the one being written)
    _J["fh"] = open(_J["file"], "a", buffering=1, encoding="utf-8")
    for f in _files()[:-int(CFG["journal"]["keep_files"])]: f.unlink(missing_ok=True)

def _write(kind: str, data: dict):
    with _LOCK:
        _J["id"] += 1
        rec = {"id": _J["id"], "t": _now(), "kind": kind, **{(f"{k}_" if k in ("id", "t", "kind") else k): v for k, v in data.items()}}   # a field named id, t or kind is kept as id_, t_, kind_ (llama.crash carries the crash's kind_)
        if _J["fh"].tell() > float(CFG["journal"]["max_mb"]) * 1e6: _rotate()
        _J["fh"].write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        _RECENT.append(rec)

def _scan(after: int, limit: int = 10 ** 9) -> list:
    """Events after the id `after`, read from the files (a reader catching up); a line cut off by a crash is skipped."""
    files, out = _files(), []
    start = max([i for i, f in enumerate(files) if _first_id(f) <= after + 1], default=0)
    for f in files[start:]:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try: rec = json.loads(line)
                except ValueError: continue
                if rec.get("id", 0) > after:
                    out.append(rec)
                    if len(out) >= limit: return out
    return out

def read(after: int, limit: int) -> list:
    with _LOCK: recent = list(_RECENT)
    if recent and after >= recent[0]["id"] - 1: return [r for r in recent if r["id"] > after][:limit]
    return _scan(after, limit)

@_hook
def record(kind: str, data: dict):
    """One journal entry: kind (dotted, e.g. req.end, llama.ready, kv.save) and plain JSON fields."""
    _write(kind, data)

def event(kind: str, /, **data): record(kind, data)   # kind positional-only: a field may itself be called kind (llama.crash kind=memory)

# --- Hooks from the serving path (main.py) ---

_TIMING = (("prompt_n", "prompt_n"), ("prompt_ms", "prompt_ms"), ("cache_n", "cache_n"), ("predicted_n", "gen_n"), ("predicted_ms", "gen_ms"), ("draft_n", "draft_n"), ("spec_n", "draft_n"), ("draft_n_accepted", "draft_ok"), ("draft_accept", "draft_ok"), ("spec_accept", "draft_ok"))   # llama-server timing names -> journal names (builds name the draft counts differently)

@_hook
def request_start(rid: str, kind: str, model: str, session: str):
    _REQ[rid] = {"t0": time.time(), "api": kind, "model": model, "session": session}
    _write("req.start", {"rid": rid, "job": _job(rid), "api": kind, "model": model, "session": session})

@_hook
def request_running(rid: str):
    """The request's own llama-server call starts: the time before it was spent loading the model or waiting for other requests (wait_s)."""
    r = _REQ.get(rid)
    if r is not None and "run_t" not in r: r["run_t"] = time.time()

@_hook
def timings(rid: str, t: dict):
    r = _REQ.get(rid)
    if r is None or not t: return
    r["timings"] = {}
    for src, dst in _TIMING:
        if t.get(src) is not None: r["timings"].setdefault(dst, t[src])

@_hook
def response_timings(rid: str, resp):
    """timings() from a non-streamed llama-server reply (main.py's infer_post), read before the request is settled."""
    if "json" in resp.headers.get("content-type", ""): timings(rid, resp.json().get("timings") or {})

@_hook
def request_end(rid: str, outcome: str):
    r = _REQ.pop(rid, None)
    if r is None: return
    now = time.time()
    _write("req.end", {"rid": rid, "job": _job(rid), "api": r["api"], "model": r["model"], "session": r["session"], "outcome": outcome, "seconds": round(now - r["t0"], 3), "wait_s": round(r.get("run_t", now) - r["t0"], 3), **r.get("timings", {})})

def observe(kind: str, arg: str = "arg"):
    """Decorator for an async function whose result is worth journaling (main.py: ring_save, ring_restore): one event per call with its first argument (named arg), its seconds and the result's plain fields.
    An error is journaled and raised again - the decorated function behaves exactly as before."""
    def wrap(fn):
        if not ON: return fn
        @functools.wraps(fn)
        async def inner(*a, **k):
            t0 = time.time()
            try: res = await fn(*a, **k)
            except Exception as e:
                record(kind, {arg: a[0] if a else None, "seconds": round(time.time() - t0, 3), "ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
                raise
            record(kind, {arg: a[0] if a else None, "seconds": round(time.time() - t0, 3), **({f: v for f, v in res.items() if v is None or isinstance(v, (str, int, float, bool))} if isinstance(res, dict) else {"result": str(res)[:200]})})
            return res
        return inner
    return wrap

@_hook
def pinned(fn: str) -> bool:
    """A KV ring file the manager pinned, by file name or by session (main.py's ring pruning keeps it)."""
    return fn in PINS or any(p.startswith("session:") and fn.endswith(f"__{p[8:]}.bin") for p in PINS)

def _save_pins(): PINS_PATH.write_text(json.dumps(sorted(PINS), indent=1))

# --- Sampler and collectors ---

async def _sampler():
    while True:
        await asyncio.sleep(float(CFG["sample_s"]))
        try: _write("mem", HOOKS["sample"]())
        except Exception as e: _hook_error(e)

async def _c_prometheus(c: dict, st: dict) -> int:
    """Metrics in Prometheus text format from c.url; names matching the regex c.match (default: all) go into one event. Histogram buckets are left out unless c.buckets is true. c.engine: only while that engine runs."""
    eng = HOOKS["engines"].ENGINES.get(c["engine"]) if c.get("engine") else None
    if c.get("engine") and not (eng and eng.alive()): return 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=3.0, read=10.0, write=5.0, pool=3.0)) as cl: r = await cl.get(c["url"])
    r.raise_for_status()
    vals, rx = {}, re.compile(c.get("match", ""))
    for line in r.text.splitlines():
        m = re.match(r"^([^\s{#]+(?:\{[^}]*\})?)\s+(\S+)", line)
        if not m or ("_bucket" in m.group(1) and not c.get("buckets")) or not rx.search(m.group(1)): continue
        vals[m.group(1)] = _num(m.group(2))
    if vals: _write(c.get("kind", "metrics"), {"source": c["name"], "values": vals})
    return 1 if vals else 0

async def _c_log(c: dict, st: dict) -> int:
    """New lines of the file c.path since the last read; each line matching the regex c.pattern becomes an event of kind c.kind (default log.match) with the pattern's named groups (numbers as numbers).
    The first read starts at the end of the file (c.from_start: at its beginning); a file that shrank (rotated, rewritten) is read again from its start."""
    p = Path(os.path.expandvars(c["path"]))
    if not p.exists(): return 0
    size, off = p.stat().st_size, st.get("offset")
    off = (0 if c.get("from_start") else size) if off is None else (0 if off > size else off)
    def chunk():
        with open(p, "rb") as f:
            f.seek(off)
            return f.read(int(c.get("max_kb", 1024)) * 1024)
    data = await asyncio.to_thread(chunk)
    cut = data.rfind(b"\n") + 1   # whole lines only; the rest is read next time
    st["offset"], n = off + cut, 0
    rx = st.setdefault("rx", re.compile(c["pattern"]))
    for line in data[:cut].decode("utf-8", "replace").splitlines():
        m = rx.search(line)
        if m: _write(c.get("kind", "log.match"), {"source": c["name"], **{k: _num(v) for k, v in m.groupdict().items()}}); n += 1
    return n

async def _c_stat(c: dict, st: dict) -> int:
    """The size and modification time of the file c.path, as an event whenever either changed."""
    p = Path(os.path.expandvars(c["path"]))
    s = p.stat() if p.exists() else None
    key = (s.st_size, s.st_mtime) if s else None
    if key == st.get("key"): return 0
    st["key"] = key
    _write(c.get("kind", "file.stat"), {"source": c["name"], "path": str(p), "exists": bool(s), "size": s.st_size if s else 0, "mtime": s.st_mtime if s else None})
    return 1

COLLECT = {"prometheus": _c_prometheus, "log_regex": _c_log, "file_stat": _c_stat}

async def _collect_loop():
    due = {}
    while True:
        await asyncio.sleep(5)
        for c in (c for c in CFG["collectors"] if c.get("enabled", True)):
            if time.time() < due.get(c["name"], 0): continue
            due[c["name"]] = time.time() + float(c.get("every_s", 60))
            st = _COLLECTORS.setdefault(c["name"], {"type": c["type"], "runs": 0, "events": 0, "last": None, "error": ""})
            try: n = await COLLECT[c["type"]](c, st)
            except Exception as e: st["error"] = f"{type(e).__name__}: {e}"[:300]; continue
            st.update(runs=st["runs"] + 1, events=st["events"] + n, last=_now(), error="")

# --- Actions ---

def action(name: str):
    def reg(fn):
        ACTIONS[name] = fn
        return fn
    return reg

@action("warm")
async def _warm(args: dict) -> dict:
    """Loads args.model now with args.options (num_ctx, kv_cache_type ... - the ones its users send, or the next request reloads it) and args.session, so the next request does not wait for the load."""
    await HOOKS["warm"](args["model"], args.get("options") or {}, args.get("session") or "default")
    return {"model": args["model"]}

@action("unload")
async def _unload(args: dict) -> dict:
    await HOOKS["unload"]()
    return {}

@action("kv_save")
async def _kv_save(args: dict) -> dict: return await HOOKS["kv_save"](args.get("session") or "default")

@action("kv_drop")
async def _kv_drop(args: dict) -> dict:
    fn = os.path.basename(str(args.get("file", "")))
    if not (fn.startswith("ring__") and fn.endswith(".bin")): raise ValueError(f"not a KV ring file: {fn!r}")
    if pinned(fn) and not args.get("force"): raise ValueError(f"{fn} is pinned - unpin it first, or send force")
    Path(HOOKS["cache_dir"], fn).unlink(missing_ok=True)
    return {"dropped": fn}

def _pin_key(args: dict) -> str: return os.path.basename(str(args["file"])) if args.get("file") else f"""session:{args["session"]}"""

@action("pin")
async def _pin(args: dict) -> dict:
    PINS.add(_pin_key(args))
    _save_pins()
    return {"pins": sorted(PINS)}

@action("unpin")
async def _unpin(args: dict) -> dict:
    PINS.discard(_pin_key(args))
    _save_pins()
    return {"pins": sorted(PINS)}

def _read_through(files: list, chunk: int, busy) -> tuple:
    """Reads files start to end so the kernel keeps them in its page cache; pauses while the node answers a request, so it never competes with one for the disk. (bytes, seconds reading, seconds paused)."""
    n, paused, t0 = 0, 0.0, time.time()
    for f in files:
        with open(f, "rb", buffering=0) as fh:
            while True:
                while busy(): time.sleep(1); paused += 1
                b = fh.read(chunk)
                if not b: break
                n += len(b)
    return n, time.time() - t0 - paused, paused

@action("prefetch")
async def _prefetch(args: dict) -> dict:
    """Reads a model's files into the page cache, so its next load reads RAM instead of the disk. The cache is the kernel's: other reads can push it out again.
    gb_s far above the disk's speed means the files were already cached."""
    m = HOOKS["find"](args["model"])
    if not m: raise ValueError(f"""no model {args["model"]!r} on this node""")
    if PREFETCH["running"]: raise ValueError(f"""a prefetch of {PREFETCH["running"]} is running""")
    PREFETCH["running"] = m["name"]
    try: n, secs, paused = await asyncio.to_thread(_read_through, m["files"], int(CFG["prefetch"]["chunk_mb"]) * 1024 * 1024, HOOKS["busy"])
    finally: PREFETCH["running"] = None
    PREFETCH["last"] = {"model": m["name"], "gb": round(n / 1e9, 2), "seconds": round(secs, 1), "paused_s": round(paused), "gb_s": round(n / 1e9 / max(secs, 1e-3), 2), "t": _now()}
    return PREFETCH["last"]

@action("engine_start")
async def _engine_start(args: dict) -> dict:
    e = HOOKS["engines"].ENGINES[args["name"]]
    return {"started": await e.ensure(), "status": e.status}

@action("engine_stop")
async def _engine_stop(args: dict) -> dict:
    await HOOKS["engines"].ENGINES[args["name"]].stop("asked through fleet")
    return {}

async def _run_action(did: str, name: str, args: dict) -> dict:
    t0 = time.time()
    try: res = {"id": did, "ok": True, "result": await ACTIONS[name](args)}
    except Exception as e: res = {"id": did, "ok": False, "error": f"""{type(e).__name__}: {getattr(e, "detail", "") or e}"""[:400]}
    event("directive.done", directive=did, action=name, seconds=round(time.time() - t0, 2), **{k: v for k, v in res.items() if k != "id"})
    return res

# --- Routes ---

@router.api_route("/fleet/events", methods=["GET", "POST"])
async def fleet_events(request: Request):
    """Events after `after` (an id of this epoch; 0 = from the oldest kept), at most `limit`. gap: events between `after` and first_id were deleted by rotation before this reader got them."""
    require_on()
    b = (await request.json()) if request.method == "POST" else dict(request.query_params)
    after, limit = int(b.get("after") or 0), max(1, min(int(b.get("limit") or 1000), 10000))
    out = await asyncio.to_thread(read, after, limit)   # a reader catching up scans files
    files = _files()
    first = _first_id(files[0]) if files else 1
    return {"node": HOOKS["node"], "epoch": _J["epoch"], "now": _now(), "first_id": first, "last_id": _J["id"], "gap": after + 1 < first, "more": bool(out) and out[-1]["id"] < _J["id"], "events": out}

def _ring() -> list:
    """The KV ring files (ring__<model>__<kv>__fa<flash>__<session>.bin) with what their names say, and their pins."""
    out = []
    for e in HOOKS["ring"]():
        parts = e["file"][6:-4].rsplit("__", 3)
        model, kv, fa, session = parts if len(parts) == 4 else (e["file"], "", "", "")
        out.append({"file": e["file"], "model": model, "kv": kv, "fa": fa[2:], "session": session, "size_mb": round(e["size"] / 1e6, 1), "age_s": round(time.time() - e["mtime"]), "pinned": bool(pinned(e["file"]))})
    return out

@router.get("/fleet/state")
async def fleet_state():
    require_on()
    eng = HOOKS["engines"].ENGINES
    return {"node": HOOKS["node"], "epoch": _J["epoch"], "now": _now(), "last_id": _J["id"], "busy": bool(HOOKS["busy"]()), "resident": HOOKS["resident"](), "memory": HOOKS["sample"](), "models": await asyncio.to_thread(HOOKS["models"]),
            "kv_ring": _ring(), "pins": sorted(PINS), "engines": [{"name": n, "status": e.status, "alive": e.alive(), "active": e.active, "exclusive": e.spec.get("exclusive", True)} for n, e in eng.items()], "prefetch": PREFETCH,
            "fleet": {"hook_errors": HOOK_ERRORS, "settings": CFG, "actions": sorted(ACTIONS), "collectors": {n: {k: v for k, v in s.items() if k != "rx"} for n, s in _COLLECTORS.items()}, "journal": [{"file": f.name, "mb": round(f.stat().st_size / 1e6, 2)} for f in _files()]},
            **{k: fn() for k, fn in EXTRA_STATE.items()}}

@router.post("/fleet/apply")
async def fleet_apply(request: Request):
    """{"action", "args", "id", "dry_run", "source", "wait"}. Refused when fleet.json does not allow it (or, under idle_only, when a disruptive action meets a busy node); dry_run (or apply.mode dry_run) only journals it.
    Runs in the background unless wait is true (then up to 600 s); the outcome is the directive.done event with the same id."""
    require_on()
    b = await request.json()
    name, args, did = str(b.get("action") or ""), b.get("args") or {}, str(b.get("id") or uuid.uuid4().hex[:10])
    if name not in ACTIONS: raise HTTPException(status_code=400, detail=f"unknown action {name!r} - known: {sorted(ACTIONS)}")
    a = CFG["apply"]
    why = "this node accepts no actions (fleet.json apply.mode off)" if a["mode"] == "off" else f"this node does not accept {name} (fleet.json apply.allow)" if name not in a["allow"] else "busy: the node is answering requests (fleet.json apply.idle_only; args.force overrides)" if a.get("idle_only") and name in DISRUPTIVE and HOOKS["busy"]() and not args.get("force") else ""
    dry = bool(b.get("dry_run")) or a["mode"] == "dry_run"
    event("directive", directive=did, action=name, args=args, dry_run=dry, source=str(b.get("source") or ""), refused=why)
    if why: return {"id": did, "ok": False, "refused": why}
    if dry: return {"id": did, "ok": True, "dry_run": True}
    task = spawn(_run_action(did, name, args))
    if b.get("wait"):
        done, _ = await asyncio.wait({task}, timeout=600)
        if done: return task.result()
    return {"id": did, "ok": True, "started": True}

# --- Lifecycle (main.py's lifespan) ---

def _load_cfg() -> dict:
    c = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    return {**DEFAULTS, **{k: ({**DEFAULTS[k], **{kk: vv for kk, vv in v.items() if not kk.startswith("_")}} if isinstance(DEFAULTS.get(k), dict) and isinstance(v, dict) else v) for k, v in c.items() if not k.startswith("_")}}   # _doc keys are notes for people

async def start(**hooks):
    """First in main.py's lifespan, so everything after it is journaled. hooks: node, port, busy(), sample(), resident(), models(), find(name), ring(), cache_dir, engines (the module), warm(model, options, session), unload(), kv_save(session), cancel(id prefix)."""
    global _RECENT
    if not ON: print("[FLEET] off (FLEET=0)", flush=True); return
    CFG.clear(); CFG.update(_load_cfg()); HOOKS.update(hooks)
    _RECENT = collections.deque(maxlen=int(CFG["journal"]["memory_events"]))
    _open()
    PINS.update(json.loads(PINS_PATH.read_text()) if PINS_PATH.exists() else [])
    event("node.start", node=hooks["node"], config=str(CONFIG_PATH))
    spawn(_sampler()); spawn(_collect_loop())
    print(f"""[FLEET] journal {_J["file"].name} (epoch {_J["epoch"]}, last id {_J["id"]}) | memory sample every {CFG["sample_s"]} s | actions: {CFG["apply"]["mode"]} {CFG["apply"]["allow"]} | collectors: {[c["name"] for c in CFG["collectors"] if c.get("enabled", True)] or "none"}""", flush=True)

async def stop():
    if not ON or _J["fh"] is None: return
    for t in list(_TASKS): t.cancel()
    event("node.stop", node=HOOKS["node"])
    _J["fh"].close()
