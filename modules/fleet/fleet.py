"""modules/fleet/fleet.py - the node side of Fleet (the portal's Fleet tool is the other side): the journal's reader, what this node holds now, and the actions a manager may ask for.
State - GET /fleet/state. Actions - POST /fleet/apply: warm, unload, kv_save, kv_drop, pin, unpin, prefetch, engine_start, engine_stop. fleet.json (node settings under "fleet") decides which ones this node accepts and whether they only report what they would do - the node owner's choice, not the manager's. Every request for an action is journaled (directive), and so is its outcome (directive.done).
Collectors - Prometheus endpoints (colibri's /metrics; {port} in a url is the named service's port), log lines matched by a regex, file sizes; each reading becomes a journal event. Tests - fleet_tests.py (POST /fleet/test)."""
import os, re, json, time, uuid, asyncio
from pathlib import Path
import httpx
from core import settings, journal, services, jobs, modules, hardware
from core.http import Router, HTTPException, json_body

router = Router()
CFG: dict = {}
PINS_PATH = settings.STATE / "pins.json"
DISRUPTIVE = {"warm", "unload", "prefetch", "engine_start", "engine_stop"}   # actions that take the model, the memory or the disk away from a request in progress: refused while busy under apply.idle_only (args.force overrides)
HOOKS: dict = {}                 # filled by setup(): node, port, busy, cancel
EXTRA_STATE: dict = {}           # name -> function returning a part of /fleet/state (fleet_tests.py adds "tests")
ACTIONS: dict = {}               # action name -> async function(args) -> result dict
PINS: set = set()
PREFETCH = {"running": None, "last": None}
_COLLECTORS: dict = {}           # collector name -> its status (runs, events, last, error, file offset)

def event(kind: str, /, **data): journal.record(kind, data)
def spawn(coro): return jobs.spawn(coro)
def _num(v):
    try: return int(v)
    except (TypeError, ValueError):
        try: return float(v)
        except (TypeError, ValueError): return v

def pinned(fn: str) -> bool:
    """A KV ring file the manager pinned, by file name or by session (the llama module's ring pruning keeps it)."""
    return fn in PINS or any(p.startswith("session:") and fn.endswith(f"__{p[8:]}.bin") for p in PINS)

def _save_pins(): PINS_PATH.write_text(json.dumps(sorted(PINS), indent=1))
def _llama(): return modules.ADAPTERS.get("llama")
def _ring_mod():
    from modules.llama import module as m
    return m

# --- Sampler and collectors ---

async def _sampler():
    while True:
        await asyncio.sleep(float(CFG["sample_s"]))
        try: journal.record("mem", _sample())
        except Exception as e: journal.hook_error(e)

async def _c_prometheus(c: dict, st: dict) -> int:
    """Metrics in Prometheus text format from c.url; names matching the regex c.match (default: all) go into one event. Histogram buckets are left out unless c.buckets is true. c.engine: only while that engine runs."""
    eng = services.get(c["engine"]) if c.get("engine") else None
    if c.get("engine") and not (eng and eng.ready()): return 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=3.0, read=10.0, write=5.0, pool=3.0), **(eng.connector() if eng else {})) as cl: r = await cl.get(c["url"].replace("{port}", str(eng.port if eng else "")))
    r.raise_for_status()
    vals, rx = {}, re.compile(c.get("match", ""))
    for line in r.text.splitlines():
        m = re.match(r"^([^\s{#]+(?:\{[^}]*\})?)\s+(\S+)", line)
        if not m or ("_bucket" in m.group(1) and not c.get("buckets")) or not rx.search(m.group(1)): continue
        vals[m.group(1)] = _num(m.group(2))
    if vals: journal.record(c.get("kind", "metrics"), {"source": c["name"], "values": vals})
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
        if m: journal.record(c.get("kind", "log.match"), {"source": c["name"], **{k: _num(v) for k, v in m.groupdict().items()}}); n += 1
    return n

async def _c_stat(c: dict, st: dict) -> int:
    """The size and modification time of the file c.path, as an event whenever either changed."""
    p = Path(os.path.expandvars(c["path"]))
    s = p.stat() if p.exists() else None
    key = (s.st_size, s.st_mtime) if s else None
    if key == st.get("key"): return 0
    st["key"] = key
    journal.record(c.get("kind", "file.stat"), {"source": c["name"], "path": str(p), "exists": bool(s), "size": s.st_size if s else 0, "mtime": s.st_mtime if s else None})
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
            st.update(runs=st["runs"] + 1, events=st["events"] + n, last=journal.now(), error="")

# --- Actions ---

def action(name: str):
    def reg(fn):
        ACTIONS[name] = fn
        return fn
    return reg

@action("warm")
async def _warm(args: dict) -> dict:
    """Loads args.model now with args.options (num_ctx, kv_cache_type ... - the ones its users send, or the next request reloads it) and args.session, so the next request does not wait for the load."""
    a = modules.resolve("chat", args["model"])
    if not a: raise ValueError(f"""no model {args["model"]!r} on this node""")
    job = jobs.submit(jobs.Job("warm", a, {"model": args["model"], "options": args.get("options") or {}, "cache_session": args.get("session") or "default"}, rid=f"warm_{uuid.uuid4().hex[:8]}", model=args["model"], session=args.get("session") or "default"))
    while (ev := await job.out.get()) is not None:
        if ev.get("type") == "error": raise RuntimeError(ev["message"])
    return {"model": args["model"]}

@action("unload")
async def _unload(args: dict) -> dict:
    for s in [s for s in services.running() if s.name != "colibri:worker"]: await s.stop("asked through fleet")
    return {}

@action("kv_save")
async def _kv_save(args: dict) -> dict:
    a = _llama()
    if not a: raise ValueError("no llama module on this node")
    async with a.main.lock: return await _ring_mod().ring_save(a.main, args.get("session") or "default")   # a load in progress holds the lock until its server answers, so a save sent meanwhile waits instead of failing to connect

@action("kv_drop")
async def _kv_drop(args: dict) -> dict:
    fn = os.path.basename(str(args.get("file", "")))
    if not (fn.startswith("ring__") and fn.endswith(".bin")): raise ValueError(f"not a KV ring file: {fn!r}")
    if pinned(fn) and not args.get("force"): raise ValueError(f"{fn} is pinned - unpin it first, or send force")
    Path(_ring_mod().cache_dir(), fn).unlink(missing_ok=True)
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
    m = _ring_mod().find_model(args["model"]) if _llama() else None
    if not m: raise ValueError(f"""no model {args["model"]!r} on this node""")
    if PREFETCH["running"]: raise ValueError(f"""a prefetch of {PREFETCH["running"]} is running""")
    PREFETCH["running"] = m["name"]
    try: n, secs, paused = await asyncio.to_thread(_read_through, m["files"], int(CFG["prefetch"]["chunk_mb"]) * 1024 * 1024, jobs.busy)
    finally: PREFETCH["running"] = None
    PREFETCH["last"] = {"model": m["name"], "gb": round(n / 1e9, 2), "seconds": round(secs, 1), "paused_s": round(paused), "gb_s": round(n / 1e9 / max(secs, 1e-3), 2), "t": journal.now()}
    return PREFETCH["last"]

@action("engine_start")
async def _engine_start(args: dict) -> dict:
    from core import api
    s = services.REG[args["name"]]
    await api.warm(args["name"])
    return {"started": s.ready(), "status": s.status}

@action("engine_stop")
async def _engine_stop(args: dict) -> dict:
    await services.REG[args["name"]].stop("asked through fleet")
    return {}

async def _run_action(did: str, name: str, args: dict) -> dict:
    t0 = time.time()
    try: res = {"id": did, "ok": True, "result": await ACTIONS[name](args)}
    except Exception as e: res = {"id": did, "ok": False, "error": f"""{type(e).__name__}: {getattr(e, "detail", "") or e}"""[:400]}
    event("directive.done", directive=did, action=name, seconds=round(time.time() - t0, 2), **{k: v for k, v in res.items() if k != "id"})
    return res

# --- Routes ---

@router.route("/fleet/events", ("GET", "POST"), scope="read")
async def fleet_events(request):
    """Events after `after` (an id of this epoch; 0 = from the oldest kept), at most `limit`. gap: events between `after` and first_id were deleted by rotation before this reader got them."""
    if not journal.ON: raise HTTPException(status_code=503, detail="the journal is off on this node (FLEET=0)")
    b = (await json_body(request)) if request.method == "POST" else dict(request.query_params)
    after, limit = int(b.get("after") or 0), max(1, min(int(b.get("limit") or 1000), 10000))
    out = await asyncio.to_thread(journal.read, after, limit)
    files = journal.files()
    first = journal.first_id(files[0]) if files else 1
    return {"node": settings.NODE_ID, "epoch": journal.J["epoch"], "now": journal.now(), "first_id": first, "last_id": journal.J["id"], "gap": after + 1 < first, "more": bool(out) and out[-1]["id"] < journal.J["id"], "events": out}

def _ring() -> list:
    """The KV ring files (ring__<model>__<kv>__fa<flash>__<session>.bin) with what their names say, and their pins."""
    out = []
    for e in (_ring_mod().ring_entries() if _llama() else []):
        parts = e["file"][6:-4].rsplit("__", 3)
        model, kv, fa, session = parts if len(parts) == 4 else (e["file"], "", "", "")
        out.append({"file": e["file"], "model": model, "kv": kv, "fa": fa[2:], "session": session, "size_mb": round(e["size"] / 1e6, 1), "age_s": round(time.time() - e["mtime"]), "pinned": bool(pinned(e["file"]))})
    return out

def _resident() -> dict:
    a = _llama()
    if not a: return {"alive": False, "model": None}
    m = a.main
    return {"alive": m.ready(), "model": m.model["name"] if m.model else None, "ctx": m.meta.get("ctx", 0), "kv": m.meta.get("kv"), "loras": m.loras, "session": m.session, "in_flight": m.active, **{k: m.plan.get(k) for k in ("params", "est", "experiments")}}

def _sample() -> dict:
    a = _llama()
    n = hardware.now()
    return {**n, "ram_total_gb": hardware.facts()["ram_gb"], "model": a.main.model["name"] if a and a.main.ready() and a.main.model else None, "in_flight": sum(s.active for s in services.REG.values()), "rss_gb": a.main.rss_gb() if a else 0, "engines": [s.name for s in services.running() if not s.name == "llama:main"]}

@router.get("/fleet/state")
async def fleet_state(request):
    return {"node": settings.NODE_ID, "epoch": journal.J["epoch"], "now": journal.now(), "last_id": journal.J["id"], "busy": jobs.busy(), "resident": _resident(), "memory": _sample(), "models": await asyncio.to_thread(lambda: [{"name": m["name"], "size_gb": round(m.get("size", 0) / 1e9, 2), "source": m.get("source", m.get("adapter")), "mtime": m.get("mtime"), "adapter": m["adapter"]} for m in modules.models()]),
            "kv_ring": _ring(), "pins": sorted(PINS), "engines": [{"name": s.name, "status": s.status, "alive": s.running(), "active": s.active, "exclusive": False} for s in services.REG.values() if s.name != "llama:main"], "prefetch": PREFETCH,
            "fleet": {"hook_errors": journal.HOOK_ERRORS, "settings": CFG, "actions": sorted(ACTIONS), "collectors": {n: {k: v for k, v in s.items() if k != "rx"} for n, s in _COLLECTORS.items()}, "journal": [{"file": f.name, "mb": round(f.stat().st_size / 1e6, 2)} for f in journal.files()]},
            **{k: fn() for k, fn in EXTRA_STATE.items()}}

@router.post("/fleet/apply")
async def fleet_apply(request):
    """{"action", "args", "id", "dry_run", "source", "wait"}. Refused when fleet settings do not allow it (or, under idle_only, when a disruptive action meets a busy node); dry_run (or apply.mode dry_run) only journals it.
    Runs in the background unless wait is true (then up to 600 s); the outcome is the directive.done event with the same id."""
    b = await json_body(request)
    name, args, did = str(b.get("action") or ""), b.get("args") or {}, str(b.get("id") or uuid.uuid4().hex[:10])
    if name not in ACTIONS: raise HTTPException(status_code=400, detail=f"unknown action {name!r} - known: {sorted(ACTIONS)}")
    a = CFG["apply"]
    why = "this node accepts no actions (apply.mode off)" if a["mode"] == "off" else f"this node does not accept {name} (apply.allow)" if name not in a["allow"] else "busy: the node is answering requests (apply.idle_only; args.force overrides)" if a.get("idle_only") and name in DISRUPTIVE and jobs.busy() and not args.get("force") else ""
    dry = bool(b.get("dry_run")) or a["mode"] == "dry_run"
    event("directive", directive=did, action=name, args=args, dry_run=dry, source=str(b.get("source") or ""), refused=why)
    if why: return {"id": did, "ok": False, "refused": why}
    if dry: return {"id": did, "ok": True, "dry_run": True}
    task = spawn(_run_action(did, name, args))
    if b.get("wait"):
        done, _ = await asyncio.wait({task}, timeout=600)
        if done: return task.result()
    return {"id": did, "ok": True, "started": True}

# --- Setup ---

def setup(cfg: dict, port: int):
    CFG.clear(); CFG.update(cfg)
    HOOKS.update(node=settings.NODE_ID, port=port, busy=jobs.busy, cancel=jobs.cancel)
    PINS.update(json.loads(PINS_PATH.read_text()) if PINS_PATH.exists() else [])
    def wire_pins():
        if _llama(): _ring_mod().PINNED["fn"] = pinned   # whichever of the two modules became active first
    modules.WATCHERS.append(wire_pins)
    wire_pins()
    spawn(_sampler()); spawn(_collect_loop())
    print(f"""[FLEET] memory sample every {CFG["sample_s"]} s | actions: {CFG["apply"]["mode"]} {CFG["apply"]["allow"]} | collectors: {[c["name"] for c in CFG["collectors"] if c.get("enabled", True)] or "none"}""", flush=True)
