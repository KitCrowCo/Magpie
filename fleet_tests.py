"""fleet_tests.py - tests a manager runs on this node through Fleet (POST /fleet/test), measured through the node's own API exactly as a user's request is, with bench.py's ask() - a test and a bench run measure the same thing.
Kinds (TESTS):
  perf      prompts of about `lengths` tokens, each sent `repeat` times in a row (run 1 is cold for that prompt, later runs show what the kept cache saves): time to first token, prefill and decode speed, cache reuse
  prefix    one prompt of about `length` tokens, then the same prompt with a short addition: what reusing a cached prefix saves
  fidelity  greedy answers to fixed prompts (bench.py's set, or `prompts`) with a hash each, and one hash over the whole set: two runs with the same set hash gave byte-identical answers
  load      unloads the main model, optionally prefetches its file (prefetch: true), then one short prompt: what a load costs, cold or prefetched
Common params: model (or engine: an engine's name - its OpenAI-compatible API through /engines/<name>), num_ctx (keep it equal to what the model's users send, or the model reloads), max_tokens, think, timeout_s.
Every measurement is a journal event test.<kind> carrying the test id; test.start and test.end bracket the run. Its requests carry X-Request-Id fleettest_<test id>.<n>, so their req.* events belong to the test (job fleettest_<test id>).
One test runs at a time; others wait in order. Before each measurement a test waits for the node to be idle (fleet.json tests.idle_wait_s), so it measures the node, not the node plus someone else's request."""
import re, time, uuid, types, asyncio, hashlib, collections
from fastapi import APIRouter, Request, HTTPException
import fleet
from bench import bench   # bench/bench.py, standard library only

router = APIRouter()
TESTS: dict = {}
CUR = {"id": None, "kind": None, "started": None, "cancel": False}
DONE = collections.deque(maxlen=20)
_Q: dict = {"queue": None, "worker": None}

def test(kind: str):
    def reg(fn):
        TESTS[kind] = fn
        return fn
    return reg

def _args(p: dict, rid: str) -> types.SimpleNamespace:
    """bench.py's settings for one request: wire ollama for this node's own models, openai for an engine (p.engine)."""
    url = f"""http://127.0.0.1:{fleet.HOOKS["port"]}"""
    return types.SimpleNamespace(wire="openai" if p.get("engine") else "ollama", url=f"""{url}/engines/{p["engine"]}""" if p.get("engine") else url, model=str(p.get("model", "")), key="", think=str(p.get("think", "")), num_ctx=int(p.get("num_ctx") or 0), timeout=float(p.get("timeout_s", 36000)), headers={"X-Request-Id": rid})

async def _idle():
    deadline = time.time() + float(fleet.CFG["tests"].get("idle_wait_s", 600))
    while fleet.HOOKS["busy"]() and time.time() < deadline and not CUR["cancel"]: await asyncio.sleep(2)

async def _ask(t: dict, prompt: str, n: int, temperature: float = 0.0) -> dict:
    """One measured request (bench.ask in a thread); an error is a result, not a stop."""
    await _idle()
    try: return await asyncio.to_thread(bench.ask, _args(t["params"], f"""fleettest_{t["id"]}.{n}"""), prompt, int(t["params"].get("max_tokens", 128)), temperature)
    except Exception as e: return {"error": f"{type(e).__name__}: {e}"[:300]}

def _plain(r: dict) -> dict: return {**{k: v for k, v in r.items() if k not in ("text", "thinking")}, "text_head": r.get("text", "")[:160]}

def _items(v, sep: str) -> list: return v if isinstance(v, list) else [x.strip() for x in str(v).split(sep) if x.strip()]   # a list param given as a JSON list or as text ("128,2048"; prompts one per line)

# --- Tests ---

@test("perf")
async def _perf(t: dict, p: dict) -> dict:
    n, rows = 0, []
    for length in [int(x) for x in _items(p.get("lengths", "128,2048"), ",")]:
        for run in range(1, int(p.get("repeat", 2)) + 1):
            if CUR["cancel"]: return {"runs": len(rows)}
            n += 1
            r = await _ask(t, bench.notes_prompt(length), n)
            fleet.event("test.perf", test=t["id"], target_tokens=length, run=run, **_plain(r))
            rows.append(r)
    return {"runs": len(rows), "errors": sum(1 for r in rows if r.get("error"))}

@test("prefix")
async def _prefix(t: dict, p: dict) -> dict:
    base, out = bench.notes_prompt(int(p.get("length", 4096))), {}
    for n, (step, prompt) in enumerate((("first", base), ("extended", base + "\n\nThen add one sentence about the weather.")), 1):
        r = await _ask(t, prompt, n)
        fleet.event("test.prefix", test=t["id"], step=step, target_tokens=int(p.get("length", 4096)), **_plain(r))
        out[step] = r.get("ttft_s")
    return {"ttft_first_s": out.get("first"), "ttft_extended_s": out.get("extended")}

@test("fidelity")
async def _fidelity(t: dict, p: dict) -> dict:
    prompts, hashes = _items(p.get("prompts") or bench.FIDELITY, "\n"), []
    for i, q in enumerate(prompts):
        if CUR["cancel"]: break
        r = await _ask(t, q, i + 1)
        h = hashlib.sha256((r.get("thinking", "") + "\x00" + r.get("text", "")).encode()).hexdigest()
        hashes.append(h)
        fleet.event("test.fidelity", test=t["id"], i=i, sha256=h, prompt=q[:100], **_plain(r))
    return {"prompts": len(hashes), "set_sha256": hashlib.sha256("".join(hashes).encode()).hexdigest()}

@test("load")
async def _load(t: dict, p: dict) -> dict:
    if p.get("engine"): raise ValueError("load measures this node's own llama.cpp models; for an engine, stop it (engine_stop) and run perf")
    await _idle()
    await fleet.HOOKS["unload"]()
    pre = await fleet.ACTIONS["prefetch"]({"model": p["model"]}) if p.get("prefetch") else None
    r = await _ask(t, "Say OK.", 1)
    fleet.event("test.load", test=t["id"], prefetch=pre, **_plain(r))
    return {"ttft_s": r.get("ttft_s"), "prefetch": pre}

# --- Queue ---

async def _worker():
    while True:
        t = await _Q["queue"].get()
        CUR.update(id=t["id"], kind=t["kind"], started=time.time(), cancel=False)
        fleet.event("test.start", test=t["id"], test_kind=t["kind"], params=t["params"])
        t0 = time.time()
        try: res = {"ok": True, "summary": await TESTS[t["kind"]](t, t["params"])}
        except Exception as e: res = {"ok": False, "error": f"""{type(e).__name__}: {getattr(e, "detail", "") or e}"""[:400]}
        fleet.event("test.end", test=t["id"], test_kind=t["kind"], seconds=round(time.time() - t0, 1), cancelled=CUR["cancel"], **res)
        DONE.append({"id": t["id"], "kind": t["kind"], "seconds": round(time.time() - t0, 1), "cancelled": CUR["cancel"], **res})
        CUR.update(id=None, kind=None, started=None, cancel=False)

def submit(kind: str, params: dict, tid: str = "") -> dict:
    if kind not in TESTS: raise HTTPException(status_code=400, detail=f"unknown test {kind!r} - known: {sorted(TESTS)}")
    if _Q["queue"] is None: _Q["queue"] = asyncio.Queue()
    if _Q["worker"] is None or _Q["worker"].done(): _Q["worker"] = fleet.spawn(_worker())
    t = {"id": re.sub(r"[^A-Za-z0-9_-]", "_", str(tid or uuid.uuid4().hex[:8]))[:40], "kind": kind, "params": params}
    _Q["queue"].put_nowait(t)
    fleet.event("test.queued", test=t["id"], test_kind=kind, params=params)
    return {"id": t["id"], "queued": _Q["queue"].qsize(), "running": CUR["id"]}

def status() -> dict: return {"kinds": sorted(TESTS), "running": {k: CUR[k] for k in ("id", "kind", "started")} if CUR["id"] else None, "queued": _Q["queue"].qsize() if _Q["queue"] else 0, "recent": list(DONE)}

fleet.EXTRA_STATE["tests"] = status

@router.post("/fleet/test")
async def fleet_test(request: Request):
    """{"kind", "params", "id"} -> queued; the results are journal events (test.*)."""
    fleet.require_on()
    if not fleet.CFG["tests"].get("allow", True): raise HTTPException(status_code=403, detail="this node runs no tests (fleet.json tests.allow)")
    b = await request.json()
    return submit(str(b.get("kind") or ""), b.get("params") or {}, str(b.get("id") or ""))

@router.post("/fleet/test/cancel")
async def fleet_test_cancel(request: Request):
    """Stops the running test after its current measurement and drops that measurement's request (a test waiting in the queue still runs)."""
    fleet.require_on()
    if not CUR["id"]: return {"cancelled": None}
    CUR["cancel"] = True
    dropped = fleet.HOOKS["cancel"](f"""fleettest_{CUR["id"]}""")
    return {"cancelled": CUR["id"], "requests_dropped": dropped}
