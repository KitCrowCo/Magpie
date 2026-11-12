"""core/batches.py - work that outlives whoever asked for it: a batch is a plan (data) of steps this node runs by itself, one item at a time, with every result kept on disk.
A portal, a browser or this node may restart in the middle: the batch is still there by its id, with what it has finished, and goes on from the next unfinished item. Nothing in a plan is code. A measurement set, a list of files to ingest and a night's generation list are all the same thing: steps over this node's own capabilities and status pages.
A plan: {"name", "params": {name: {"default", "label", "pick"}} (a parameter without a default must be given), "texts": {name: [lines]}, "on_restart": "resume" (default) | "hold", "steps": [step]}. A step is one of:
  {"id", "read": ["/node", "/budget", ...]}                      this node's own GET answers, kept as they are
  {"id", "capability", "body", "each": {var: [values] | {"text": <texts name>}}, "make": {var: {"fill": <texts name>, "tokens", "line", "then"}}, "keep": [...], "hash": bool, "show": [...]}
each: one item per combination, the first variable outermost. make: a variable built per item (fill: lines of a text repeated to about that many tokens, the same every time). body, each and make take {param} and {var} placeholders (one "{name}" alone keeps the value's type).
keep: which parts of the answer are stored ("text", "thinking", "usage", "result", "tool_calls", "text_head"); hash: also a sha256 over thinking and text, and one over the step's hashes in order, so two runs can be compared byte for byte. show: {column label: dotted path into the item's record}, what a page lists per item, for people.
Files, per batch, under state/batches/<id>/: plan.json, status.json, results.jsonl (one line per item, appended as each ends)."""
import json, time, uuid, hashlib, asyncio, itertools, shutil
import httpx
from core import settings, jobs, modules, journal, auth
from core.adapters import render

ROOT = settings.STATE / "batches"
QUEUE: asyncio.Queue = None   # batch ids waiting their turn: one runs at a time, so a measurement is never shared with another batch
RUNNING = {"id": "", "job": None}

def _dir(bid: str): return ROOT / bid
def status(bid: str) -> dict: return json.loads((_dir(bid) / "status.json").read_text())
def _plan(bid: str) -> dict: return json.loads((_dir(bid) / "plan.json").read_text())
def results(bid: str, start: int = 0) -> list: return [json.loads(l) for l in (_dir(bid) / "results.jsonl").read_text().splitlines()[start:]] if (_dir(bid) / "results.jsonl").exists() else []
def view() -> list: return sorted((status(d.name) for d in ROOT.glob("*/") if (d / "status.json").exists()), key=lambda s: s["created"], reverse=True)

def _save(st: dict, **changes):
    st.update(changes)
    (_dir(st["id"]) / "status.json").write_text(json.dumps(st, indent=1))

# --- Items ---

def _fill(texts: dict, spec: dict, env: dict) -> str:
    """Lines of one text, numbered and repeated to about spec["tokens"] tokens (4 characters each), then spec["then"]: the same text every time for the same inputs."""
    lines, want, out, i = texts[spec["fill"]], int(render(spec["tokens"], env)) * 4 - 200, [], 0
    while sum(map(len, out)) < want:
        out.append(spec.get("line", "{line}").replace("{n}", str(i + 1)).replace("{line}", lines[i % len(lines)]))
        i += 1
    return "\n".join(out) + spec.get("then", "")

def items(plan: dict, params: dict) -> list:
    """Every item of the plan, in order: [{"step", "i", "vars", "read" | ("capability", "body")}]."""
    out = []
    for st in plan["steps"]:
        if "read" in st:
            out.append({"step": st["id"], "i": 0, "vars": {}, "read": st["read"]})
            continue
        each = {k: plan["texts"][v["text"]] if isinstance(v, dict) else render(v, params) for k, v in (st.get("each") or {}).items()}
        for i, combo in enumerate(itertools.product(*each.values()) if each else [()]):
            env = {**params, **dict(zip(each, combo))}
            env.update({k: _fill(plan.get("texts", {}), m, env) for k, m in (st.get("make") or {}).items()})
            out.append({"step": st["id"], "i": i, "vars": dict(zip(each, combo)), "capability": st["capability"], "body": render(st["body"], env)})
    return out

async def _run_item(bid: str, n: int, it: dict, st: dict) -> dict:
    """One item's record. A failure is recorded and the batch goes on: one bad item does not cost the rest."""
    rec, t0 = {"n": n, "step": it["step"], "i": it["i"], "vars": it["vars"], "t": journal.now()}, time.time()
    try:
        if "read" in it:
            async with httpx.AsyncClient(timeout=60.0, trust_env=False) as c: rec["read"] = {p: (await c.get(f"""http://127.0.0.1:{settings.owner("bind.port", settings.PORT)}{p}""", headers={"Authorization": f"Bearer {auth.LOCAL_KEY}"})).json() for p in it["read"]}
            return {**rec, "outcome": "served"}
        from core import api
        a = modules.resolve(it["capability"], str(it["body"].get("model") or ""))
        if not a: raise LookupError(f"""nothing on this node does {it["capability"]} for model {it["body"].get("model")!r}""")
        job = jobs.Job(it["capability"], a, it["body"], rid=f"{bid}.{n}", model=str(it["body"].get("model") or ""), key="batch")
        job.background, RUNNING["job"] = True, job
        out = await api.collect(jobs.submit(job))
        keep = st.get("keep") or ["text", "usage", "result"]
        rec.update({k: out[k] for k in ("text", "thinking", "usage", "result", "tool_calls") if k in keep and out.get(k)}, outcome="failed" if out["error"] else job.outcome or "served", **({"error": out["error"]["message"]} if out["error"] else {}), **({"text_head": out["text"][:200]} if "text_head" in keep else {}))
        if st.get("hash"): rec["sha256"] = hashlib.sha256((out["thinking"] + "\x00" + out["text"]).encode()).hexdigest()
    except Exception as e: rec.update(outcome="failed", error=f"{type(e).__name__}: {e}"[:600])
    finally: RUNNING["job"] = None
    return {**rec, "seconds": round(time.time() - t0, 2)}

# --- A batch ---

def _summary(plan: dict, recs: list) -> dict:
    """Per step: items done and failed, and for a hashed step one hash over its answers in order (equal set hashes = byte-identical runs; none while an item of the step has failed)."""
    out = {}
    for st in plan["steps"]:
        mine = [r for r in recs if r["step"] == st["id"]]
        out[st["id"]] = {"done": len(mine), "failed": sum(r["outcome"] != "served" for r in mine), **({"set_hash": hashlib.sha256("".join(r.get("sha256", "-") for r in mine).encode()).hexdigest()} if st.get("hash") and mine and all("sha256" in r for r in mine) else {})}
    return out

async def _run(bid: str):
    plan, st = _plan(bid), status(bid)
    todo, steps, done = items(plan, st["params"]), {s["id"]: s for s in plan["steps"]}, len(results(bid))
    _save(st, state="running", total=len(todo), done=done, started=st.get("started") or journal.now())
    journal.event("batch.start", id=bid, name=plan.get("name", ""), total=len(todo), at=done)
    with open(_dir(bid) / "results.jsonl", "a", buffering=1) as f:
        for n, it in enumerate(todo[done:], start=done):
            if status(bid)["state"] != "running": break   # cancelled from outside
            _save(st, step=it["step"], item=it["vars"])
            rec = await _run_item(bid, n, it, steps[it["step"]])
            if status(bid)["state"] != "running": break   # cancelled while it ran: the item is not recorded, so a resume runs it again
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            _save(st, done=n + 1, summary=_summary(plan, results(bid)))
    st = status(bid)
    if st["state"] == "running": _save(st, state="done", ended=journal.now(), step="", item={})
    journal.event("batch.end", id=bid, state=status(bid)["state"], done=status(bid)["done"])

async def worker():
    """Runs queued batches one at a time. At start, a batch this node was running when it stopped goes on from its next item ("on_restart": "resume") or waits to be resumed ("hold"), with a line in its results either way."""
    global QUEUE
    QUEUE = asyncio.Queue()
    ROOT.mkdir(parents=True, exist_ok=True)
    for st in reversed(view()):
        if st["state"] not in ("running", "queued"): continue
        hold = st["state"] == "running" and _plan(st["id"]).get("on_restart") == "hold"
        if st["state"] == "running":
            with open(_dir(st["id"]) / "restarts.jsonl", "a") as f: f.write(json.dumps({"t": journal.now(), "at_item": st.get("done", 0), "then": "held" if hold else "resumed"}) + "\n")
        _save(st, state="interrupted" if hold else "queued", restarts=st.get("restarts", 0) + (st["state"] == "running"))
        if not hold: QUEUE.put_nowait(st["id"])
    while True:
        bid = await QUEUE.get()
        if not (_dir(bid) / "status.json").exists() or status(bid)["state"] != "queued": continue
        RUNNING["id"] = bid
        try: await _run(bid)
        except Exception as e:
            _save(status(bid), state="failed", error=f"{type(e).__name__}: {e}"[:600], ended=journal.now())
            print(f"[BATCH] {bid} failed: {type(e).__name__}: {e}", flush=True)
        finally: RUNNING["id"] = ""

def submit(plan: dict, params: dict, source: str = "") -> dict:
    """Checks the plan (every item can be built from the parameters given), stores it and queues it. ValueError says what is wrong."""
    if not plan.get("steps"): raise ValueError("a plan needs steps")
    given = {**{k: v.get("default") for k, v in (plan.get("params") or {}).items() if isinstance(v, dict) and "default" in v}, **(params or {})}
    if miss := ", ".join(k for k, v in (plan.get("params") or {}).items() if "default" not in v and given.get(k) in (None, "")): raise ValueError(f"this plan needs: {miss}")
    todo = items(plan, given)
    if gone := sorted({f"""{it["capability"]} with model {it["body"].get("model")!r}""" for it in todo if "capability" in it and not modules.resolve(it["capability"], str(it["body"].get("model") or ""))}): raise ValueError(f"""nothing on this node does {", ".join(gone)}""")
    bid = f"""{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:4]}"""
    _dir(bid).mkdir(parents=True)
    (_dir(bid) / "plan.json").write_text(json.dumps(plan, indent=1))
    _save({"id": bid, "name": plan.get("name", ""), "params": given, "state": "queued", "created": journal.now(), "source": source, "total": len(todo), "done": 0, "restarts": 0})
    QUEUE.put_nowait(bid)
    return status(bid)

def cancel(bid: str) -> dict:
    """Stops after the item now running, which is cancelled; what is done stays, and resume goes on from there."""
    st = status(bid)
    if st["state"] in ("queued", "running"): _save(st, state="cancelled", ended=journal.now())
    if RUNNING["id"] == bid and RUNNING["job"]: jobs.cancel(RUNNING["job"].id)
    return status(bid)

def resume(bid: str) -> dict:
    st = status(bid)
    if st["state"] in ("queued", "running", "done"): raise ValueError(f"""batch {bid} is {st["state"]}""")
    _save(st, state="queued", ended=None)
    QUEUE.put_nowait(bid)
    return status(bid)

def remove(bid: str): shutil.rmtree(_dir(bid))
