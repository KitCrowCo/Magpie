"""core/jobs.py - every piece of work is a Job, whatever engine does it: the same queue, the same status, the same cancel, the same statistics.
A job runs in a task of its own and writes events to a queue the HTTP answer reads from. So a client going away, a cancel and a kill all work the same way, and none of them relies on the engine.
The scheduler holds requests here instead of in each engine's own queue. It prefers work for what is already loaded, so two models alternating on one node do not reload for every request; after scheduler.max_hold_s the oldest waiting job wins and what stands in its way finishes its running work and is stopped (nothing running is ever cut off, unless scheduler.drain_max_s says so). A job with priority above 0 does not wait for max_hold_s.
Events: {"type": "text"|"thinking", "text"} {"type": "tool_calls", "calls"} {"type": "progress", ...} {"type": "result", "data"} {"type": "usage", ...} {"type": "done", "reason"} {"type": "error", "message", "status"}."""
import re, time, uuid, asyncio, collections
from core import settings, budget, services, journal, contract

JOBS: dict = {}                            # id -> a job not yet finished
DONE = collections.deque(maxlen=200)       # the latest finished jobs, their results kept for background callers
COUNTS = {"served": 0, "cancelled": 0, "failed": 0}
QUEUE: list = []                           # jobs waiting for a slot, in arrival order
LISTENERS: list = []                       # fn(job), called as each job ends (modules/draft takes its samples here)
_KICK = asyncio.Event()
_TASKS: set = set()
_LOADING: dict = {}                        # service name -> the task loading it

def spawn(coro) -> asyncio.Task:
    t = asyncio.create_task(coro)
    _TASKS.add(t)
    t.add_done_callback(_TASKS.discard)
    return t

def kick(): _KICK.set()
def job_of(rid: str) -> str: return rid.split(".", 1)[0]   # the portal sends <job id>.<random>: the job is everything before the first dot

def clean_id(rid: str) -> str: return re.sub(r"[^A-Za-z0-9._:-]", "_", rid or "")[:96]

class Job:
    def __init__(self, capability: str, adapter, body: dict, rid: str = "", model: str = "", session: str = "default", priority: int = 0, key: str = ""):
        self.id = clean_id(rid) or uuid.uuid4().hex[:12]
        self.capability, self.adapter, self.body, self.model, self.session, self.priority, self.key = capability, adapter, body, model, session, int(priority), key
        self.phase, self.t0, self.t_grant, self.t_first, self.out, self.task, self.outcome, self.reason = "queued", time.time(), None, None, asyncio.Queue(), None, None, ""
        self.pieces, self.timings, self.service, self.need, self.cancelled, self.result, self.record, self.granted, self.wait_load, self.background, self.holds_slot = 0, {}, None, None, False, None, {}, asyncio.Event(), None, False, False
        JOBS[self.id] = self

    def public(self) -> dict: return {"id": self.id, "capability": self.capability, "kind": self.capability, "adapter": self.adapter.name, "model": self.model, "session": self.session, "phase": self.phase, "priority": self.priority, "age_s": round(time.time() - self.t0, 1), "service": self.service.name if self.service else None, "timings": self.timings, "progress": self.record.get("progress")}

# --- Running one job ---

def submit(job: Job) -> Job:
    """Starts the job's task; its events are read from job.out, ending with None."""
    if job.capability != "warm": journal.event("req.start", rid=job.id, job=job_of(job.id), api=job.capability, model=job.model, session=job.session, adapter=job.adapter.name)
    job.task = spawn(_run(job))
    return job

async def _run(job: Job):
    try:
        if job.adapter.needs_service(job):
            job.need = job.adapter.need(job)
            QUEUE.append(job)
            kick()
            await job.granted.wait()
            if job.wait_load: raise job.wait_load   # the load this job waited for failed
            job.phase, job.t_grant = "running", time.time()
            job.service.last_use = time.time()
        else: job.phase, job.t_grant = "running", time.time()
        async for ev in job.adapter.run(job, job.service):
            if job.t_first is None and ev.get("type") in ("text", "thinking", "tool_calls", "result"): job.t_first = time.time()
            t = ev.get("type")
            if t == "usage":
                job.timings.update({k: v for k, v in ev.items() if k != "type"})
                continue   # given once, complete, just before done
            if t == "progress": job.record["progress"] = ev
            elif t == "result": job.result = ev["data"] = contract.conform(job.capability, job.adapter.capabilities.get(job.capability, {}), ev["data"], job.adapter.name)
            elif t in ("text", "thinking"): job.pieces += 1
            elif t == "done": await job.out.put({"type": "usage", **_timings(job)})
            await job.out.put(ev)
        job.outcome = "served"
    except asyncio.CancelledError:
        job.outcome, job.reason = "cancelled", job.reason or "cancelled"
        await job.out.put({"type": "done", "reason": "cancelled", "error": "cancelled through the control channel"})
    except Exception as e:
        job.outcome, job.reason = "failed", str(getattr(e, "detail", "") or f"{type(e).__name__}: {e}")[:600]
        await job.out.put({"type": "error", "message": job.reason, "status": getattr(e, "status_code", 500)})
    finally:
        if job in QUEUE: QUEUE.remove(job)
        if job.holds_slot:
            job.service.active -= 1
            job.service.last_use = time.time()
            job.adapter.on_job_end(job)
        try: _finish(job)
        finally:   # a fault in the record keeping must not leave a caller waiting
            await job.out.put(None)
            kick()

def _timings(job: Job) -> dict:
    """The same figures for every job, whatever its program reports: what it says when it says it, otherwise measured here."""
    t, now = job.timings, time.time()
    since = (job.t_first or now) * 1000
    t.setdefault("gen_n_source", "program" if t.get("gen_n") is not None else "counted")
    if t.get("prompt_ms") is None and job.t_first and job.t_grant: t["prompt_ms"] = round((job.t_first - job.t_grant) * 1000, 1)
    if t.get("gen_n") is None and job.pieces: t["gen_n"] = job.pieces
    if not t.get("gen_ms") and job.pieces: t["gen_ms"] = round(now * 1000 - since, 1)
    t.update(total_ms=round((now - job.t0) * 1000, 1), wait_ms=round(((job.t_grant or now) - job.t0) * 1000, 1), **({"ttft_ms": round((job.t_first - job.t0) * 1000, 1)} if job.t_first else {}), **({"tps": round(t["gen_n"] / (t["gen_ms"] / 1000), 2)} if t.get("gen_n") and t.get("gen_ms") else {}))
    return t

def _finish(job: Job):
    JOBS.pop(job.id, None)
    COUNTS[job.outcome] += 1
    now = time.time()
    wait = round((job.t_grant or now) - job.t0, 3)
    t = job.timings
    if job.capability == "warm": pass
    elif job.adapter.journal_family == "llama": journal.event("req.end", rid=job.id, job=job_of(job.id), api=job.capability, model=job.model, session=job.session, outcome=job.outcome, seconds=round(now - job.t0, 3), wait_s=wait, adapter=job.adapter.name, **t)
    else: journal.event("engine.req", name=job.adapter.name, path=job.capability, method="POST", status={"served": 200, "cancelled": 499}.get(job.outcome, 500), seconds=round(now - (job.t_grant or job.t0), 3), wait_s=wait, rid=job.id, gen=True, model=job.model, session=job.session, outcome=job.outcome, **t)
    DONE.append({"id": job.id, "kind": job.capability, "adapter": job.adapter.name, "model": job.model, "outcome": job.outcome, "reason": job.reason, "seconds": round(now - job.t0, 1), "wait_s": wait, "ended": journal.now(), "result": job.result if job.background else None, "timings": t})
    for fn in LISTENERS:
        try: fn(job)
        except Exception as e: journal.hook_error(e)

# --- Cancel and kill: independent of the engine ---

def cancel(prefix: str) -> list:
    """One job, or every job of one portal job (an id prefix up to the first dot)."""
    ids = [k for k in JOBS if k == prefix or k.startswith(prefix + ".")]
    for k in ids:
        j = JOBS[k]
        j.phase, j.cancelled = "cancelling", True
        j.adapter.on_cancel(j)
        j.task.cancel()
    return ids

async def kill(scope: str = "requests", service: str = "") -> dict:
    """requests: drops every job. service: also kills that one service. server: also kills every service, at once, no state saved - the next job starts what it needs again."""
    ids = cancel_all()
    targets = [services.get(service)] if scope == "service" and services.get(service) else services.running() if scope == "server" else []
    for s in targets: await s.kill(f"control channel, scope {scope}")
    print(f"[CONTROL] kill ({scope}): {len(ids)} job(s) dropped, killed {[s.name for s in targets]}", flush=True)
    kick()
    return {"cancelled": len(ids), "killed": [s.name for s in targets], "server_killed": bool(targets)}

def cancel_all() -> list: return [i for k in list(JOBS) for i in cancel(k)]
def busy() -> bool: return bool(JOBS)
def view() -> list: return [j.public() for j in JOBS.values()]

# --- Scheduler ---
# adapter.need(job) answers {"service": Service, "ram_gb", "gpu_gb", "keep", "label", "load": async fn(job) that brings the service up in the configuration this job needs}. A job can run when its service is ready, not closing, has a free slot, and its configuration fits (adapter.fits).

def _grant(job: Job):
    """Takes the slot at once, so the next job in the same pass sees it taken."""
    job.service = job.need["service"]
    job.service.active += 1
    job.holds_slot = True
    QUEUE.remove(job)
    job.granted.set()

def _fail_waiting(job: Job, e: Exception):
    job.wait_load, job.service = e, None
    QUEUE.remove(job)
    job.granted.set()

async def _load(job: Job, evict: list):
    """Stops what must go, then brings the job's service up. A failed load fails every job waiting for that service, with the reason."""
    s, need = job.need["service"], job.need
    try:
        for name in evict: await services.get(name).stop(f"""{need.get("label") or s.name} needs the memory""")
        if s.running() and not job.adapter.fits(s, job): await s.stop("restarting with a new configuration")
        budget.lease(s.name, need.get("ram_gb", 0.0), need.get("gpu_gb", 0.0), keep=need.get("keep", False), why=job.capability)
        job.phase = "loading"
        await need["load"](job)
        if not s.ready(): raise RuntimeError(f"""{s.name} did not come up: {s.status}""")
    except Exception as e:
        print(f"[SCHED] loading {s.name} failed: {type(e).__name__}: {e}", flush=True)
        budget.release(s.name)
        for w in [w for w in QUEUE if w.need["service"] is s]: _fail_waiting(w, e)
    finally:
        _LOADING.pop(s.name, None)
        kick()

def _tick():
    """Two passes. First, everything that can run on what is already up runs: that is what keeps two alternating models from reloading per request. Then the jobs left, oldest first, may load - only into memory no running work holds, unless they have waited past max_hold_s (or carry a priority), which closes what stands in the way."""
    hold, now = float(settings.get("scheduler.max_hold_s", 900)), time.time()
    for job in list(QUEUE):
        s = job.need["service"]
        if s.name not in _LOADING and s.ready() and not s.closing and s.active < s.slots and job.adapter.fits(s, job): _grant(job)
    for job in list(QUEUE):
        s = job.need["service"]
        if s.name in _LOADING or (s.ready() and not s.closing and job.adapter.fits(s, job)): continue   # being loaded, or the right service with every slot working
        urgent = job.priority > 0 or now - job.t0 > hold
        if s.running() and s.active:                                                                    # the same service in another configuration, still working
            if urgent: s.closing = True
            continue
        if s.closing: continue
        solo = budget.alone(s.name, job.need)
        b = budget.fits(job.need.get("ram_gb", 0.0) - (budget.LEASES.get(s.name, {}).get("ram_gb", 0.0) if s.running() else 0.0), services.REG, exclude=s.name, alone=solo)
        if b["busy"]:
            job.record["blocked"] = b
            if urgent:
                for n in b["busy"]: services.get(n).closing = True
            continue
        if not b["fits"] and not b["evict"]:
            if not settings.get("scheduler.wait_for_memory", False): _fail_waiting(job, RuntimeError(f"""not enough memory for {s.name}: short {b["short_gb"]} GB with everything stoppable stopped (see resources in node.json, and /budget)"""))
            continue
        budget.lease(s.name, job.need.get("ram_gb", 0.0), job.need.get("gpu_gb", 0.0), keep=job.need.get("keep", False), why=job.capability)
        s.alone = solo
        _LOADING[s.name] = spawn(_load(job, b["evict"]))

def _close_idle():
    """A service asked to make way whose work has ended stops now; so does one past its idle_stop_s with nothing waiting for it. A process that died by itself is noticed here too."""
    for s in services.REG.values(): s.reap()
    for s in services.running():
        if s.name in _LOADING or s.active: continue
        idle = float(s.spec.get("idle_stop_s", 0))
        if s.closing: _LOADING[s.name] = spawn(_stop(s, "making way for waiting work"))
        elif idle and s.last_use and time.time() - s.last_use > idle and not any(j.need["service"] is s for j in QUEUE): _LOADING[s.name] = spawn(_stop(s, f"idle for {idle:.0f} s"))

async def _stop(s, why: str):
    try: await s.stop(why)
    finally:
        _LOADING.pop(s.name, None)
        kick()

async def scheduler():
    while True:
        _KICK.clear()
        _close_idle()
        _tick()
        try: await asyncio.wait_for(_KICK.wait(), 2.0)
        except asyncio.TimeoutError: pass
