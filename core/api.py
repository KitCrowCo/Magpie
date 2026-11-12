"""core/api.py - the node's own interface. A portal sees capacity and ability; how each program underneath wants to be talked to stays here. Starts nothing unless asked to do work. Files are in core/files.py.
GET  /health, /node, /node/capabilities, /node/log, /budget, /models?capability=   POST /node/restart
POST /run/<capability>           any work. stream true/absent: NDJSON events (?shape=events: always, for every capability); ?background=1: answers a job id at once and the work goes on without its caller
GET  /info/<name>                POST /act/<name>   a program's status and controls, by what they are
POST /plan                       what one capability and model would take here, and the ways to make it fit   GET /knobs?model=   every adjustable value, typed
GET  /assets/kinds, /assets/<kind>   PUT, DELETE /assets/<kind>/<name>   things a manager makes here, by kind (a knowledge group); what one is made of is listed with the kind
GET  /jobs, /jobs/<id>?wait=s    DELETE /jobs/<id> (a kept answer)   POST /jobs/<id>/cancel, /control/cancel, /control/kill   GET /control/status
GET  /services, /services/<name>/log   POST /services/<name>/<start|stop|kill|restart>
GET  /modules, /modules/<name>   POST /modules/<name>/<install|activate|deactivate>   DELETE /modules/<name>
GET  /policy                     PUT /policy, PATCH /settings, POST /policy/rollback
GET  /keys                       POST /keys, DELETE /keys/<id>"""
import os, time, json, asyncio, signal
from core import settings, auth, hardware, budget, services, jobs, journal, modules, drivers, contract, plan
from core.http import Router, Request, HTTPException, json_body, ndjson

router = Router()
STARTED = time.time()
HEARTBEAT_S = 15.0

# --- Rendering events ---

def _opt(body: dict, key: str, default=None):
    o = body.get("options") if isinstance(body.get("options"), dict) else {}
    return o.get(key, body.get(key, default))

def make_job(request: Request, capability: str, body: dict, adapter: str = ""):
    """The job for one request; 404 with what would help when nothing here can do it. Nothing is started by this."""
    model = str(body.get("model") or "")
    a = modules.resolve(capability, model, adapter or str(body.get("adapter") or ""))
    if not a:
        able = modules.capabilities().get(capability)
        raise HTTPException(status_code=404, detail=f"no model {model!r} for {capability} on this node (see /models)" if able else f"this node has nothing active that does {capability} (see /modules)")
    job = jobs.Job(capability, a, body, rid=request.headers.get("x-request-id", ""), model=model, session=str(_opt(body, "cache_session") or "default"), priority=int(_opt(body, "priority", 0) or 0), key=request.state.key.get("name", ""))
    job.record["passive"] = request.headers.get("x-passive-learning", os.getenv("PASSIVE_LEARNING_DEFAULT", "false")).lower() == "true"
    return job

async def events(job, heartbeat: bool = True):
    """A job's events until it ends. A client that goes away cancels the job, unless it runs in the background. None marks a quiet heartbeat_s, so a renderer can keep a long wait's connection open."""
    try:
        while True:
            try: ev = await asyncio.wait_for(job.out.get(), HEARTBEAT_S) if heartbeat else await job.out.get()
            except asyncio.TimeoutError:
                yield {"type": "heartbeat"}
                continue
            if ev is None: return
            yield ev
    finally:
        if job.id in jobs.JOBS and not job.background: jobs.cancel(job.id)

async def collect(job) -> dict:
    """A job's whole answer, for a caller that does not stream: a result as it is, or the text gathered."""
    text, thinking, calls, result, usage, reason, err = "", "", [], None, {}, "stop", None
    async for ev in events(job, heartbeat=False):
        t = ev.get("type")
        if t == "text": text += ev["text"]
        elif t == "thinking": thinking += ev["text"]
        elif t == "tool_calls": calls += ev["calls"]
        elif t == "result": result = ev["data"]
        elif t == "usage": usage.update({k: v for k, v in ev.items() if k != "type"})
        elif t == "done": reason = ev.get("reason", reason)
        elif t == "error": err = ev
    return {"text": text, "thinking": thinking, "tool_calls": calls, "result": result, "usage": usage, "reason": reason, "error": err}

def raise_error(job_err: dict):
    if job_err: raise HTTPException(status_code=int(job_err.get("status") or 500), detail=job_err.get("message", "failed"))

async def events_stream(job):
    """This node's own stream: its events as lines, the same for every capability and every program behind it."""
    yield json.dumps({"type": "job", "id": job.id}) + "\n"
    async for ev in events(job): yield json.dumps(ev) + "\n"

SHAPES = {"events": events_stream}   # how a stream may be asked for (?shape=); a module adds more

# --- Work ---

@router.get("/interface")
async def interface(request):
    """What a portal can rely on here, whichever programs are behind it."""
    return {"magpie": 2, "contract": contract.CONTRACT, "events": contract.EVENTS, "timings": contract.TIMINGS, "shapes": sorted(SHAPES), "uncontracted": sorted(c for c in modules.capabilities() if c not in contract.CONTRACT)}

@router.post("/run/{capability}", scope="infer")
async def run(request: Request, capability: str):
    """Any capability, one way in. Streaming capabilities answer lines as they come when the body has stream true or absent, in this node's events or the ?shape= asked for; the rest answer their result as JSON, with its timings. ?background=1 answers a job id at once: poll /jobs/<id>."""
    body, shape = await json_body(request), request.query_params.get("shape", "events")
    if shape not in SHAPES: raise HTTPException(status_code=400, detail=f"no shape {shape}: {sorted(SHAPES)}")
    if request.query_params.get("background"):
        if (known := _job_state(jobs.clean_id(request.headers.get("x-request-id", "")))): return {"job": known["id"], "phase": known["phase"], "existing": True}   # asked again under the same id (a caller picking its work up): the one job, not a second
        job = jobs.submit(make_job(request, capability, body))
        job.background = True
        jobs.spawn(_keep_answer(job))
        return {"job": job.id, "phase": job.phase}
    job = jobs.submit(make_job(request, capability, body))
    if (job.adapter.capabilities.get(capability, {}).get("streams") and body.get("stream", True)) or request.query_params.get("shape") == "events": return ndjson(SHAPES[shape](job))
    out = await collect(job)
    raise_error(out["error"])
    if out["result"] is not None: return {**out["result"], "timings": out["usage"]} if isinstance(out["result"], dict) else out["result"]
    return {"model": job.model, "job": job.id, "message": {"role": "assistant", "content": out["text"], **({"thinking": out["thinking"]} if out["thinking"] else {}), **({"tool_calls": out["tool_calls"]} if out["tool_calls"] else {})}, "done": True, "done_reason": out["reason"], "usage": out["usage"]}

@router.get("/jobs")
async def jobs_list(request: Request): return {"jobs": jobs.view(), "queued": [j.id for j in jobs.QUEUE], "recent": [{k: v for k, v in d.items() if k != "result"} for d in list(jobs.DONE)[-30:]], "counts": jobs.COUNTS}

# Work that goes on without its caller (?background=1): its whole answer is written to <state>/jobs/<id>.json as it ends, so it is there for whoever asks by id - after the caller, or this node, has restarted.
# The id is the caller's own (X-Request-Id), so a caller that lost track asks again under the same id and gets the same job. An answer stays until it is deleted, or for jobs.kept_days.

ANSWERS = settings.STATE / "jobs"

async def _keep_answer(job):
    out = await collect(job)
    ANSWERS.mkdir(parents=True, exist_ok=True)
    (ANSWERS / f"{job.id}.json").write_text(json.dumps({"id": job.id, "phase": "done", "capability": job.capability, "model": job.model, "outcome": job.outcome, "reason": job.reason, "ended": journal.now(), **out}, ensure_ascii=False))

def _job_state(job_id: str):
    """A job by id: running, kept on disk, or among the latest finished; None when unknown."""
    if not job_id: return None
    if job_id in jobs.JOBS: return jobs.JOBS[job_id].public()
    if (ANSWERS / f"{job_id}.json").exists(): return json.loads((ANSWERS / f"{job_id}.json").read_text())
    return next(({**d, "phase": "done"} for d in reversed(jobs.DONE) if d["id"] == job_id), None)

def sweep_answers():
    """At start: kept answers older than jobs.kept_days are removed."""
    for f in ANSWERS.glob("*.json"):
        if time.time() - f.stat().st_mtime > float(settings.get("jobs.kept_days", 7)) * 86400: f.unlink()

@router.get("/jobs/{job_id}")
async def job_get(request: Request, job_id: str):
    """?wait=<seconds>: answers when the job has ended, or after that long with where it is."""
    if job_id in jobs.JOBS and (wait := float(request.query_params.get("wait", 0))): await asyncio.wait([jobs.JOBS[job_id].task], timeout=wait)
    if job_id not in jobs.JOBS and not (ANSWERS / f"{job_id}.json").exists(): await asyncio.sleep(0.05)   # a job that has just ended: its answer is being written
    if not (d := _job_state(job_id)): raise HTTPException(status_code=404, detail=f"no job {job_id} (answers of work without a caller are kept until deleted; other finished jobs for the latest 200)")
    return d

@router.delete("/jobs/{job_id}", scope="infer")
async def job_forget(request: Request, job_id: str):
    (ANSWERS / f"{jobs.clean_id(job_id)}.json").unlink(missing_ok=True)
    return {"removed": job_id}

@router.post("/jobs/{job_id}/cancel", scope="infer")
async def job_cancel(request: Request, job_id: str): return {"cancelled": jobs.cancel(job_id)}

# --- Control channel: answers at once, whatever is running ---

def control_status_doc() -> dict:
    return {"node": settings.NODE_ID, "uptime_s": round(time.time() - STARTED), "requests": jobs.view(), "queued": [j.id for j in jobs.QUEUE], "recent": [{k: v for k, v in d.items() if k not in ("result", "timings")} for d in list(jobs.DONE)[-20:]], "counts": jobs.COUNTS, "services": services.view(), "engines": [{"name": s.name, "alive": s.running()} for s in services.REG.values()], "memory": {**hardware.now(), "ram_total_gb": hardware.facts()["ram_gb"]}, "budget": budget.view()}

@router.get("/control/status")
async def control_status(request: Request): return control_status_doc()

@router.post("/control/cancel", scope="infer")
async def control_cancel(request: Request):
    rid = str((await json_body(request)).get("request_id") or "")
    if not rid: raise HTTPException(status_code=400, detail="request_id needed")
    ids = jobs.cancel(rid)
    print(f"""[CONTROL] cancel {rid}: {ids or "no such request"}""", flush=True)
    return {"cancelled": bool(ids), "ids": ids}

@router.post("/control/kill")
async def control_kill(request: Request):
    """scope requests (default): drops every job. service (with "service"): also kills that service. server: also kills every service at once, no state saved; the next job starts what it needs."""
    b = await json_body(request)
    return await jobs.kill(str(b.get("scope") or "requests"), str(b.get("service") or ""))

# --- Node ---

@router.get("/health", scope="open")
async def health(request: Request): return {"status": "ok"}

@router.get("/node")
async def node(request: Request):
    return {"node": settings.NODE_ID, "magpie": 2, "uptime_s": round(time.time() - STARTED), "auth_open": auth.is_open(), "hardware": hardware.facts(), "now": hardware.now(), "speed": json.loads(p.read_text()) if (p := settings.STATE / "speed.json").exists() else None, "locations": settings.locations(), "budget": budget.view(), "modules": modules.view(), "capabilities": modules.capabilities(), "could": modules.could(), "services": services.view(), "jobs": jobs.view(), "queued": len(jobs.QUEUE), "counts": jobs.COUNTS, "policy_version": settings.manager_view()["version"], "journal": {"last_id": journal.J["id"], "epoch": journal.J["epoch"], "hook_errors": journal.HOOK_ERRORS}, "adapters": {a.name: a.status() for a in modules.ADAPTERS.values()}}

@router.get("/node/capabilities")
async def node_caps(request: Request): return {"capabilities": modules.capabilities(), "could": modules.could()}

@router.get("/node/log")
async def node_log(request: Request):
    """The node's own output (magpie.log in the logs location), last n lines; each program it runs has its own under /services/<name>/log."""
    p = settings.location("logs") / "magpie.log"
    return {"log": str(p), "tail": p.read_text(errors="replace").splitlines()[-int(request.query_params.get("n", 200)):] if p.exists() else []}

def supervised() -> str:
    """What starts this process again when it exits: systemd (sets INVOCATION_ID), run.sh (sets MAGPIE_SUPERVISED), a container's restart policy; blank when it was started by hand."""
    return "systemd" if os.getenv("INVOCATION_ID") else "run.sh" if os.getenv("MAGPIE_SUPERVISED") else "container" if hardware.facts()["container"] else ""

@router.post("/node/restart")
async def node_restart(request: Request):
    """Shuts down as at any stop (running work ends, services stop) and is started again by what supervises it, with the current settings and modules. Refused when nothing would start it again."""
    by = supervised()
    if not by: raise HTTPException(status_code=409, detail="this node was started by hand: nothing would start it again - use ./run.sh or magpie.py autostart on")
    journal.event("node.restart", by=by)
    asyncio.get_running_loop().call_later(1.0, os.kill, os.getpid(), signal.SIGTERM)
    return {"restarting": True, "by": by}

@router.post("/node/speed")
async def node_speed(request: Request): return await asyncio.to_thread(hardware.speed, True)

@router.get("/budget")
async def budget_view(request: Request): return budget.view()

# --- Assets a manager makes here: one set of routes for every kind a module registers (a knowledge group today), so a portal needs no knowledge of the module ---

def _kind(kind: str) -> dict:
    if kind not in modules.ASSET_KINDS: raise HTTPException(status_code=404, detail=f"nothing active on this node manages assets of kind {kind!r} - kinds: {sorted(modules.ASSET_KINDS)}")
    return modules.ASSET_KINDS[kind]

@router.get("/assets/kinds")
async def asset_kinds(request: Request): return {"kinds": {k: {x: v[x] for x in ("label", "about", "fields", "module", "holds") if x in v} for k, v in modules.ASSET_KINDS.items()}}

@router.get("/assets/{kind}")
async def asset_list(request: Request, kind: str): return {"kind": kind, "assets": await _kind(kind)["list"](request)}

@router.put("/assets/{kind}/{name}")
async def asset_set(request: Request, kind: str, name: str):
    """Makes the asset or changes it; the body is the kind's fields (GET /assets/kinds)."""
    return await _kind(kind)["set"](request, name)

@router.delete("/assets/{kind}/{name}")
async def asset_remove(request: Request, kind: str, name: str):
    """?confirm=1 removes it; &data=1 also deletes what it holds."""
    return await _kind(kind)["remove"](request, name)

@router.post("/plan", scope="read")
async def plan_make(request: Request):
    """{"capability", "model", "want": {...}, "adapter"}: what doing this would take here, what the node has, and the ways to make it fit (core/plan.py). Starts nothing and changes nothing."""
    b = await json_body(request)
    try: return await asyncio.to_thread(plan.make, str(b.get("capability") or "chat"), str(b.get("model") or ""), b.get("want") or {}, str(b.get("adapter") or ""))
    except LookupError as e: raise HTTPException(status_code=404, detail=str(e))

@router.get("/knobs")
async def knobs_view(request: Request):
    """Every adjustable value, typed (core/plan.py); ?model= gives one model's own. Set one with PATCH /settings {"path": <key>, "value"}; a null value puts it back on auto."""
    return {"knobs": plan.knobs(request.query_params.get("model", ""), request.query_params.get("capability", ""))}

# --- Models ---

@router.get("/models")
async def models_list(request):
    """Flat list under "models"; an adapter that keeps a richer listing of its own (the image worker's image and video lists) adds its keys, so one connection serves every picker."""
    extra = {}
    for a in modules.ADAPTERS.values():
        if hasattr(a, "listed"): extra.update(a.listed())
    return {**extra, "models": modules.models(request.query_params.get("capability", ""), refresh=bool(request.query_params.get("refresh")))}

# --- A program's status and its own controls, by what they are rather than which program ---

def _holder(kind: str, name: str):
    a = next((a for a in modules.ADAPTERS.values() if name in (a.spec.get(kind) or {})), None)
    if not a: raise HTTPException(status_code=404, detail=f"nothing active here offers {name}")
    return a

@router.get("/info/{name}")
async def info(request, name: str):
    """Never starts anything."""
    return await _holder("info", name).info(name)

@router.post("/act/{name}", scope="infer")
async def act(request, name: str):
    a = _holder("actions", name)
    c = a.spec["actions"][name]
    params = {**request.query_params, **(await json_body(request))}
    if a.service.ready(): return (await a.service.client().request(c.get("method", "POST"), c["path"], params=params, timeout=600.0)).json()
    if not c.get("starts"): return {"status": "stopped"}
    out = await collect(jobs.submit(jobs.Job(f"_act:{name}", a, {"_query": params}, key=request.state.key.get("name", ""))))
    raise_error(out["error"])
    return out["result"]

# --- Services ---

@router.get("/services")
async def services_list(request): return {"services": services.view()}

@router.get("/services/{name}/log")
async def service_log(request, name: str):
    s = services.get(name)
    if not s: raise HTTPException(status_code=404, detail=f"no service {name}")
    return {"log": str(s.log), "tail": s.log.read_text(errors="replace").splitlines()[-int(request.query_params.get("n", 100)):] if s.log.exists() else []}

async def warm(name: str):
    """Up through the scheduler, so it is counted against the node's memory: an adapter's program, or a model's service."""
    a = modules.ADAPTERS.get(name) or getattr(services.get(name), "adapter", None) or modules.ADAPTERS.get(name.split(":")[0])
    if not a: raise HTTPException(status_code=400, detail=f"nothing on this node knows how to start {name}")
    model = name.split(":", 1)[1] if ":" in name and name.split(":")[0] in modules.ADAPTERS else ""
    job = jobs.submit(jobs.Job("warm", a, {"model": model}, model=model))
    while (ev := await job.out.get()) is not None:
        if ev.get("type") == "error": raise HTTPException(status_code=int(ev.get("status") or 500), detail=ev["message"])

@router.post("/services/{name}/{verb}")
async def service_verb(request, name: str, verb: str):
    """stop finishes running work first; kill waits for nothing. start and restart bring it up as work would, or from its own command."""
    s = services.get(name)
    if not s: raise HTTPException(status_code=404, detail=f"no service {name} - known: {sorted(services.REG)}")
    if verb == "kill": await s.kill("asked through /services")
    elif verb == "stop":
        s.closing = s.running()
        jobs.kick()
    elif verb in ("start", "restart"):
        if verb == "restart" and s.running(): await s.stop("asked through /services")
        if s.spec.get("cmd") and name not in modules.ADAPTERS:
            budget.lease(name, s.spec.get("ram_gb", 1.0), keep=bool(s.spec.get("keep")))
            await s.start()
            s.measure()
        elif name.startswith("llama:"): raise HTTPException(status_code=400, detail=f"{name} starts with its model: POST /run/<capability> or fleet warm")
        else: await warm(name)
    else: raise HTTPException(status_code=400, detail="verb must be start, stop, restart or kill")
    return s.info()

# --- Modules: what could run here, what it takes, and bringing it about from afar ---

@router.get("/modules")
async def modules_list(request): return {"modules": modules.view(), "capabilities": modules.capabilities(), "could": modules.could()}

@router.get("/modules/{name}")
async def module_get(request, name: str):
    if name not in modules.ALL: raise HTTPException(status_code=404, detail=f"no module {name}")
    return modules.describe(name)

@router.post("/modules/{name}/activate")
async def module_activate(request, name: str):
    try: d = modules.activate(name)
    except ValueError as e: raise HTTPException(status_code=409, detail=str(e))
    modules.remember_active(name)
    return d

@router.post("/modules/{name}/deactivate")
async def module_deactivate(request, name: str): return await modules.deactivate(name)

@router.post("/modules/{name}/install")
async def module_install(request, name: str):
    """{"driver" (blank = the best that fits), "options", "activate" (default true)}: progress lines as it goes. Each module's dependencies live in its own environment. Installing another driver replaces this module's and leaves the rest of the node alone."""
    if name not in modules.ALL: raise HTTPException(status_code=404, detail=f"no module {name}")
    b = await json_body(request)
    async def lines():
        ok = False
        for dep in modules.ALL[name]["spec"].get("needs", []):
            if not modules._ready(dep):
                async for p in drivers.install(dep, modules.ALL[dep]["spec"]): yield json.dumps(p) + "\n"
        async for p in drivers.install(name, modules.ALL[name]["spec"], str(b.get("driver") or ""), b.get("options")):
            ok = ok or p.get("status") == "success"
            yield json.dumps(p) + "\n"
        if ok and b.get("activate", True) and not modules.ALL[name]["active"]:
            modules.activate(name)
            modules.remember_active(name)
            yield json.dumps({"status": "active", "capabilities": modules.ALL[name]["spec"].get("provides", [])}) + "\n"
    return ndjson(lines())

@router.delete("/modules/{name}")
async def module_remove(request, name: str):
    """Its driver record and environment. Models and results are not touched."""
    if not request.query_params.get("confirm"): raise HTTPException(status_code=400, detail="destructive - send ?confirm=1")
    await modules.deactivate(name)
    drivers.remove(name)
    return modules.describe(name)

# --- Policy: the rules a manager may change, and the versions kept of them ---

@router.get("/policy")
async def policy(request: Request): return {**settings.manager_view(), "effective": {k: settings.get(k) for k in ("scheduler", "resources", "defaults", "locations")}}

@router.put("/policy")
async def policy_set(request: Request):
    b = await json_body(request)
    try: v = settings.set_manager(b.get("rules") or {}, str(b.get("source") or request.state.key.get("name", "")))
    except ValueError as e: raise HTTPException(status_code=403, detail=str(e))
    journal.event("policy.set", version=v["version"], source=b.get("source"))
    jobs.kick()
    return v

@router.post("/policy/rollback")
async def policy_rollback(request: Request):
    try: return settings.rollback((await json_body(request))["version"])
    except (ValueError, KeyError) as e: raise HTTPException(status_code=400, detail=str(e))

@router.route("/settings", ("PATCH",))
async def setting_set(request):
    """{"path": "colibri.ram_gb", "value": 20}: one value, with every other kept and the old version saved. A service using it picks it up at its next start, which the scheduler brings about when it is idle."""
    b = await json_body(request)
    try: v = settings.set_path(str(b["path"]), b.get("value"), str(b.get("source") or request.state.key.get("name", "")))
    except (ValueError, KeyError) as e: raise HTTPException(status_code=403, detail=str(e))
    journal.event("policy.set", version=v["version"], path=b["path"])
    jobs.kick()
    return {"path": b["path"], "value": settings.get(str(b["path"])), "version": v["version"]}

# --- Keys ---

@router.get("/keys")
async def keys_list(request: Request): return {"keys": auth.list_keys(), "open": auth.is_open()}

@router.post("/keys")
async def keys_new(request: Request):
    """The key itself appears in this one answer only."""
    b = await json_body(request)
    try: k = auth.new_key(str(b.get("name") or "unnamed"), str(b.get("scope") or "infer"))
    except ValueError as e: raise HTTPException(status_code=400, detail=str(e))
    return {"key": k, "scope": b.get("scope") or "infer"}

@router.delete("/keys/{key_id}")
async def keys_revoke(request: Request, key_id: str): return {"revoked": auth.revoke(key_id)}
