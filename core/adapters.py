"""core/adapters.py - an adapter is how one program does some of this node's work. The rest of Magpie only sees capabilities, models and services, never how a program wants to be run.
Adapter is the interface. WorkerAdapter runs any HTTP program from data alone (settings "<module>.adapters.<name>", shipped in module.json), so a new engine or a hyper-specific working setup is a data block, not new code. It reads its data when it starts the program, so a manager's change applies at the next start, by itself. Its request maps use one placeholder rule, the same as the portal's profiles: "{key}" alone keeps the value's type; a field whose placeholder has no value is dropped.
Spec keys: capabilities {name: {"path", "method", "body" (a map; absent = the job's body as sent), "body_add" (fields always sent), "stream" (progress lines), "result" (dotted path into the answer)}}, runtime {"cmd", "cwd", "env", "health", "start_timeout_s", ...} (core/services.py; placeholders core/drivers.py), memory {"ram_gb", "gpu_gb", "keep"}, models {"root" (a location), "glob", "dirs_with", "list_path"}, idle_stop_s, cancel {"path"}, slots, info {name: {"path", "stopped" (answer while stopped), "remember"}} (GET /info/<name>: never starts it), actions {name: {"path", "method", "starts"}} (POST /act/<name>; starts: bring it up first, else "stopped" is the answer)."""
import re, json, types
from pathlib import Path
import httpx
from core import settings, services, budget, drivers
from core.http import HTTPException

_MISSING = object()

def render(node, params: dict):
    """Fills placeholders. A string that is exactly "{key}" gives the raw value; a placeholder with no value drops that field."""
    if isinstance(node, dict): return {k: v for k, v in ((k, render(v, params)) for k, v in node.items()) if v is not _MISSING}
    if isinstance(node, list): return [v for v in (render(x, params) for x in node) if v is not _MISSING]
    if isinstance(node, str):
        keys = re.findall(r"\{(\w+)\}", node)
        if not keys: return node
        if any(k not in params or params[k] is None for k in keys): return _MISSING
        return params[keys[0]] if re.fullmatch(r"\{\w+\}", node) else re.sub(r"\{(\w+)\}", lambda m: str(params[m.group(1)]), node)
    return node

def dig(data, path: str):
    for k in path.split("."):
        if isinstance(data, list) and k.isdigit() and int(k) < len(data): data = data[int(k)]
        elif isinstance(data, dict): data = data.get(k)
        else: return None
    return data

class Adapter:
    """name: unique on this node. capabilities: {name: spec}. journal_family: "llama" journals jobs as req.end, anything else as engine.req."""
    journal_family, capabilities, owner = "engine", {}, None
    def __init__(self, name: str, spec: dict = None): self.name, self._spec = name, spec or {}
    @property
    def spec(self) -> dict: return self._spec
    def models(self) -> list: return []                  # [{"name", "capabilities", "size", ...}]. Never starts anything.
    def needs_service(self, job) -> bool: return True
    def need(self, job) -> dict: raise NotImplementedError
    def fits(self, service, job) -> bool: return True    # does the running service's configuration satisfy this job?
    def plan(self, capability: str, model: str, want: dict) -> dict:
        """What doing this would take here, without starting anything (core/plan.py adds what the node has and the ways to make it fit). {"service", "ram_gb" (what it would be given), "gpu_gb", "alone", "keep", "source": "estimated" | "measured" | "program" (the program's own planner), and optionally "min_ram_gb" (below this it cannot start), "parts" {label: GB}, "ok_now" (the program's own verdict for the node as it is), "notes" [..], "ways" [{"label", "set" {settings path: value}, "cost", "fits"}]}. This default asks need(), which every adapter has."""
        n = self.need(types.SimpleNamespace(capability=capability, model=model, body=want, session="default", record={}))
        sv, lease = n["service"], budget.LEASES.get(n["service"].name) or {}
        held = sv.running() and lease.get("measured")
        return {"service": sv.name, "ram_gb": round(lease["ram_gb"] if held else float(n.get("est_gb", n.get("ram_gb", 0.0))), 2), "gpu_gb": round(float(n.get("gpu_gb", 0.0)), 2), "alone": budget.alone(sv.name, n), "keep": bool(n.get("keep")), "source": "measured" if held else "estimated"}
    def knob_auto(self, model: str = "") -> dict: return {}   # {setting name: what "auto" comes to right now}, for knobs the program decides
    def on_cancel(self, job): pass                       # anything beyond closing the connection (an engine's own stop call)
    def on_job_end(self, job): pass
    def status(self) -> dict: return {}
    async def run(self, job, service):                   # yields events (core/jobs.py)
        raise NotImplementedError
        yield

# --- A program driven entirely by data ---

class WorkerAdapter(Adapter):
    def __init__(self, name: str, owner: str):
        super().__init__(name)
        self.owner = owner
        self.service = services.Service(name, {"health": "/health"}, events="engine", owner=owner)

    @property
    def spec(self) -> dict: return settings.get(f"{self.owner}.adapters.{self.name}", {}) or {}
    @property
    def capabilities(self) -> dict:
        opts = (drivers.state(self.owner) or {}).get("options", {})
        return {k: v for k, v in self.spec.get("capabilities", {}).items() if drivers.when(v, opts)}

    def runtime(self) -> dict:
        """What to start, with this moment's settings, locations and driver filled in."""
        return drivers.fill(self.spec.get("runtime", {}), self.owner)

    def need(self, job) -> dict:
        mem = self.spec.get("memory", {})
        async def load(j):
            rt = self.runtime()
            self.service.spec.update(rt, idle_stop_s=self.spec.get("idle_stop_s", 0), slots=self.spec.get("slots", 1))
            self.service.slots = int(self.spec.get("slots", 1))
            if not await self.service.start(rt["cmd"], rt.get("env"), sig=json.dumps(rt, sort_keys=True)): raise RuntimeError(f"""{self.name} is not running ({self.service.status}): {" | ".join(self.service.tail(3))}""")
            self.service.measure()
            await self._remember()
        return {"service": self.service, "est_gb": float(mem.get("ram_gb", 1.0)), "ram_gb": min(float(mem.get("ram_gb", 1.0)), budget.can_free(services.REG, self.name)), "gpu_gb": float(mem.get("gpu_gb", 0.0)), "keep": bool(mem.get("keep")), "label": self.name, "load": load}

    def fits(self, service, job) -> bool: return service.sig == json.dumps(self.runtime(), sort_keys=True)   # a changed setting or a moved location: restart when idle; the program swaps its own models

    def on_cancel(self, job):
        """The program's own stop call, when it has one: closing the connection alone does not stop a blocking GPU job."""
        c = self.spec.get("cancel")
        if c and self.service.ready(): post_later(self.service, c["path"])

    async def run(self, job, service):
        if job.capability == "warm":
            yield {"type": "done", "reason": "stop"}
            return
        if job.capability.startswith("_act:"):   # an action that needed the program up: it is up now
            a = self.spec["actions"][job.capability[5:]]
            r = await service.client().request(a.get("method", "POST"), a["path"], params=job.body.get("_query"), timeout=600.0)
            yield {"type": "result", "data": r.json()}
            yield {"type": "done", "reason": "stop"}
            return
        cap = self.capabilities[job.capability]
        body = {**{cap.get("rename", {}).get(k, k): v for k, v in (render(cap["body"], job.body) if "body" in cap else job.body).items()}, **cap.get("body_add", {})}   # rename: {contract field: the program's own name for it} (an image program's "model_path" for the contract's "model")
        method, path = cap.get("method", "POST"), render(cap.get("path", "/"), job.body)
        if cap.get("stream"):
            async with service.client().stream(method, path, json=body) as r:
                if r.status_code >= 400: raise HTTPException(status_code=r.status_code, detail=(await r.aread()).decode(errors="replace")[:500])
                async for line in r.aiter_lines():
                    if line.strip(): yield {"type": "progress", **json.loads(line)}
            yield {"type": "done", "reason": "stop"}
            return
        r = await service.client().request(method, path, json=body if method != "GET" else None)
        if r.status_code >= 400: raise HTTPException(status_code=r.status_code, detail=r.text[:800])
        data = r.json() if "json" in r.headers.get("content-type", "") else {"text": r.text}
        yield {"type": "result", "data": dig(data, cap["result"]) if cap.get("result") else data}
        yield {"type": "done", "reason": "stop"}

    # --- Lists, without starting anything ---

    def _memo(self, key: str) -> Path:
        d = settings.STATE / "memo"
        d.mkdir(exist_ok=True)
        return d / f"{self.name}.{key}.json"

    async def _remember(self):
        """A program that lists things is asked once it is up; the answers are kept, so the lists work while it is stopped."""
        m = self.spec.get("models") or {}
        asks = {**({"models": m["list_path"]} if m.get("list_path") else {}), **{k: v["path"] for k, v in (self.spec.get("info") or {}).items() if v.get("remember")}}
        for key, path in asks.items():
            try:
                r = await self.service.client().get(path, timeout=15.0)
                if r.status_code == 200: self._memo(key).write_text(r.text)
            except (httpx.HTTPError, OSError) as e: print(f"[ADAPTER] {self.name}: could not read {path}: {e}", flush=True)

    def listed(self) -> dict: return json.loads(self._memo("models").read_text()) if self._memo("models").exists() else {}

    async def info(self, name: str) -> dict:
        """One of this program's status answers: from the program when it is up, else what it last said, marked stale, else its answer for stopped."""
        i = self.spec["info"][name]
        if self.service.ready():
            d = (await self.service.client().get(i["path"], timeout=10.0)).json()
            if i.get("remember"): self._memo(name).write_text(json.dumps(d))
            return d
        return {**json.loads(self._memo(name).read_text()), "stale": True} if i.get("remember") and self._memo(name).exists() else i.get("stopped", {"status": "stopped"})

    def models(self) -> list:
        m = self.spec.get("models") or {}
        if not m.get("root"): return []
        root, caps = settings.location(m["root"]), m.get("capabilities") or list(self.capabilities)
        if not root.exists(): return []
        out = [{"name": str(p.relative_to(root)), "capabilities": caps, "size": p.stat().st_size, "format": p.suffix.lstrip("."), "mtime": p.stat().st_mtime} for g in m.get("glob", []) for p in sorted(root.glob(g)) if p.is_file()]
        for d in sorted(root.glob("*/")) if m.get("dirs_with") else []:
            if (d / m["dirs_with"]).exists(): out.append({"name": d.name, "capabilities": caps, "size": sum(f.stat().st_size for f in d.rglob("*") if f.is_file()), "format": "folder", "mtime": d.stat().st_mtime})
        return out

def post_later(service, path: str):
    """A fire-and-forget POST to a service's own control route; used from places that cannot wait."""
    from core import jobs
    async def go():
        try: await service.client().post(path, timeout=10.0)
        except httpx.HTTPError as e: print(f"[ADAPTER] {service.name} {path}: {e}", flush=True)
    jobs.spawn(go())
