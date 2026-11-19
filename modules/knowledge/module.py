"""modules/knowledge/module.py - knowledge servers behind this node. Each instance of a server (LightRAG today) is reached at /knowledge/<instance>/<path>, for an allowed list of paths only: nothing else of the server is exposed, and it listens on loopback or a private container network.
What a group can do is offered as contract capabilities (knowledge_query, knowledge_add, knowledge_status, knowledge_documents, knowledge_remove; the group is the "model" a caller names), asked through /run like all other work; how each is asked of the program is this module's capability table (module.json), not code.
A health check or a status read while it is stopped answers "stopped" and starts nothing; real work starts it. Groups (instances) are managed from afar: GET /knowledge lists them, PUT /knowledge/<group> adds or changes one (its chat and embedding models, its memory), DELETE /knowledge/<group> removes one - all without a restart. A portal reaches a group through this node's own connection; its knowledge profile names the paths. Its own LLM and embedding calls come back through this node, so they share the queue, the budget and the kill switch with everything else. Each instance's data is under the knowledge location."""
import re, json, time, base64, shutil, asyncio, hashlib
import httpx
from core import settings, services, journal, jobs, drivers, auth, modules, api
from core.adapters import Adapter, render, dig
from core.http import Router, HTTPException, StreamingResponse, json_body

router = Router()
READ_ONLY = ("/health", "/documents/pipeline_status", "/documents/track_status", "/documents/status_counts", "/openapi.json")   # answered "stopped" rather than starting the server

def _safe_name(name: str) -> str:
    """A document name every program and file system takes: letters (any script), digits, . _ - kept; anything else becomes _; the extension stays. Programs that refuse names they think unsafe (path signs, odd characters) are given one they accept rather than refusing the document."""
    stem, dot, ext = name.rpartition(".") if "." in name.strip(".") else (name, "", "")
    clean = lambda x: re.sub(r"[^\w.-]+", "_", x).strip("._")   # \w keeps letters of every script; a run of anything else is one _ (a "__" already in a name stays: it marks a folder)
    return (clean(stem)[:180] or "document") + (dot + clean(ext)[:16] if clean(ext) else "")

def _sha(b: dict) -> str: return hashlib.sha256((b.get("data") or "\x00".join(b.get("pieces") or [b.get("text") or ""])).encode()).hexdigest()   # what an add sends, whichever way it sends it

class Instance(Adapter):
    def __init__(self, name: str, spec: dict):
        super().__init__(f"knowledge:{name}", spec)
        self.inst, self.owner = name, "knowledge"
        self.service = services.Service(self.name, {"idle_stop_s": spec.get("idle_stop_s", 0), "health": "/health"}, events="engine", owner="knowledge")
        self.service.adapter = self
        self.mem = spec.get("memory", {})

    @property
    def capabilities(self) -> dict: return settings.get("knowledge.capabilities", {}) or {}   # how each contract capability is asked of the knowledge program: data (module.json), so another program is another table

    def models(self) -> list:
        """The group itself is what a caller names: an asset of kind "knowledge", with what it holds and the query options its program takes."""
        return [{"name": self.inst, "kind": "knowledge", "capabilities": list(self.capabilities), "size": 0, "notes": self.spec.get("notes", ""), "options": settings.get("knowledge.query_options", {}) or {}}]

    def need(self, job) -> dict:
        async def load(j):
            rt = drivers.fill(settings.merge(settings.get("knowledge.runtime", {}), self.spec.get("runtime", {})), "knowledge")
            rt["env"] = {k: v.replace("{instance}", self.inst) for k, v in rt.get("env", {}).items()}
            chat, embed = _choice(self.spec, "chat"), _choice(self.spec, "embed")
            if miss := [k for k, c in (("chat", chat), ("embedding", embed)) if not c.get("model")]: raise RuntimeError(f"""{self.name} has no {" and no ".join(miss)} model: choose them in the group's settings""")
            rt["env"].update(LLM_MODEL=f"kimi~{self.inst}~chat", EMBEDDING_MODEL=f"kimi~{self.inst}~embed", EMBEDDING_DIM=str(embed.get("dim") or await _measure_dim(_routed(embed))), **{_PROGRAM()[k]["env"]: str(v) for k, v in (self.spec.get("program") or {}).items() if k in _PROGRAM()})
            if (vision := _choice(self.spec, "vision")).get("model"): rt["env"].update(VLM_PROCESS_ENABLE="true", VLM_LLM_MODEL=f"kimi~{self.inst}~vision", VLM_LLM_BINDING=rt["env"].get("LLM_BINDING", "ollama"), VLM_LLM_BINDING_HOST=rt["env"].get("LLM_BINDING_HOST", ""))   # pictures in documents described by this model, through the same door as the chat model (LightRAG from 2026.05; an older program ignores these)
            self.service.spec.update(rt)
            if not await self.service.start(rt["cmd"], rt["env"]): raise RuntimeError(f"""{self.name} is not running ({self.service.status}): {" | ".join(self.service.tail(3))}""")
            self.service.measure()
        return {"service": self.service, "ram_gb": float(self.mem.get("ram_gb", 2.0)), "label": self.name, "load": load}

    async def run(self, job, service):
        """A contract capability, asked of the program the way this module's capability table says. ("warm" only brings the server up, for the calls that still pass in proxy() below.)
        "stored" in the table answers from a file the program keeps in the group's folder, where the program has no route for it; "forget" clears the record of what was added once the program has emptied the group.
        Two things the program does not do are done here, for every program: a list asked for without a page is gathered over all its pages ("pages" in the table), and adding under a name the group already holds replaces what it holds under that name, or does nothing when the content is the same as last time ("replace"; the names and content hashes added are kept in the group's folder, added.json)."""
        if job.capability == "warm":
            yield {"type": "done", "reason": "stop"}
            return
        cap, b = self.capabilities[job.capability], job.body
        if self._queued(): self.retry_later()
        if cap.get("upload"): b = {**b, "name": _safe_name(str(b["name"]))}
        if cap.get("stored"): data = await asyncio.to_thread(self._stored, cap["stored"], b)
        elif cap.get("pages") and "page" not in b: data = {cap["pages"]["list"]: await self._all(service, job.capability, b)}
        elif cap.get("replace") and b.get("name") and (same := await self._replace(service, b)): data = same
        else:
            data = await self._send(service, cap["path"], cap["upload"]["field"], str(b["name"]), base64.b64decode(b["data"]), float(cap.get("timeout_s", 600))) if cap.get("upload") else await self._ask(service, cap.get("method", "POST"), render(cap["path"], b), self._body(cap, b), float(cap.get("timeout_s", 120)), render(cap["query"], {**cap.get("query_add", {}), **b}) if cap.get("query") else None)
            data = data if isinstance(data, dict) else {cap.get("as", "items"): data}   # a program that answers a bare list: the list under the name the table gives
            if (q := cap.get("queue_when")) and all(dig(data, k) == v for k, v in q.items()): data = self._queue(job.capability, b, data)
            if cap.get("upload"): data = {**data, "name": b["name"]}
            if cap.get("replace") and b.get("name"): self._added({**self._added(), str(b["name"]): _sha(b)})
            if cap.get("forget"): self._added({})   # the group was emptied: nothing counts as added before, so the next sync sends everything
        asked = {}   # each extra path asked once, however many fields are taken from it
        for key, more in (cap.get("also") or {}).items():
            got = asked[more["path"]] = asked.get(more["path"]) or await self._ask(service, "GET", more["path"], None, 30.0)
            data = {**data, key: {k: dig(got, k) for k in more["take"]} if isinstance(more["take"], list) else dig(got, more["take"])}   # "take": one field, or a list of fields given together
        if cap.get("shows_queued"): data = {**data, "queued": self._queued()}
        if cap.get("text"): yield {"type": "text", "text": str(dig(data, cap["text"]) or "")}
        yield {"type": "result", "data": data}
        yield {"type": "done", "reason": "stop"}

    def _body(self, cap: dict, b: dict) -> dict:
        body = {**cap.get("body_add", {}), **(render(cap["body"], b) if "body" in cap else {}), **((b.get("options") or {}) if cap.get("options") else {})}
        if b.get("pieces") and cap.get("pieces"): body = {**body, "text": cap["pieces"]["separator"].join(b["pieces"]), **render(cap["pieces"]["body"], {"separator": cap["pieces"]["separator"]})}   # one document whose chunks are exactly those pieces
        return body

    async def _all(self, service, capability: str, b: dict) -> list:
        """Every page of a list, in order."""
        cap, out, page = self.capabilities[capability], [], 1
        while True:
            d = await self._ask(service, cap.get("method", "POST"), cap["path"], self._body(cap, {**b, "page": page, "page_size": cap["pages"]["size"]}), float(cap.get("timeout_s", 120)))
            out += dig(d, cap["pages"]["list"]) or []
            if not dig(d, cap["pages"]["more"]): return out
            page += 1

    def _stored(self, how: dict, b: dict) -> dict:
        """Something the program keeps in the group's own folder and offers no route for, read from its file ("stored" in the table): {"file": a pattern under the group's folder, "key": which entry (filled from the request), "take": the part of the entry that is the text}. The newest file matching is read; nothing is written."""
        files = sorted((settings.location("knowledge") / self.inst).glob(how["file"]), key=lambda f: f.stat().st_mtime)
        if not files: raise HTTPException(status_code=404, detail=f"""{self.name}: no file {how["file"]} in its folder - it holds nothing yet, or its program keeps this elsewhere (the capability table says where)""")
        entry = json.loads(files[-1].read_text()).get(str(render(how["key"], b)))
        if entry is None: raise HTTPException(status_code=404, detail=f"""{self.name}: {files[-1].name} has no entry {render(how["key"], b)}""")
        return {"id": render(how["key"], b), "text": str(dig(entry, how["take"]) if how.get("take") else entry)}

    # --- Asked while the program was busy: kept and sent again ---

    def _queue_file(self): return settings.location("knowledge") / self.inst / "queued.json"
    def _queued(self) -> list: return json.loads(self._queue_file().read_text()) if self._queue_file().exists() else []

    def _keep(self, items: list):
        self._queue_file().parent.mkdir(parents=True, exist_ok=True)
        self._queue_file().write_text(json.dumps(items, indent=1))

    def _queue(self, capability: str, b: dict, said: dict) -> dict:
        """A request the program refused because it was busy ("queue_when" in the table), kept in the group's folder and sent again until taken. The same request asked again is kept once."""
        item = {"capability": capability, "body": {k: v for k, v in b.items() if k != "model"}, "t": time.time(), "said": str(dig(said, "message") or "")[:300]}
        if not any(i["capability"] == capability and i["body"] == item["body"] for i in self._queued()): self._keep([*self._queued(), item])
        journal.event("knowledge.queued", group=self.inst, capability=capability)
        self.retry_later()
        return {**said, "status": "queued", "queued": len(self._queued()), "note": "the program is busy: kept, and sent again as soon as it takes it"}

    def retry_later(self):
        if not getattr(self, "_retrying", None) or self._retrying.done(): self._retrying = jobs.spawn(self._retry())

    async def _retry(self):
        """Every RETRY_S, the oldest kept request sent again; one taken is dropped from the list, one refused as busy again stays. A request the program refuses for another reason is dropped and said in the journal."""
        while self._queued():
            await asyncio.sleep(RETRY_S)
            item, cap = self._queued()[0], self.capabilities.get(self._queued()[0]["capability"], {})
            try:
                await _ensure(self)
                got = await self._ask(self.service, cap.get("method", "POST"), render(cap["path"], item["body"]), self._body(cap, item["body"]), float(cap.get("timeout_s", 120)))
                if all(dig(got, k) == v for k, v in (cap.get("queue_when") or {}).items()): continue
                journal.event("knowledge.queued.sent", group=self.inst, capability=item["capability"], status=str(dig(got, "status") or ""))
            except Exception as e: journal.event("knowledge.queued.failed", group=self.inst, capability=item["capability"], error=f"{type(e).__name__}: {e}"[:300])   # a kept request that cannot be sent is not kept forever
            self._keep([i for i in self._queued() if i != item])

    def _added(self, new: dict = None) -> dict:
        """What has been added under which name, with the hash of its content: read, or written when new is given."""
        f = settings.location("knowledge") / self.inst / "added.json"
        if new is None: return json.loads(f.read_text()) if f.exists() else {}
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(new, indent=1))
        return new

    async def _replace(self, service, b: dict):
        """Before adding under a name: {"status": "unchanged"} when the same content went in under it last time and the program still holds it; otherwise what the group holds under that name is removed (and None answered, so the add goes on)."""
        name, held = str(b["name"]), self._added()
        docs, rm = await self._all(service, "knowledge_documents", {}), self.capabilities["knowledge_remove"]
        ids = [d["id"] for d in docs if d.get("file_path") == name]
        if ids and held.get(name) == _sha(b) and not b.get("again"): return {"status": "unchanged", "name": name}   # only when the program still holds it: what it lost is sent again
        if ids: await self._ask(service, rm.get("method", "POST"), rm["path"], self._body(rm, {"ids": ids, "delete_file": True}), float(rm.get("timeout_s", 120)))
        return None

    async def _send(self, service, path: str, field: str, name: str, content: bytes, timeout: float) -> dict:
        r = await service.client().post(path, files={field: (name, content)}, headers={"X-API-Key": self.spec["api_key"]} if self.spec.get("api_key") else {}, timeout=httpx.Timeout(connect=10.0, read=timeout, write=120.0, pool=10.0))
        if r.status_code >= 400: raise HTTPException(status_code=r.status_code, detail=f"{self.name}: {r.text[:600]}")
        return r.json()

    async def _ask(self, service, method: str, path: str, body, timeout: float, query: dict = None) -> dict:
        r = await service.client().request(method, path, params=query, json=body if method != "GET" else None, headers={"X-API-Key": self.spec["api_key"]} if self.spec.get("api_key") else {}, timeout=httpx.Timeout(connect=10.0, read=timeout, write=60.0, pool=10.0))
        if r.status_code >= 400: raise HTTPException(status_code=r.status_code, detail=f"{self.name}: {r.text[:600]}")
        return r.json()

INSTANCES: dict = {}
RETRY_S = 15   # how often a request kept while the program was busy is sent again
_PROGRAM = lambda: settings.get("knowledge.program_settings", {}) or {}   # {field: {"env", "label", "type", "default", "hint"}}: the program's own settings a group may set, given to it as environment

async def _ensure(inst: Instance):
    """Brings the server up through the scheduler, so it is counted against the node's memory."""
    if inst.service.ready(): return
    job = jobs.submit(jobs.Job("warm", inst, {}, rid=""))
    while (ev := await job.out.get()) is not None:
        if ev.get("type") == "error": raise HTTPException(status_code=503, detail=ev["message"])

async def proxy(request, instance: str, path: str = ""):
    inst = INSTANCES.get(instance)
    p = "/" + path
    if not inst: raise HTTPException(status_code=404, detail=f"no knowledge instance {instance!r} - known: {sorted(INSTANCES)}")
    allowed = settings.get("knowledge.allowed", [])
    if not any(p == a or p.startswith(a.rstrip("/") + "/") for a in allowed): raise HTTPException(status_code=403, detail=f"{p} is not among the paths this node exposes (knowledge.allowed in the node settings)")
    if not inst.service.ready():
        if any(p == r or p.startswith(r + "/") for r in READ_ONLY): return {"status": "stopped", "busy": False, "note": "the knowledge server is stopped; it starts with real work"} if p != "/health" else {"status": "stopped"}
        await _ensure(inst)
    inst.service.active += 1
    inst.service.last_use = time.time()
    t0 = inst.service.last_use
    headers = {k: v for k, v in request.headers.items() if k.lower() in ("content-type", "accept")}
    if inst.spec.get("api_key"): headers["X-API-Key"] = inst.spec["api_key"]
    try: r = await inst.service.client().send(inst.service.client().build_request(request.method, p, params=request.query_params, content=await request.body(), headers=headers), stream=True)
    except httpx.HTTPError as x:
        inst.service.active -= 1
        raise HTTPException(status_code=502, detail=f"{inst.name}: {type(x).__name__}: {x}")
    async def body():
        try:
            async for chunk in r.aiter_bytes(): yield chunk
        finally:
            await r.aclose()
            inst.service.active -= 1
            inst.service.last_use = time.time()
            journal.event("knowledge.req", name=inst.name, path=p, method=request.method, status=r.status_code, seconds=round(inst.service.last_use - t0, 3))
    return StreamingResponse(body(), status_code=r.status_code, media_type=r.headers.get("content-type"))

# --- Groups: one knowledge server each, managed from afar ---

GROUP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")   # no dots: a group name is one step of a settings path

def _choice(spec: dict, kind: str) -> dict:
    """A group's model choice for kind ("chat", "embed", or "vision" - optional, describes pictures in documents): {"source": "" (this node) or one of remote.sources, "model": the name that machine uses, "dim": embedding size when known}."""
    return {"source": "", "model": "", **(spec.get(kind) or {})}

_LACKS: dict = {}   # what _lacks found, and the install it was found in

def _lacks() -> dict:
    """{field: why} for the group fields this node's knowledge program cannot act on, judged from what is installed (its own settings in its code), so a choice it would ignore says so instead of doing nothing."""
    env = settings.STATE / "envs" / "lightrag"
    if not env.exists(): return {"vision": "the knowledge program is not installed on this node yet"}
    code = next(iter(env.glob("lib/python*/site-packages/lightrag")), None)
    if code and _LACKS.get("at") == code.stat().st_mtime: return _LACKS["is"]   # read again only when the program is installed again
    _LACKS["at"] = code.stat().st_mtime if code else None
    _LACKS["is"] = {} if code and any("VLM_PROCESS_ENABLE" in f.read_text(errors="ignore") for f in code.rglob("*.py")) else {"vision": "the installed knowledge program does not describe pictures (LightRAG from 2026.05 does): update the program on this node, then the choice takes effect"}
    return _LACKS["is"]

def _standing(rest: str) -> str:
    """What "kimi~<group>~<kind>" names now: the group's current choice for kind, as this node routes it. The knowledge program is given these standing names, so its models are chosen here (and can change) instead of fixed in it at start."""
    group, _, kind = rest.rpartition("~")
    if group not in INSTANCES: raise HTTPException(status_code=404, detail=f"no knowledge group {group!r} for the name kimi~{rest}")
    if not (c := _choice(INSTANCES[group].spec, kind)).get("model"): raise HTTPException(status_code=404, detail=f"knowledge group {group} has no {kind} model set")
    return _routed(c)

def _routed(c: dict) -> str: return f"""{c["source"]}/{c["model"]}""" if c.get("source") else c.get("model", "")   # the name this node routes: a source's models are offered here as <source>/<model> (modules/remote)

async def _measure_dim(model: str) -> int:
    """The embedding size of model, asked of this node itself: a knowledge server needs it, and only a model's own answer is certain."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=300.0, write=10.0, pool=5.0), trust_env=False) as c: r = await c.post(f"""http://127.0.0.1:{settings.owner("bind.port", settings.PORT)}/api/embed""", headers={"Authorization": f"Bearer {auth.LOCAL_KEY}"}, json={"model": model, "input": ["dimension probe"]})
    if r.status_code != 200: raise RuntimeError(f"embedding model {model} did not answer ({r.status_code}): {r.text[:200]}")
    return len(r.json()["embeddings"][0])

def _same(a: dict, b: dict) -> bool: return (a.get("source", ""), a.get("model", "")) == (b.get("source", ""), b.get("model", ""))
def _stored(name: str) -> bool: return any((settings.location("knowledge") / name / "rag_storage").glob("*")) if (settings.location("knowledge") / name / "rag_storage").exists() else False

def _view(inst) -> dict: return {"name": inst.inst, "status": inst.service.status, "chat": _choice(inst.spec, "chat"), "embed": _choice(inst.spec, "embed"), "vision": _choice(inst.spec, "vision"), "ram_gb": inst.mem.get("ram_gb", 2.0), "notes": inst.spec.get("notes", ""), **(inst.spec.get("program") or {}), "queued": len(inst._queued()), "stored": _stored(inst.inst), "data": str(settings.location("knowledge") / inst.inst)}

async def groups(request): return {"groups": [_view(i) for i in INSTANCES.values()], "sources": sorted((settings.get("remote.sources", {}) or {}).keys())}

def _fields() -> list: return [{"name": "chat", "label": "Model that reads and answers", "pick": "model:chat", "required": True}, {"name": "embed", "label": "Model that embeds", "pick": "model:embed", "required": True, "hint": "Changing it later makes what is stored unfindable: the documents then have to be added again."}, {"name": "vision", "label": "Model that describes pictures in documents", "pick": "model:vision", "hint": "Optional. Pictures, charts and scanned pages in what is added are described by this model and the description is stored with the text. Empty: pictures are skipped. Needs a model that sees (a llama model with its mmproj file beside it lists as vision) and a knowledge program from 2026.05 or later; an older one ignores it."}, {"name": "ram_gb", "label": "Memory for its server (GB)", "type": "number", "default": 2.0}, {"name": "notes", "label": "What it holds", "type": "long", "hint": "For people and for models choosing where to look."}, *({"name": k, "advanced": True, **{x: v for x, v in f.items() if x != "env"}} for k, f in _PROGRAM().items())]   # what a knowledge group is made of: its models, then its program's own settings (_PROGRAM, "advanced"; read when asked, after settings are loaded)

def _fields_now() -> list:
    """The group fields, each marked with what this node cannot act on yet ("unavailable": why), asked each time the kinds are listed."""
    lacks = _lacks()
    return [{**f, **({"unavailable": lacks[f["name"]]} if f["name"] in lacks else {})} for f in _fields()]

async def _asset_list(request) -> list: return [{**_view(i), "chat": _routed(_choice(i.spec, "chat")), "embed": _routed(_choice(i.spec, "embed")), "vision": _routed(_choice(i.spec, "vision"))} for i in INSTANCES.values()]   # the groups in the shape of /assets: each field as it is set

async def group_set(request, group: str):
    """{"chat": {"source", "model"}, "embed": {"source", "model", "dim" (optional: else measured at start)}, "ram_gb", "notes" (what the group holds, for people and agents), "embed_change_ok"}: adds the group or changes it. source "" is this node; otherwise one of remote.sources. A running group stops and starts again with the change at its next use. Changing the embedding model of a group that holds data needs embed_change_ok: what is stored was embedded with the old one."""
    if not GROUP_NAME.match(group): raise HTTPException(status_code=400, detail="a group name is letters, digits, - and _ (up to 64)")
    b, old = await json_body(request), INSTANCES.get(group)
    known = set((settings.get("remote.sources", {}) or {}).keys())
    for k in ("chat", "embed", "vision"):   # a model may also be named the way this node lists it: "<source>/<model>" for a source's, the bare name for its own
        if isinstance(b.get(k), str): b[k] = {"source": b[k].split("/", 1)[0], "model": b[k].split("/", 1)[1]} if b[k].split("/", 1)[0] in known and "/" in b[k] else {"source": "", "model": b[k]}
    pick = {k: {"source": str((b.get(k) or {}).get("source") or ""), "model": str((b.get(k) or {}).get("model") or "").strip(), **({"dim": int(b[k]["dim"])} if (b.get(k) or {}).get("dim") else {})} for k in ("chat", "embed", "vision") if k != "vision" or (b.get(k) or {}).get("model")}
    if bad := [c["source"] for c in pick.values() if c["source"] and c["source"] not in known]: raise HTTPException(status_code=400, detail=f"no model source named {bad[0]} on this node - sources: {sorted(known)}")
    if old and _stored(group) and not _same(_choice(old.spec, "embed"), pick["embed"]) and not b.get("embed_change_ok"): raise HTTPException(status_code=409, detail=f"""{group} holds data embedded with {_routed(_choice(old.spec, "embed"))}: searches with another embedding model find nothing. Remove its data and add the documents again, or confirm the change.""")
    program = {k: (int(b[k]) if f.get("type") == "integer" else float(b[k]) if f.get("type") == "number" else str(b[k])) for k, f in _PROGRAM().items() if b.get(k) not in (None, "")}   # only what is set: the rest stays the program's own default
    spec = {"memory": {"ram_gb": float(b.get("ram_gb") or 2.0)}, "notes": str(b.get("notes") or "")[:2000], **pick, **({"program": program} if program else {})}
    settings.set_path(f"knowledge.instances.{group}", spec, str(b.get("source") or request.state.key.get("name", "")))
    if old:
        fixed = lambda s: (_choice(s, "embed"), s.get("program"), s.get("memory"), bool(_choice(s, "vision").get("model")))   # what the program is started with; the chat and vision models it reaches by standing names, so changing those needs no restart
        if old.service.running() and fixed(old.spec) != fixed(spec): await old.service.stop("group settings changed")
        old._spec, old.mem = spec, spec["memory"]
    else: INSTANCES[group] = modules.ADAPTERS[f"knowledge:{group}"] = Instance(group, spec)
    journal.event("knowledge.group.set", group=group, chat=_routed(pick["chat"]), embed=_routed(pick["embed"]))
    return _view(INSTANCES[group])

async def group_rename(request, group: str):
    """{"to": new name}: the group and everything it stored under a new name (its folder moved, its settings moved, its program stopped first). Refused when the new name is taken, on this node or as a folder."""
    to, inst = str((await json_body(request)).get("to") or "").strip(), INSTANCES.get(group)
    if not inst: raise HTTPException(status_code=404, detail=f"no knowledge group {group}")
    if not GROUP_NAME.match(to): raise HTTPException(status_code=400, detail="a group name is letters, digits, - and _ (up to 64)")
    if to in INSTANCES or (to.lower() != group.lower() and (settings.location("knowledge") / to).exists()): raise HTTPException(status_code=409, detail=f"{to} is already a group or a folder in this node's knowledge place: nothing was moved")
    if settings.owner(f"knowledge.instances.{group}") is not None: raise HTTPException(status_code=409, detail=f"{group} is set in this node's own node.json: only its owner renames it there")
    if inst.service.running(): await inst.service.stop("group renamed")
    old_dir, new_dir = settings.location("knowledge") / group, settings.location("knowledge") / to
    if old_dir.exists(): (old_dir.rename(old_dir.with_name(f".{to}.renaming")).rename(new_dir)) if to.lower() == group.lower() else old_dir.rename(new_dir)   # through a third name when only the case changes, for filesystems that ignore case
    who = str(request.state.key.get("name", ""))
    settings.set_path(f"knowledge.instances.{to}", inst.spec, who)
    settings.set_path(f"knowledge.instances.{group}", None, who)
    INSTANCES.pop(group)
    modules.ADAPTERS.pop(inst.name, None)
    services.REG.pop(inst.name, None)
    INSTANCES[to] = modules.ADAPTERS[f"knowledge:{to}"] = Instance(to, inst.spec)
    journal.event("knowledge.group.renamed", group=group, to=to)
    return _view(INSTANCES[to])

async def group_remove(request, group: str):
    """?confirm=1 removes the group; &data=1 also deletes what it stored (documents, graph, embeddings). Without data its folder stays, and a group of the same name finds it again."""
    if not request.query_params.get("confirm"): raise HTTPException(status_code=400, detail="destructive - send ?confirm=1")
    inst = INSTANCES.get(group)
    if not inst: raise HTTPException(status_code=404, detail=f"no knowledge group {group}")
    if settings.owner(f"knowledge.instances.{group}") is not None: raise HTTPException(status_code=409, detail=f"{group} is set in this node's own node.json: only its owner removes it there")
    if inst.service.running(): await inst.service.stop("group removed")
    settings.set_path(f"knowledge.instances.{group}", None, str(request.state.key.get("name", "")))
    INSTANCES.pop(group)
    modules.ADAPTERS.pop(inst.name, None)
    services.REG.pop(inst.name, None)
    gone = request.query_params.get("data") == "1" and (settings.location("knowledge") / group).exists()
    if gone: shutil.rmtree(settings.location("knowledge") / group)
    journal.event("knowledge.group.removed", group=group, data_deleted=gone)
    return {"removed": group, "data_deleted": gone}

router.add("/knowledge", groups, ("GET",), "read")
router.add("/knowledge/{group}", group_set, ("PUT",), "admin")
router.add("/knowledge/{group}", group_remove, ("DELETE",), "admin")
router.add("/knowledge/{group}/rename", group_rename, ("POST",), "admin")
api.ALIASES["kimi"] = _standing
router.add("/knowledge/{instance}/{path:path}", proxy, ("GET", "POST", "PUT", "DELETE"), "infer")

def setup(ctx):
    for name, spec in (ctx.cfg("instances") or {}).items():
        if spec.get("enabled", True):
            INSTANCES[name] = Instance(name, spec)
            ctx.add_adapter(INSTANCES[name])   # each group is an adapter of its own: its name is the "model" a knowledge request names
    ctx.add_router(router)
    ctx.add_asset_kind("knowledge", {"label": "Knowledge group", "about": "A body of documents that can be asked. Each group has its own server and data folder, and its own two models: one that reads and answers, one that embeds.",
        "fields": _fields(), "fields_now": _fields_now,
        "holds": True, "keep_case": True, "list": _asset_list, "set": group_set, "remove": group_remove, "rename": group_rename})   # holds: removing one may also delete what it stored (?data=1)
    print(f"""[KNOWLEDGE] instances: {sorted(INSTANCES) or 'none'}""", flush=True)
