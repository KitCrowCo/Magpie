"""modules/knowledge/module.py - knowledge servers behind this node. Each instance of a server (LightRAG today) is reached at /knowledge/<instance>/<path>, for an allowed list of paths only: nothing else of the server is exposed, and it listens on loopback or a private container network.
What a group can do is offered as contract capabilities (knowledge_query, knowledge_add, knowledge_status, knowledge_documents, knowledge_remove; the group is the "model" a caller names), asked through /run like all other work; how each is asked of the program is this module's capability table (module.json), not code.
A health check or a status read while it is stopped answers "stopped" and starts nothing; real work starts it. Groups (instances) are managed from afar: GET /knowledge lists them, PUT /knowledge/<group> adds or changes one (its chat and embedding models, its memory), DELETE /knowledge/<group> removes one - all without a restart. A portal reaches a group through this node's own connection; its knowledge profile names the paths. Its own LLM and embedding calls come back through this node, so they share the queue, the budget and the kill switch with everything else. Each instance's data is under the knowledge location."""
import re, json, time, base64, shutil, hashlib
import httpx
from core import settings, services, journal, jobs, drivers, auth, modules
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
            rt["env"].update(LLM_MODEL=_routed(chat), EMBEDDING_MODEL=_routed(embed), EMBEDDING_DIM=str(embed.get("dim") or await _measure_dim(_routed(embed))))
            self.service.spec.update(rt)
            if not await self.service.start(rt["cmd"], rt["env"]): raise RuntimeError(f"""{self.name} is not running ({self.service.status}): {" | ".join(self.service.tail(3))}""")
            self.service.measure()
        return {"service": self.service, "ram_gb": float(self.mem.get("ram_gb", 2.0)), "label": self.name, "load": load}

    async def run(self, job, service):
        """A contract capability, asked of the program the way this module's capability table says. ("warm" only brings the server up, for the calls that still pass in proxy() below.)
        Two things the program does not do are done here, for every program: a list asked for without a page is gathered over all its pages ("pages" in the table), and adding under a name the group already holds replaces what it holds under that name, or does nothing when the content is the same as last time ("replace"; the names and content hashes added are kept in the group's folder, added.json)."""
        if job.capability == "warm":
            yield {"type": "done", "reason": "stop"}
            return
        cap, b = self.capabilities[job.capability], job.body
        if cap.get("upload"): b = {**b, "name": _safe_name(str(b["name"]))}
        if cap.get("pages") and "page" not in b: data = {cap["pages"]["list"]: await self._all(service, job.capability, b)}
        elif cap.get("replace") and b.get("name") and (same := await self._replace(service, b)): data = same
        else:
            data = await self._send(service, cap["path"], cap["upload"]["field"], str(b["name"]), base64.b64decode(b["data"]), float(cap.get("timeout_s", 600))) if cap.get("upload") else await self._ask(service, cap.get("method", "POST"), render(cap["path"], b), self._body(cap, b), float(cap.get("timeout_s", 120)), render(cap["query"], {**cap.get("query_add", {}), **b}) if cap.get("query") else None)
            data = data if isinstance(data, dict) else {cap.get("as", "items"): data}   # a program that answers a bare list: the list under the name the table gives
            if cap.get("upload"): data = {**data, "name": b["name"]}
            if cap.get("replace") and b.get("name"): self._added({**self._added(), str(b["name"]): _sha(b)})
        for key, more in (cap.get("also") or {}).items(): data = {**data, key: dig(await self._ask(service, "GET", more["path"], None, 30.0), more["take"])}
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
    """A group's model choice for kind ("chat" or "embed"): {"source": "" (this node) or one of remote.sources, "model": the name that machine uses, "dim": embedding size when known}."""
    return {"source": "", "model": "", **(spec.get(kind) or {})}

def _routed(c: dict) -> str: return f"""{c["source"]}/{c["model"]}""" if c.get("source") else c.get("model", "")   # the name this node routes: a source's models are offered here as <source>/<model> (modules/remote)

async def _measure_dim(model: str) -> int:
    """The embedding size of model, asked of this node itself: a knowledge server needs it, and only a model's own answer is certain."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=300.0, write=10.0, pool=5.0), trust_env=False) as c: r = await c.post(f"""http://127.0.0.1:{settings.owner("bind.port", settings.PORT)}/api/embed""", headers={"Authorization": f"Bearer {auth.LOCAL_KEY}"}, json={"model": model, "input": ["dimension probe"]})
    if r.status_code != 200: raise RuntimeError(f"embedding model {model} did not answer ({r.status_code}): {r.text[:200]}")
    return len(r.json()["embeddings"][0])

def _same(a: dict, b: dict) -> bool: return (a.get("source", ""), a.get("model", "")) == (b.get("source", ""), b.get("model", ""))
def _stored(name: str) -> bool: return any((settings.location("knowledge") / name / "rag_storage").glob("*")) if (settings.location("knowledge") / name / "rag_storage").exists() else False

def _view(inst) -> dict: return {"name": inst.inst, "status": inst.service.status, "chat": _choice(inst.spec, "chat"), "embed": _choice(inst.spec, "embed"), "ram_gb": inst.mem.get("ram_gb", 2.0), "notes": inst.spec.get("notes", ""), "stored": _stored(inst.inst), "data": str(settings.location("knowledge") / inst.inst)}

async def groups(request): return {"groups": [_view(i) for i in INSTANCES.values()], "sources": sorted((settings.get("remote.sources", {}) or {}).keys())}

async def _asset_list(request) -> list: return [{**_view(i), "chat": _routed(_choice(i.spec, "chat")), "embed": _routed(_choice(i.spec, "embed"))} for i in INSTANCES.values()]   # the groups in the shape of /assets: each field as it is set

async def group_set(request, group: str):
    """{"chat": {"source", "model"}, "embed": {"source", "model", "dim" (optional: else measured at start)}, "ram_gb", "notes" (what the group holds, for people and agents), "embed_change_ok"}: adds the group or changes it. source "" is this node; otherwise one of remote.sources. A running group stops and starts again with the change at its next use. Changing the embedding model of a group that holds data needs embed_change_ok: what is stored was embedded with the old one."""
    if not GROUP_NAME.match(group): raise HTTPException(status_code=400, detail="a group name is letters, digits, - and _ (up to 64)")
    b, old = await json_body(request), INSTANCES.get(group)
    known = set((settings.get("remote.sources", {}) or {}).keys())
    for k in ("chat", "embed"):   # a model may also be named the way this node lists it: "<source>/<model>" for a source's, the bare name for its own
        if isinstance(b.get(k), str): b[k] = {"source": b[k].split("/", 1)[0], "model": b[k].split("/", 1)[1]} if b[k].split("/", 1)[0] in known and "/" in b[k] else {"source": "", "model": b[k]}
    pick = {k: {"source": str((b.get(k) or {}).get("source") or ""), "model": str((b.get(k) or {}).get("model") or "").strip(), **({"dim": int(b[k]["dim"])} if (b.get(k) or {}).get("dim") else {})} for k in ("chat", "embed")}
    if bad := [c["source"] for c in pick.values() if c["source"] and c["source"] not in known]: raise HTTPException(status_code=400, detail=f"no model source named {bad[0]} on this node - sources: {sorted(known)}")
    if old and _stored(group) and not _same(_choice(old.spec, "embed"), pick["embed"]) and not b.get("embed_change_ok"): raise HTTPException(status_code=409, detail=f"""{group} holds data embedded with {_routed(_choice(old.spec, "embed"))}: searches with another embedding model find nothing. Remove its data and add the documents again, or confirm the change.""")
    spec = {"memory": {"ram_gb": float(b.get("ram_gb") or 2.0)}, "notes": str(b.get("notes") or "")[:2000], **pick}
    settings.set_path(f"knowledge.instances.{group}", spec, str(b.get("source") or request.state.key.get("name", "")))
    if old:
        if old.service.running(): await old.service.stop("group settings changed")
        old._spec, old.mem = spec, spec["memory"]
    else: INSTANCES[group] = modules.ADAPTERS[f"knowledge:{group}"] = Instance(group, spec)
    journal.event("knowledge.group.set", group=group, chat=_routed(pick["chat"]), embed=_routed(pick["embed"]))
    return _view(INSTANCES[group])

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
router.add("/knowledge/{instance}/{path:path}", proxy, ("GET", "POST", "PUT", "DELETE"), "infer")

def setup(ctx):
    for name, spec in (ctx.cfg("instances") or {}).items():
        if spec.get("enabled", True):
            INSTANCES[name] = Instance(name, spec)
            ctx.add_adapter(INSTANCES[name])   # each group is an adapter of its own: its name is the "model" a knowledge request names
    ctx.add_router(router)
    ctx.add_asset_kind("knowledge", {"label": "Knowledge group", "about": "A body of documents that can be asked. Each group has its own server and data folder, and its own two models: one that reads and answers, one that embeds.",
        "fields": [{"name": "chat", "label": "Model that reads and answers", "pick": "model:chat", "required": True}, {"name": "embed", "label": "Model that embeds", "pick": "model:embed", "required": True, "hint": "Changing it later makes what is stored unfindable: the documents then have to be added again."}, {"name": "ram_gb", "label": "Memory for its server (GB)", "type": "number", "default": 2.0}, {"name": "notes", "label": "What it holds", "type": "long", "hint": "For people and for models choosing where to look."}],
        "holds": True, "list": _asset_list, "set": group_set, "remove": group_remove})   # holds: removing one may also delete what it stored (?data=1)
    print(f"""[KNOWLEDGE] instances: {sorted(INSTANCES) or 'none'}""", flush=True)
