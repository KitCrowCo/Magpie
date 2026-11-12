"""modules/remote/module.py - models another server serves, offered on this node as <source>/<model>. The node itself makes the call, so a contained program here (a knowledge server) reaches a stronger machine through its one approved route, this node's port, and every call shares this node's queue, records and kill switch.
A source is any server answering Ollama's /api/tags, /api/chat, /api/generate and /api/embed: another Magpie node (its compat module, with a key of that node) or an Ollama. Sources are settings (remote.sources), read on every model list refresh and every call, so a manager's change applies without a restart. Naming a source is this node's approval to contact it."""
import json, time, fnmatch, asyncio
import httpx
from core import settings, journal, jobs
from core.adapters import Adapter
from core.http import HTTPException, json_body

LISTS: dict = {}   # source -> {"t", "models": [...], "error"}

def _sources() -> dict: return {n: s for n, s in (settings.get("remote.sources", {}) or {}).items() if isinstance(s, dict) and s.get("url") and s.get("enabled", True)}
def _headers(src: dict) -> dict: return {"Authorization": f"""Bearer {src["key"]}"""} if src.get("key") else {}
def _split(model: str) -> tuple: return tuple(model.split("/", 1)) if "/" in model else ("", model)

def _caps(m: dict) -> list:
    """What a listed model can do: the source's own word when it gives one (Magpie peers, newer Ollama), else by name - embedding models by remote.embed_names, everything else chat."""
    if m.get("capabilities"): return [c if c != "completion" else "chat" for c in m["capabilities"]] + (["generate"] if "completion" in m["capabilities"] else [])
    return ["embed"] if any(fnmatch.fnmatch(m["name"].lower(), g) for g in settings.get("remote.embed_names", [])) else ["chat", "generate"]

# --- Model lists ---

async def _read(name: str, src: dict):
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0), trust_env=False) as c: r = await c.get(f"""{src["url"].rstrip("/")}/api/tags?capability=*""", headers=_headers(src))
        r.raise_for_status()
        LISTS[name] = {"t": time.time(), "url": src["url"], "error": "", "models": [{"name": f"""{name}/{m["name"]}""", "capabilities": _caps(m), "size": m.get("size", 0), "format": (m.get("details") or {}).get("format", ""), "source": name} for m in r.json().get("models", []) if m.get("adapter") != "remote"]}   # a peer's own remote models are left out: two nodes naming each other would otherwise list each other's lists without end
    except (httpx.HTTPError, ValueError) as e: LISTS[name] = {**LISTS.get(name, {"models": []}), "t": time.time(), "url": src["url"], "error": f"{type(e).__name__}: {e}"}

async def _refresher():
    """Reads a new or changed source within seconds, and every source again each refresh_s."""
    while True:
        srcs, every = _sources(), max(10, int(settings.get("remote.refresh_s", 60)))
        for n in [n for n in LISTS if n not in srcs or LISTS[n].get("url") != srcs[n]["url"]]: LISTS.pop(n)
        await asyncio.gather(*(_read(n, s) for n, s in srcs.items() if time.time() - LISTS.get(n, {}).get("t", 0) >= every))
        await asyncio.sleep(5)

# --- The adapter ---

class Remote(Adapter):
    capabilities = {"chat": {}, "generate": {}, "embed": {}}
    owner = "remote"
    def models(self) -> list: return [m for v in LISTS.values() for m in v.get("models", [])]
    def needs_service(self, job) -> bool: return False   # nothing runs here: the source does the work
    def status(self) -> dict: return {n: {"models": len(v.get("models", [])), "error": v.get("error", ""), "read_s_ago": round(time.time() - v["t"]) if v.get("t") else None} for n, v in LISTS.items()}

    async def run(self, job, service):
        name, model = _split(job.body.get("model") or job.model or "")
        src = _sources().get(name)
        if not src: raise RuntimeError(f"""no remote source {name!r} for model {job.body.get("model")!r} - sources: {sorted(_sources())}""")
        url, t0 = src["url"].rstrip("/"), time.time()
        journal.event("remote.call", source=name, model=model, capability=job.capability)
        timeout = httpx.Timeout(connect=10.0, read=float(src.get("read_timeout_s", 900)), write=60.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as c:
            if job.capability == "embed":
                r = await c.post(f"{url}/api/embed", headers=_headers(src), json={**job.body, "model": model})
                if r.status_code != 200: raise RuntimeError(f"""{name} answered {r.status_code}: {r.text[:300]}""")
                yield {"type": "result", "data": {"embeddings": r.json().get("embeddings")}}
                yield {"type": "done", "reason": "stop"}
                return
            path, field = ("/api/chat", None) if job.capability == "chat" else ("/api/generate", "response")
            async with c.stream("POST", url + path, headers=_headers(src), json={**job.body, "model": model, "stream": True}) as r:
                if r.status_code != 200: raise RuntimeError(f"""{name} answered {r.status_code}: {(await r.aread()).decode(errors="replace")[:300]}""")
                async for line in r.aiter_lines():
                    if not line.strip(): continue
                    d = json.loads(line)
                    if d.get("error"): raise RuntimeError(f"""{name}: {d["error"]}""")
                    msg = d.get("message") or {}
                    if think := msg.get("thinking") or d.get("thinking"): yield {"type": "thinking", "text": think}
                    if text := (d.get(field) if field else msg.get("content")): yield {"type": "text", "text": text}
                    if calls := msg.get("tool_calls"): yield {"type": "tool_calls", "calls": [{"name": tc["function"]["name"], "arguments": tc["function"].get("arguments") or {}} for tc in calls]}
                    if d.get("done"):
                        yield {"type": "usage", **({"gen_n": d["eval_count"], "gen_n_source": "program"} if d.get("eval_count") is not None else {}), **({"gen_ms": d["eval_duration"] / 1e6} if d.get("eval_duration") else {}), **({"prompt_n": d["prompt_eval_count"], "prompt_ms": (d.get("prompt_eval_duration") or 0) / 1e6} if d.get("prompt_eval_count") is not None else {})}
                        yield {"type": "done", "reason": d.get("done_reason") or "stop"}
                        return
        yield {"type": "done", "reason": f"{name} closed the stream after {round(time.time() - t0)} s without finishing"}

# --- Sources as things a manager makes (core/api.py /assets/source) ---

NAME = __import__("re").compile(r"^[A-Za-z0-9_-]{1,40}$")

async def _source_list(request) -> list: return [{"name": n, "url": s.get("url", ""), "status": "answers" if LISTS.get(n) and not LISTS[n].get("error") else (LISTS.get(n) or {}).get("error") or "not read yet", "models": len(LISTS.get(n, {}).get("models", [])), "enabled": s.get("enabled", True)} for n, s in (settings.get("remote.sources", {}) or {}).items() if isinstance(s, dict)]

async def _source_set(request, name: str):
    """{"url", "key" (empty keeps the stored one), "enabled"}: a manager rule, read again within seconds."""
    if not NAME.match(name): raise HTTPException(status_code=400, detail="a source name is letters, digits, - and _ (up to 40): it is how its models are named here (<source>/<model>)")
    b, old = await json_body(request), (settings.get(f"remote.sources.{name}", {}) or {})
    if not str(b.get("url") or old.get("url") or "").strip(): raise HTTPException(status_code=400, detail="a source needs its address (http://host:11434 for an Ollama)")
    url = str(b.get("url") or old["url"]).strip().rstrip("/")
    settings.set_path(f"remote.sources.{name}", {"url": url if "://" in url else f"http://{url}", "key": b.get("key") or old.get("key", ""), "enabled": b.get("enabled", True) not in (False, "false", "0")}, request.state.key.get("name", ""))
    return {"name": name, "set": True}

async def _source_remove(request, name: str):
    if not request.query_params.get("confirm"): raise HTTPException(status_code=400, detail="send ?confirm=1")
    settings.set_path(f"remote.sources.{name}", None, request.state.key.get("name", ""))
    return {"removed": name}

def setup(ctx):
    ctx.add_adapter(Remote("remote"))
    ctx.add_asset_kind("source", {"label": "Model source", "about": "Another server whose models this node uses as its own, named <source>/<model>: an Ollama on another machine, or another Magpie node's compat port. Work for them is done there; this node only passes it on, so its knowledge groups and other modules can use them.",
        "fields": [{"name": "url", "label": "Address", "hint": "host:port - an Ollama is usually host:11434"}, {"name": "key", "label": "API key", "type": "password", "hint": "only if that server asks for one; empty keeps the stored one"}], "list": _source_list, "set": _source_set, "remove": _source_remove})
    jobs.spawn(_refresher())
    print(f"""[REMOTE] sources: {sorted(_sources()) or 'none'}""", flush=True)
