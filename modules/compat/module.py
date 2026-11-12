"""modules/compat/module.py - the common wire formats on the one port, so any program that speaks Ollama's or OpenAI's API (LightRAG, bench tools) can use this node, and its every call still goes through the queue, the accounting and the kill switch. /api/... Ollama. /v1/... OpenAI."""
import os, re, time, json, uuid
from pathlib import Path
import httpx
from core import jobs, services, modules, api, journal, files as store
from core.http import Router, HTTPException, HTMLResponse, json_body, ndjson, sse

router = Router()

def ollama_line(job, ev: dict, key: str = "message") -> dict | None:
    """One event as a line of the portal's chat stream (Ollama's shape)."""
    now, t = journal.now(), ev.get("type")
    msg = lambda **m: {"model": job.model, "created_at": now, "job": job.id, key: ({"role": "assistant", "content": "", **m} if key == "message" else m.get("content", "")), **({"thinking": m["thinking"]} if key != "message" and "thinking" in m else {}), "done": False}
    if t == "text": return msg(content=ev["text"])
    if t == "thinking": return msg(thinking=ev["text"])
    if t == "tool_calls": return msg(tool_calls=ev["calls"])
    if t == "heartbeat": return msg()                     # an empty piece: valid to every reader, and keeps a long wait's connection alive
    if t == "usage":
        job.record["usage"] = {k: v for k, v in ev.items() if k != "type"}
        return None
    if t == "done":
        u, g, gms, pms = job.record.get("usage", {}), None, None, None
        g, gms, pn, pms = u.get("gen_n"), u.get("gen_ms"), u.get("prompt_n"), u.get("prompt_ms")
        return {**msg(), "done": True, "done_reason": ev.get("reason", "stop"), "total_duration": int((time.time() - job.t0) * 1e9), **({"error": ev["error"]} if ev.get("error") else {}), **({"eval_count": g} if g is not None else {}), **({"eval_duration": int(gms * 1e6)} if gms else {}), **({"prompt_eval_count": pn, "prompt_eval_duration": int((pms or 0) * 1e6), "cache_n": u.get("cache_n")} if pn is not None else {})}
    if t == "error": return {**msg(), "done": True, "done_reason": "error", "error": ev["message"]}
    if t == "progress": return {"model": job.model, "job": job.id, "done": False, **{k: v for k, v in ev.items() if k != "type"}}
    return None

async def ollama_stream(job, key: str = "message"):
    async for ev in api.events(job):
        if (line := ollama_line(job, ev, key)) is not None: yield json.dumps(line) + "\n"

def _job(request, capability: str, body: dict, adapter: str = ""): return jobs.submit(api.make_job(request, capability, body, adapter))

# --- Ollama ---

@router.post("/api/chat", scope="infer")
async def ol_chat(request):
    body = await json_body(request)
    if not body.get("messages") and body.get("model") and "keep_alive" in body: return {"model": body["model"], "created_at": time.time(), "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "load"}   # Ollama's bare preload call: nothing is loaded for it, the first real request does that
    job = _job(request, "chat", body)
    if body.get("stream", True): return ndjson(ollama_stream(job))
    out = await api.collect(job)
    api.raise_error(out["error"])
    return {"model": job.model, "created_at": time.time(), "message": {"role": "assistant", "content": out["text"], **({"thinking": out["thinking"]} if out["thinking"] else {}), **({"tool_calls": out["tool_calls"]} if out["tool_calls"] else {})}, "done": True, "done_reason": out["reason"], **_final(out["usage"], job)}

@router.post("/api/generate", scope="infer")
async def ol_generate(request):
    body = await json_body(request)
    job = _job(request, "generate", body)
    if body.get("stream", True): return ndjson(ollama_stream(job, "response"))
    out = await api.collect(job)
    api.raise_error(out["error"])
    return {"model": job.model, "created_at": time.time(), "response": out["text"], **({"thinking": out["thinking"]} if out["thinking"] else {}), "done": True, "done_reason": out["reason"], **_final(out["usage"], job)}

def _final(u: dict, job) -> dict: return {"total_duration": int((time.time() - job.t0) * 1e9), **({"eval_count": u["gen_n"]} if u.get("gen_n") is not None else {}), **({"eval_duration": int(u["gen_ms"] * 1e6)} if u.get("gen_ms") else {}), **({"prompt_eval_count": u["prompt_n"], "prompt_eval_duration": int((u.get("prompt_ms") or 0) * 1e6), "cache_n": u.get("cache_n")} if u.get("prompt_n") is not None else {})}

def _tag(m: dict) -> dict:
    a = modules.ADAPTERS.get(m["adapter"])
    if m["adapter"] == "llama":
        from modules.llama import module as llama
        hit = next((x for x in llama.scan_models() if x["name"] == m["name"]), None)
        if hit: return {**llama.tags_entry(hit), "capabilities": m["capabilities"], "adapter": "llama"}
    return {"name": m["name"], "model": m["name"], "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(m.get("mtime", 0))), "size": m.get("size", 0), "source": m["adapter"], "adapter": m["adapter"], "capabilities": m["capabilities"], "loaded": bool(a and (s := services.get(f"""{m["adapter"]}:{m["name"]}""")) and s.ready()), "details": {"format": m.get("format", ""), "ctx": m.get("ctx")}}

@router.get("/api/tags")
async def ol_tags(request):
    """Text models only by default (what a chat picker wants); ?capability= for others, ?capability=* for all."""
    c = request.query_params.get("capability", "chat")
    return {"models": [_tag(m) for m in modules.models("" if c == "*" else c) + ([m for m in modules.models("embed")] if c == "chat" else [])]}

@router.get("/api/ps")
async def ol_ps(request): return {"models": [{"name": s.meta.get("model", s.name), "model": s.meta.get("model", s.name), "service": s.name, "size": 0, "size_vram": 0, "context_length": s.meta.get("ctx", 0), "kv_cache_type": s.meta.get("kv"), "rss_gb": s.rss_gb(), "plan": (s.info().get("plan"))} for s in services.running()]}

@router.post("/api/show")
async def ol_show(request):
    body = await json_body(request)
    name = body.get("model") or body.get("name") or ""
    a = modules.resolve("chat", name) or modules.resolve("embed", name)
    if not a: raise HTTPException(status_code=404, detail=f"Model not found: {name}")
    if a.name != "llama": return {"modelfile": f"FROM {name}", "details": next((m for m in modules.models() if m["name"] == name), {}), "model_info": {}}
    from modules.llama import module as llama
    m = llama.require_model(name)
    info = await llama.model_meta(m)
    return {"modelfile": f"""FROM {m["name"]}""", "details": {**llama.tags_entry(m)["details"], "bpw": info.get("bpw"), "params_b": info.get("params_b"), "file_type": info.get("file_type")}, "model_info": info}

@router.delete("/api/delete")
async def ol_delete(request):
    """A model on this node's list, by name. Refused without confirm, and while it is loaded."""
    body = await json_body(request)
    name = body.get("name") or body.get("model") or ""
    if not body.get("confirm"): raise HTTPException(status_code=400, detail="destructive - send confirm=true")
    from modules.llama import module as llama
    m = llama.require_model(name)
    a = llama.A["adapter"]
    if a.main.model and a.main.model["path"] == m["path"] and a.main.ready(): raise HTTPException(status_code=409, detail="model is currently loaded - load a different one first")
    shared = m["source"] == "ollama_store" and any(o["path"] == m["path"] and o["manifest"] != m["manifest"] for o in llama.scan_models())   # another tag points at the same blob
    freed = 0
    if not shared:
        for f in m["files"]: freed += llama._fsize(f); Path(f).unlink(missing_ok=True)
    if m["manifest"]: Path(m["manifest"]).unlink(missing_ok=True)
    return {"status": "success", "freed_bytes": freed, "shared_blob_kept": shared}

def _quant_match(fn: str, quant: str) -> bool: return any(t == quant or t.endswith("_" + quant) for t in re.split(r"[-.]", os.path.basename(fn).upper()))

@router.post("/api/pull")
async def ol_pull(request):
    """{"repo", "quant" or "filenames"}. A quant needs the repo's file list, which is one more request to the same host."""
    b = await json_body(request)
    repo, quant, names = b.get("repo") or b.get("name"), b.get("quant", ""), b.get("filenames") or ([b["filename"]] if b.get("filename") else [])
    if not repo or not (names or quant): raise HTTPException(status_code=400, detail="need repo plus quant or filename(s)")
    if not names:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as c: files = [s["rfilename"] for s in (await c.get(f"https://huggingface.co/api/models/{repo}")).json().get("siblings", [])]
        names = sorted(f for f in files if f.lower().endswith(".gguf") and "mmproj" not in f.lower() and _quant_match(f, quant.upper()))
        if not names: raise HTTPException(status_code=404, detail=f"no GGUF files match quant {quant} in {repo}")
    async def lines():
        async for p in store.pull({"location": "models", "source": "hf", "repo": repo, "files": names, "folder": "gguf_local/" + repo.replace("/", "__")}): yield json.dumps(p) + "\n"
    return ndjson(lines())

@router.post("/api/embed", scope="infer")
async def ol_embed(request):
    body = await json_body(request)
    out = await api.collect(_job(request, "embed", body))
    api.raise_error(out["error"])
    return out["result"]

@router.post("/api/embeddings", scope="infer")
async def ol_embeddings(request):
    body = await json_body(request)
    out = await api.collect(_job(request, "embed", {**body, "input": body.get("prompt", "")}))
    api.raise_error(out["error"])
    return {"embedding": out["result"]["embeddings"][0]}

# --- OpenAI ---

def _oai_chunk(job, cid: str, delta: dict, finish=None) -> str: return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": int(job.t0), "model": job.model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n"

def _oai_calls(calls: list) -> list: return [{"id": c.get("id") or f"call_{i}", "index": i, "type": "function", "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"] if isinstance(c["function"]["arguments"], str) else json.dumps(c["function"]["arguments"])}} for i, c in enumerate(calls)]

def _from_openai(body: dict) -> dict:
    """An OpenAI request in this node's shape: max_tokens and sampling fields become options; reasoning_effort becomes think."""
    o = {**(body.get("options") or {}), **{k: body[k] for k in ("temperature", "top_p", "seed", "stop", "presence_penalty", "frequency_penalty") if k in body}}
    if body.get("max_tokens") is not None or body.get("max_completion_tokens") is not None: o["num_predict"] = body.get("max_tokens") if body.get("max_tokens") is not None else body["max_completion_tokens"]
    return {**body, "options": o, **({"think": body["reasoning_effort"]} if body.get("reasoning_effort") else {})}

async def _oai_stream(job, want_usage: bool = True):
    cid, u = f"chatcmpl-{uuid.uuid4().hex[:12]}", {}
    async for ev in api.events(job):
        t = ev.get("type")
        if t == "text": yield _oai_chunk(job, cid, {"content": ev["text"]})
        elif t == "thinking": yield _oai_chunk(job, cid, {"reasoning_content": ev["text"]})
        elif t == "tool_calls": yield _oai_chunk(job, cid, {"tool_calls": _oai_calls(ev["calls"])})
        elif t == "heartbeat": yield ": still working\n\n"
        elif t == "usage": u = ev
        elif t == "error": yield "data: " + json.dumps({"error": {"message": ev["message"]}}) + "\n\n"
        elif t == "done":
            yield _oai_chunk(job, cid, {}, ev.get("reason", "stop"))
            if want_usage: yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "choices": [], "usage": {"prompt_tokens": u.get("prompt_n"), "completion_tokens": u.get("gen_n"), "total_tokens": (u.get("prompt_n") or 0) + (u.get("gen_n") or 0)}, "timings": {"prompt_n": u.get("prompt_n"), "prompt_ms": u.get("prompt_ms"), "cache_n": u.get("cache_n"), "predicted_n": u.get("gen_n"), "predicted_ms": u.get("gen_ms")}}) + "\n\n"
    yield "data: [DONE]\n\n"

async def _oai_answer(request, body: dict, adapter: str = ""):
    job = _job(request, "chat", _from_openai(body), adapter)
    if body.get("stream"): return sse(_oai_stream(job, bool((body.get("stream_options") or {}).get("include_usage"))))
    out = await api.collect(job)
    api.raise_error(out["error"])
    u = out["usage"]
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(job.t0), "model": job.model, "choices": [{"index": 0, "message": {"role": "assistant", "content": out["text"], **({"reasoning_content": out["thinking"]} if out["thinking"] else {}), **({"tool_calls": _oai_calls(out["tool_calls"])} if out["tool_calls"] else {})}, "finish_reason": out["reason"]}], "usage": {"prompt_tokens": u.get("prompt_n"), "completion_tokens": u.get("gen_n"), "total_tokens": (u.get("prompt_n") or 0) + (u.get("gen_n") or 0)}, "timings": {"prompt_n": u.get("prompt_n"), "prompt_ms": u.get("prompt_ms"), "cache_n": u.get("cache_n"), "predicted_n": u.get("gen_n"), "predicted_ms": u.get("gen_ms")}}

@router.post("/v1/chat/completions", scope="infer")
async def oai_chat(request): return await _oai_answer(request, await json_body(request))

@router.get("/v1/models")
async def oai_models(request): return {"object": "list", "data": [{"id": m["name"], "object": "model", "created": int(m.get("mtime") or 0), "owned_by": m["adapter"]} for m in modules.models("chat")]}

@router.post("/v1/embeddings", scope="infer")
async def oai_embeddings(request):
    out = await api.collect(_job(request, "embed", await json_body(request)))
    api.raise_error(out["error"])
    r = out["result"]
    return {"object": "list", "model": r["model"], "data": [{"object": "embedding", "index": i, "embedding": e} for i, e in enumerate(r["embeddings"])], "usage": {"prompt_tokens": r.get("prompt_eval_count"), "total_tokens": r.get("prompt_eval_count")}}

# --- Diagnostic page ---

@router.get("/chat", scope="open")
async def chat_page(request): return HTMLResponse((Path(__file__).parent / "chat.html").read_text())

def setup(ctx): ctx.add_router(router)

api.SHAPES.update(ollama=ollama_stream, openai=_oai_stream)
