"""core/files.py - the node's files as something a manager administers from afar. Magpie is a remote administration handler, and what happens to a model, an image, a video or a log is the same few operations: find it, move it, rename it, copy it, send it, fetch it, delete it.
A location is a named place (models, outputs, logs ... or any name a manager makes); none is fixed at install. Programs are told their locations when they start, so moving one takes effect at their next start. A long location never has a file replaced or removed unless told to (overwrite, confirm); a scratch one is shared and clears itself after scratch_ttl_h. A name already taken gets the next free one (-2, -3) when free is asked for. Paths cannot leave their location.
A send or fetch contacts only the host it names, by plain HTTP with no client library."""
import os, json, time, shutil, asyncio
from pathlib import Path
import httpx
from core import settings, journal, hardware, jobs
from core.http import Router, HTTPException, StreamingResponse, json_body, ndjson

router = Router()
SOURCES = {"hf": "https://huggingface.co/{repo}/resolve/{ref}/{file}"}

# --- Paths ---

def _base(name: str) -> Path:
    if name not in settings.locations(): raise HTTPException(status_code=404, detail=f"no location {name!r} - known: {sorted(settings.locations())}")
    return settings.location(name)

def resolve(name: str, rel: str = "") -> Path:
    base = _base(name).resolve()
    p = (base / rel).resolve()
    if not p.is_relative_to(base): raise HTTPException(status_code=400, detail=f"{rel!r} leaves location {name}")
    return p

def _ref(d: dict) -> Path: return resolve(str(d.get("location") or ""), str(d.get("path") or ""))

def free_name(p: Path) -> Path:
    if not p.exists(): return p
    return next(c for c in (p.with_name(f"{p.stem}-{i}{p.suffix}") for i in range(2, 100000)) if not c.exists())

def _target(p: Path, b: dict) -> Path:
    """Where a file may land: refused when the name is taken, unless overwrite or free is asked for."""
    if not p.exists() or b.get("overwrite"): return p
    if b.get("free"): return free_name(p)
    raise HTTPException(status_code=409, detail=f"{p.name} exists; send overwrite, or free for the next free name")

def _entry(f: Path, base: Path) -> dict:
    st = f.stat()
    return {"path": str(f.relative_to(base)), "dir": f.is_dir(), "size": st.st_size if f.is_file() else None, "mtime": st.st_mtime}

# --- Locations ---

@router.get("/locations")
async def locations(request):
    return {"locations": [{"name": n, **v, "exists": Path(v["path"]).exists(), "disk": hardware.disk(v["path"])} for n, v in settings.locations().items()]}

def _move_tree(src: Path, dst: Path) -> dict:
    """Everything in src into dst; refuses before touching anything if a name is already taken there."""
    dst.mkdir(parents=True, exist_ok=True)
    items = list(src.iterdir()) if src.exists() else []
    if taken := [i.name for i in items if (dst / i.name).exists()]: raise HTTPException(status_code=409, detail=f"already at the destination: {taken[:10]}")
    for i in items: shutil.move(str(i), dst / i.name)
    return {"moved": len(items)}

@router.get("/browse", scope="admin")
async def browse(request):
    """?path=<folder on this machine> (default: the user's home): its folders, for a manager choosing where a location goes. Only names and free space; nothing is read or changed."""
    at = Path(os.path.expanduser(request.query_params.get("path") or "~")).resolve()
    if not at.is_dir(): raise HTTPException(status_code=404, detail=f"no folder {at}")
    try: dirs = sorted(d.name for d in at.iterdir() if d.is_dir() and not d.name.startswith("."))
    except PermissionError: dirs = []
    return {"path": str(at), "parent": str(at.parent) if at != at.parent else "", "folders": dirs, "disk": hardware.disk(str(at))}

@router.put("/locations/{name}")
async def location_set(request, name: str):
    """{"path", "kind", "move"}: makes a location or changes one. With move, what is there goes along, once no service is working. Programs see a new path at their next start; the journal follows at once."""
    b, cur = await json_body(request), settings.locations().get(name, {})
    new = str(b.get("path") or cur.get("path") or settings.DATA / "locations" / name)
    if jobs.busy() and b.get("move"): raise HTTPException(status_code=409, detail="work is running: wait, or cancel it")
    moved = {}
    if b.get("move") and cur.get("path") and new != cur["path"]:
        logs = name == "logs"
        if logs: journal.reopen()
        moved = await asyncio.to_thread(_move_tree, Path(cur["path"]), Path(new))
        if logs: settings.set_path(f"locations.{name}", {"path": new, "kind": b.get("kind") or cur.get("kind", "long")}, "location set"); journal.start()
    settings.set_path(f"locations.{name}", {"path": new, "kind": b.get("kind") or cur.get("kind", "long")}, "location set")
    Path(new).mkdir(parents=True, exist_ok=True)
    journal.event("location.set", name=name, path=new, moved=moved.get("moved", 0))
    jobs.kick()
    return {"name": name, **settings.locations()[name], **moved}

# --- Files ---

@router.get("/files/{location}")
async def file_list(request, location: str):
    base, q = _base(location), request.query_params
    start = resolve(location, q.get("under", ""))
    walk = start.rglob("*") if q.get("recursive") else start.iterdir() if start.exists() else []
    return {"files": [_entry(f, base) for f in sorted(walk)][:int(q.get("limit", 20000))]}

@router.get("/files/{location}/{path:path}")
async def file_get(request, location: str, path: str):
    p = resolve(location, path)
    if not p.is_file(): raise HTTPException(status_code=404, detail=f"no file {path} in {location}")
    def chunks():
        with open(p, "rb") as f:
            while b := f.read(1 << 20): yield b
    return StreamingResponse(chunks(), media_type="application/octet-stream", headers={"Content-Length": str(p.stat().st_size), "X-File-Name": p.name})

@router.put("/files/{location}/{path:path}", scope="infer")
async def file_put(request, location: str, path: str):
    """The request body is the file. Written beside the target and renamed, so a cut connection leaves no half file. ?overwrite=1 or ?free=1 when the name may be taken."""
    p = _target(resolve(location, path), request.query_params)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp, n = p.with_name(p.name + ".part"), 0
    with open(tmp, "wb") as f:
        async for chunk in request.stream():
            n += len(chunk)
            f.write(chunk)
    tmp.replace(p)
    return {"location": location, "path": str(p.relative_to(_base(location).resolve())), "size": n}

@router.post("/files/move")
async def file_move(request):
    """{"from": {"location", "path"}, "to": {"location", "path"}, "overwrite", "free"}: rename within a location, or move between them."""
    b = await json_body(request)
    src, dst = _ref(b["from"]), _ref(b["to"])
    if not src.exists(): raise HTTPException(status_code=404, detail=f"""no such path: {b["from"]}""")
    dst = _target(dst, b)
    dst.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(shutil.move, str(src), dst)
    journal.event("file.move", src=str(src), dst=str(dst))
    return {"moved": str(dst.name)}

@router.post("/files/copy")
async def file_copy(request):
    b = await json_body(request)
    src, dst = _ref(b["from"]), _ref(b["to"])
    if not src.exists(): raise HTTPException(status_code=404, detail=f"""no such path: {b["from"]}""")
    dst = _target(dst, b)
    dst.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(shutil.copytree if src.is_dir() else shutil.copy2, src, dst)
    return {"copied": str(dst.name)}

@router.post("/files/delete")
async def file_delete(request):
    """{"location", "path", "confirm"}. A whole location cannot be deleted this way."""
    b = await json_body(request)
    if not b.get("confirm"): raise HTTPException(status_code=400, detail="destructive - send confirm=true")
    p = _ref(b)
    if p == _base(b["location"]).resolve(): raise HTTPException(status_code=400, detail="refusing to delete a whole location")
    if not p.exists(): raise HTTPException(status_code=404, detail=f"""no such path: {b.get('path')}""")
    size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
    await asyncio.to_thread(shutil.rmtree, p) if p.is_dir() else p.unlink()
    journal.event("file.delete", path=str(p), freed=size)
    return {"status": "success", "freed_bytes": size}

async def _send_one(c: httpx.AsyncClient, f: Path, url: str, key: str, q: dict):
    def body():
        with open(f, "rb") as fh:
            while b := fh.read(1 << 20): yield b
    async def abody():
        for b in body(): yield b
    r = await c.put(url, content=abody(), params=q, headers={"Authorization": f"Bearer {key}"} if key else {})
    if r.status_code != 200: raise HTTPException(status_code=502, detail=f"{url}: HTTP {r.status_code} {r.text[:200]}")

@router.post("/files/send")
async def file_send(request):
    """{"from": {"location", "path"}, "to": {"url": another node, "key", "location", "path"}, "overwrite", "free", "delete_after"}: a file or a folder to another node's location, streamed. The source goes only after everything arrived, and only if delete_after is sent."""
    b = await json_body(request)
    src, to = _ref(b["from"]), b["to"]
    if not src.exists(): raise HTTPException(status_code=404, detail=f"""no such path: {b["from"]}""")
    files, sent, q = [src] if src.is_file() else sorted(f for f in src.rglob("*") if f.is_file()), 0, {k: "1" for k in ("overwrite", "free") if b.get(k)}
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=20.0, read=600.0, write=600.0, pool=20.0)) as c:
        for f in files:
            rel = f"""{to.get("path") or src.name}""" if src.is_file() else f"""{to.get("path") or src.name}/{f.relative_to(src)}"""
            await _send_one(c, f, f"""{to["url"].rstrip("/")}/files/{to["location"]}/{rel}""", to.get("key", ""), q)
            sent += f.stat().st_size
    if b.get("delete_after"): await asyncio.to_thread(shutil.rmtree, src) if src.is_dir() else src.unlink()
    journal.event("file.send", src=str(src), to=to["url"], bytes=sent, files=len(files))
    return {"sent_files": len(files), "bytes": sent}

# --- Fetching (models, assets, anything by URL) ---

async def pull_file(url: str, dest: Path, headers: dict = None):
    """Yields progress dicts while downloading url to dest, resuming a partial download; a finished dest is not fetched again."""
    if dest.exists():
        yield {"file": dest.name, "completed": dest.stat().st_size, "total": dest.stat().st_size, "already": True}
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".incomplete")
    have = part.stat().st_size if part.exists() else 0
    async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(connect=20.0, read=120.0, write=20.0, pool=20.0)) as c:
        async with c.stream("GET", url, headers={**(headers or {}), **({"Range": f"bytes={have}-"} if have else {})}) as r:
            if r.status_code >= 400 and r.status_code != 416: raise RuntimeError(f"HTTP {r.status_code} for {url}")
            if r.status_code != 416:
                if have and r.status_code != 206: have = 0   # the server would not resume: start over
                total, last = have + int(r.headers.get("content-length") or 0), 0.0
                with open(part, "ab" if have else "wb") as f:
                    async for chunk in r.aiter_bytes(1 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        if time.time() - last > 2:
                            last = time.time()
                            yield {"file": dest.name, "completed": have, "total": total}
    part.rename(dest)
    yield {"file": dest.name, "completed": have, "total": have}

async def pull(body: dict):
    """{"location": "models", "folder", "files": [names], "source": "hf", "repo", "ref"} or {"urls": [{"url", "to"}]}. One progress line per step, ending with status success or error."""
    loc = body.get("location", "models")
    try:
        src = {**SOURCES, **(settings.get("sources", {}) or {})}.get(body.get("source", "hf"))
        items = [(u["url"], resolve(loc, u["to"])) for u in body.get("urls", [])] + [(src.format(repo=body.get("repo", ""), ref=body.get("ref", "main"), file=f), resolve(loc, str(Path(body.get("folder") or body.get("repo", "").replace("/", "__")) / f))) for f in body.get("files", [])]
        if not items: raise ValueError("nothing to pull (files with a repo, or urls)")
        token = settings.get("sources_token", "")
        yield {"status": f"""pulling {len(items)} file(s), {hardware.disk(items[0][1])["free_gb"]} GB free""", "done": False}
        for url, dest in items:
            async for p in pull_file(url, dest, {"Authorization": f"Bearer {token}"} if token and "huggingface.co" in url else None): yield {"status": f"""downloading {p["file"]}""", "done": False, **p}
            journal.event("file.fetch", file=str(dest), url=url)
        yield {"status": "success", "done": True}
    except (Exception, HTTPException) as e:
        journal.event("file.fetch", ok=False, error=str(getattr(e, "detail", e))[:300])
        yield {"status": f"""error: {getattr(e, "detail", e)}""", "done": True}

@router.post("/files/fetch")
async def file_fetch(request):
    body = await json_body(request)
    async def lines():
        async for p in pull(body): yield json.dumps(p) + "\n"
    return ndjson(lines())

# --- Scratch: only scratch locations, only past their age limit, journaled ---

async def sweep():
    while True:
        await asyncio.sleep(3600)
        ttl = float(settings.get("scratch_ttl_h", 72)) * 3600
        for n, v in settings.locations().items():
            if v.get("kind") != "scratch": continue
            for f in Path(v["path"]).rglob("*"):
                if f.is_file() and time.time() - f.stat().st_mtime > ttl:
                    f.unlink(missing_ok=True)
                    journal.event("file.expired", location=n, path=str(f))
