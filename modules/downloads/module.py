"""modules/downloads/module.py - files brought onto this machine from an address, as things a manager makes (asset kind "download").
One is {"url", "location": one of this node's places (models, audio_models, ...), "folder": a folder inside it, "token": sent as a Bearer header for a repository that asks for one (used for this fetch, never stored)}; its name is the file's name there.
A repository folder is fetched whole: a Hugging Face address of a folder (https://huggingface.co/<owner>/<repository>/tree/<revision>[/<path>]) becomes the folder <name> holding every file under it, as the repository lists them (its API, the same host); the record keeps the list and each file is fetched as a single one would be.
The node fetches in the background, into <name>.part, continuing from where an earlier try stopped (a Range request) when the server allows it; only a finished file gets its name. A name that exists is refused: nothing is replaced. Removing one deletes that file, and only a file this module brought (it keeps a record beside its state).
Programs see the file like any other in that place - a model list reads the folder - so nothing here knows what the file is for."""
import re, json, time, asyncio
from pathlib import Path
from urllib.parse import urlsplit
import httpx
from core import settings, journal
from core.http import HTTPException, json_body

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,180}$")
FOLDER = re.compile(r"^[A-Za-z0-9._+-]+(/[A-Za-z0-9._+-]+)*$")
LIVE: dict = {}    # name -> {"task", "got", "size", "error", "t0"}: downloads under way in this process

def _file() -> Path: return settings.STATE / "downloads.json"
def _known() -> dict: return json.loads(_file().read_text()) if _file().exists() else {}
def _keep(d: dict): _file().write_text(json.dumps(d, indent=1))
def _at(rec: dict, name: str) -> Path: return settings.location(rec["location"]) / rec.get("folder", "") / name

def _allowed(url: str):
    host, allow = (urlsplit(url).hostname or "").lower(), [str(h).lower().lstrip(".") for h in settings.get("downloads.allow_hosts", []) or []]
    if urlsplit(url).scheme not in ("http", "https") or not host: raise HTTPException(status_code=400, detail=f"not a web address: {url[:200]}")
    if allow and not any(host == h or host.endswith("." + h) for h in allow): raise HTTPException(status_code=403, detail=f"{host} is not among the hosts this node downloads from (downloads.allow_hosts)")

_TREE = re.compile(r"^https://huggingface\.co/(?!api/)([^/]+/[^/]+)/tree/([^/]+)/?(.*)$")   # a repository folder: owner/repository, revision, path inside
def _client() -> httpx.AsyncClient: return httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0), trust_env=False, follow_redirects=True)
def _auth(token: str) -> dict: return {"Authorization": f"Bearer {token}"} if token else {}

async def _tree(url: str, token: str) -> list:
    """The files under a repository folder, as the repository lists them: [{"path" (inside the folder), "url", "size"}]."""
    repo, rev, sub = _TREE.match(url).groups()
    async with _client() as c: r = await c.get(f"""https://huggingface.co/api/models/{repo}/tree/{rev}/{sub}""".rstrip("/") + "?recursive=true", headers=_auth(token))
    if r.status_code != 200: raise RuntimeError(f"the repository answered {r.status_code} when asked for the folder's files" + (": it asks for a token, or the token is not accepted" if r.status_code in (401, 403) else ""))
    return [{"path": x["path"][len(sub):].lstrip("/") if sub else x["path"], "url": f"""https://huggingface.co/{repo}/resolve/{rev}/{x["path"]}""", "size": x.get("size") or 0} for x in r.json() if x.get("type") == "file"]

async def _get(c: httpx.AsyncClient, url: str, dest: Path, token: str, live: dict):
    """One file: appended to <dest>.part from where it stands, renamed when whole. live counts the bytes."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    async with c.stream("GET", url, headers={**_auth(token), **({"Range": f"bytes={have}-"} if have else {})}) as r:
        if r.status_code not in (200, 206): raise RuntimeError(f"the server answered {r.status_code} for {url}" + (": it asks for a token, or the token is not accepted" if r.status_code in (401, 403) else ""))
        if r.status_code == 200: have = 0   # the server sends the whole file: start over
        want, live["got"] = have + int(r.headers.get("content-length") or 0), live["got"] + have
        with open(part, "ab" if have else "wb") as f:
            async for chunk in r.aiter_bytes(int(settings.get("downloads.chunk_bytes", 1048576))):
                f.write(chunk)
                live["got"] += len(chunk)
                have += len(chunk)
    if want and have < want: raise RuntimeError(f"""the connection closed at {have} of {want} bytes of {dest.name}; asking for it again continues from there""")
    part.rename(dest)

async def _fetch(name: str, rec: dict, token: str):
    """The download itself: one file, or every file of a repository folder (those already whole are kept), each continued from where it stood."""
    dest, live = _at(rec, name), LIVE[name]
    try:
        async with _client() as c:
            if not rec.get("files"):
                live["size"] = 0
                await _get(c, rec["url"], dest, token, live)
            else:
                todo = [x for x in rec["files"] if not (dest / x["path"]).exists()]
                live.update(size=sum(x["size"] for x in rec["files"]), got=sum(x["size"] for x in rec["files"] if (dest / x["path"]).exists()))
                for x in todo: await _get(c, x["url"], dest / x["path"], token, live)
        _keep({**_known(), name: {**rec, "bytes": sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()) if dest.is_dir() else dest.stat().st_size, "done": time.time(), "error": ""}})
        journal.event("download.done", name=name, url=rec["url"][:500], bytes=dest.stat().st_size, seconds=round(time.time() - live["t0"], 1))
    except asyncio.CancelledError:
        _keep({**_known(), name: {**rec, "error": "stopped; asking for it again continues from where it is"}})
        raise
    except Exception as e:
        _keep({**_known(), name: {**rec, "error": f"{type(e).__name__}: {e}"}})
        journal.event("download.failed", name=name, url=rec["url"][:500], error=f"{type(e).__name__}: {e}")
    finally: LIVE.pop(name, None)

# --- The asset kind (core/api.py /assets/download) ---

async def _list(request) -> list:
    def status(n, rec):
        if n in LIVE: return f"""downloading: {LIVE[n]["got"] / 1e6:.0f} of {LIVE[n]["size"] / 1e6:.0f} MB{f", {len(rec['files'])} files" if rec.get("files") else ""}""" if LIVE[n].get("size") else "starting"
        return rec["error"] if rec.get("error") else "here" if _at(rec, n).exists() else "the file is gone from its place"
    return [{"name": n, "url": rec["url"], "location": rec["location"], "folder": rec.get("folder", ""), "status": status(n, rec), "bytes": rec.get("bytes", 0), "path": str(_at(rec, n)), "live": n in LIVE, "got": LIVE.get(n, {}).get("got", 0), "size": LIVE.get(n, {}).get("size", 0)} for n, rec in _known().items()]   # live, got, size: a download under way, for a progress bar

async def _set(request, name: str):
    """Starts a download, or takes one up again that stopped. {"url", "location", "folder", "token"}. A repository folder's address fetches every file in it into the folder <name>."""
    b, old = await json_body(request), _known().get(name, {})
    rec = {"url": str(b.get("url") or old.get("url") or "").strip(), "location": str(b.get("location") or old.get("location") or "models"), "folder": str(b.get("folder") if b.get("folder") is not None else old.get("folder", "")).strip("/")}
    if not NAME.match(name): raise HTTPException(status_code=400, detail="its name is the file's name here: letters, digits, . _ + -")
    if rec["folder"] and not FOLDER.match(rec["folder"]): raise HTTPException(status_code=400, detail="the folder is a path inside the place, of plain names")
    if rec["location"] not in settings.locations(): raise HTTPException(status_code=400, detail=f"""this node has no place named {rec["location"]}; its places: {", ".join(settings.locations())}""")
    _allowed(rec["url"])
    if name in LIVE: raise HTTPException(status_code=409, detail=f"{name} is being downloaded")
    if _TREE.match(rec["url"]):
        try: rec["files"] = old.get("files") if old.get("url") == rec["url"] and old.get("files") else await _tree(rec["url"], str(b.get("token") or ""))
        except (RuntimeError, httpx.HTTPError) as e: raise HTTPException(status_code=502, detail=f"the folder's files could not be listed: {type(e).__name__}: {e}")
        if not rec["files"]: raise HTTPException(status_code=404, detail="the repository lists no files in that folder")
    if _at(rec, name).exists() and not (rec.get("files") and old.get("files") and not old.get("done")): raise HTTPException(status_code=409, detail=f"{_at(rec, name)} exists already: nothing is replaced. Remove it first, or give this one another name.")   # a folder download that stopped is taken up again in its own folder
    _keep({**_known(), name: {**rec, "error": ""}})
    LIVE[name] = {"got": 0, "size": 0, "t0": time.time()}
    LIVE[name]["task"] = asyncio.get_running_loop().create_task(_fetch(name, rec, str(b.get("token") or "")))
    journal.event("download.start", name=name, url=rec["url"][:500], to=str(_at(rec, name)), by=request.state.key.get("name", ""))
    return {"name": name, "set": True, "to": str(_at(rec, name))}

async def _remove(request, name: str):
    """?confirm=1: the download is stopped if under way, and the file it brought (and its unfinished part) deleted. Only what this module brought. ?forget=1 instead: only the record goes (the list line); the file stays where it is, as any other model file."""
    if not request.query_params.get("confirm"): raise HTTPException(status_code=400, detail="send ?confirm=1")
    known = _known()
    if name not in known: raise HTTPException(status_code=404, detail=f"no download named {name}")
    if request.query_params.get("forget"):
        if name in LIVE: raise HTTPException(status_code=409, detail=f"{name} is still being downloaded: stop it (which deletes its part) or let it finish, then forget it")
        _keep({k: v for k, v in _known().items() if k != name})
        journal.event("download.forgotten", name=name, path=str(_at(known[name], name)), by=request.state.key.get("name", ""))
        return {"forgotten": name, "file_kept": str(_at(known[name], name))}
    if name in LIVE: LIVE[name]["task"].cancel()
    dest = _at(known[name], name)
    if known[name].get("files"):   # a repository folder: the files it listed (and their unfinished parts), then the folders left empty; anything else put there stays
        for x in known[name]["files"]:
            for f in (dest / x["path"], dest / (x["path"] + ".part")): f.unlink(missing_ok=True)
        for d in sorted((p for p in dest.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)) + [dest]:
            if d.exists() and not any(d.iterdir()): d.rmdir()
    else:
        for f in (dest, dest.with_name(dest.name + ".part")): f.unlink(missing_ok=True)
    _keep({k: v for k, v in _known().items() if k != name})
    journal.event("download.removed", name=name, path=str(dest), by=request.state.key.get("name", ""))
    return {"removed": name}

def _fields() -> list:
    """The form of a download, asked for each time it is drawn: the place is picked from this node's places as they are now, each shown with its folder, so the choice says where the file lands (typing a name still works)."""
    return [{"name": "url", "label": "Address of the file, or of a repository folder", "required": True, "hint": "A file, e.g. https://huggingface.co/<owner>/<repository>/resolve/main/<file>.gguf - or a whole folder as the repository shows it, e.g. https://huggingface.co/Systran/faster-whisper-tiny/tree/main: every file in it is fetched into a folder named as this download"}, {"name": "location", "label": "Which place", "default": "text_models", "choices": [[n, f"""{n} - {v["path"]}"""] for n, v in settings.locations().items()], "hint": "One of this machine's places (Where it keeps things): text_models for language models (GGUF), image_models, audio_models, or models (the folder holding them all). text_models is the models folder itself on a machine that does not split models by kind."}, {"name": "folder", "label": "Folder inside it", "default": "gguf_local", "hint": "Where the program that will use it looks: gguf_local for GGUF models."}, {"name": "token", "label": "Access token", "type": "password", "hint": "Only for a repository that asks for one. Used for this download and not kept."}]

def setup(ctx):
    ctx.add_asset_kind("download", {"label": "Downloaded file", "about": "A file this machine fetches from an address into one of its places: a model file from a repository, for example. Its name is the file's name there. It continues a download that stopped, replaces nothing, and contacts only the address given.",
        "fields": _fields(), "fields_now": _fields,
        "plain_names": True, "list": _list, "set": _set, "remove": _remove})   # plain_names: the name is the file's own name, taken as written (a manager's form must not reshape it)
