"""modules/downloads/module.py - files brought onto this machine from an address, as things a manager makes (asset kind "download").
One is {"url", "location": one of this node's places (models, audio_models, ...), "folder": a folder inside it, "token": sent as a Bearer header for a repository that asks for one (used for this fetch, never stored)}; its name is the file's name there.
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

async def _fetch(name: str, rec: dict, token: str):
    """The download itself: appended to <name>.part from where it stands, renamed when whole."""
    dest, live = _at(rec, name), LIVE[name]
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        have = part.stat().st_size if part.exists() else 0
        head = {**({"Authorization": f"Bearer {token}"} if token else {}), **({"Range": f"bytes={have}-"} if have else {})}
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0), trust_env=False, follow_redirects=True) as c:
            async with c.stream("GET", rec["url"], headers=head) as r:
                if r.status_code not in (200, 206): raise RuntimeError(f"the server answered {r.status_code}" + (": it asks for a token, or the token is not accepted" if r.status_code in (401, 403) else ""))
                if r.status_code == 200: have = 0   # the server sends the whole file: start over
                live.update(got=have, size=have + int(r.headers.get("content-length") or 0))
                with open(part, "ab" if have else "wb") as f:
                    async for chunk in r.aiter_bytes(int(settings.get("downloads.chunk_bytes", 1048576))):
                        f.write(chunk)
                        live["got"] += len(chunk)
        if live["size"] and live["got"] < live["size"]: raise RuntimeError(f"""the connection closed at {live["got"]} of {live["size"]} bytes; asking for it again continues from there""")
        part.rename(dest)
        _keep({**_known(), name: {**rec, "bytes": dest.stat().st_size, "done": time.time(), "error": ""}})
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
        if n in LIVE: return f"""downloading: {LIVE[n]["got"] / 1e6:.0f} of {LIVE[n]["size"] / 1e6:.0f} MB""" if LIVE[n].get("size") else "starting"
        return rec["error"] if rec.get("error") else "here" if _at(rec, n).exists() else "the file is gone from its place"
    return [{"name": n, "url": rec["url"], "location": rec["location"], "folder": rec.get("folder", ""), "status": status(n, rec), "bytes": rec.get("bytes", 0), "path": str(_at(rec, n))} for n, rec in _known().items()]

async def _set(request, name: str):
    """Starts a download, or takes one up again that stopped. {"url", "location", "folder", "token"}."""
    b, old = await json_body(request), _known().get(name, {})
    rec = {"url": str(b.get("url") or old.get("url") or "").strip(), "location": str(b.get("location") or old.get("location") or "models"), "folder": str(b.get("folder") if b.get("folder") is not None else old.get("folder", "")).strip("/")}
    if not NAME.match(name): raise HTTPException(status_code=400, detail="its name is the file's name here: letters, digits, . _ + -")
    if rec["folder"] and not FOLDER.match(rec["folder"]): raise HTTPException(status_code=400, detail="the folder is a path inside the place, of plain names")
    if rec["location"] not in settings.locations(): raise HTTPException(status_code=400, detail=f"""this node has no place named {rec["location"]}; its places: {", ".join(settings.locations())}""")
    _allowed(rec["url"])
    if name in LIVE: raise HTTPException(status_code=409, detail=f"{name} is being downloaded")
    if _at(rec, name).exists(): raise HTTPException(status_code=409, detail=f"{_at(rec, name)} exists already: nothing is replaced. Remove it first, or give this one another name.")
    _keep({**_known(), name: {**rec, "error": ""}})
    LIVE[name] = {"got": 0, "size": 0, "t0": time.time()}
    LIVE[name]["task"] = asyncio.get_running_loop().create_task(_fetch(name, rec, str(b.get("token") or "")))
    journal.event("download.start", name=name, url=rec["url"][:500], to=str(_at(rec, name)), by=request.state.key.get("name", ""))
    return {"name": name, "set": True, "to": str(_at(rec, name))}

async def _remove(request, name: str):
    """?confirm=1: the download is stopped if under way, and the file it brought (and its unfinished part) deleted. Only what this module brought."""
    if not request.query_params.get("confirm"): raise HTTPException(status_code=400, detail="send ?confirm=1")
    known = _known()
    if name not in known: raise HTTPException(status_code=404, detail=f"no download named {name}")
    if name in LIVE: LIVE[name]["task"].cancel()
    dest = _at(known[name], name)
    for f in (dest, dest.with_name(dest.name + ".part")): f.unlink(missing_ok=True)
    _keep({k: v for k, v in _known().items() if k != name})
    journal.event("download.removed", name=name, path=str(dest), by=request.state.key.get("name", ""))
    return {"removed": name}

def setup(ctx):
    ctx.add_asset_kind("download", {"label": "Downloaded file", "about": "A file this machine fetches from an address into one of its places: a model file from a repository, for example. Its name is the file's name there. It continues a download that stopped, replaces nothing, and contacts only the address given.",
        "fields": [{"name": "url", "label": "Address of the file", "required": True, "hint": "The file itself, e.g. https://huggingface.co/<owner>/<repository>/resolve/main/<file>.gguf"}, {"name": "location", "label": "Which place", "default": "models", "hint": "One of this machine's places (Where it keeps things): models, audio_models, image_models ..."}, {"name": "folder", "label": "Folder inside it", "default": "gguf_local", "hint": "Where the program that will use it looks: gguf_local for GGUF models."}, {"name": "token", "label": "Access token", "type": "password", "hint": "Only for a repository that asks for one. Used for this download and not kept."}],
        "plain_names": True, "list": _list, "set": _set, "remove": _remove})   # plain_names: the name is the file's own name, taken as written (a manager's form must not reshape it)
