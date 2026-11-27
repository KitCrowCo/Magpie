"""core/channels.py - any other way into this node, opened by a manager: whether a path is shaped by Magpie or passed straight through is decided here, never by what runs behind it.
A channel is a port this node listens on, with Magpie holding both ends, so it is counted, can be closed, and accepts only who it was told to.
mode "shape": a virtual port serving this node's whole interface (keys apply, a client sees the shapes of /api, /v1, /run), for something that only knows one fixed port. mode "pass": bytes straight to one program ("service") or a local port ("to_port"), for when nothing needs changing and the program's own protocol is what is wanted. It carries no key, so "allow" (addresses or networks, or "*") must be said. While a pass connection is open the service counts it as working: it holds a slot, and is not stopped under it."""
import time, asyncio, ipaddress
from core import settings, services, jobs, journal, contain
from core.http import Router, HTTPException, json_body
from core.netns import join

router = Router()
OPEN: dict = {}   # name -> {"spec", "server", "stats"}

def _allowed(addr: str, allow: list) -> bool: return "*" in allow or any(ipaddress.ip_address(addr) in ipaddress.ip_network(a, strict=False) for a in allow)

async def _conn(name: str, r, w):
    c = OPEN[name]
    spec, st, s = c["spec"], c["stats"], None
    peer = w.get_extra_info("peername")[0]
    if not _allowed(peer, spec.get("allow", ["*"])) or (spec["mode"] == "pass" and spec.get("service") and not ((s := services.get(spec["service"])) and s.ready())):
        st["refused"] += 1
        journal.event("channel.refused", name=name, peer=peer)
        return w.close()
    if s: s.active, s.last_use = s.active + 1, time.time()
    st["open"], st["total"] = st["open"] + 1, st["total"] + 1
    journal.event("channel.open", name=name, peer=peer)
    try:
        pr, pw = await (s.open() if s else asyncio.open_connection("127.0.0.1", int(spec.get("to_port") or contain.node_port())))
        await join(r, w, pr, pw, lambda n: st.__setitem__("in", st["in"] + n), lambda n: st.__setitem__("out", st["out"] + n))
    except OSError as e: journal.event("channel.failed", name=name, error=str(e)[:200])
    finally:
        w.close()
        st["open"] -= 1
        if s:
            s.active, s.last_use = s.active - 1, time.time()
            jobs.kick()

def _check(spec: dict):
    if spec.get("mode") not in ("pass", "shape"): raise HTTPException(status_code=400, detail="mode must be pass or shape")
    if spec["mode"] == "pass" and not spec.get("allow"): raise HTTPException(status_code=400, detail="""a pass channel carries no key: say who may use it in allow (addresses, networks, or "*")""")
    if spec["mode"] == "pass" and not (spec.get("service") or spec.get("to_port")): raise HTTPException(status_code=400, detail="a pass channel needs service or to_port")
    for a in spec.get("allow", []):
        if a != "*":
            try: ipaddress.ip_network(a, strict=False)
            except ValueError as e: raise HTTPException(status_code=400, detail=str(e))

async def open_channel(name: str, spec: dict) -> int:
    await close_channel(name)
    srv = await asyncio.start_server(lambda r, w: _conn(name, r, w), spec.get("host", "0.0.0.0"), int(spec.get("port") or 0))
    OPEN[name] = {"spec": spec, "server": srv, "stats": {"open": 0, "total": 0, "refused": 0, "in": 0, "out": 0}}
    port = srv.sockets[0].getsockname()[1]
    journal.event("channel.listen", name=name, port=port, mode=spec["mode"], service=spec.get("service"))
    print(f"""[CHANNEL] {name}: {spec["mode"]} on port {port}""", flush=True)
    return port

async def close_channel(name: str):
    if c := OPEN.pop(name, None):
        c["server"].close()
        journal.event("channel.close", name=name)

async def start():
    """The channels a manager has set, reopened at each start."""
    for name, spec in (settings.get("channels", {}) or {}).items(): await open_channel(name, spec)

def view() -> list: return [{"name": n, "port": c["server"].sockets[0].getsockname()[1] if c["server"].sockets else None, **{k: v for k, v in c["spec"].items()}, **c["stats"]} for n, c in OPEN.items()]

@router.get("/channels")
async def channels(request): return {"channels": view()}

@router.route("/channels/{name}", ("PUT",))
async def channel_set(request, name: str):
    """{"mode", "port" (0 or absent = a free one), "host", "allow", "service" or "to_port"}. Kept: it reopens when the node restarts."""
    spec = await json_body(request)
    _check(spec)
    if "." in name: raise HTTPException(status_code=400, detail="a channel name has no dots")
    try: settings.set_path(f"channels.{name}", spec, request.state.key.get("name", ""))
    except ValueError as e: raise HTTPException(status_code=403, detail=str(e))
    try: return {"name": name, "port": await open_channel(name, spec)}
    except OSError as e: raise HTTPException(status_code=409, detail=f"cannot listen: {e}")

@router.route("/channels/{name}", ("DELETE",))
async def channel_delete(request, name: str):
    try: settings.set_path(f"channels.{name}", None, request.state.key.get("name", ""))
    except ValueError as e: raise HTTPException(status_code=403, detail=str(e))
    await close_channel(name)
    return {"name": name, "status": "closed"}
