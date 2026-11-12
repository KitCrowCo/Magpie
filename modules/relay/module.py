"""modules/relay/module.py - carries a program's raw connection to another machine, so a program that only knows "127.0.0.1:9100" (colibri's cluster workers, say) can be spread across machines without being told.
A link is a loopback listener on this node, which every contained program can reach. By default ("via": "tunnel") what connects to it is carried in one HTTP request to a peer's /relay/tunnel/<name>: request body one way, response body the other, at the same time, with that node's key, its TLS if it has any, and its log. A peer reaches only names under relay.expose. With "via": "port" it goes straight to a pass channel the peer opened (core/channels.py), for traffic where the tunnel's cost matters. Links and exposures are settings: a manager decides who talks to whom and how.
The link's local port goes to the program as {link:<name>} in its settings, or read it from GET /relay."""
import ssl, socket, asyncio
from urllib.parse import urlsplit
from core import settings, services, journal, contain
from core.netns import join
from core.http import Router, HTTPException, Response, json_body

router = Router()
LINKS: dict = {}
STATS = {"in": 0, "out": 0, "refused": 0, "bytes_in": 0, "bytes_out": 0}

def _nodelay(w: asyncio.StreamWriter):
    s = w.get_extra_info("socket")
    if s is not None: s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

# --- The side a peer reaches: one request, both directions ---

class Tunnel(Response):
    """Answers with the target's bytes while still reading the request's. ASGI allows both at once; no extra library, and it works over the node's TLS."""
    def __init__(self, target, name: str): self.target, self.name = target, name

    async def __call__(self, scope, receive, send):
        try: r, w = await (self.target.open() if isinstance(self.target, services.Service) else asyncio.open_connection("127.0.0.1", self.target))
        except OSError as e:
            await send({"type": "http.response.start", "status": 502, "headers": []})
            await send({"type": "http.response.body", "body": str(e).encode()})
            return
        _nodelay(w)
        STATS["in"] += 1
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/octet-stream")]})
        async def down():
            while b := await r.read(1 << 16):
                STATS["bytes_out"] += len(b)
                await send({"type": "http.response.body", "body": b, "more_body": True})
        async def up():
            while True:
                m = await receive()
                if m["type"] == "http.disconnect": return
                if b := m.get("body"):
                    STATS["bytes_in"] += len(b)
                    w.write(b)
                    await w.drain()
                if not m.get("more_body"):
                    w.write_eof()
                    await asyncio.Event().wait()   # the other way may still be sending
        tasks = [asyncio.ensure_future(down()), asyncio.ensure_future(up())]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in tasks: t.cancel()
        w.close()
        await send({"type": "http.response.body", "body": b"", "more_body": False})

def _target(name: str):
    """What that name reaches: a running service, or a port on this machine."""
    v = (settings.get("relay.expose", {}) or {}).get(name)
    if isinstance(v, str):
        s = services.get(v)
        return s if s and s.ready() else None
    return int(v or 0) or None

@router.post("/relay/tunnel/{name}", scope="infer")
async def tunnel(request, name: str):
    target = _target(name)
    if not target:
        STATS["refused"] += 1
        journal.event("relay.refused", name=name, peer=str(request.client))
        raise HTTPException(status_code=403, detail=f"{name} is not exposed here, or is not running")
    return Tunnel(target, name)

# --- The side a local program connects to ---

async def _chunks(r: asyncio.StreamReader, w: asyncio.StreamWriter):
    """The program's bytes as one chunked request body."""
    try:
        while b := await r.read(1 << 16):
            w.write(b"%x\r\n" % len(b) + b + b"\r\n")
            await w.drain()
        w.write(b"0\r\n\r\n")
        await w.drain()
    except ConnectionError: pass

async def _unchunk(r: asyncio.StreamReader, w: asyncio.StreamWriter):
    """The peer's chunked answer as the program's bytes."""
    try:
        while True:
            n = int((await r.readline()).strip() or b"0", 16)
            if n == 0: break
            w.write(await r.readexactly(n))
            await w.drain()
            await r.readexactly(2)
    except (ConnectionError, asyncio.IncompleteReadError, ValueError): pass
    finally:
        try: w.close()
        except Exception: pass

async def _carry(link: dict, r: asyncio.StreamReader, w: asyncio.StreamWriter):
    u = urlsplit(link["peer"])
    try:
        if link.get("via") == "port":
            STATS["out"] += 1
            return await join(r, w, *await asyncio.open_connection(u.hostname, u.port))
        pr, pw = await asyncio.open_connection(u.hostname, u.port or (443 if u.scheme == "https" else 80), ssl=ssl.create_default_context(cafile=link.get("cafile")) if u.scheme == "https" else None)
        _nodelay(pw); _nodelay(w)
        pw.write(f"""POST /relay/tunnel/{link["name"]} HTTP/1.1\r\nHost: {u.netloc}\r\nAuthorization: Bearer {link.get("key", "")}\r\nTransfer-Encoding: chunked\r\nContent-Type: application/octet-stream\r\n\r\n""".encode())
        await pw.drain()
        head = (await asyncio.wait_for(pr.readuntil(b"\r\n\r\n"), 30)).decode(errors="replace")
        if " 200 " not in head.splitlines()[0]: raise ConnectionError(head.splitlines()[0])
        STATS["out"] += 1
        up = asyncio.ensure_future(_chunks(r, pw))
        await _unchunk(pr, w)
        up.cancel()
        pw.close()
    except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError, OSError) as e:
        journal.event("relay.failed", link=link.get("name"), peer=link["peer"], error=str(e)[:200])
        w.close()

async def open_link(l: dict) -> int:
    """A loopback listener carried to that name on that peer. local_port 0 takes a free one."""
    srv = await asyncio.start_server(lambda r, w: _carry(l, r, w), "127.0.0.1", int(l.get("local_port") or 0))
    port = srv.sockets[0].getsockname()[1]
    contain.REACH.add(port)
    LINKS[l["name"]] = {**l, "local_port": port, "server": srv}
    journal.event("relay.link", name=l["name"], peer=l["peer"], local_port=port)
    return port

@router.get("/relay")
async def status(request): return {"expose": sorted(settings.get("relay.expose", {}) or {}), "links": [{"name": n, "peer": l["peer"], "local_port": l["local_port"]} for n, l in LINKS.items()], "stats": STATS}

@router.post("/relay/link")
async def add_link(request):
    """{"name", "peer": "http://host:port", "key", "local_port", "via": "tunnel" or "port"}: from now on, connecting to that local port here reaches <name> on the peer. Lasts until restart unless it is also under relay.links in the settings."""
    b = await json_body(request)
    return {"name": b["name"], "local_port": await open_link(b)}

async def _start():
    for l in settings.get("relay.links", []) or []: await open_link(l)

def setup(ctx):
    ctx.add_router(router)
    return _start()
