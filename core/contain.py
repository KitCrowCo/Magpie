"""core/contain.py - nothing a program does reaches outside unless this node approved it. Each program runs in a network of its own (core/netns.py) with no route anywhere. What it gets are the routes made here: Magpie to the program, the program to Magpie's own port and relay links, and one proxy that opens only approved hosts and writes every attempt to the journal.
Settings: network.default "contained" or "host" (no shell, for a manager that wants nothing in the path); network.services {name: mode}; network.allow {"*" or a service name: [host patterns]}; network.require true: refuse to run a program when this kernel cannot contain it. A change of mode applies at the program's next start.
Contained here means the network. Files are not: a program can read what this user can, and connect to a Unix socket on disk that this user may use."""
import os, sys, json, socket, asyncio, fnmatch, hashlib, tempfile, collections, subprocess
from pathlib import Path
from core import settings, journal
from core.http import Router
from core.netns import join

router = Router()
NETNS = Path(__file__).with_name("netns.py")
EGRESS_PORT = 17171                         # the proxy's port inside a program's own network
REACH: set = set()                          # ports on this machine every contained program may connect to, added by relay links
ATTEMPTS = collections.deque(maxlen=200)    # the latest outside contacts asked for, approved or not
_SERVERS: dict = {}                         # socket path -> the server behind it, one per route
_KEEP: set = set()
_AVAILABLE = []

def _dir() -> Path:
    """Unix socket paths have a short limit, so these live under the system's temporary folder rather than wherever the data was moved to; only this user can enter it."""
    d = Path(tempfile.gettempdir()) / f"""mgp{os.geteuid()}_{hashlib.sha256(str(settings.DATA).encode()).hexdigest()[:8]}"""
    d.mkdir(mode=0o700, exist_ok=True)
    return d

def _path(name: str, kind: str) -> Path: return _dir() / f"""{hashlib.sha256(name.encode()).hexdigest()[:12]}.{kind}"""

def available() -> bool:
    """Whether this kernel lets a program be contained, tried once. Android and macOS do not; there a program runs as "host" and /network says so."""
    if not _AVAILABLE: _AVAILABLE.append(sys.platform == "linux" and subprocess.run([sys.executable, "-I", str(NETNS), "--probe"], capture_output=True).returncode == 0)
    return _AVAILABLE[0]

def mode(name: str) -> str:
    want = (settings.get("network.services", {}) or {}).get(name) or settings.get("network.default", "contained")
    if want != "contained" or available(): return want
    if settings.get("network.require"): raise RuntimeError(f"network.require is set and this kernel cannot contain {name}")
    return "host"

# --- Routes a contained program gets ---

def _listen(path: Path, handler):
    """A route exists from the moment this returns, so a program started right after finds it."""
    if str(path) in _SERVERS: return
    path.unlink(missing_ok=True)
    s = socket.socket(socket.AF_UNIX)
    s.bind(str(path))
    s.listen(128)
    t = asyncio.ensure_future(asyncio.start_unix_server(handler, sock=s))
    _SERVERS[str(path)] = t

def _reach(port: int) -> str:
    """A route to a port on this machine, opened once however many programs use it."""
    p = _path(f"port:{port}", "r")
    async def h(r, w):
        try: await join(r, w, *await asyncio.open_connection("127.0.0.1", port))
        except OSError: w.close()
    _listen(p, h)
    return str(p)

def approved(name: str, host: str) -> bool:
    a = settings.get("network.allow", {}) or {}
    return any(fnmatch.fnmatch(host, pat) for pat in [*a.get("*", []), *a.get(name, [])])

def _egress(name: str) -> str:
    """The one way out: a proxy that answers CONNECT for approved hosts. A program ignoring its proxy setting has no route at all."""
    p = _path(name, "e")
    async def h(r, w):
        try:
            head = (await r.readuntil(b"\r\n\r\n")).decode(errors="replace").split("\r\n")[0].split()
            target = head[1] if len(head) > 1 else ""
            host, _, port = target.rpartition(":") if head[0] == "CONNECT" else (target, "", "")
            ok = head[0] == "CONNECT" and approved(name, host)
            ATTEMPTS.append({"t": journal.now(), "service": name, "host": host, "approved": ok})
            journal.event("net.allowed" if ok else "net.refused", service=name, host=host, method=head[0])
            if not ok:
                print(f"[NETWORK] refused: {name} asked for {target} (approve with network.allow)", flush=True)
                w.write(b"HTTP/1.1 403 Not approved by this node\r\nContent-Length: 0\r\n\r\n")
                await w.drain()
                return w.close()
            pr, pw = await asyncio.open_connection(host, int(port))
            w.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            await w.drain()
            sent = [0, 0]
            await join(r, w, pr, pw, lambda n: sent.__setitem__(0, sent[0] + n), lambda n: sent.__setitem__(1, sent[1] + n))
            journal.event("net.closed", service=name, host=host, sent=sent[0], received=sent[1])
        except (OSError, ValueError, asyncio.IncompleteReadError, asyncio.LimitOverrunError): w.close()
    _listen(p, h)
    return str(p)

def node_port() -> int: return int(settings.owner("bind.port", settings.PORT))

def wrap(name: str, port: int, cmd: list, env: dict) -> tuple:
    """(command, env, the Unix socket that reaches the program, or None when it runs on this machine's own loopback)."""
    if mode(name) != "contained": return cmd, env, None
    sock = _path(name, "s")
    plan = {"sock": str(sock), "port": port, "reach": {**{str(p): _reach(p) for p in {node_port(), *REACH} if p != port}, str(EGRESS_PORT): _egress(name)}}
    proxy = f"http://127.0.0.1:{EGRESS_PORT}"
    return [sys.executable, "-I", str(NETNS), json.dumps(plan), *map(str, cmd)], {**(env or {}), **{k: proxy for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")}, "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}, str(sock)

# --- Looking ---

@router.get("/network")
async def network(request): return {"contained_possible": available(), "default": settings.get("network.default", "contained"), "require": bool(settings.get("network.require")), "services": settings.get("network.services", {}) or {}, "allow": settings.get("network.allow", {}) or {}, "attempts": list(ATTEMPTS)}
