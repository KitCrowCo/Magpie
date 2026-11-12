"""magpie.py - a node of the portal's resource pool. Starts as a manager of nothing: the modules named in node.json (or activated later by a manager) decide what this node can do."""
import os
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")   # for every program this node starts
import sys, asyncio, logging, argparse, contextlib
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException
from core import settings

settings.load()
from core import auth, journal, hardware, services, jobs, modules, drivers, files, api, contain, channels, autostart

class _QuietPolls(logging.Filter):
    """Drops access lines for successful status polls (each managing portal reads these every few seconds); any other status on those routes is still logged."""
    def filter(self, r): return not (isinstance(r.args, tuple) and len(r.args) >= 5 and str(r.args[2]).startswith(("/fleet/events", "/fleet/state", "/control/status", "/node", "/health")) and r.args[4] == 200)
logging.getLogger("uvicorn.access").addFilter(_QuietPolls())

async def _http_error(request: Request, e: HTTPException): return JSONResponse({"error": e.detail, "detail": e.detail}, status_code=e.status_code)

# --- Output log ---

LOG_MAX_BYTES = 20_000_000   # magpie.log moves to magpie.log.1 past this size (one earlier file kept)

class _Tee:
    """A stream that also writes to the node's own log file, so its output can be read without the system journal (no group membership, no root) - by the portal's node page or tail -f."""
    def __init__(self, stream, f): self.stream, self.f = stream, f
    def write(self, s):
        self.stream.write(s)
        self.f.write(s)
        if self.f.tell() > LOG_MAX_BYTES: _rotate(self.f)
        return len(s)
    def flush(self): self.stream.flush(); self.f.flush()
    def __getattr__(self, name): return getattr(self.stream, name)

def _open_log():
    p = settings.location("logs") / "magpie.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    return open(p, "a", buffering=1, errors="replace")

def _rotate(f):
    """Keeps the previous file as magpie.log.1 and continues in a new magpie.log; both streams switch together."""
    f.flush()
    p = settings.location("logs") / "magpie.log"
    p.replace(p.with_name("magpie.log.1"))
    sys.stdout.f = sys.stderr.f = _open_log()
    f.close()

def _log_to_file():
    """From here on, everything this process prints also goes to <logs location>/magpie.log, a plain file owned by the user the node runs as."""
    f = _open_log()
    sys.stdout, sys.stderr = _Tee(sys.stdout, f), _Tee(sys.stderr, f)

@contextlib.asynccontextmanager
async def lifespan(app):
    auth.load()
    journal.start()
    journal.event("node.start", node=settings.NODE_ID, config=str(settings.NODE_FILE))
    print(f"""[NODE] {settings.NODE_ID}: {hardware.facts()["cpu"]}, {hardware.facts()["ram_gb"]} GB, {hardware.facts()["arch"]}""", flush=True)
    modules.discover()
    modules._APP["app"] = app
    for name in modules.wanted():
        try: modules.activate(name)
        except ValueError as e: print(f"[MODULE] {e}", flush=True)   # a module that cannot run here does not stop the node: /modules says why
    print(f"""[NODE] active: {[n for n, m in modules.ALL.items() if m["active"]] or "nothing yet"} | capabilities: {sorted(modules.capabilities()) or "none"}""", flush=True)
    sched = jobs.spawn(jobs.scheduler())
    jobs.spawn(files.sweep())
    api.sweep_answers()
    await channels.start()
    if not contain.available(): print("[NETWORK] this kernel cannot contain programs: they run with this machine's network, and /network says so", flush=True)
    yield
    sched.cancel()
    jobs.cancel_all()
    for s in services.running(): await s.stop("node shutting down")
    journal.event("node.stop", node=settings.NODE_ID)
    journal.stop()

app = Starlette(routes=[*api.router.routes, *files.router.routes, *channels.router.routes, *contain.router.routes], lifespan=lifespan, exception_handlers={HTTPException: _http_error})

def _bind() -> dict:
    """node.json "bind": {"host", "port", "tls_cert", "tls_key"}. With a certificate the API key does not cross the network in the clear."""
    b = settings.owner("bind", {})
    return {"host": b.get("host", settings.HOST), "port": int(b.get("port", settings.PORT)), **({"ssl_certfile": b["tls_cert"], "ssl_keyfile": b["tls_key"]} if b.get("tls_cert") else {})}

# --- First setup ---

async def setup(a):
    """Finds out what this machine is and what fits, then installs what it is for: --for text,speech,image,knowledge, or the modules marked auto. --plan only reports. Anything left out can be installed later, from here or from a portal."""
    if a.allow_system: settings._OWNER.setdefault("setup", {})["allow_system"] = True
    modules.discover()
    f = hardware.facts()
    print(f"""{f["cpu"]} | {f["threads"]} threads | {f["ram_gb"]} GB | {f["arch"]} | GPU {[g["vendor"] for g in f["gpus"]] or "none"} | system packages: {drivers.manager() or "no manager"}, {"allowed" if drivers.system_allowed() else "not allowed (--allow-system)"}""", flush=True)
    uses = set(a.for_.split(",")) if a.for_ else None
    chosen = []
    for n, m in modules.ALL.items():
        if not m["spec"].get("drivers"): continue
        p = drivers.plan(n, m["spec"])
        want = bool(p["driver"]) and (set(m["spec"].get("for", [])) & uses if uses else m["spec"].get("auto") and p["recommended"])
        print(f"""  {n:12} {"-> " + p["driver"] if p["driver"] else "blocked":14} {"install now" if want else ""} {m["spec"].get("label", "")[:60]}""", flush=True)
        if want: chosen.append(n)
    if a.plan: return 0
    failed = []
    for n in chosen:
        for name in [*modules.ALL[n]["spec"].get("needs", []), n]:
            if not modules.ALL[name]["spec"].get("drivers") or modules._ready(name): continue
            async for p in drivers.install(name, modules.ALL[name]["spec"]):
                if p.get("status"): print(f"""[{name}] {p["status"]}""", flush=True)
        if modules._ready(n): modules.remember_active(n)
        else: failed.append(n)
    if failed: print(f"Not installed: {failed}. What worked stays; try again, change the options, or install from a portal.", flush=True)
    if a.autostart: print("\n".join(autostart.on()), flush=True)
    return 1 if failed else 0

def main():
    ap = argparse.ArgumentParser(description="A node of a Portal Server's resource pool.")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("setup", help="find out what fits this machine and install it")
    s.add_argument("--plan", action="store_true", help="report only")
    s.add_argument("--for", dest="for_", default="", help="what this node is for: text,speech,image,knowledge")
    s.add_argument("--allow-system", action="store_true", help="may install system packages (build tools, drivers)")
    s.add_argument("--autostart", action="store_true", help="start the node with the machine from now on (magpie.py autostart on)")
    sub.add_parser("autostart", help="start the node with the machine, without a terminal kept open").add_argument("do", choices=["on", "off", "status"], nargs="?", default="status")
    a = ap.parse_args()
    if a.cmd == "autostart": return print("\n".join(getattr(autostart, a.do)()))
    _log_to_file()
    if a.cmd == "setup":
        auth.load()
        return asyncio.run(setup(a))
    uvicorn.run(app, **_bind())

if __name__ == "__main__": sys.exit(main())
