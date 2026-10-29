"""engines.py - inference engines other than llama-server on this node (colibri first), configured in engines.json (path: ENGINES_CONFIG).
An engine is a server process with its own API (colibri: OpenAI-compatible).
A request to /engines/<name>/<path> starts the engine when it is not running, is forwarded to it unchanged, and its answer is streamed back as it comes; after idle_stop_s with no request the engine is stopped.
Callers use the engine's own API through this prefix (in the portal: the colibri connection profile with Base Path /engines/<name>), so one port serves llama.cpp models and every engine.
An exclusive engine needs the node's memory to itself: stop_main (given by main.py) stops the main llama-server before the engine starts, and main.py calls stop_exclusive() before it starts llama-server, which waits for the engine's running requests to finish and then stops it.
The two never hold the memory at once.
Logs: MODELS_DIR/engines/<name>.log."""
import os, json, time, asyncio, threading, subprocess
from pathlib import Path
import httpx
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import fleet   # journal events: engine.start, engine.up, engine.failed, engine.stop, engine.req

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/root/.ollama/models"))
LOG_DIR = MODELS_DIR / "engines"
CONFIG_PATH = Path(os.getenv("ENGINES_CONFIG", Path(__file__).parent / "engines.json"))
router = APIRouter()
ENGINES: dict = {}
_HOOKS: dict = {"stop_main": None}   # set by start(): async callable that stops the main llama-server and returns once its memory is free
_TASKS: set = set() # background tasks are held here: the event loop keeps only weak references
_HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "accept-encoding", "content-encoding"}   # hop-by-hop and encoding headers are not forwarded either way

def _now() -> str: return time.strftime("%Y-%m-%dT%H:%M:%S")

# --- Engines ---

ECHO = os.getenv("ENGINES_ECHO", "1") != "0"   # 0 = engine output goes only to its log file, not to Magpie's own output

def _pump(name: str, stream, log: Path):
    """Copies an engine's output line by line to its log file and (unless ENGINES_ECHO=0) to Magpie's output prefixed with its name, so `docker compose logs -f` shows engines live."""
    with open(log, "a", buffering=1, errors="replace") as f:
        for line in iter(stream.readline, ""):
            f.write(line)
            if ECHO: print(f"[{name}] {line.rstrip()}", flush=True)

class Engine:
    def __init__(self, spec: dict):
        self.spec, self.name, self.port = spec, spec["name"], int(spec["port"])
        self.proc, self.status, self.last_use, self.active, self.lock = None, "stopped", 0.0, 0, asyncio.Lock()
        self.log = LOG_DIR / f"{self.name}.log"
        self.http = httpx.AsyncClient(base_url=f"http://127.0.0.1:{self.port}", timeout=httpx.Timeout(connect=10.0, read=float(spec.get("read_timeout_s", 36000)), write=60.0, pool=10.0))

    def alive(self) -> bool: return self.proc is not None and self.proc.poll() is None
    def tail(self, n: int = 15) -> list: return self.log.read_text(errors="replace").splitlines()[-n:] if self.log.exists() else []

    def info(self) -> dict:
        dead = self.proc is not None and not self.alive()
        return {"name": self.name, "port": self.port, "status": f"exited ({self.proc.returncode})" if dead and self.status == "up" else self.status, "pid": self.proc.pid if self.alive() else None, "active": self.active, "idle_s": round(time.time() - self.last_use) if self.last_use else None, "exclusive": self.spec.get("exclusive", True), "idle_stop_s": self.spec.get("idle_stop_s", 0), "cmd": self.spec["cmd"], **({"log_tail": self.tail(8)} if dead else {})}

    async def ensure(self) -> bool:
        """Running and answering, or started now and waited for up to start_timeout_s. False, with the reason in status and the log, when it exits or never answers."""
        async with self.lock:
            if self.alive() and self.status == "up": return True
            if self.alive(): await self._kill()   # a start that never answered
            if self.spec.get("exclusive", True) and _HOOKS["stop_main"]: await _HOOKS["stop_main"]()
            cmd = [a.replace("{port}", str(self.port)) for a in self.spec["cmd"]]
            with open(self.log, "a") as f: f.write(f"""\n=== {_now()} {" ".join(cmd)}\n""")
            self.proc, self.status = subprocess.Popen(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1", **self.spec.get("env", {})}, cwd=self.spec.get("cwd") or None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1), "starting"
            threading.Thread(target=_pump, args=(self.name, self.proc.stdout, self.log), daemon=True).start()
            print(f"[ENGINE] starting {self.name}: {' '.join(cmd)}", flush=True)
            fleet.event("engine.start", name=self.name, cmd=cmd, exclusive=self.spec.get("exclusive", True))
            deadline = time.time() + float(self.spec.get("start_timeout_s", 1800))
            while self.proc.poll() is None and time.time() < deadline:
                try:
                    if await self._ready():
                        self.status, self.last_use = "up", time.time()
                        print(f"[ENGINE] {self.name} up on port {self.port}", flush=True)
                        fleet.event("engine.up", name=self.name, pid=self.proc.pid)
                        return True
                except httpx.HTTPError: pass
                await asyncio.sleep(2)
            self.status = f"failed (exit {self.proc.returncode})" if self.proc.poll() is not None else f"""failed (no answer in {self.spec.get("start_timeout_s", 1800)} s)"""
            print(f"[ENGINE] {self.name} {self.status} - see {self.log}", flush=True)
            fleet.event("engine.failed", name=self.name, status=self.status, tail=self.tail(8))
            if self.alive(): await self._kill()
            return False

    async def _ready(self) -> bool:
        """health "tcp": the port accepts a connection (engines with a raw protocol, such as colibri's expert workers); otherwise a GET of the health path answers 200."""
        if self.spec.get("health") != "tcp": return (await self.http.get(self.spec.get("health", "/v1/models"))).status_code == 200
        try: _, w = await asyncio.open_connection("127.0.0.1", self.port)
        except OSError: return False
        w.close()
        return True

    async def _kill(self):
        self.proc.terminate()
        try: await asyncio.to_thread(self.proc.wait, 30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            await asyncio.to_thread(self.proc.wait)

    async def stop(self, why: str = ""):
        async with self.lock:
            if self.alive():
                print(f"[ENGINE] stopping {self.name}{' - ' + why if why else ''}", flush=True)
                await self._kill()
                fleet.event("engine.stop", name=self.name, why=why)
            self.proc, self.status = None, "stopped"

# --- Lifecycle (called from main.py) ---

async def start(stop_main=None):
    """Registers every enabled engine in engines.json; none is started here (the first request starts it). Values may name environment variables as ${NAME};
    an engine with an unset name is skipped with a log line. stop_main: async callable that stops the main llama-server and returns once its memory is free."""
    if not CONFIG_PATH.exists(): print(f"[ENGINE] {CONFIG_PATH} not found - no other engines", flush=True); return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _HOOKS["stop_main"] = stop_main
    for spec in (json.loads(os.path.expandvars(json.dumps(s))) for s in json.loads(CONFIG_PATH.read_text())["engines"] if s.get("enabled", True)):
        if "${" in json.dumps(spec): print(f"""[ENGINE] {spec["name"]}: unset environment variable in {spec} - engine skipped""", flush=True); continue
        ENGINES[spec["name"]] = Engine(spec)
    _TASKS.add(asyncio.create_task(_idle_loop()))
    print(f"[ENGINE] registered (started on first request): {list(ENGINES) or 'none'}", flush=True)

async def stop_exclusive():
    """For main.py, before it starts llama-server: waits for each running exclusive engine's requests to finish, then stops it."""
    for e in ENGINES.values():
        if not (e.alive() and e.spec.get("exclusive", True)): continue
        while e.active: await asyncio.sleep(1)
        await e.stop("llama-server needs the memory")

def any_exclusive_running() -> bool: return any(e.alive() and e.spec.get("exclusive", True) for e in ENGINES.values())

async def stop():
    for t in _TASKS: t.cancel()
    for e in ENGINES.values(): await e.stop("shutdown")

async def _idle_loop():
    while True:
        await asyncio.sleep(30)
        for e in list(ENGINES.values()):
            if e.alive() and not e.active and e.spec.get("idle_stop_s") and time.time() - e.last_use > e.spec["idle_stop_s"]: await e.stop(f"""idle for {e.spec["idle_stop_s"]} s""")

# --- Routes ---

def _engine(name: str) -> Engine:
    if name not in ENGINES: raise HTTPException(status_code=404, detail=f"""no engine {name!r} - registered: {", ".join(ENGINES) or "none"} (enable it in {CONFIG_PATH.name})""")
    return ENGINES[name]

@router.get("/api/engines/status")
async def engines_status(): return {"config": str(CONFIG_PATH), "engines": [e.info() for e in ENGINES.values()]}

@router.post("/api/engines/{name}/start")
async def engine_start(name: str): return {"started": await _engine(name).ensure(), **_engine(name).info()}

@router.post("/api/engines/{name}/stop")
async def engine_stop(name: str):
    await _engine(name).stop("asked through /api/engines")
    return _engine(name).info()

@router.api_route("/engines/{name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def engine_proxy(name: str, path: str, request: Request):
    """One request forwarded to the engine (started first when needed), its answer streamed back as it arrives (server-sent events included)."""
    e = _engine(name)
    if not await e.ensure(): raise HTTPException(status_code=503, detail=f"""engine {name} is not running ({e.status}) - log {e.log}: {" | ".join(e.tail(3))}""")
    e.active += 1
    e.last_use = t0 = time.time()
    try: r = await e.http.send(e.http.build_request(request.method, f"/{path}", params=request.query_params, content=await request.body(), headers={k: v for k, v in request.headers.items() if k.lower() not in _HOP}), stream=True)
    except httpx.HTTPError as x:
        e.active -= 1
        raise HTTPException(status_code=502, detail=f"engine {name}: {type(x).__name__}: {x}")
    async def body():
        n = 0
        try:
            async for chunk in r.aiter_bytes(): n += len(chunk); yield chunk
        finally:
            await r.aclose()
            e.active, e.last_use = e.active - 1, time.time()
            fleet.event("engine.req", name=name, path=path, method=request.method, status=r.status_code, seconds=round(time.time() - t0, 3), bytes=n, rid=request.headers.get("x-request-id", ""))
    return StreamingResponse(body(), status_code=r.status_code, headers={k: v for k, v in r.headers.items() if k.lower() not in _HOP})

