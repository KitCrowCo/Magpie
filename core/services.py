"""core/services.py - every process this node runs for someone is a Service: one state machine, one way to stop it, one log, one kill switch that needs no cooperation from the program inside.
A process runs in its own process group, so a kill takes its children too, and in a network of its own (core/contain.py), so nothing reaches it, and it reaches nothing, except through Magpie.
Looking at a service never starts it. Only the scheduler (core/jobs.py) starts one for work, or an admin through POST /services/<name>/start."""
import os, time, signal, socket, asyncio, threading, subprocess, collections
from pathlib import Path
import httpx
from core import settings, budget, journal, hardware, contain

REG: dict = {}   # name -> Service
ECHO = os.getenv("ENGINES_ECHO", "1") != "0"   # 0 = a service's output goes only to its log file, not to Magpie's own output

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def get(name: str): return REG.get(name)
def running() -> list: return [s for s in REG.values() if s.running()]
def view() -> list: return [s.info() for s in REG.values()]

class Service:
    """spec: {"cmd": [...], "env", "cwd", "port" (0 or absent = a free one), "health" (a GET path answering 200, or "tcp"), "start_timeout_s", "slots", "stop_timeout_s"}. events: the journal family (<events>.start, .up, .failed, .stop, .exit), so existing readers keep working."""
    def __init__(self, name: str, spec: dict, events: str = "engine", owner: str = ""):
        self.name, self.spec, self.events, self.owner = name, spec, events, owner
        self.alone = False    # set by the scheduler at each load: while up, nothing else runs beside it (core/budget.py)
        self.adapter = None   # what brings this service up through the scheduler when it is not one of modules.ADAPTERS (a knowledge instance)
        self.proc, self.status, self.detail, self.sig, self.t_up = None, "stopped", "", None, 0.0
        self.active, self.slots, self.last_use, self.closing, self.starts, self.exits = 0, int(spec.get("slots", 1)), 0.0, False, 0, []
        self.port = int(spec.get("port") or 0)
        self.http, self.sock = None, None
        self.tail_buf = collections.deque(maxlen=200)
        self.lock, self.meta = asyncio.Lock(), {}
        REG[name] = self

    # --- Looking (never starts anything) ---

    @property
    def log(self) -> Path:
        d = settings.location("logs") / "services"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"""{self.name.replace(":", "_")}.log"""

    def alive(self) -> bool: return self.proc is not None and self.proc.poll() is None
    def running(self) -> bool: return self.status in ("up", "starting") and self.alive()
    def ready(self) -> bool: return self.status == "up" and self.alive()
    def tail(self, n: int = 15) -> list: return list(self.tail_buf)[-n:]
    def _ev(self, what: str, **f): journal.event(f"{self.events}.{what}", name=self.name, **f)

    def rss_gb(self) -> float:
        """The whole process group's resident memory."""
        if not self.alive(): return 0.0
        pids, seen = [self.proc.pid], 0.0
        while pids:
            pid = pids.pop()
            seen += hardware.rss_gb(pid)
            try: pids += [int(c) for t in Path(f"/proc/{pid}/task").iterdir() for c in (t / "children").read_text().split()]
            except OSError: pass
        return round(seen, 2)

    def info(self) -> dict:
        dead = self.proc is not None and not self.alive()
        return {"name": self.name, "owner": self.owner, "status": f"exited ({self.proc.returncode})" if dead and self.status == "up" else self.status, "detail": self.detail, "pid": self.proc.pid if self.alive() else None, "network": "contained" if self.sock else "host", "sock": self.sock, "port": self.port or None, "active": self.active, "slots": self.slots, "closing": self.closing, "idle_s": round(time.time() - self.last_use) if self.last_use else None, "rss_gb": self.rss_gb(), "uptime_s": round(time.time() - self.t_up) if self.ready() else 0, "starts": self.starts, "last_exits": self.exits[-3:], "lease": budget.LEASES.get(self.name), "sig": self.sig, **self.meta, **({"log_tail": self.tail(8)} if dead or self.status.startswith("failed") else {})}

    # --- Starting ---

    def base(self) -> str: return f"http://127.0.0.1:{self.port}"

    def connector(self) -> dict:
        """What an httpx client needs to reach this program. No kept-alive connections: on this path they save nothing, and a program closing an idle one would fail the next request."""
        limits = httpx.Limits(max_keepalive_connections=0)
        return {"transport": httpx.AsyncHTTPTransport(uds=self.sock, limits=limits)} if self.sock else {"limits": limits}

    def client(self, read_s: float = 36000.0) -> httpx.AsyncClient:
        """All four httpx timeout fields, so a hung read cannot wait forever."""
        if self.http is None or self.http.is_closed or self.http_for != (self.base(), self.sock):
            self.http, self.http_for = httpx.AsyncClient(base_url=self.base(), timeout=httpx.Timeout(connect=10.0, read=read_s, write=60.0, pool=10.0), **self.connector()), (self.base(), self.sock)
        return self.http

    async def open(self) -> tuple:
        """A raw connection to the program: (reader, writer)."""
        return await (asyncio.open_unix_connection(self.sock) if self.sock else asyncio.open_connection("127.0.0.1", self.port))

    def _pump(self, stream):
        with open(self.log, "a", buffering=1, errors="replace") as f:
            for line in iter(stream.readline, ""):
                f.write(line)
                self.tail_buf.append(line.rstrip())
                if ECHO: print(f"[{self.name}] {line.rstrip()}", flush=True)

    def spawn(self, cmd: list, env: dict = None, cwd=None):
        """Starts the process in a group of its own; env entries set to None are removed."""
        self.tail_buf.clear()
        with open(self.log, "a") as f: f.write(f"""\n=== {time.strftime("%Y-%m-%dT%H:%M:%S")} {" ".join(map(str, cmd))}\n""")
        cmd, env, self.sock = contain.wrap(self.name, self.port, cmd, env)
        self.proc = subprocess.Popen(cmd, env={k: v for k, v in {**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}.items() if v is not None}, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1, start_new_session=True)
        threading.Thread(target=self._pump, args=(self.proc.stdout,), daemon=True).start()
        self.starts += 1

    async def probe(self) -> bool:
        """One readiness check. health "tcp": the port accepts a connection; otherwise a GET of the health path answers 200."""
        h = self.spec.get("health", "/health")
        if h == "tcp":
            try: _, w = await self.open()
            except OSError: return False
            w.close()
            return True
        try: return (await self.client().get(h, timeout=5.0)).status_code == 200
        except httpx.HTTPError: return False

    async def start(self, cmd: list = None, env: dict = None, sig=None) -> bool:
        """Starts and waits for readiness, up to start_timeout_s. False, with the reason in status and the log, when it exits or never answers. The caller decides when: looking at a service must not lead here."""
        async with self.lock:
            if self.ready() and (sig is None or sig == self.sig): return True
            if self.alive(): await self._stop_process("restarting with a new configuration")
            self.closing, self.status, self.detail, self.sig = False, "starting", "", sig
            t0 = time.time()
            self.port = int(self.spec.get("port") or free_port())
            cmd = [str(a).replace("{port}", str(self.port)) for a in (cmd or self.spec["cmd"])]
            self.spawn(cmd, {**self.spec.get("env", {}), **(env or {})}, self.spec.get("cwd") or None)
            print(f"""[SERVICE] starting {self.name}: {" ".join(cmd)}""", flush=True)
            self._ev("start", cmd=cmd, sig=sig)
            deadline = time.time() + float(self.spec.get("start_timeout_s", 1800))
            while self.proc.poll() is None and time.time() < deadline:
                if await self.probe():
                    self.status, self.last_use, self.t_up = "up", time.time(), time.time()
                    print(f"[SERVICE] {self.name} up in {time.time() - t0:.1f}s", flush=True)
                    self._ev("up", pid=self.proc.pid if self.proc else None, seconds=round(time.time() - t0, 2))
                    return True
                await asyncio.sleep(float(self.spec.get("probe_s", 2)))
            self.status = f"failed (exit {self.proc.returncode})" if self.proc and self.proc.poll() is not None else f"""failed (no answer in {self.spec.get("start_timeout_s", 1800)} s)"""
            print(f"[SERVICE] {self.name} {self.status} - see {self.log}", flush=True)
            self._ev("failed", status=self.status, tail=self.tail(8))
            if self.alive(): await self._kill_group()
            budget.release(self.name)
            return False

    # --- Stopping ---

    def _signal(self, sig):
        try: os.killpg(self.proc.pid, sig)
        except ProcessLookupError: pass

    async def _kill_group(self):
        self._signal(signal.SIGKILL)
        await asyncio.to_thread(self.proc.wait)

    async def _stop_process(self, why: str):
        self._signal(signal.SIGTERM)
        try: await asyncio.to_thread(self.proc.wait, float(self.spec.get("stop_timeout_s", 30)))
        except subprocess.TimeoutExpired: await self._kill_group()
        self.exits.append({"t": journal.now(), "why": why, "code": self.proc.returncode})

    async def before_stop(self): pass   # a chance to save state (the llama module saves its KV context here); kill() does not call it

    async def stop(self, why: str = ""):
        """A polite stop: state saved first, then TERM, then KILL after stop_timeout_s. The scheduler only calls this on a service with no running work."""
        async with self.lock:
            if self.alive():
                print(f"""[SERVICE] stopping {self.name}{" - " + why if why else ""}""", flush=True)
                await self.before_stop()
                await self._stop_process(why)
            self._ev("stop", why=why)
            self.status, self.closing, self.sig = "stopped", False, None
            budget.release(self.name)

    async def kill(self, why: str = "killed"):
        """At once, without asking the program: SIGKILL to the whole group. Works when the program is wedged, and does not wait for this service's lock, since a wedged start or stop may be holding it."""
        if self.alive():
            await self._kill_group()
            self.exits.append({"t": journal.now(), "why": why, "code": self.proc.returncode})
        print(f"[SERVICE] {self.name} killed ({why})", flush=True)
        self._ev("stop", why=why, killed=True)
        self.status, self.closing, self.sig = "stopped", False, None
        budget.release(self.name)

    def reap(self):
        """Called on every scheduler pass: a process that died by itself is noticed here, whether or not anything was asked of it, and its memory comes off the account."""
        if not self.running(): self.closing = False
        if self.status == "up" and self.proc and self.proc.poll() is not None: self.on_exit()

    def on_exit(self):
        code = self.proc.returncode
        print(f"[SERVICE] {self.name} exited by itself (code {code}) - see {self.log}", flush=True)
        self.exits.append({"t": journal.now(), "why": "exited by itself", "code": code})
        self._ev("exit", code=code, tail=self.tail(8))
        self.status, self.sig = f"exited ({code})", None
        budget.release(self.name)

    def measure(self, gpu_gb: float = 0.0):
        """Replaces this service's estimate with what it was found to hold."""
        budget.measured(self.name, self.rss_gb(), gpu_gb)
