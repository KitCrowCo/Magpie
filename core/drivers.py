"""core/drivers.py - one install does everything. A module does not name its dependencies; it lists drivers, each a way of running it, best first. This node finds the first driver its hardware allows and installs only what that driver needs, each in an environment of its own, so nothing one module needs can break another.
"Best" means: runs well here, with the fewest added dependencies. A program already on the machine is a driver that needs nothing. Open standards come first (Vulkan, CPU, Metal on a Mac); a driver for anything else is one more data block in a module.json, by whoever wants it.
A driver: {"name", "detect": {"bins", "files"} (present = usable as it is), "requires" (core/hardware.unmet: what no install can fix), "programs": [needed to build], "files": [headers and the like], "system": {manager: [packages that provide those]}, "options": {name: {"default", "hint", "choices"}} (no default = someone must decide before install), "install": [steps], "provides": {name: path}, "env", "credits": {"project", "url", "license"}}.
Steps: {"pip": [...], "index", "when"} into this module's venv; {"git", "ref", "to", "build": [argv...]}; {"run": argv, "cwd", "when"}; {"each": option, "run": argv}; {"copy": path, "to"}; {"line": text, "file"} (added once); {"fetch": url, "to", "when"}.
Placeholders: {src} {env} {python} {dir} {jobs} {cpu_level} {gfx} {node_port} {local_key} {opt:name} {loc:name} {bin:name} {link:name}. Changing the machine itself (system packages) needs the owner's yes: setup.allow_system in node.json, the --allow-system flag, or running in a container or on Termux, where the machine is this node's."""
import collections, os, re, sys, json, time, shutil, asyncio
from pathlib import Path
from core import settings, hardware, journal

def _rec(name: str) -> Path:
    d = settings.STATE / "drivers"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.json"

def state(name: str) -> dict:
    """What is installed for this module: {"driver", "options", "provides", "python", "t"}. The owner can say so outright, for something built by hand: "<module>": {"provides": {...}} in node.json."""
    given = settings.owner(f"{name}.provides")
    if given: return {"driver": "given", "options": {}, "provides": given, "python": settings.owner(f"{name}.python") or sys.executable}
    return json.loads(_rec(name).read_text()) if _rec(name).exists() else {}

def cpu_level() -> str:
    """The highest CPU level a build can target here."""
    f = hardware.facts()
    if f["arch"] not in ("x86_64", "AMD64"): return "native"
    return "x86-64-v4" if "avx512f" in f["cpu_features"] else "x86-64-v3" if "avx2" in f["cpu_features"] else "x86-64-v2"

def jobs_for_build() -> int: return max(1, min(hardware.facts()["threads"] or 1, int(hardware.facts()["ram_gb"] // 3)))   # a compile job can take 2-3 GB: more jobs than that runs a small node out of memory

def fill(value, module: str, drv: dict = None, opts: dict = None, item: str = None):
    """Placeholders in any string of a nested value, for this module's installed driver. {port} is left for the service to fill."""
    if isinstance(value, dict): return {k: fill(v, module, drv, opts, item) for k, v in value.items()}
    if isinstance(value, list): return [fill(v, module, drv, opts, item) for v in value]
    if not isinstance(value, str): return value
    st, env = state(module), settings.STATE / "envs" / module
    o = {**(st.get("options") or {}), **(opts or {})}
    prov = {**(st.get("provides") or {})}
    gfx = (hardware.facts().get("gfx") or [""])[0]
    plain = {"src": str(settings.STATE / "src"), "env": str(env), "python": st.get("python") or (str(env / "bin" / "python") if env.exists() else sys.executable), "dir": str(_mdir(module)), "jobs": str(jobs_for_build()), "cpu_level": cpu_level(), "gfx": gfx, "item": item or "", "node_port": str(settings.owner("bind.port", settings.PORT)), "local_key": _local_key()}
    def sub(m):
        kind, name = m.group(1), m.group(2)
        if kind is None: return plain.get(name, m.group(0))
        if kind == "opt": return str(o.get(name, m.group(0)))
        if kind == "loc": return str(settings.location(name))
        if kind == "link": return str(_link(name))
        if kind == "bin": return str(fill(prov.get(name, m.group(0)), module, drv, opts) if name in prov else m.group(0))
        return m.group(0)
    return re.sub(r"\{(?:(opt|loc|bin|link):)?(\w+)\}", sub, value)

def _local_key() -> str:
    from core import auth
    return auth.LOCAL_KEY

def _link(name: str) -> int:
    from modules.relay import module as relay
    return relay.LINKS[name]["local_port"]

def _mdir(module: str) -> Path:
    from core import modules
    return modules.ALL[module]["dir"] if module in modules.ALL else settings.HOME

# --- What fits this machine ---

def manager() -> str:
    """The system package manager here. pkg is Termux's."""
    if "com.termux" in os.environ.get("PREFIX", ""): return "pkg"
    return next((m for m in ("apt-get", "dnf", "pacman", "apk", "brew") if shutil.which(m)), "")

def system_allowed() -> bool: return bool(settings.owner("setup.allow_system")) or hardware.facts()["container"] or hardware.facts()["phone"]

def missing(d: dict) -> list: return [p for p in d.get("programs", []) if not shutil.which(p)] + [f for f in d.get("files", []) if not Path(f).exists()]

def detected(d: dict) -> bool:
    det = d.get("detect")
    return bool(det) and all(shutil.which(b) for b in det.get("bins", [])) and all(Path(f).exists() for f in det.get("files", []))

def judge(d: dict) -> dict:
    """One driver on this machine: usable, what blocks it, what could fix that. Hardware limits and missing programs are told apart, since only one of them an install can solve."""
    blocks = hardware.unmet(d.get("requires", {}))
    if d.get("detect") and not detected(d): blocks = [*blocks, {"block": "not present on this machine", "resolve": ""}]
    miss, pkgs = missing(d), (d.get("system") or {}).get(manager(), [])
    if miss and not pkgs: blocks = [*blocks, {"block": f"""needs {miss} and has no package list for {manager() or 'this system'}""", "resolve": f"install {miss} by hand"}]
    return {"driver": d["name"], "ok": not blocks and (not miss or system_allowed()), "blocks": blocks, "needs_system": {"missing": miss, "packages": pkgs, "manager": manager(), "allowed": system_allowed()} if miss else None, "decide": [k for k, v in d.get("options", {}).items() if "default" not in v], "options": d.get("options", {}), "credits": d.get("credits")}

def choose(spec: dict, wanted: str = "") -> dict:
    """The driver to use: the one asked for; else the first this machine allows as it stands; else the first it could allow, given the owner's yes for system packages."""
    ds = spec.get("drivers", [])
    if wanted: return next((d for d in ds if d["name"] == wanted), None)
    return next((d for d in ds if judge(d)["ok"]), None) or next((d for d in ds if not judge(d)["blocks"]), None)

def plan(name: str, spec: dict) -> dict:
    """What installing this module here would mean, without doing any of it: what a manager sees before deciding. better: drivers ahead of the pick that lack only system packages the owner has not allowed."""
    ds, pick = spec.get("drivers", []), choose(spec)
    ahead = ds[:ds.index(pick)] if pick else []
    return {"driver": pick["name"] if pick else None, "needs_install": bool(pick and "install" in pick), "decide": judge(pick)["decide"] if pick else [], "needs_system": judge(pick)["needs_system"] if pick else None, "options": pick.get("options", {}) if pick else {}, "better": [j for j in map(judge, ahead) if not j["blocks"]], "drivers": [judge(d) for d in ds], "recommended": not hardware.unmet(spec.get("recommend", {}))}

# --- Installing ---

def when(step: dict, opts: dict) -> bool:
    """A step's or capability's condition on the install options: {"opt", "has" (one of a comma list), "is", or "isnt"}."""
    w = step.get("when")
    if not w: return True
    v = str(opts.get(w["opt"], ""))
    return w["has"] in v.split(",") if "has" in w else v != str(w["isnt"]) if "isnt" in w else v == str(w.get("is"))

async def _run(argv: list, cwd=None, env: dict = None):
    proc = await asyncio.create_subprocess_exec(*map(str, argv), cwd=cwd, env={**os.environ, "HF_HUB_DISABLE_TELEMETRY": "1", **(env or {})}, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    tail, first = collections.deque(maxlen=4), None    # first: the first line that says error, and the two after it
    async for line in proc.stdout:
        s = line.decode(errors="replace").rstrip()[:300]
        if s.strip(): tail.append(s.strip())
        if first is None and re.search(r"\berror\b", s, re.I): first = []
        if first is not None and len(first) < 3 and s.strip(): first.append(s.strip())
        yield {"line": s}
    if await proc.wait(): raise RuntimeError(f"""exit {proc.returncode}: {" ".join(map(str, argv))[:200]} | {" | ".join([*(first or []), *[x for x in tail if x not in (first or [])]])}""")

async def _system(d: dict):
    """Brings in what the driver needs from the system, if anything is missing and the owner allows it."""
    miss = missing(d)
    if not miss: return
    m, pkgs = manager(), (d.get("system") or {}).get(manager(), [])
    if not system_allowed(): raise RuntimeError(f"""needs {miss}: run {m} install {" ".join(pkgs)} (or allow it: setup.allow_system in node.json)""")
    sudo = [] if m == "pkg" or m == "brew" or os.geteuid() == 0 else ["sudo", "-n"]
    if m == "apt-get":
        async for p in _run([*sudo, m, "update", "-q"]): yield p
    async for p in _run([*sudo, m, *({"pacman": ["-S", "--noconfirm"], "apk": ["add"]}.get(m, ["install", "-y"] if m != "brew" else ["install"])), *pkgs], env={"DEBIAN_FRONTEND": "noninteractive"}): yield p
    if still := missing(d): raise RuntimeError(f"still missing after installing {pkgs}: {still}")

async def _step(step: dict, name: str, d: dict, opts: dict):
    f = lambda v, item=None: fill(v, name, d, opts, item)
    env_dir = settings.STATE / "envs" / name
    if "pip" in step:
        if not env_dir.exists():
            yield {"status": f"making environment {env_dir.name}"}
            async for p in _run([sys.executable, "-m", "venv", env_dir]): yield p
        yield {"status": f"""pip: {" ".join(step["pip"])[:200]}"""}
        async for p in _run([env_dir / "bin" / "pip", "install", "--no-cache-dir", *(["--index-url", f(step["index"])] if step.get("index") else []), *f(step["pip"])]): yield p
    if "git" in step:
        to = settings.STATE / "src" / step["to"]
        to.parent.mkdir(parents=True, exist_ok=True)
        if not to.exists():
            yield {"status": f"""fetching {step["git"]}"""}
            async for p in _run(["git", "clone", "--depth", "1", *(["--branch", f(step["ref"])] if step.get("ref") and step["ref"] != "master" else []), step["git"], to]): yield p
        for b in step.get("build", []):
            yield {"status": " ".join(map(str, f(b)))[:200]}
            async for p in _run(f(b), cwd=to): yield p
    if "each" in step:
        for item in str(opts.get(step["each"], "")).replace(",", " ").split():
            async for p in _run(f(step["run"], item), cwd=f(step.get("cwd"))): yield p
    elif "run" in step:
        yield {"status": " ".join(map(str, f(step["run"])))[:200]}
        async for p in _run(f(step["run"]), cwd=f(step.get("cwd") or "{dir}")): yield p
    if "copy" in step:
        dst = Path(f(step["to"]))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f(step["copy"]), dst)
    if "line" in step:
        p = Path(f(step["file"]))
        if step["line"] not in p.read_text(): p.write_text(p.read_text().rstrip("\n") + "\n" + step["line"] + "\n")
    if "fetch" in step:
        from core import files
        yield {"status": f"""fetching {step["fetch"]}"""}
        async for p in files.pull_file(f(step["fetch"]), Path(f(step["to"]))): yield p

async def install(name: str, spec: dict, wanted: str = "", options: dict = None):
    """Installs a driver for this module and records it. Yields progress lines, ending with status success or error. Nothing is recorded until every step worked, so a failed install leaves the module as it was."""
    d = choose(spec, wanted)
    try:
        if not d: raise RuntimeError(f"""no driver for {name} fits this machine: {[(j["driver"], [b["block"] for b in j["blocks"]]) for j in map(judge, spec.get("drivers", []))]}""")
        j = judge(d)
        if j["blocks"]: raise RuntimeError(f"""driver {d["name"]}: {"; ".join(b["block"] for b in j["blocks"])}""")
        opts = {**{k: v["default"] for k, v in d.get("options", {}).items() if "default" in v}, **(options or {})}
        if und := [k for k in d.get("options", {}) if k not in opts]: raise RuntimeError(f"needs a decision before install: {und}")
        opts = {k: fill(v, name, d, {}) if isinstance(v, str) else v for k, v in opts.items()}
        yield {"status": f"""driver {d["name"]} for {name}""", "options": opts}
        async for p in _system(d): yield p
        for step in d.get("install", []):
            if when(step, opts):
                async for p in _step(step, name, d, opts): yield p
        env_dir = settings.STATE / "envs" / name
        rec = {"driver": d["name"], "options": opts, "provides": fill(d.get("provides", {}), name, d, opts), "env": d.get("env", {}), "python": str(env_dir / "bin" / "python") if env_dir.exists() else sys.executable, "t": round(time.time()), "credits": d.get("credits")}
        _rec(name).write_text(json.dumps(rec, indent=1))
        hardware.facts(refresh=True)
        journal.event("module.install", module=name, driver=d["name"], ok=True, options=opts)
        yield {"status": "success", "driver": d["name"], "done": True}
    except Exception as e:
        journal.event("module.install", module=name, ok=False, error=str(e)[:400])
        yield {"status": f"error: {e}", "done": True}

def adopt(name: str, spec: dict) -> bool:
    """A driver with nothing to install (a program already here, or nothing needed) is simply recorded."""
    d = choose(spec)
    if not d or d.get("install"): return False
    _rec(name).write_text(json.dumps({"driver": d["name"], "options": {}, "provides": d.get("provides", {}), "env": d.get("env", {}), "python": sys.executable, "t": round(time.time()), "credits": d.get("credits")}, indent=1))
    journal.event("module.install", module=name, driver=d["name"], ok=True, adopted=True)
    return True

def remove(name: str):
    """Takes away what an install added for this module alone: its record and environment. Source trees stay, as another driver may reuse them; models never go with it."""
    _rec(name).unlink(missing_ok=True)
    shutil.rmtree(settings.STATE / "envs" / name, ignore_errors=True)
