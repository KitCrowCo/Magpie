"""core/modules.py - what this node can do is whatever modules are active. A module is a folder: module.json (what it provides, its drivers, its settings and which of them must be decided before install, data-only adapters) and optionally module.py (setup(ctx), for anything data cannot say). None is required: with nothing active the node still reports what it is and can install anything that fits.
Each module is in one state, and a manager can see all of them before deciding anything:
  blocked      no driver fits this machine (why, and what would solve it)
  installable  a driver fits; "decide" lists what must be chosen first; POST /modules/<name>/install starts it from afar
  installed    ready, not active
  active       its capabilities are on offer
modules/ ships with the code; node.json "module_dirs" adds more. "always": true marks modules with nothing to install (the node's own tools).
The capability registry answers "who can do this, for this model" - the only question the portal needs answered."""
import sys, json, time, inspect, importlib
from pathlib import Path
from core import settings, hardware, journal, jobs, drivers, services
from core.adapters import WorkerAdapter

ALL: dict = {}        # name -> {"dir", "spec", "active", "error"}
ADAPTERS: dict = {}   # adapter name -> Adapter
ASSET_KINDS: dict = {}   # kind -> what its module registered (Ctx.add_asset_kind)
WATCHERS: list = []   # fn() after each activation
_CACHE = {"t": 0.0, "models": []}
_APP: dict = {"app": None}

def _active_path() -> Path: return settings.STATE / "active_modules.json"

class Ctx:
    """What a module's setup() gets."""
    settings, hardware, journal = settings, hardware, journal
    def __init__(self, name: str, folder: Path): self.name, self.dir = name, folder
    def cfg(self, path: str = "", default=None): return settings.get(f"{self.name}.{path}" if path else self.name, default)
    def add_adapter(self, a): ADAPTERS[a.name] = a
    def add_asset_kind(self, kind: str, spec: dict): ASSET_KINDS[kind] = {**spec, "module": self.name}   # a kind of thing this module lets a manager make, change and remove here (core/api.py /assets): {"label", "about", "fields": [what one is made of], "list", "set", "remove": async fn(request, name)}
    def add_router(self, r):
        if _APP["app"]: _APP["app"].router.routes[0:0] = r.routes
    def bin(self, name: str) -> str: return drivers.state(self.name).get("provides", {}).get(name, "")

def discover():
    for root in [settings.HOME / "modules", *[Path(p) for p in settings.owner("module_dirs", [])]]:
        for f in sorted(root.glob("*/module.json")):
            spec = json.loads(f.read_text())
            ALL[spec.get("name", f.parent.name)] = {"dir": f.parent, "spec": spec, "active": False, "error": ""}
    for name, m in ALL.items(): settings.add_defaults(name, m["spec"].get("defaults", {}))

def wanted() -> list:
    """What should be active: what needs nothing, what the owner lists, and what was activated here before."""
    return list(dict.fromkeys([*(n for n, m in ALL.items() if m["spec"].get("always")), *settings.owner("modules", []), *(json.loads(_active_path().read_text()) if _active_path().exists() else [])]))

def _ready(name: str) -> bool: return not ALL[name]["spec"].get("drivers") or bool(drivers.state(name)) or drivers.adopt(name, ALL[name]["spec"])

def state_of(name: str) -> str:
    m = ALL[name]
    if m["active"]: return "active"
    if _ready(name): return "installed"
    return "installable" if drivers.choose(m["spec"]) else "blocked"

def describe(name: str) -> dict:
    m, st = ALL[name], drivers.state(name)
    d = {"name": name, "label": m["spec"].get("label", name), "state": state_of(name), "provides": m["spec"].get("provides", []), "needs": m["spec"].get("needs", []), "settings": {k: {**v, "value": settings.get(f"{name}.{k}")} for k, v in m["spec"].get("settings", {}).items()}, "error": m["error"], "credits": st.get("credits") or m["spec"].get("credits")}
    if st: d["installed"] = {k: st.get(k) for k in ("driver", "options", "t")}
    if m["spec"].get("drivers"): d["plan"] = drivers.plan(name, m["spec"])
    return d

def view() -> list: return [describe(n) for n in ALL]

def activate(name: str) -> dict:
    """Loads one module and what it needs first. A module not yet installed answers why, and what installing it would take: nothing half-loaded."""
    if name not in ALL: raise ValueError(f"no module {name!r} - known: {sorted(ALL)}")
    m = ALL[name]
    if m["active"]: return describe(name)
    if not _ready(name): raise ValueError(f"{name} is {state_of(name)}: POST /modules/{name}/install first")
    for dep in m["spec"].get("needs", []): activate(dep)
    for key, a in (settings.get(f"{name}.adapters", {}) or {}).items():
        if isinstance(a, dict) and a.get("enabled", True): ADAPTERS[key] = WorkerAdapter(key, name)
    if (m["dir"] / "module.py").exists():
        if str(settings.HOME) not in sys.path: sys.path.insert(0, str(settings.HOME))
        r = importlib.import_module(f"""modules.{m["dir"].name}.module""").setup(Ctx(name, m["dir"]))
        if inspect.isawaitable(r): jobs.spawn(r)   # a module's slower startup work runs beside the node's own
    m["active"], m["error"] = True, ""
    journal.event("module.active", module=name)
    for fn in WATCHERS: fn()
    print(f"[MODULE] {name} active", flush=True)
    return describe(name)

def remember_active(name: str, on: bool = True):
    cur = set(json.loads(_active_path().read_text()) if _active_path().exists() else [])
    _active_path().write_text(json.dumps(sorted(cur | {name} if on else cur - {name})))

async def deactivate(name: str) -> dict:
    """Its capabilities stop being offered and its services stop, now. Its files stay."""
    for a in [a for a in ADAPTERS.values() if getattr(a, "owner", None) == name]: ADAPTERS.pop(a.name)
    for k in [k for k, v in ASSET_KINDS.items() if v["module"] == name]: ASSET_KINDS.pop(k)   # what it let a manager make is no longer offered either
    for s in [s for s in services.REG.values() if s.owner == name and s.running()]: await s.stop(f"module {name} deactivated")
    ALL[name]["active"] = False
    remember_active(name, False)
    journal.event("module.inactive", module=name)
    return describe(name)

# --- Capabilities and models ---

def capabilities() -> dict:
    """{capability: [adapter names]}, for active adapters. This is what a portal connection learns the node can do now."""
    out = {}
    for a in ADAPTERS.values():
        for c in [*a.capabilities, *a.spec.get("info", {}), *a.spec.get("actions", {})]: out.setdefault(c, []).append(a.name)
    return out

def could() -> dict:
    """{capability: [{"module", "state", "decide"}]} for what is not on offer yet: what this node could do if a manager asks, and what that would take."""
    on, out = capabilities(), {}
    for n, m in ALL.items():
        if m["active"]: continue
        st = state_of(n)
        for c in m["spec"].get("provides", []):
            if c not in on: out.setdefault(c, []).append({"module": n, "state": st, **({"decide": drivers.choose(m["spec"]) and drivers.judge(drivers.choose(m["spec"]))["decide"]} if st == "installable" else {})})
    return out

def models(capability: str = "", refresh: bool = False) -> list:
    """Every model every active adapter can serve, one list. Reads files only. Briefly cached: a listing is often asked for many times per page."""
    if refresh or time.time() - _CACHE["t"] > 5: _CACHE.update(t=time.time(), models=[{"modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(m.get("mtime", 0))), "details": {"format": m.get("format", "")}, **m, "adapter": a.name} for a in ADAPTERS.values() for m in a.models()])
    return [m for m in _CACHE["models"] if not capability or capability in m.get("capabilities", [])]

def resolve(capability: str, model: str = "", adapter: str = ""):
    """The adapter for one request: named outright; else the one holding this model; else node policy (defaults.<capability>); else the only or first one able."""
    able = [a for a in ADAPTERS.values() if capability in a.capabilities]
    if not able: return None
    if adapter: return next((a for a in able if a.name == adapter), None)
    if model:
        tag = lambda n: n if ":" in n else f"{n}:latest"
        hit = next((m for m in models(capability) if tag(m["name"]) == tag(model)), None) or next((m for m in models(capability, refresh=True) if tag(m["name"]) == tag(model)), None)
        if hit: return ADAPTERS[hit["adapter"]]
    return next((a for a in able if a.name == settings.get(f"defaults.{capability}")), None) or (able[0] if not model or len(able) == 1 else None)
