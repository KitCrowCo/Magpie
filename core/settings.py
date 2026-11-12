"""core/settings.py - where things live, and the layered settings every part reads.
Layers, weakest first: module defaults (module.json "defaults") < manager rules (written by a managing portal through PUT /policy) < the node owner's node.json. The node owner always wins, and node.json "manager_may" lists which top-level keys a manager may set at all.
Every manager write keeps the version before it (state/manager/<n>.json), so any change can be rolled back; nothing a manager sets can touch auth. On a node with an empty node.json, everything but auth and bind is the manager's to set."""
import os, re, json, time, copy
from pathlib import Path

HOME = Path(__file__).resolve().parent.parent
os.environ.setdefault("MAGPIE_HOME", str(HOME))
DATA = Path(os.getenv("MAGPIE_DATA") or HOME / "data")      # where everything lives unless a manager moves it (locations)
STATE = Path(os.getenv("MAGPIE_STATE") or DATA / "state")   # the node's own records: keys, rules, what is installed. Not movable while running
NODE_FILE = Path(os.getenv("MAGPIE_NODE_FILE") or HOME / "node.json")
PORT, HOST = int(os.getenv("SERVER_PORT", 8790)), os.getenv("SERVER_HOST", "0.0.0.0")
NODE_ID = os.getenv("NODE_ID") or os.uname().nodename
NEVER_MANAGED = ("auth", "manager_may", "bind")   # the node owner's alone, whatever manager_may says
_DEFAULTS: dict = {}
_OWNER: dict = {}
_MANAGER: dict = {"version": 0, "rules": {}}
_MERGED: dict = {}
_ENV = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")

def merge(a: dict, b: dict) -> dict:
    """b over a, recursively for dicts; keys starting with _ are notes for people and are dropped."""
    out = dict(a)
    for k, v in b.items():
        if k.startswith("_"): continue
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out

def expand(v):
    """${NAME} and ${NAME:-default} filled from the environment, in any string of a nested value. An unset name with no default stays as written, so a caller can see what is missing (missing_env)."""
    if isinstance(v, dict): return {k: expand(x) for k, x in v.items()}
    if isinstance(v, list): return [expand(x) for x in v]
    if isinstance(v, str): return _ENV.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) if m.group(2) is not None else m.group(0)), v)
    return v

def missing_env(v) -> list: return sorted({m[0] for m in _ENV.findall(json.dumps(v))})

def _rebuild():
    _MERGED.clear()
    _MERGED.update(merge(merge(_DEFAULTS, _MANAGER["rules"]), _OWNER))

def load():
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "manager").mkdir(exist_ok=True)
    _OWNER.clear()
    if NODE_FILE.exists(): _OWNER.update(expand(json.loads(NODE_FILE.read_text())))
    cur = STATE / "manager" / "current.json"
    if cur.exists(): _MANAGER.update(json.loads(cur.read_text()))
    _rebuild()

def add_defaults(name: str, d: dict):
    """A module's shipped defaults, under its own name."""
    _DEFAULTS[name] = merge(_DEFAULTS.get(name, {}), expand(d))
    _rebuild()

# --- Locations: named places for files, none of them fixed at install ---

LOCATIONS = {"models": "models", "loras": "models/loras", "cache": "cache", "outputs": "outputs", "logs": "logs", "vault": "vault", "image_models": "models/image", "audio_models": "models/audio", "knowledge": "knowledge"}

def locations() -> dict:
    """{name: {"path", "kind"}}: the defaults under DATA, then what a manager or the owner set under "locations". kind long: nothing changes files here unless told to. kind scratch: shared and transient, cleared after its age limit."""
    out = {n: {"path": str(DATA / p), "kind": "long"} for n, p in LOCATIONS.items()}
    for n, v in (get("locations", {}) or {}).items(): out[n] = {**out.get(n, {"kind": "long", "path": str(DATA / "locations" / n)}), **(v if isinstance(v, dict) else {"path": v})}
    for v in out.values(): v["path"] = os.path.expanduser(v["path"])   # "~/..." means the home of the user Magpie runs as, so one node.json fits every machine
    return out

def location(name: str) -> Path: return Path(locations()[name]["path"])

def get(path: str, default=None):
    """A dotted path through the merged settings: get("scheduler.max_hold_s", 900)."""
    node = _MERGED
    for k in path.split("."):
        if not isinstance(node, dict) or k not in node: return default
        node = node[k]
    return node

def owner(path: str, default=None):
    node = _OWNER
    for k in path.split("."):
        if not isinstance(node, dict) or k not in node: return default
        node = node[k]
    return node

# --- Manager rules ---

def manager_view() -> dict: return {"version": _MANAGER["version"], "rules": _MANAGER["rules"], "may_set": owner("manager_may", ["*"]), "never": list(NEVER_MANAGED), "owner_keys": sorted(k for k in _OWNER if not k.startswith("_")), "history": sorted(int(f.stem) for f in (STATE / "manager").glob("[0-9]*.json"))}

def _refused(rules: dict) -> list:
    may = owner("manager_may", ["*"])
    return [k for k in rules if k in NEVER_MANAGED or not ("*" in may or k in may)]

def set_manager(rules: dict, source: str = "") -> dict:
    """Replaces the manager layer whole, so the portal's view and the node's are never partly merged. The version it replaces is kept. Raises ValueError for keys this node does not let a manager set."""
    if bad := _refused(rules): raise ValueError(f"this node does not let a manager set: {bad}")
    d = STATE / "manager"
    (d / f"""{_MANAGER["version"]}.json""").write_text(json.dumps(_MANAGER, indent=1))
    _MANAGER.update(version=_MANAGER["version"] + 1, rules=rules, t=time.time(), source=source)
    (d / "current.json").write_text(json.dumps(_MANAGER, indent=1))
    _rebuild()
    return manager_view()

def set_path(path: str, value, source: str = "") -> dict:
    """One value in the manager layer, by dotted path, with everything else kept. None removes it."""
    rules, node, *keys = copy.deepcopy(_MANAGER["rules"]), None, *path.split(".")
    node = rules
    for k in keys[:-1]: node = node.setdefault(k, {})
    if value is None: node.pop(keys[-1], None)
    else: node[keys[-1]] = value
    return set_manager(rules, source or f"set {path}")

def rollback(version: int) -> dict:
    p = STATE / "manager" / f"{int(version)}.json"
    if not p.exists(): raise ValueError(f"no saved manager rules version {version}")
    return set_manager(json.loads(p.read_text())["rules"], source=f"rollback to {version}")
