"""core/plan.py - what it would take to do something here, and what can be adjusted: both answered without starting anything.
make(): for one capability and model - what it needs (its adapter's plan), what this node has and who holds the rest, and the ways to make it fit. A way is either something the scheduler does by itself or a set of knob values a manager applies (PATCH /settings); none is applied here.
knobs(): every adjustable value as one typed list, from each module's own declaration ("settings" in its module.json) and the node's own (CORE). A knob nobody set is on auto: the node decides, and says what auto comes to when it can.
A portal draws both with one generic panel and holds no program's setting names: an adapter adds a knob, or a way, by adding data."""
import types
from core import settings, budget, services, modules, hardware

CORE = {   # the node's own knobs, in the same shape a module declares its settings
    "scheduler.alone": {"type": "list", "label": "Run alone", "when": "now", "hint": "service names (globs, e.g. colibri:*) that run by themselves: everything else on this node stops while one is up, and it stops when other work needs the node"},
    "scheduler.max_hold_s": {"type": "number", "label": "Longest wait behind loaded work (s)", "when": "now", "default": 900, "min": 0, "hint": "after this, the oldest waiting job wins: what stands in its way finishes its running work and stops"},
    "scheduler.wait_for_memory": {"type": "boolean", "label": "Wait when memory is short", "when": "now", "default": False, "hint": "off: a job that cannot fit with everything stoppable stopped fails at once and says by how much"},
    "jobs.kept_days": {"type": "number", "label": "Days an unfetched answer is kept", "when": "start", "default": 7, "min": 0, "hint": "answers of work that went on without its caller, removed at start when older than this"},
    "resources.reserve_gb": {"type": "number", "label": "Memory kept for the system (GB)", "when": "now", "default": 2.5, "min": 0, "hint": "never promised to any service"},
}

def _dig(node, path: str):
    for k in path.split("."):
        if not isinstance(node, dict) or k not in node: return None
        node = node[k]
    return node

# --- What the node has ---

def have(exclude: str = "") -> dict:
    """Memory now, who holds what, and what stopping would free: idle services the scheduler may stop for memory by itself, and everything else (running alone)."""
    rows = [{"service": s.name, "owner": s.owner, "ram_gb": round((budget.LEASES.get(s.name) or {}).get("ram_gb", s.rss_gb()), 2), "measured": bool((budget.LEASES.get(s.name) or {}).get("measured")), "kept": bool((budget.LEASES.get(s.name) or {}).get("keep")), "working": s.active > 0 or s.status != "up", "model": s.meta.get("model", "")} for s in services.running() if s.name != exclude]
    free = budget.free_gb()
    return {"ram_total_gb": hardware.facts()["ram_gb"], "ram_avail_gb": round(hardware.mem_avail_gb(), 1), "reserve_gb": budget.reserve_gb(), "free_gb": round(free, 1), "free_if_idle_stopped_gb": round(free + sum(r["ram_gb"] for r in rows if not r["kept"] and not r["working"]), 1), "free_if_alone_gb": round(free + sum(r["ram_gb"] for r in rows), 1), "holders": rows}

# --- A plan for one piece of work ---

def _alone_set(a, model: str, service: str) -> dict:
    """The knob that makes this run alone: the program's own per-model setting when its module declares one, else this service's name added to scheduler.alone."""
    if a.owner and "alone" in modules.ALL[a.owner]["spec"].get("settings", {}): return {f"{a.owner}.models.{model}.alone": True}
    return {"scheduler.alone": sorted({*(settings.get("scheduler.alone", []) or []), service})}

def make(capability: str, model: str = "", want: dict = None, adapter: str = "") -> dict:
    """{"fits": "yes" | "yes_if" | "no", "needs", "have", "ways", "will"}. LookupError when nothing here does this."""
    a = modules.resolve(capability, model, adapter)
    if not a: raise LookupError(f"no model {model!r} for {capability} on this node" if modules.capabilities().get(capability) else f"nothing active on this node does {capability}")
    if not a.needs_service(types.SimpleNamespace(capability=capability, model=model, body=want or {}, session="default", record={})): return {"fits": "yes", "running": True, "needs": {"service": "", "ram_gb": 0.0, "gpu_gb": 0.0, "alone": False, "keep": False, "source": "estimated", "elsewhere": True}, "have": have(), "ways": [], "will": "Nothing starts on this node for it: the work is done where the model is."}
    p = a.plan(capability, model, {**(want or {}), "_room_gb": have()["free_if_alone_gb"]})
    h, sv = have(p["service"]), services.get(p["service"])
    if sv and sv.ready(): return {"fits": "yes", "running": True, "needs": p, "have": h, "ways": [], "will": f"""{p["service"]} is up: work for it starts at once."""}
    need, ok_now, others = float(p.get("min_ram_gb") or p["ram_gb"]), p.get("ok_now") is not False, ", ".join(r["service"] for r in h["holders"])
    idle, busy = [r["service"] for r in h["holders"] if not r["kept"] and not r["working"]], [r["service"] for r in h["holders"] if r["working"]]
    ways = []
    if not (p["alone"] and h["holders"]): ways.append({"id": "as_is", "label": "Start it as things are", "fits": p["ok_now"] if "ok_now" in p else need <= h["free_gb"], "auto": True, "set": {}})   # the program's own verdict for the node as it is, when it gave one
    if idle and not p["alone"]: ways.append({"id": "stop_idle", "label": f"""Stop idle services first ({", ".join(idle)})""", "fits": need <= h["free_if_idle_stopped_gb"], "auto": True, "frees_gb": round(h["free_if_idle_stopped_gb"] - h["free_gb"], 1), "set": {}, "cost": "each starts again when work needs it"})
    if h["holders"] or p["alone"]: ways.append({"id": "alone", "label": f"""Run alone: stop everything else while it is up ({others or "nothing else is running"})""", "fits": need <= h["free_if_alone_gb"], "auto": p["alone"], "frees_gb": round(h["free_if_alone_gb"] - h["free_gb"], 1), "set": {} if p["alone"] else _alone_set(a, model, p["service"]), "cost": "work for the stopped services waits until this one is idle" + (f"""; working now, so it waits for them: {", ".join(busy)}""" if busy else "")})
    less = round(h["reserve_gb"] - (need - h["free_if_alone_gb"]) - 0.1, 1)
    if not any(w["fits"] for w in ways) and less >= 0.5: ways.append({"id": "reserve", "label": f"""Keep less memory for the system: {h["reserve_gb"]} GB -> {less} GB""", "fits": True, "auto": False, "set": {"resources.reserve_gb": less}, "cost": "less room for everything this node does not manage; the machine swaps if that runs out"})
    ways += [{"id": f"way{i}", "auto": False, **w} for i, w in enumerate(p.get("ways", []))]
    auto, manual = next((w for w in ways if w["auto"] and w["fits"]), None), next((w for w in ways if not w["auto"] and w["fits"]), None)
    short, why = round(need - h["free_if_alone_gb"], 1), "" if ok_now else " (its own planner refuses the node as it is)"
    will = f"""At the next request this node will: {auto["label"]}.""" if auto else f"Does not fit as set{why}: apply one of the ways below." if manual else f"Does not fit as set{why}: short by about {short} GB even alone."
    return {"fits": "yes" if auto else "yes_if" if manual else "no", "running": False, "needs": p, "need_gb": round(need, 2), "have": h, "ways": ways, "will": will}

# --- Knobs ---

def _knob(path: str, name: str, d: dict, module: str, applies_to: str, inherit: str = "", auto: str = "") -> dict:
    """One adjustable value. value: what is set at exactly this path (None = auto). current: what is in effect. set_by: who set it; an owner's value cannot be changed from a portal."""
    by = "owner" if settings.owner(path) is not None else "manager" if _dig(settings.manager_view()["rules"], path) is not None else "auto"
    base = settings.get(inherit) if inherit else d.get("default", _dig(settings._DEFAULTS, path))
    return {"key": path, "name": name, "module": module, "applies_to": applies_to, "label": d.get("label") or name, "type": d.get("type") or {bool: "boolean", int: "integer", float: "number", list: "list", dict: "object"}.get(type(base), "text"), **{k: d[k] for k in ("choices", "min", "max", "pick") if k in d}, "value": settings.get(path) if by != "auto" else None, "current": settings.get(path, base), "inherits": base, "set_by": by, "auto": auto or d.get("auto", ""), "when": d.get("when", "runtime"), "effect": d.get("hint", "")}

def knobs(model: str = "", capability: str = "") -> list:
    """Without a model: the node's knobs and every active module's. With one: the knobs of that model alone (a module marks those "per": "asset"; they are kept under <module>.models.<model>), each falling back to the module-wide value."""
    if model:
        a = modules.resolve(capability or "chat", model) or next((modules.ADAPTERS[m["adapter"]] for m in modules.models() if m["name"] == model), None)
        if not a or not a.owner: return []
        autos = a.knob_auto(model)
        return [_knob(f"{a.owner}.models.{model}.{k}", k, d, a.owner, "asset", inherit=f"{a.owner}.{k}", auto=str(autos.get(k, ""))) for k, d in modules.ALL[a.owner]["spec"].get("settings", {}).items() if d.get("per") == "asset"]
    return [_knob(k, k, d, "node", "node") for k, d in CORE.items()] + [_knob(f"{n}.{k}", k, d, n, "module") for n, m in modules.ALL.items() if m["active"] for k, d in m["spec"].get("settings", {}).items()]
