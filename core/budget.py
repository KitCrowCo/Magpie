"""core/budget.py - the node's memory as one account shared by every service, instead of each engine assuming it owns the machine.
A lease is what a running or starting service holds: its estimate until it is up, what it was measured to hold after that. fits() says whether a new load can go ahead and, if not, which idle services would have to go first.
node.json "resources": {"mode": "full" | "carve", "reserve_gb": kept for the OS, "carve_gb": kept for things this node does not control}. Only "full" assumes everything else on the machine is Magpie's."""
import fnmatch
from core import settings, hardware

LEASES: dict = {}   # service name -> {"ram_gb", "gpu_gb", "keep", "measured", "why"}

def reserve_gb() -> float:
    r = settings.get("resources", {}) or {}
    return float(r.get("reserve_gb", 2.5)) + (float(r.get("carve_gb", 4)) if r.get("mode") == "carve" else 0.0)

def lease(name: str, ram_gb: float, gpu_gb: float = 0.0, keep: bool = False, why: str = ""): LEASES[name] = {"ram_gb": float(ram_gb), "gpu_gb": float(gpu_gb), "keep": keep, "measured": False, "why": why}

def measured(name: str, ram_gb: float, gpu_gb: float = 0.0):
    if name in LEASES: LEASES[name].update(ram_gb=float(ram_gb), gpu_gb=float(gpu_gb), measured=True)

def release(name: str): LEASES.pop(name, None)

def free_gb() -> float:
    """What can be promised right now: memory the machine says is available, less the reserve, less what starting services are about to take (a measured lease is already inside the machine's own number)."""
    return hardware.mem_avail_gb() - reserve_gb() - sum(l["ram_gb"] for l in LEASES.values() if not l["measured"])

def held_by_others(name: str) -> tuple:
    """(ram, gpu) leased by every service but this one: what a planner must plan around."""
    others = [l for n, l in LEASES.items() if n != name]
    return round(sum(l["ram_gb"] for l in others), 2), round(sum(l["gpu_gb"] for l in others), 2)

def alone(name: str, need: dict) -> bool:
    """Whether this service runs by itself, everything else stopped while it is up: its adapter says so (need["alone"], from that program's own settings), or its name matches one of scheduler.alone (globs)."""
    return bool(need.get("alone")) or any(fnmatch.fnmatch(name, p) for p in settings.get("scheduler.alone", []) or [])

def can_free(services: dict, name: str = "", alone: bool = False) -> float:
    """The most this service could be given: free now, plus what everything stoppable holds (everything, when it runs alone), plus what it holds itself. An estimate is capped at this, so a rough number never refuses a load the real load decides."""
    return round(free_gb() + sum(LEASES.get(s.name, {}).get("ram_gb", 0.0) for s in services.values() if s.running() and (alone or s.name == name or not LEASES.get(s.name, {}).get("keep"))) - 0.05, 2)

def fits(need_gb: float, services: dict, exclude: str = "", alone: bool = False) -> dict:
    """{"fits": bool, "evict": [idle service names to stop, oldest use first], "busy": [names that would also have to go but are working or starting]}. A kept service is never chosen for its memory. alone: every other running service has to go, kept or not; a running service that itself runs alone always has to go. A need of 0 or less asks for no memory."""
    run = sorted((s for s in services.values() if s.name != exclude and s.running()), key=lambda s: (s.active > 0, s.last_use))
    must = [s for s in run if alone or s.alone]
    evict, busy = [s.name for s in must if not s.active and s.status == "up"], [s.name for s in must if s.active or s.status != "up"]
    short = need_gb - free_gb() - sum((LEASES.get(s.name) or {}).get("ram_gb", 0.0) for s in must)
    if need_gb <= 0 or short <= 0: return {"fits": not busy, "evict": evict, "busy": busy, "short_gb": 0}
    for s in (s for s in run if s not in must):
        l = LEASES.get(s.name) or {}
        if l.get("keep"): continue
        if short <= 0: break
        (busy if s.active else evict).append(s.name)
        short -= l.get("ram_gb", 0.0)
    return {"fits": short <= 0 and not busy, "evict": evict if short <= 0 else [], "busy": busy, "short_gb": round(max(short, 0), 2)}

def view() -> dict:
    r = settings.get("resources", {}) or {}
    return {"mode": r.get("mode", "full"), "reserve_gb": reserve_gb(), "ram_total_gb": hardware.facts()["ram_gb"], "ram_avail_gb": round(hardware.mem_avail_gb(), 1), "free_to_promise_gb": round(free_gb(), 1), "leases": LEASES}
