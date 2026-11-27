"""core/pools.py - one standing name served by several machines (D-12 in claude/data/decisions.md, I-78 in claude/data/issues.md).
A pool is a node setting, written by the portal from its own pools (Corvid spread_view.py): pools.<name> = {"candidates": [{"source": "" for this node or a remote source's name, "model": the name there, "slots": requests it takes at once (absent: as many as that machine says it serves the model with, its listing's "slots"), "enabled": true}], "class": "interactive" | "normal" | "background", "tags": [...], "portal": the portal's version of the definition (so the portal sees which machines hold its current one)}.
A program on this node names "pool~<name>" as its model (a knowledge server is given it through its group's standing name). Each request is sent, as it arrives, to a candidate that can do what the request asks (chat, embed, ...) where it would finish first (its requests here per slot, times how long one has taken there lately: took_s), and carries the pool's class as its priority to wherever it runs (a remote candidate's node applies the same class rules to it). So one pool may hold chat and embedding models together (an ingestion pool): a knowledge server given it for both gets each request answered by a member that does that.
A candidate of this node with its own "slots" asks that many requests at once of the model here: wanted() is what its program serves it with (llama: parallel slots), whatever the machine's general rule says.
Classes: interactive is priority 1, normal 0, background -1. The scheduler (core/jobs.py) runs a background request only while no other request waits for the same program, and only up to scheduler.background_share of that program's slots (0 pauses background work; what is running finishes)."""
from core import settings, jobs

CLASS_PRIORITY = {"interactive": 1, "normal": 0, "background": -1}

def _pools() -> dict: return settings.get("pools", {}) or {}
def _listing() -> dict:
    from core import modules   # here, not at the top: modules loads the adapters, which read pools
    return {m["name"]: m for m in modules.models()}
def routed(c: dict) -> str: return f"""{c["source"]}/{c["model"]}""" if c.get("source") else c["model"]
def _live(p: dict) -> list: return [c for c in p.get("candidates", []) if c.get("enabled", True) and c.get("model")]
def _caps(c: dict, have: dict) -> set: return set(have.get(routed(c), {}).get("capabilities", []))   # what a candidate can do as this node lists it now (a remote one once its source's list is read)
def _able(p: dict, capability: str, have: dict) -> list: return [c for c in _live(p) if not capability or capability in _caps(c, have)]
def slots(c: dict, have: dict = None) -> int: return max(1, int(c.get("slots") or 0) or int((have if have is not None else _listing()).get(routed(c), {}).get("slots") or 1))   # a candidate's own number, else what the machine serving it lists for that model
def wanted(model: str) -> int: return max((int(c.get("slots") or 0) for p in _pools().values() for c in _live(p) if not c.get("source") and c["model"] == model), default=0)   # the requests at once a pool asks of a model of this node (0: none asked); its program serves that many (llama: parallel)
def load(model: str) -> int: return sum(1 for j in jobs.JOBS.values() if j.model == model)   # requests here for that model, waiting or running

def pick(name: str, capability: str = "") -> dict:
    """What "pool~<name>" stands for, for one request of capability now: {"model": of the candidates that do it, the one where it would finish first (ties: the first listed), "priority": the pool's class}."""
    p, have = _pools().get(name), _listing()
    if not p or not _live(p): raise KeyError(f"""no pool {name!r} with an enabled candidate on this node (pools: {", ".join(sorted(_pools())) or "none"}) - the portal sends pools to the machines (Settings, Operations, Resource pools)""")
    if not (able := _able(p, capability, have)): raise KeyError(f"""pool {name!r} has no member that does {capability} as this node lists them now: {", ".join(f"{routed(c)} ({', '.join(sorted(_caps(c, have))) or 'not listed here yet'})" for c in _live(p))}""")
    took = {routed(c): took_s(routed(c), capability) for c in able}
    guess = min((t for t in took.values() if t), default=1.0)   # a member not measured yet is taken to be as quick as the quickest: it gets work, and so a measure
    best = min(((load(routed(c)) // slots(c, have) + 1) * (took[routed(c)] or guess), i, routed(c)) for i, c in enumerate(able))
    return {"model": best[2], "priority": CLASS_PRIORITY.get(p.get("class", "normal"), 0)}

def took_s(model: str, capability: str = "", last: int = 20) -> float:
    """How long a request of this kind took on this member lately (working time, the wait before it left out; mean of its last served ones), 0 when none has finished yet. A pool sends each request where it would finish first - its queue per slot times this - so a slower machine gets less of a flood instead of an equal share that holds everyone up."""
    t = [d["seconds"] - (d.get("wait_s") or 0) for d in reversed(jobs.DONE) if d["model"] == model and d["outcome"] == "served" and (not capability or d["kind"] == capability)][:last]
    return sum(t) / len(t) if t else 0.0

def capacity(name: str, capability: str = "") -> int:
    have = _listing()
    return sum(slots(c, have) for c in _able(_pools().get(name) or {}, capability, have))

def advertised(name: str, capability: str = "") -> int:
    """What a program is told it may send at once for this pool (for one capability: chat for a knowledge server's model requests): every slot of every candidate that does it, times pools_advertise (default 4), so the program always has work waiting here and the node decides how fast it drains."""
    return capacity(name, capability) * int(settings.get("pools_advertise", 4) or 4)

def listed(models: list, capability: str = "") -> list:
    """Each pool as a model this node offers ("pool~<name>"), so any picker (a knowledge group's chat or embedding model, a role's binding) can choose it like a model: it can do what any enabled candidate can do, as the node lists them."""
    have, out = {m["name"]: m for m in models}, []
    for n, p in _pools().items():
        caps = set().union(*[_caps(c, have) for c in _live(p)]) if _live(p) else set()
        if caps and (not capability or capability in caps): out.append({"name": f"pool~{n}", "capabilities": sorted(caps), "size": max(have.get(routed(c), {}).get("size", 0) for c in _live(p)), "format": "pool", "adapter": "pool", "slots": sum(slots(c, have) for c in _live(p)), "tags": p.get("tags") or [], "about": f"""{p.get("class", "normal")} work over {len(_live(p))} member(s): {", ".join(routed(c) for c in _live(p))}"""})
    return out

def view() -> dict:
    """Each pool as it stands: its class, the portal's version of it, and per candidate what it does, its slots and the requests here for it now."""
    have = _listing()
    return {n: {"class": p.get("class", "normal"), "portal": p.get("portal", ""), "tags": p.get("tags") or [], "advertised": advertised(n, "chat") or advertised(n), "candidates": [{"model": routed(c), "does": sorted(_caps(c, have)), "slots": slots(c, have), "auto": not c.get("slots"), "here": load(routed(c)), "enabled": c.get("enabled", True)} for c in p.get("candidates", [])]} for n, p in _pools().items()}
