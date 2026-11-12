"""modules/colibri/module.py - colibri as an adapter: any checkpoint folder under colibri.dir is a model on this node's one list, and asking for it is all a client ever does. The served id colibri insists on, its memory budget flag, its slots and its environment are this module's business.
Requests wait in the node's queue, never in colibri's, so a queue timeout inside colibri cannot fire and a queued request can be cancelled. A listing reads the folder and never starts anything.
The expert map: /experts/<model> reads the heat colibri has learned (.coli_usage), and takes a portal's placement for this node, turned into the same file format for PIN, so a pool-wide plan reaches this engine as an ordinary start setting."""
import os, json, subprocess
from pathlib import Path
from core import settings, budget, services, journal, drivers
from core.adapters import Adapter
from core.http import Router, HTTPException, json_body
from core.services import Service
from modules.llm import oai

router, A = Router(), {"adapter": None}
USAGE_VERSION = 1

# --- Expert history files (format: colibri c/route_trace.h - "<layer> <expert> <count>", with -1 and -2 header records) ---

def read_usage(path: Path) -> dict:
    """{"layers", "experts", "heat": {layer: {expert: count}}}. Header records have a negative layer. An unreadable or binary file gives an empty heat, with the reason."""
    out, heat = {"layers": None, "experts": None}, {}
    try:
        for line in path.read_text().splitlines():
            l, e, c = (int(x) for x in line.split()[:3])
            if l == -1: out.update(layers=e, experts=c)
            elif l >= 0: heat.setdefault(l, {})[e] = c
    except (OSError, ValueError, UnicodeDecodeError) as x: out["note"] = f"{type(x).__name__}: {x}"[:200]
    return {**out, "heat": heat}

def write_pins(path: Path, pairs: list, heat: dict):
    """A placement as a history file colibri's PIN setting reads: every pair ranked by its real heat when known, else even. colibri ranks by count and fills PIN_GB, so the ranking is what the pin order means."""
    rows = sorted({(int(l), int(e)) for l, e in pairs})
    path.write_text("".join(f"""{l} {e} {heat.get(l, {}).get(e, 0) + 1}\n""" for l, e in rows))

# --- The adapter ---

class ColibriAdapter(Adapter):
    capabilities = {"chat": {"streams": True}}

    def __init__(self, cfg):
        super().__init__("colibri", cfg)
        self.owner, self.exp_dir = "colibri", settings.STATE / "experts"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.svcs = {}

    @property
    def dir(self) -> Path: return settings.location("models") / "colibri"

    def _folders(self) -> list: return sorted(d for d in self.dir.glob("*/") if (d / "config.json").exists()) if self.dir.exists() else []

    def models(self) -> list: return [{"name": d.name, "capabilities": ["chat"], "size": sum(f.stat().st_size for f in d.glob("*.safetensors")), "format": "colibri", "mtime": d.stat().st_mtime, "ctx": self._cfg(d.name)["ctx"]} for d in self._folders()]

    def _cfg(self, model: str) -> dict:
        """Settings for one model: this module's defaults, then its entry under colibri.models."""
        return settings.merge(settings.get("colibri", {}), settings.get(f"colibri.models.{model}", {}) or {})

    def _service(self, model: str) -> Service:
        c = self._cfg(model)
        if model not in self.svcs: self.svcs[model] = Service(f"colibri:{model}", {"health": c.get("health", "/v1/models"), "start_timeout_s": c.get("start_timeout_s", 36000), "idle_stop_s": c.get("idle_stop_s", 18000), "slots": int(c.get("kv_slots", 1)), "probe_s": 2}, events="engine", owner="colibri")
        return self.svcs[model]

    def _ram(self, c: dict) -> float:
        """Its memory budget: the setting when one is given (too large for this node is an error that says so), otherwise all the node can give it, since colibri uses what it is told it has."""
        return float(c["ram_gb"]) if c.get("ram_gb") else budget.can_free(services.REG, alone=bool(c.get("alone", True)))

    # --- Planning (nothing starts) ---

    def _own_plan(self, model: str, c: dict, ctx: int, ram) -> dict:
        """colibri's own resource plan for this model, as it sees the node right now. {"error"} when its planner does not answer."""
        cmd = [*drivers.fill(c["cmd"], "colibri")[:2], "plan", "--model", str(self.dir / model), "--ctx", str(ctx), "--json", *(["--ram", f"{float(ram):.0f}"] if ram else [])]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env={**os.environ, **(drivers.state("colibri").get("env") or {}), **c.get("env", {})})
        try: return json.loads(r.stdout)
        except ValueError: return {"error": (r.stderr or r.stdout).strip()[-400:] or f"exit {r.returncode}"}

    def _floor(self, p: dict) -> tuple:
        """(GB always held, GB for one expert slot per layer, GB shared with the GPU) from colibri's plan: below their sum it refuses to start."""
        t = p["tiers"]["ram"]
        return (t["dense_bytes"] + t["runtime_bytes"]) / 1e9, (p["model"].get("per_cap_bytes") or 0) / 1e9, (p["tiers"]["vram"]["budget_bytes"] / 1e9 if p["memory"].get("unified") else 0.0)

    def plan(self, capability: str, model: str, want: dict) -> dict:
        if not (self.dir / model / "config.json").exists(): raise LookupError(f"no colibri model folder {model!r} under {self.dir}")
        c, room = self._cfg(model), float(want.get("_room_gb") or 0)
        ctx, ram = int(want.get("ctx") or c.get("ctx", 32768)), want.get("ram_gb") or c.get("ram_gb")
        out = {"service": self._service(model).name, "ram_gb": round(float(ram) if ram else budget.can_free(services.REG, alone=bool(c.get("alone", True))), 1), "gpu_gb": 0.0, "alone": bool(c.get("alone", True)), "keep": False, "settings": {"ctx": ctx, "ram_gb": ram or "auto", "kv_slots": c.get("kv_slots", 1)}}
        p = self._own_plan(model, c, ctx, ram)
        if p.get("error"): return {**out, "source": "estimated", "notes": [f"""colibri's own planner did not answer, so only its memory setting is known: {p["error"]}"""]}
        fixed, slot, shared = self._floor(p)
        t, notes, ways = p["tiers"]["ram"], [*p.get("warnings", [])], []
        if ram and fixed + slot + shared > float(ram): notes.append(f"its memory setting ({float(ram):.0f} GB) is below what it needs to start ({fixed + slot + shared:.1f} GB): raise it or put it back on auto")
        c2 = ctx
        while room and fixed + slot + shared > room and c2 > 4096:   # a smaller context shrinks what is always held: offered as a way when the node cannot hold this one even alone
            c2 //= 2
            p2 = self._own_plan(model, c, c2, ram)
            if p2.get("error"): break
            f2, s2, g2 = self._floor(p2)
            if f2 + s2 + g2 <= room:
                ways.append({"label": f"Context {c2} tokens: needs {f2 + s2 + g2:.1f} GB", "set": {f"colibri.models.{model}.ctx": c2}, "cost": f"prompts longer than {c2} tokens are refused", "fits": True})
                break
        return {**out, "source": "program", "min_ram_gb": round(fixed + slot + shared, 2), "ok_now": t["cache_slots_per_layer"] >= 1, "parts": {"weights always in memory": round(t["dense_bytes"] / 1e9, 2), f"runtime and state for {ctx} tokens of context": round(t["runtime_bytes"] / 1e9, 2), "one expert slot per layer (the least it runs with)": round(slot, 2), **({"shared with the GPU": round(shared, 2)} if shared else {})}, "program": {"expert_slots_per_layer_now": t["cache_slots_per_layer"], "sees_available_gb": round(p["memory"]["available_bytes"] / 1e9, 1), "bottleneck": p.get("expected_bottleneck", ""), "experts_on_disk_gb": round(p["tiers"]["disk"]["cold_expert_bytes"] / 1e9, 1)}, "notes": notes, "ways": ways}

    def knob_auto(self, model: str = "") -> dict:
        c = self._cfg(model) if model else settings.get("colibri", {})
        return {"ram_gb": f"""all this node can give: {budget.can_free(services.REG, alone=bool(c.get("alone", True))):.1f} GB now"""}

    def need(self, job) -> dict:
        model = job.model
        if not (self.dir / model / "config.json").exists(): raise HTTPException(status_code=404, detail=f"no colibri model folder {model!r} under {self.dir}")
        c, sv = self._cfg(model), self._service(model)
        ram = self._ram(c)
        async def load(j):
            pin = self.exp_dir / f"{model}.pin"
            cmd = drivers.fill(c["cmd"], "colibri")
            fill = {"model_path": str(self.dir / model), "ram_gb": f"{ram:.0f}", "kv_slots": str(c.get("kv_slots", 1)), "ctx": str(c.get("ctx", 32768))}
            env = {**(drivers.state("colibri").get("env") or {}), **c.get("env", {}), **({"PIN": str(pin), **({"PIN_GB": str(c["pin_gb"])} if c.get("pin_gb") else {})} if pin.exists() else {})}
            sv.spec["read_timeout_s"] = c.get("read_timeout_s", 360000)
            sv.meta = {"model": model, "ram_gb": round(ram), "ctx": c.get("ctx", 32768), "pinned": pin.exists()}
            if not await sv.start([a.format(**fill, port="{port}") for a in cmd], env, sig=self._sig(c, ram)): raise RuntimeError(f"""colibri {model} did not start ({sv.status}): {" | ".join(sv.tail(3))} - log {sv.log}""")
            sv.measure()
            r = await sv.client().get("/v1/models", timeout=15.0)
            ids = [m.get("id") for m in (r.json().get("data") or [])] if r.status_code == 200 else []
            sv.meta["served_id"] = ids[0] if ids else model
        return {"service": sv, "ram_gb": ram, "alone": bool(c.get("alone", True)), "label": model, "load": load}

    def _sig(self, c: dict, ram: float) -> str: return json.dumps([c.get("ctx"), c.get("kv_slots"), ram, c.get("env"), c.get("pin_gb")])
    def fits(self, service, job) -> bool:
        """A setting a manager changed takes effect here: the next job restarts it, once idle. Left unset, the memory it was given stands until something else needs a restart."""
        c = self._cfg(job.model)
        return service.sig == self._sig(c, self._ram(c) if c.get("ram_gb") else json.loads(service.sig or "[0,0,0]")[2])

    async def run(self, job, sv):
        if job.capability == "warm":
            yield {"type": "done", "reason": "stop"}
            return
        c = self._cfg(job.model)
        spec = c["chat"]
        body = {**job.body, "_served_model": sv.meta.get("served_id", job.model), "options": {**spec.get("defaults", {}), **(job.body.get("options") or {})}}
        if "think" not in body and spec.get("think"): body["think"] = False
        async for ev in oai.chat_events(sv.client(float(c.get("read_timeout_s", 360000))), body, spec):
            if ev["type"] == "done":
                job.record["full"], job.record["calls"] = ev.pop("_full", None), ev.pop("_calls", None)
            yield ev

# --- Routes ---

@router.get("/experts/{model}")
async def experts_get(request, model: str):
    """What colibri has learned and what this node was told to hold, for a manager planning placement across nodes."""
    a = A["adapter"]
    d = a.exp_dir / f"{model}.json"
    return {"model": model, "usage": read_usage(a.dir / model / ".coli_usage"), "placement": json.loads(d.read_text()) if d.exists() else None}

@router.put("/experts/{model}")
async def experts_put(request, model: str):
    """{"pairs": [[layer, expert], ...], "source"}: the experts this node should keep resident. Applies at the engine's next start (nothing is interrupted); null clears it."""
    a, b = A["adapter"], await json_body(request)
    if not (a.dir / model / "config.json").exists(): raise HTTPException(status_code=404, detail=f"no colibri model folder {model!r}")
    d, pin = a.exp_dir / f"{model}.json", a.exp_dir / f"{model}.pin"
    if b.get("pairs") is None:
        d.unlink(missing_ok=True); pin.unlink(missing_ok=True)
        return {"cleared": True}
    write_pins(pin, b["pairs"], read_usage(a.dir / model / ".coli_usage")["heat"])
    d.write_text(json.dumps({"pairs": b["pairs"], "source": b.get("source", ""), "t": journal.now()}))
    journal.event("experts.placement", model=model, pairs=len(b["pairs"]), source=b.get("source", ""))
    return {"model": model, "pairs": len(b["pairs"]), "applies": "at the next engine start"}

def setup(ctx):
    A["adapter"] = ColibriAdapter(ctx.cfg())
    ctx.add_adapter(A["adapter"])
    ctx.add_router(router)
    for d in A["adapter"]._folders(): A["adapter"]._service(d.name)   # listed from the start, so a manager can see and warm them
    if w := ctx.cfg("worker"):
        if w.get("enabled"): Service("colibri:worker", {**w, "cmd": drivers.fill(w["cmd"], "colibri")}, events="engine", owner="colibri")
