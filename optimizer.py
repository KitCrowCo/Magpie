# optimizer.py - launch planner for the llama.cpp supervisor (main.py).
# Node facts, goals, fit constants and experiments live in optimizer_rules.json, never here.
# Scope note: this planner is llama.cpp-specific by design (GGUF metadata, llama-server flags). Another engine
# would get its own planner behind the same setup()/plan()/satisfies() surface, not branches inside this one.
#
# Flow per request: setup() resolves constraints + given values + candidate ranges -> satisfies() decides whether the
# running server already covers the request -> plan() searches the ranges lexicographically in goal order.
# Learned state (optimizer_learned.json) is discarded automatically whenever node facts or the rules file change.
import os, re, json, glob, math, shutil, hashlib, platform, subprocess
from pathlib import Path
import httpx

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/root/.ollama/models"))
RULES_PATH = Path(os.getenv("OPTIMIZER_RULES", Path(__file__).parent / "optimizer_rules.json"))
LEARNED_PATH, PROFILE_PATH = MODELS_DIR / "optimizer_learned.json", MODELS_DIR / "node_profile.json"
GB, MODES = 1e9, ("ON", "SET", "UPPER", "LOWER", "OFF")
MANAGER: dict = {}
_INFO: dict = {}
_MODELS: list = []
EXTERNAL: dict = {"gpu_gb": 0.0, "ram_gb": 0.0, "why": ""}   # memory held by other processes this supervisor started (aux slots), taken off every plan's budgets

def reserve_external(gpu_gb: float, ram_gb: float, why: str): EXTERNAL.update(gpu_gb=gpu_gb, ram_gb=ram_gb, why=why)

# --- File Helpers ---

def _read(p, default=""):
    try: return Path(p).read_text().strip()
    except OSError: return default

def _json(p): return json.loads(Path(p).read_text()) if Path(p).exists() else {}
def load_rules() -> dict: return json.loads(RULES_PATH.read_text())

def _sysctl(k, default=""):
    try: return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True, timeout=5).stdout.strip() or default
    except (OSError, subprocess.SubprocessError): return default

def _boot() -> str: return _read("/proc/sys/kernel/random/boot_id") or _sysctl("kern.boottime")

def _ram_speed():
    try: out = subprocess.run(["dmidecode", "-t", "memory"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError): return None
    v = re.findall(r"Configured (?:Memory|Clock) Speed: (\d+)", out)
    return int(v[0]) if v else None

def _extra_tags() -> set: return {t for t in os.getenv("LLAMA_EXTRA_TAGS", "").lower().split(",") if t}

# --- System Facts (probed once, cached) ---

def _probe_darwin(rules: dict) -> dict:
    """Apple silicon: unified memory, Metal backend. vram = the GPU wired limit (sysctl iogpu.wired_limit_mb), or the macOS default share when unset."""
    total, wired = int(_sysctl("hw.memsize", "0")), int(_sysctl("iogpu.wired_limit_mb", "0") or 0) * 1024 * 1024
    tags = {rules["arch_tags"].get(platform.machine(), platform.machine()), "darwin", "metal", "apple_silicon"}
    if os.getenv("LLAMA_NODE_TAGS"): tags = {t for t in os.getenv("LLAMA_NODE_TAGS").lower().split(",") if t}
    return {"tags": sorted(tags | _extra_tags()), "backend": "metal", "vendor": "apple", "device": _sysctl("machdep.cpu.brand_string"), "apu": False, "gfx": [], "vram": wired or int(total * (0.75 if total >= 36 * GB else 0.667)), "gtt": 0, "ttm_limit": 0,
            "mem_total": total, "mem_avail": total, "ram_mts": None, "cpu": _sysctl("machdep.cpu.brand_string"), "cores": int(_sysctl("hw.perflevel0.physicalcpu") or _sysctl("hw.physicalcpu") or os.cpu_count()),
            "threads": os.cpu_count(), "avx2": False, "avx512": False, "kernel_cmdline": "", "disk_free": shutil.disk_usage(MODELS_DIR).free}

def _probe(rules: dict) -> dict:
    if platform.system() == "Darwin": return _probe_darwin(rules)
    mem = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in _read("/proc/meminfo").splitlines() if l.startswith(("MemTotal", "MemAvailable"))}
    cpu = {}
    for l in _read("/proc/cpuinfo").splitlines():
        k, _, v = l.partition(":")
        cpu.setdefault(k.strip(), v.strip())
    flags = cpu.get("flags") or cpu.get("Features", "")
    cards = [{"vendor": _read(f"{d}/vendor"), "device": _read(f"{d}/device"), "vram": int(_read(f"{d}/mem_info_vram_total", "0")), "gtt": int(_read(f"{d}/mem_info_gtt_total", "0"))} for d in sorted(glob.glob("/sys/class/drm/card[0-9]/device"))]
    gpu = next((c for c in cards if c["vendor"] in rules["vendors"]), None)
    vend = rules["vendors"][gpu["vendor"]] if gpu else None
    apu = bool(gpu) and (gpu["device"] in rules["apu_device_ids"] or (vend == "amd" and "radeon" in (cpu.get("model name") or "").lower()))
    backend = os.getenv("LLAMA_BACKEND") or ("hip" if vend == "amd" and os.path.exists("/opt/rocm") else "vulkan" if vend == "amd" and glob.glob("/usr/share/vulkan/icd.d/*") else "cuda" if vend == "nvidia" else "cpu")
    gfx = sorted({f"gfx{v // 10000}{(v // 100) % 100:x}{v % 100:x}" for p in glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties") for l in _read(p).splitlines() if l.startswith("gfx_target_version") and (v := int(l.split()[1]))})
    ids = {(_read(f"{c}/physical_package_id"), _read(f"{c}/core_id")) for c in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology")}
    tags = {rules["arch_tags"].get(platform.machine(), platform.machine()), backend, *gfx} | ({vend, "apu" if apu else "discrete_gpu", f"""dev:{gpu["device"]}"""} if gpu else {"no_gpu"}) | ({"cpu_only"} if backend == "cpu" else set())
    metrics = {"vram_gb": (gpu["vram"] if gpu else 0) / GB, "ram_gb": mem["MemTotal"] / GB}
    if gpu: tags |= {d["tag"] for d in rules.get("derived_tags", []) if metrics[d["metric"]] >= d.get("ge", -1) and metrics[d["metric"]] < d.get("lt", 1e12)}
    if os.getenv("LLAMA_NODE_TAGS"): tags = {t for t in os.getenv("LLAMA_NODE_TAGS").lower().split(",") if t}
    return {"tags": sorted(tags | _extra_tags()), "backend": backend, "vendor": vend, "device": gpu["device"] if gpu else None, "apu": apu, "gfx": gfx, "vram": gpu["vram"] if gpu else 0, "gtt": gpu["gtt"] if gpu else 0,
            "ttm_limit": int(_read("/sys/module/ttm/parameters/pages_limit", "0") or 0) * 4096, "mem_total": mem["MemTotal"], "mem_avail": mem["MemAvailable"], "ram_mts": _ram_speed(), "cpu": cpu.get("model name") or cpu.get("Model", ""),
            "cores": len(ids) if ids != {("", "")} else os.cpu_count(), "threads": os.cpu_count(), "avx2": "avx2" in flags, "avx512": "avx512f" in flags, "kernel_cmdline": _read("/proc/cmdline"), "disk_free": shutil.disk_usage(MODELS_DIR).free}

def info(refresh: bool = False) -> dict:
    global _INFO
    if refresh or not _INFO: _INFO = _probe(load_rules())
    return _INFO

def gpu_used_bytes() -> int:
    """VRAM + GTT currently in use on amdgpu cards (0 elsewhere). Sampled before/after a load to measure what the load really took."""
    return sum(int(_read(f"{d}/mem_info_vram_used", "0")) + int(_read(f"{d}/mem_info_gtt_used", "0")) for d in glob.glob("/sys/class/drm/card[0-9]/device") if _read(f"{d}/vendor") == "0x1002")

def node_env() -> dict:
    """The environment the rules give llama-server on this node (node tags and enabled experiments, no model-specific rules) - for helper llama-server processes such as draft and embedding slots."""
    rules = load_rules()
    cfg = _merge(rules["rules"], set(info()["tags"]), set(enabled_experiments(rules, resolve_constraints(rules, {}).get("experiments", []))))
    return {k: os.environ.get(k, v) for k, v in cfg["env"].items() if v is not None}

# --- GPU Kernels: does this build carry code for this GPU? ---

TARGETS_PATH = Path(os.getenv("LLAMA_BUILD_TARGETS", Path(__file__).parent / "AMDGPU_TARGETS.txt"))   # written by the Dockerfile's build stage

def _override_gfx(v: str) -> str:
    """HSA_OVERRIDE_GFX_VERSION as the gfx name the runtime then reports, and whose kernels it loads: 11.0.0 -> gfx1100."""
    m = re.fullmatch(r"\s*(\d+)\.(\d+)\.(\d+)\s*", v)
    return f"gfx{m.group(1)}{int(m.group(2)):x}{int(m.group(3)):x}" if m else f"unreadable override {v!r}"

def gpu_check(env: dict) -> str:
    """'' when llama-server started with env has kernels for this node's GPU: the GPU's gfx name - or the one HSA_OVERRIDE_GFX_VERSION makes it report - is among the build's AMDGPU targets; otherwise one line on what is missing and what fixes it.
    A process started without the override its GPU needs fails on its first matrix multiply with 'no kernel image is available for execution on the device'."""
    inf = info()
    if inf["backend"] != "hip" or not inf["gfx"]: return ""
    ov = env.get("HSA_OVERRIDE_GFX_VERSION")
    runs_as = {_override_gfx(ov)} if ov else set(inf["gfx"])
    if not TARGETS_PATH.exists(): return f"""build targets unknown ({TARGETS_PATH} is missing - the image was built before it was recorded); GPU {inf["gfx"]} runs as {sorted(runs_as)} (HSA_OVERRIDE_GFX_VERSION={ov or "unset"})"""
    targets = {t for t in re.split(r"[;,\s]+", TARGETS_PATH.read_text()) if t}
    if runs_as <= targets: return ""
    return f"""NO KERNELS for {sorted(runs_as - targets)} in this build (targets {sorted(targets)}; GPU {inf["gfx"]}, HSA_OVERRIDE_GFX_VERSION={ov or "unset"}) - expect "no kernel image is available" on the first matrix multiply. Fix: a rule giving the tag {inf["gfx"][0]} an HSA_OVERRIDE_GFX_VERSION of a built target (as gfx1103_hip does), or rebuild with --build-arg AMDGPU_TARGETS="{";".join(sorted(targets | set(inf["gfx"])))}" when this ROCm version supports that target."""

# --- Rule Matching + Experiments ---

def enabled_experiments(rules: dict, listed: list) -> list:
    """Experiments with default true, plus listed names, minus names listed with a leading '-'. Unknown names raise (fail fast on typos)."""
    known, on = rules.get("experiments", {}), {k for k, v in rules.get("experiments", {}).items() if v.get("default")}
    for x in listed:
        name = x.lstrip("-+")
        if name not in known: raise ValueError(f"unknown experiment {name!r} - known: {sorted(known)}")
        (on.discard if x.startswith("-") else on.add)(name)
    return sorted(on)

def _matches(r: dict, tags: set, exps: set) -> bool:
    if r.get("experiment") and r["experiment"] not in exps: return False
    if not set(r.get("when_all", [])) <= tags: return False
    if r.get("when_any") and not set(r["when_any"]) & tags: return False
    if set(r.get("unless_any", [])) & tags: return False
    return not any(set(e) <= tags for e in r.get("exceptions", []))

def _merge(rs: list, tags: set, exps: set = frozenset()) -> dict:
    out = {"set": {}, "env": {}, "fit": {}, "allow": {}, "deny": {}, "payload": {}, "features": {}, "args": [], "drop": [], "ids": []}
    for r in (r for r in rs if _matches(r, tags, exps)):
        out["ids"].append(r["id"]); out["args"] += r.get("args", []); out["drop"] += r.get("drop", [])
        for k in ("set", "env", "fit", "allow", "payload", "features"): out[k].update(r.get(k, {}))
        for n, v in r.get("deny", {}).items(): out["deny"][n] = sorted({*out["deny"].get(n, []), *v})
    return out

# --- Constraints + Given Values ---

def _cast(v, t): return {"int": lambda x: int(float(x)), "float": float, "str": lambda x: str(x).strip().lower(), "list": lambda x: x if isinstance(x, list) else [i.strip() for i in str(x).split(",") if i.strip()], "dict": lambda x: x if isinstance(x, dict) else json.loads(x)}[t](v)

def resolve_constraints(rules: dict, request: dict) -> dict:
    """Precedence: request constraints > request aliases > manager-polled > env > rules default."""
    req_c, out = request.get("constraints") or {}, {}
    for n, sp in rules["constraints"].items():
        for v in [req_c.get(n), *[request.get(k) for k in sp["request_keys"]], MANAGER.get(n), os.getenv(sp["env"]), sp["default"]]:
            if v not in (None, ""): out[n] = _cast(v, sp["type"]); break
    return out

def _given(sp: dict, request: dict):
    for src, v in (("request", next((request[k] for k in sp["request_keys"] if request.get(k) not in (None, "")), None)), ("env", os.getenv(sp["env"])), ("default", sp["default"])):
        if v not in (None, ""): return _cast(v, sp["type"]), src
    return None, "none"

def _cands(n, dom, given, src, mode, explicit, key):
    """Candidate values of one tuned param under the mode. Bounds compare by the param's own magnitude (kv = precision order, others numeric)."""
    lo, hi = key(n, dom[0]), key(n, dom[-1])
    g = None if given is None else key(n, given)
    if explicit: lo, hi = max(lo, key(n, explicit[0])), min(hi, key(n, explicit[1]))
    elif g is not None and mode == "SET" and src != "default": return [given]
    elif g is not None and mode == "UPPER": hi = min(hi, g)
    elif g is not None and mode == "LOWER": lo = max(lo, g)
    return [d for d in dom if lo <= key(n, d) <= hi] or [min(dom, key=lambda d: abs(key(n, d) - (lo if g is None else g)))]

def _sigc(C: dict, fixed) -> str: return json.dumps([{k: v for k, v in C.items() if k != "mode"}, fixed], sort_keys=True, default=str)

def model_tags(meta: dict) -> set:
    return ({"mtp"} if meta.get("mtp") else set()) | ({"hybrid"} if meta.get("kv_layers") and meta["kv_layers"] < (meta.get("block_count") or 0) else set())

# --- Learned GPU Budget, Faults, Measured Corrections (per model; reset when node or rules change) ---

def _sig() -> str:
    i = info()
    return hashlib.sha1(json.dumps([i["vram"], i["gtt"], i["mem_total"], i["tags"], i["kernel_cmdline"], RULES_PATH.read_text(), os.getenv("LLAMA_RESET_LEARNED")]).encode()).hexdigest()[:12]

def _db() -> dict:
    d = _json(LEARNED_PATH)
    if d.get("_sig") != _sig(): d = {"_sig": _sig()}; LEARNED_PATH.write_text(json.dumps(d, indent=2))
    return d

def _save(d: dict): LEARNED_PATH.write_text(json.dumps(d, indent=2))

def _learned(name: str) -> dict:
    d = _db(); e = d.get(name, {})
    if "pending" in e:
        p = e.pop("pending")
        if isinstance(p, list) and p[1] != _boot(): e["bad_gpu_gb"] = min(e.get("bad_gpu_gb", 1e9), p[0])
        d[name] = e; _save(d)
    return e

def mark(name: str, event: str, gpu_gb: float = 0, key: str = "", ram_gb: float = 0):
    """event: pending (request started) | ok | bad (GPU-side memory death) | bad_host (system RAM exhausted - the process was OOM-killed) | fault (kernel/driver death, counted per config key and per boot, never lowers a budget) | clear (cancelled or killed by a restart).
    ram_gb is the plan's estimated system-RAM demand (host side plus GPU spill into GTT): a bad_host death caps later plans below it, and a success at or above it lifts the cap.
    A pending marker left over from an earlier BOOT counts as a hang that needed a reboot; one from the same boot (container restart) does not."""
    if not (gpu_gb or ram_gb): return
    d = _db(); e = d.setdefault(name, {})
    if event == "pending" and gpu_gb: e["pending"] = [gpu_gb, _boot()]
    else: e.pop("pending", None)
    if event == "ok":
        e["ok_gpu_gb"] = max(e.get("ok_gpu_gb", 0), gpu_gb)
        if gpu_gb >= e.get("bad_gpu_gb", 1e9): e.pop("bad_gpu_gb")
        if ram_gb >= e.get("bad_ram_gb", 1e9): e.pop("bad_ram_gb")
    if event == "bad": e["bad_gpu_gb"] = min(e.get("bad_gpu_gb", 1e9), gpu_gb)
    if event == "bad_host": e["bad_ram_gb"] = min(e.get("bad_ram_gb", 1e9), ram_gb)
    if event == "fault":
        e.setdefault("faults", {})[key] = e.get("faults", {}).get(key, 0) + 1
        bf = e.get("boot_faults", [None, 0])
        e["boot_faults"] = [_boot(), (bf[1] if bf[0] == _boot() else 0) + 1]
    _save(d)

def fault_steps(name: str) -> int:
    """GPU faults recorded for this model since the last boot - how far down the rules' fault_ladder the next plan steps. Cleared by a reboot, a rules/node change, or /api/optimizer/forget."""
    bf = _learned(name).get("boot_faults", [None, 0])
    return bf[1] if bf[0] == _boot() else 0

def forget(name: str = ""):
    d = _db()
    if name: d.pop(name, None)
    else: d = {"_sig": d["_sig"]}
    _save(d)

def _ratio_key(kv: str, fa: str, draft: bool) -> str: return f"""{kv}|{"off" if fa == "off" else "on"}{"|draft" if draft else ""}"""   # draft = an MTP/speculative draft context is allocated too

def measure(plan: dict, name: str, gpu_delta_gb: float, rss_gb: float = 0):
    """Compares what a load really took with the planner's estimate and keeps a per-model correction ratio per kv/fa/draft combination (clamped 0.7-1.6).
    Unified nodes (fit.unified) are measured by llama-server's resident memory against est.ram_gb: their GPU buffers are process (managed) memory that the sysfs VRAM/GTT counters never see.
    Other nodes are measured by the sysfs VRAM+GTT delta against est.gpu_gb."""
    est, p = plan.get("est", {}), plan.get("params", {})
    got, want, field, what = (rss_gb, est.get("ram_gb"), "ram_ratio", "llama-server resident memory") if plan.get("unified") else (gpu_delta_gb, est.get("gpu_gb"), "gpu_ratio", "GPU-side load (sysfs VRAM+GTT delta)")
    if not (got > 0.2 and want): print(f"[OPT] measured: no usable {what} ({got:.2f} GB) - estimate not checked", flush=True); return
    key = _ratio_key(p.get("kv_cache_type"), p.get("flash_attn"), plan.get("draft"))
    d = _db(); r = d.setdefault(name, {}).setdefault(field, {})
    r[key] = round(max(0.7, min(r.get(key, 1.0) * got / want, 1.6)), 3)
    _save(d)
    print(f"""[OPT] measured {what} {got:.2f} GB vs estimated {want} GB at ctx {p.get("ctx")} -> correction {r[key]} for {key}""", flush=True)

def fault_key(plan: dict) -> str:
    p = plan.get("params", {})
    return f"""{p.get("kv_cache_type")}|fa:{p.get("flash_attn")}|exp:{"+".join(plan.get("experiments", []))}"""

# --- Setup: resolve everything a request implies, before any search ---

def setup(meta: dict, request: dict, name: str = "") -> dict:
    rules = load_rules(); C = resolve_constraints(rules, request); mode = C["mode"].upper(); C["mode"] = mode
    if mode not in MODES: raise ValueError(f"mode must be one of {MODES}, got {mode}")
    fallback = rules.get("fault_ladder", [])[:fault_steps(name)] if name else []   # applied after the requested experiments, so a fault back-off wins over a request that re-enables what faulted
    exps = enabled_experiments(rules, C.get("experiments", []) + fallback)
    given = {n: _given(sp, request) for n, sp in rules["params"].items()}
    if mode == "OFF": return {"mode": mode, "C": C, "rules": rules, "given": given, "exps": exps, "sigc": _sigc(C, None), "sig": json.dumps([C, {n: g[0] for n, g in given.items() if n != "ctx"}], sort_keys=True)}
    inf = info(); cfg = _merge(rules["rules"], set(inf["tags"]) | model_tags(meta), set(exps)); f = cfg["fit"]
    L, kvt, ko = meta.get("block_count"), meta.get("kv_bytes_per_token_f16"), rules["kv_order"]
    if given["kv_cache_type"][0] not in (None, *ko): raise ValueError(f"""kv_cache_type got {given["kv_cache_type"][0]!r}, must be one of {ko}""")
    fixed = {}
    for n, sp in rules["params"].items():
        if sp.get("tuned", True): continue
        v = given[n][0] if given[n][0] is not None else cfg["set"].get(n)
        fixed[n] = inf["cores"] if v == "physical" else v
    s = {"mode": mode, "C": C, "rules": rules, "given": given, "info": inf, "cfg": cfg, "L": L, "kvt": kvt, "fixed": fixed, "exps": exps, "fallback": fallback}
    s["sigc"] = _sigc(C, fixed)
    hc = meta.get("head_count"); s["heads"] = max(hc) if isinstance(hc, list) else (hc or 0)
    if not (L and kvt): return {**s, "cands": None, "sig": json.dumps([C, fixed, {n: g[0] for n, g in given.items()}], sort_keys=True, default=str)}
    trained = meta.get("context_length") or f["ctx_unknown_cap"]
    dom = {"ctx": sorted({*range(f["ctx_min"], trained + 1, f["ctx_step"]), trained, *([given["ctx"][0]] if given["ctx"][0] and given["ctx"][0] <= trained else [])}), "ngl": list(range(L + 2)) if (f["gpu_frac"] or f["gtt_frac"]) else [0], "kv_cache_type": ko,
           "ubatch_size": [u for u in rules["ubatch_candidates"] if u <= f["max_ubatch"]] or [min(rules["ubatch_candidates"])]}
    for n in dom:
        if n in cfg["allow"]: dom[n] = [d for d in dom[n] if d in cfg["allow"][n]] or dom[n]
        dom[n] = [d for d in dom[n] if d not in cfg["deny"].get(n, [])] or dom[n]
    key, explicit = (lambda n, x: ko.index(x) if n == "kv_cache_type" else x), C.get("bounds") or {}
    s["cands"] = {n: _cands(n, dom[n], *given[n], mode, explicit.get(n), key) for n in dom}
    s["sig"] = json.dumps([C, fixed, {n: [c[0], c[-1]] for n, c in s["cands"].items() if n != "ctx"}], sort_keys=True)
    return s

def satisfies(cur: dict, s: dict) -> bool:
    """True if the running plan already sits inside this request's bounds, so no reload is needed. A plan chosen under ON is not re-planned while loaded."""
    c = s.get("cands")
    if cur.get("mode") == "OFF" or s["mode"] == "OFF" or c is None: return cur.get("sig") == s["sig"]
    p, g = cur["params"], s["given"]["ctx"][0]
    ngl_ok = p.get("ngl") is None or p.get("ngl") in c["ngl"] or (p.get("ngl") == 99 and s["L"] + 1 in c["ngl"])
    ctx_ok = g is None or s["mode"] == "ON" or (p.get("ctx") or 0) >= g or cur.get("want_ctx") == g
    return cur.get("sigc") == s["sigc"] and ngl_ok and ctx_ok and p.get("kv_cache_type") in c["kv_cache_type"] and p.get("ubatch_size") in c["ubatch_size"]

# --- Plan: lexicographic search over the candidate ranges, in goal order ---

def plan(model_bytes: int, meta: dict, s: dict, name: str) -> dict:
    C, rules, given = s["C"], s["rules"], s["given"]
    if s["mode"] == "OFF": return {"mode": "OFF", "model": name, "goals": C["goals"], "experiments": s["exps"], "params": {n: g[0] for n, g in given.items()}, "est": {}, "fits": None, "rules": [], "env": {}, "args": [], "drop": [], "payload": {}, "features": {}, "sig": s["sig"], "notes": ["optimizer OFF - given values used as-is, unset flags left to llama-server defaults"]}
    f, inf, L, kvt, c, cfg = s["cfg"]["fit"], s["info"], s["L"], s["kvt"], s["cands"], s["cfg"]
    out = {"mode": s["mode"], "model": name, "goals": C["goals"], "experiments": s["exps"], "rules": cfg["ids"], "env": {k: (None if v is None else os.environ.get(k, v)) for k, v in cfg["env"].items()}, "args": cfg["args"], "drop": cfg["drop"], "payload": cfg["payload"], "features": cfg["features"], "sig": s["sig"], "notes": [], "fits": None, "est": {}}
    if c is None: return {**out, "params": {**{n: g[0] for n, g in given.items()}, **s["fixed"]}, "notes": ["no gguf metadata - nothing optimized, given values used"]}
    share = 1 / max(C["models_expected"], 1)
    vram_b = inf["vram"] * f["gpu_frac"] * share
    gtt_eff = min(inf["gtt"], inf.get("ttm_limit") or inf["gtt"], f.get("gtt_ram_cap", 1.0) * inf["mem_total"])
    gpu_b, host_b = vram_b + gtt_eff * f["gtt_frac"] * share, (inf["mem_total"] - C["reserve_gb"] * GB) * share
    if EXTERNAL["gpu_gb"] or EXTERNAL["ram_gb"]: gpu_b, host_b, vram_b = max(0.0, gpu_b - EXTERNAL["gpu_gb"] * GB), max(0.0, host_b - EXTERNAL["ram_gb"] * GB), min(vram_b, max(0.0, gpu_b - EXTERNAL["gpu_gb"] * GB)); out["notes"].append(f"""budgets reduced by {EXTERNAL["gpu_gb"]} GB GPU and {EXTERNAL["ram_gb"]} GB RAM held by {EXTERNAL["why"]}""")
    if C.get("budget_gb"):
        scale = min(1.0, C["budget_gb"] * GB / max(host_b + vram_b, 1))
        gpu_b, host_b, vram_b = gpu_b * scale, host_b * scale, vram_b * scale
    learned = _learned(name); bad = learned.get("bad_gpu_gb")
    if bad: gpu_b = min(gpu_b, (bad - f["bad_margin_gb"]) * GB); vram_b = min(vram_b, gpu_b); out["notes"].append(f"GPU budget capped at {gpu_b / GB:.1f} GB by an earlier memory failure - open /api/optimizer/forget to reset")
    bad_ram = learned.get("bad_ram_gb")
    if bad_ram: host_b = min(host_b, (bad_ram - f["bad_margin_gb"]) * GB); out["notes"].append(f"system-RAM budget capped at {host_b / GB:.1f} GB by an earlier OOM kill - open /api/optimizer/forget to reset")
    if s.get("fallback"): out["notes"].append(f"""{fault_steps(name)} GPU fault(s) this boot - fault_ladder applied: {", ".join(s["fallback"])}""")
    wl, kf, pen, slow = model_bytes / (L + 1), rules["kv_factor"], rules["kv_penalty"], f["cpu_layer_slowdown"]
    fa_set, ratios, ram_ratios, draft = s["fixed"].get("flash_attn") or "auto", learned.get("gpu_ratio", {}), learned.get("ram_ratio", {}), "exp_mtp_spec" in cfg["ids"]
    out.update(unified=bool(f.get("unified")), draft=draft)
    fa = lambda kv: "off" if fa_set == "off" and kv in rules["full_precision_kv"] else "on"
    def ram(gpu, host, kv): return (host + gpu) * ram_ratios.get(_ratio_key(kv, fa(kv), draft), 1.0) if f.get("unified") else host + max(0, gpu - vram_b)   # system RAM the launch needs
    def cost(ctx, kv, ub, ng):
        lay, b = wl + kvt / L * kf[kv] * ctx * (1 + f.get("draft_kv_frac", 0)), min(ng, L)
        comp = (f["compute_base_gb"] + f["compute_per_256_gb"] * ub / 256 + f.get("gpu_extra_gb", 0)) * GB + (s["heads"] * ub * ctx * 4 * f.get("fa_off_factor", 2.0) if fa(kv) == "off" else 0)
        gpu = ((comp if ng else 0) + b * lay + (wl if ng > L else 0)) * ratios.get(_ratio_key(kv, fa(kv), draft), 1.0)
        return gpu, f["host_overhead_gb"] * GB + (L - b) * lay + (0 if ng > L else wl), lay, b
    best = None
    for ctx in c["ctx"]:
        for kv in c["kv_cache_type"]:
            for ub in c["ubatch_size"]:
                for ng in c["ngl"]:
                    gpu, host, lay, b = cost(ctx, kv, ub, ng)
                    if gpu > gpu_b or ram(gpu, host, kv) > host_b: continue
                    gv = {"ctx": ctx, "quality": -pen[kv], "speed": 1e9 / (lay * (b + (L - b) * slow)) * (1 + f["ubatch_bonus"] * math.log2(ub / 128))}
                    k = tuple(gv[g] for g in C["goals"])
                    if best is None or k > best[0]: best = (k, ctx, kv, ub, ng, gpu, host)
    out["fits"] = best is not None
    if best is None:
        ctx, kv, ub, ng = c["ctx"][0], c["kv_cache_type"][0], c["ubatch_size"][0], c["ngl"][0]
        gpu, host, _, _ = cost(ctx, kv, ub, ng); best = (None, ctx, kv, ub, ng, gpu, host)
        out["notes"].append("WARNING: nothing fits the estimated budgets - lowest-memory candidate used; the estimate is unverified, the real load decides")
    _, ctx, kv, ub, ng, gpu, host = best
    p = {**{n: g[0] for n, g in given.items()}, **s["fixed"], "ctx": ctx, "ngl": 99 if ng > L else ng, "kv_cache_type": kv, "ubatch_size": ub}
    p["batch_size"] = max(p.get("batch_size") or 0, ub)
    if kv not in rules["full_precision_kv"] and p.get("flash_attn") == "off": p["flash_attn"] = "on"; out["notes"].append("quantized KV needs flash attention - forced flash_attn=on")
    if "ngl" in cfg["drop"]: out["notes"].append(f"experiment llama_fit: -ngl not passed, llama.cpp fits layers itself (planner would have used {p['ngl']})")
    faults = learned.get("faults", {}).get(fault_key({"params": p, "experiments": s["exps"]}))
    if faults: out["notes"].append(f"{faults} GPU fault(s) recorded earlier with this kv/fa/experiment combination on this node")
    return {**out, "params": p, "bounds": {n: [x[0], x[-1]] for n, x in c.items()}, "est": {"ctx": ctx, "kv": kv, "ngl": ng, "gpu_gb": round(gpu / GB, 2), "host_gb": round(host / GB, 2), "ram_gb": round(ram(gpu, host, kv) / GB, 2), "gpu_budget_gb": round(gpu_b / GB, 2), "host_budget_gb": round(host_b / GB, 2)}}

def cli_args(plan: dict) -> list:
    out, drop = [], set(plan.get("drop", []))
    for n, sp in load_rules()["params"].items():
        v = plan["params"].get(n)
        if n not in drop and v not in (None, ""): out += [x for fl in sp["flags"] for x in (fl, str(v))]
    return out + list(plan.get("args", []))

def experiments_status(listed: list = None) -> dict:
    rules = load_rules()
    on = enabled_experiments(rules, listed if listed is not None else resolve_constraints(rules, {}).get("experiments", []))
    return {"enabled": on, "how": "env LLAMA_EXPERIMENTS=name,-name (or request constraints.experiments) - a leading '-' disables a default-on experiment", "registry": rules.get("experiments", {})}

# --- Startup Report + Manager Check ---

async def manager_hello():
    """Reports this node to the portal server and stores the constraints it answers with. No portal, no reply, or a bad status means skip."""
    host = os.getenv("PORTAL_SERVER")
    if not host: return
    url = (host if "://" in host else "http://" + host).rstrip("/") + "/module/ai_tools/node_hello"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=3.0, read=5.0, write=5.0, pool=3.0)) as c:
            r = await c.post(url, json={"node_id": os.getenv("NODE_ID") or platform.node(), "info": _INFO, "models": _MODELS}, headers={"Authorization": "Bearer " + os.getenv("PORTAL_TOKEN", "")})
    except httpx.HTTPError as e: print("[OPT] manager unreachable, skipping:", e); return
    if r.status_code != 200: print("[OPT] manager replied", r.status_code, "- skipping"); return
    MANAGER.clear(); MANAGER.update(r.json().get("constraints") or {})
    print("[OPT] manager constraints:", MANAGER)

async def startup(models: list):
    """models: [{name, size, meta}]. Probes the node, indexes models, writes node_profile.json, prints the report, then polls the manager."""
    global _MODELS
    rules = load_rules()
    if str(os.getenv("LLAMA_OPTIMIZE") or rules["constraints"]["mode"]["default"]).upper() == "OFF": print("[OPT] startup: LLAMA_OPTIMIZE=OFF - optimizer disabled"); return
    inf = info(True); exps = enabled_experiments(rules, resolve_constraints(rules, {}).get("experiments", []))
    cfg = _merge(rules["rules"], set(inf["tags"]), set(exps))
    _MODELS = [{"name": m["name"], "size_gb": round(m["size"] / GB, 1), "arch": m["meta"].get("general.architecture"), "layers": m["meta"].get("block_count"), "kv_layers": m["meta"].get("kv_layers"), "ctx_train": m["meta"].get("context_length"), "params_b": m["meta"].get("params_b"), "bpw": m["meta"].get("bpw"), "file_type": m["meta"].get("file_type"), "tags": sorted(model_tags(m["meta"]))} for m in models]
    env = {k: os.environ.get(k) for k in rules["report_env"]}
    PROFILE_PATH.write_text(json.dumps({"info": inf, "rules": cfg["ids"], "experiments": exps, "env": env, "models": _MODELS}, indent=2))
    print("[OPT] node:", inf["cpu"], "| cores", inf["cores"], "| RAM", round(inf["mem_total"] / GB, 1), "GB @", inf["ram_mts"], "MT/s | VRAM", round(inf["vram"] / GB, 1), "GB | GTT", round(inf["gtt"] / GB, 1), "GB | TTM limit", round(inf["ttm_limit"] / GB, 1), "GB | device", inf["device"], "| backend", inf["backend"])
    print("[OPT] tags:", inf["tags"], "| env:", {k: v for k, v in env.items() if v})
    print("[OPT] rules matched:", cfg["ids"], "" if len([i for i in cfg["ids"] if not i.startswith("exp_")]) > 1 else "- WARNING generic rules only, nothing node-specific matched these tags")
    print("[OPT] experiments enabled:", exps, "(change with LLAMA_EXPERIMENTS=name,-name)")
    print("[OPT] GPU kernels:", gpu_check({**os.environ, **node_env()}) or "ok - this build has kernels for the GPU as it runs here", flush=True)
    print("[OPT] models indexed:", [(m["name"], m["layers"], m["kv_layers"], m["ctx_train"], m["bpw"], m["tags"]) for m in _MODELS])
    await manager_hello()


