"""core/hardware.py - what this machine is and how fast it is, for anything from a phone to a mini PC. Knows nothing about AI.
facts(): probed once and kept. now(): memory, load and heat as of this moment. speed(): short measurements, run only on request and cached against the hardware. unmet(requires): why a module cannot run here, each with what would resolve it."""
import os, glob, json, time, shutil, hashlib, platform, threading
from pathlib import Path
from core import settings

_FACTS: dict = {}

def _read(p, default=""):
    try: return Path(p).read_text().strip()
    except OSError: return default

def _gpus() -> list:
    vendors = {"0x1002": "amd", "0x10de": "nvidia", "0x8086": "intel", "0x13b5": "arm", "0x5143": "qualcomm"}
    return [{"vendor": vendors.get(_read(f"{d}/vendor"), _read(f"{d}/vendor")), "device": _read(f"{d}/device"), "vram_gb": round(int(_read(f"{d}/mem_info_vram_total", "0")) / 1e9, 2), "gtt_gb": round(int(_read(f"{d}/mem_info_gtt_total", "0")) / 1e9, 2), "path": d} for d in sorted(glob.glob("/sys/class/drm/card[0-9]/device")) if _read(f"{d}/vendor")]

def facts(refresh: bool = False) -> dict:
    if _FACTS and not refresh: return _FACTS
    cpu = {}
    for l in _read("/proc/cpuinfo").splitlines():
        k, _, v = l.partition(":")
        cpu.setdefault(k.strip(), v.strip())
    flags = set((cpu.get("flags") or cpu.get("Features") or "").split())
    ids = {(_read(f"{c}/physical_package_id"), _read(f"{c}/core_id")) for c in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology")}
    mem = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in _read("/proc/meminfo").splitlines() if l.startswith(("MemTotal", "SwapTotal"))}
    gfx = sorted({f"gfx{v // 10000}{(v // 100) % 100:x}{v % 100:x}" for p in glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties") for l in _read(p).splitlines() if l.startswith("gfx_target_version") and (v := int(l.split()[1]))})
    _FACTS.clear()
    _FACTS.update({"os": platform.system().lower(), "arch": platform.machine(), "kernel": platform.release(), "cpu": cpu.get("model name") or cpu.get("Hardware") or cpu.get("Model") or platform.processor(), "cores": len(ids) if ids and ids != {("", "")} else os.cpu_count(), "threads": os.cpu_count(),
                   "cpu_features": sorted(flags & {"avx2", "avx512f", "avx_vnni", "asimd", "neon", "sve", "i8mm"}), "ram_gb": round(mem.get("MemTotal", 0) / 1e9 or _total_gb(), 1), "swap_gb": round(mem.get("SwapTotal", 0) / 1e9, 1), "gpus": _gpus(), "gfx": gfx,
                   "vulkan": bool(glob.glob("/usr/share/vulkan/icd.d/*") or glob.glob("/etc/vulkan/icd.d/*")), "rocm": os.path.exists("/opt/rocm"), "kfd": os.path.exists("/dev/kfd"), "cgroup_mem_limit_gb": _cgroup_limit(), "phone": "com.termux" in os.environ.get("PREFIX", ""), "container": os.path.exists("/.dockerenv"), "python": platform.python_version()})
    return _FACTS

def _total_gb() -> float: return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9

def _cgroup_limit():
    v = _read("/sys/fs/cgroup/memory.max")
    return round(int(v) / 1e9, 1) if v and v != "max" else None

def mem_avail_gb() -> float:
    """Memory a new process can take: the smaller of MemAvailable and, inside a container with a limit (cgroup v2), the limit minus what the container uses (inactive file cache not counted)."""
    if not os.path.exists("/proc/meminfo"): return _total_gb() * 0.85   # macOS and the like: no cheap live figure
    avail = next((int(l.split()[1]) / 1e6 for l in _read("/proc/meminfo").splitlines() if l.startswith("MemAvailable:")), 0.0)
    cg = Path("/sys/fs/cgroup")
    if _read(cg / "memory.max") in ("", "max"): return avail
    inactive = next((int(l.split()[1]) for l in _read(cg / "memory.stat").splitlines() if l.startswith("inactive_file ")), 0)
    return min(avail, (int(_read(cg / "memory.max")) - int(_read(cg / "memory.current")) + inactive) / 1e9)

def gpu_used_gb() -> float: return round(sum(int(_read(f"{d}/mem_info_vram_used", "0")) + int(_read(f"{d}/mem_info_gtt_used", "0")) for d in glob.glob("/sys/class/drm/card[0-9]/device")) / 1e9, 2)

def _temp_c():
    t = [int(v) / 1000 for z in glob.glob("/sys/class/thermal/thermal_zone*/temp") if (v := _read(z)).lstrip("-").isdigit()]
    return round(max(t), 1) if t else None

def now() -> dict:
    m = {l.split(":")[0]: int(l.split()[1]) / 1e6 for l in _read("/proc/meminfo").splitlines() if l.startswith(("SwapTotal", "SwapFree"))}
    load = os.getloadavg() if hasattr(os, "getloadavg") else (None,)
    return {"ram_avail_gb": round(mem_avail_gb(), 1), "swap_used_gb": round(m.get("SwapTotal", 0) - m.get("SwapFree", 0), 1), "gpu_used_gb": gpu_used_gb(), "load1": round(load[0], 2) if load[0] is not None else None, "temp_c": _temp_c()}

def disk(path) -> dict:
    p = Path(path)
    while not p.exists() and p != p.parent: p = p.parent
    u = shutil.disk_usage(p)
    return {"free_gb": round(u.free / 1e9, 1), "total_gb": round(u.total / 1e9, 1)}

def rss_gb(pid: int, field: str = "VmRSS") -> float:
    """A process's resident memory (VmRSS now, VmHWM peak). On unified-memory machines this includes GPU buffers. 0 when the process is gone."""
    return next((int(l.split()[1]) * 1024 / 1e9 for l in _read(f"/proc/{pid}/status").splitlines() if l.startswith(f"{field}:")), 0.0)

# --- Speed (rough, comparable between this project's nodes; run on request, a few seconds) ---

def _cpu_mb_s(seconds: float = 1.0) -> float:
    block, n, t0 = b"x" * (1 << 20), 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds: hashlib.sha256(block).digest(); n += 1
    return n / (time.perf_counter() - t0)

def _threads_mb_s(threads: int) -> float:
    out = [0.0] * threads
    def work(i): out[i] = _cpu_mb_s()
    ts = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
    for t in ts: t.start()
    for t in ts: t.join()
    return sum(out)

def _mem_gb_s() -> float:
    src, n, t0 = bytearray(1 << 28), 0, time.perf_counter()
    while time.perf_counter() - t0 < 0.5: bytes(src); n += 1
    return n * 0.268 / (time.perf_counter() - t0)

def _disk_read_gb_s(path) -> float:
    """Reads up to 1 GB of the largest file under path after asking the kernel to forget it. Reads only; None when there is no large file to read."""
    files = [(f.stat().st_size, f) for f in Path(path).rglob("*") if f.is_file() and f.stat().st_size > 1 << 27][:2000] if Path(path).exists() else []
    if not files: return None
    f = max(files)[1]
    fd = os.open(f, os.O_RDONLY)
    try:
        if hasattr(os, "posix_fadvise"): os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        n, t0 = 0, time.perf_counter()
        while n < 1 << 30 and (b := os.read(fd, 1 << 24)): n += len(b)
        return n / 1e9 / (time.perf_counter() - t0)
    finally: os.close(fd)

def speed(refresh: bool = False) -> dict:
    p, sig = settings.STATE / "speed.json", hashlib.sha256(json.dumps({k: facts()[k] for k in ("cpu", "threads", "ram_gb", "arch")}).encode()).hexdigest()[:12]
    old = json.loads(p.read_text()) if p.exists() else {}
    if old.get("sig") == sig and not refresh: return old
    out = {"sig": sig, "t": round(time.time()), "cpu_mb_s_1": round(_cpu_mb_s(), 1), "cpu_mb_s_all": round(_threads_mb_s(facts()["threads"]), 1), "mem_gb_s": round(_mem_gb_s(), 1), "disk_read_gb_s": (d := _disk_read_gb_s(settings.location("models"))) and round(d, 2)}
    p.write_text(json.dumps(out, indent=1))
    return out

# --- Requirements ---

def unmet(req: dict) -> list:
    """Why a module's "requires" block is not met here: one {"block", "resolve"} per reason. Empty when it can run. Keys: arch, os, cpu_features, min_ram_gb, gpu_vendors, needs_kfd, needs_vulkan, bins, python (import names)."""
    f, out = facts(), []
    def miss(block, resolve=""): out.append({"block": block, "resolve": resolve})
    if req.get("arch") and f["arch"] not in req["arch"]: miss(f"""architecture {f["arch"]} not in {req["arch"]}""", "needs a build for this architecture, or another adapter")
    if req.get("os") and f["os"] not in req["os"]: miss(f"""os {f["os"]} not in {req["os"]}""")
    if m := [c for c in req.get("cpu_features", []) if c not in f["cpu_features"]]: miss(f"cpu lacks {m}", "use a build for a lower CPU level")
    if f["ram_gb"] < req.get("min_ram_gb", 0): miss(f"""{f["ram_gb"]} GB RAM below {req["min_ram_gb"]}""")
    if req.get("gpu_vendors") and not {g["vendor"] for g in f["gpus"]} & set(req["gpu_vendors"]): miss(f"""no GPU from {req['gpu_vendors']}""")
    if req.get("needs_kfd") and not f["kfd"]: miss("/dev/kfd is not available", "pass the device through to this container")
    if req.get("needs_vulkan") and not f["vulkan"]: miss("no Vulkan driver found", "install the Vulkan driver (mesa-vulkan-drivers)")
    if m := [b for b in req.get("bins", []) if not (shutil.which(b) or Path(b).exists())]: miss(f"programs not found: {m}", "POST /modules/<name>/install, or set the path in node.json")
    import importlib.util as iu
    if m := [p for p in req.get("python", []) if not iu.find_spec(p)]: miss(f"python packages not found: {m}", "POST /modules/<name>/install")
    return out
