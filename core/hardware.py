"""core/hardware.py - what this machine is and how fast it is, for anything from a phone to a mini PC. Knows nothing about AI.
facts(): probed once and kept. now(): memory, load and heat as of this moment. speed(): short measurements, run only on request and cached against the hardware. unmet(requires): why a module cannot run here, each with what would resolve it."""
import ctypes, os, re, glob, json, time, shutil, hashlib, platform, threading, subprocess
from pathlib import Path
from core import settings

_FACTS: dict = {}

def _read(p, default=""):
    try: return Path(p).read_text().strip()
    except OSError: return default

def _gpus() -> list:
    vendors = {"0x1002": "amd", "0x10de": "nvidia", "0x8086": "intel", "0x13b5": "arm", "0x5143": "qualcomm"}
    return [{"vendor": (v := vendors.get(_read(f"{d}/vendor"), _read(f"{d}/vendor"))), "device": _read(f"{d}/device"), "vram_gb": round(int(_read(f"{d}/mem_info_vram_total", "0")) / 1e9, 2), "gtt_gb": round(int(_read(f"{d}/mem_info_gtt_total", "0")) / 1e9, 2), "integrated": _integrated(d, v), "path": d} for d in sorted(glob.glob("/sys/class/drm/card[0-9]/device")) if _read(f"{d}/vendor")]

def _integrated(d: str, vendor: str):
    """Whether a GPU shares the system's memory (an iGPU or APU): its "VRAM" is a carve-out of RAM and GTT is more RAM it may map. A judgement from what the driver shows, not a fact it states: amdgpu shows a VRAM vendor only for memory chips of the card's own (dGPU); Intel graphics on the i915 driver without local memory are integrated. None when it cannot tell."""
    if vendor == "amd" and os.path.exists(f"{d}/mem_info_vram_total"): return not os.path.exists(f"{d}/mem_info_vram_vendor")
    if vendor == "intel": return not os.path.exists(f"{d}/mem_info_vram_total")
    return None

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
    _FACTS.update({"os": platform.system().lower(), "arch": platform.machine(), "kernel": platform.release(), "cpu": cpu.get("model name") or cpu.get("Hardware") or cpu.get("Model") or (_win_cpu() if os.name == "nt" else platform.processor()), "cores": len(ids) if ids and ids != {("", "")} else os.cpu_count(), "threads": os.cpu_count(),
                   "cpu_features": sorted(flags & {"avx2", "avx512f", "avx_vnni", "asimd", "neon", "sve", "i8mm"}), "ram_gb": round(mem.get("MemTotal", 0) / 1e9 or _total_gb(), 1), "swap_gb": round(mem.get("SwapTotal", 0) / 1e9, 1), "gpus": _win_gpus() if os.name == "nt" else _gpus(), "gfx": gfx,
                   "vulkan": bool(glob.glob("/usr/share/vulkan/icd.d/*") or glob.glob("/etc/vulkan/icd.d/*")), "rocm": os.path.exists("/opt/rocm"), "kfd": os.path.exists("/dev/kfd"), "cgroup_mem_limit_gb": _cgroup_limit(), "phone": "com.termux" in os.environ.get("PREFIX", ""), "container": os.path.exists("/.dockerenv"), "python": platform.python_version()})
    return _FACTS

def _total_gb() -> float: return _win_mem()[0] if os.name == "nt" else os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9

# --- Windows (a node there passes work to a local Ollama: no /proc, no /sys) ---

def _win_mem() -> tuple:
    """(total, available) memory in GB from GlobalMemoryStatusEx."""
    class M(ctypes.Structure): _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), *((n, ctypes.c_ulonglong) for n in ("total", "avail", "page_total", "page_avail", "virt_total", "virt_avail", "ext"))]
    m = M()
    m.len = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.total / 1e9, m.avail / 1e9

def _win_cpu() -> str:
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k: return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()

def _win_gpus() -> list:
    """Graphics adapters as Windows lists them (Win32_VideoController). Its AdapterRAM is a 32-bit figure: above 4 GB it reads 4 GB or less, so a card's own memory is only a lower bound here; Ollama's /api/ps says what a loaded model holds on it."""
    try: got = json.loads(subprocess.run(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,PNPDeviceID | ConvertTo-Json"], capture_output=True, text=True, timeout=20).stdout or "[]")
    except (OSError, ValueError, subprocess.TimeoutExpired): return []   # no PowerShell: no adapters listed
    vendors = {"VEN_1002": "amd", "VEN_10DE": "nvidia", "VEN_8086": "intel", "VEN_5143": "qualcomm"}
    return [{"vendor": next((v for k, v in vendors.items() if k in str(g.get("PNPDeviceID") or "")), ""), "device": g.get("Name") or "", "vram_gb": round(int(g.get("AdapterRAM") or 0) / 1e9, 2), "gtt_gb": 0, "integrated": None, "name": g.get("Name") or ""} for g in (got if isinstance(got, list) else [got]) if not str(g.get("Name") or "").startswith("Microsoft")]

def _cgroup_limit():
    v = _read("/sys/fs/cgroup/memory.max")
    return round(int(v) / 1e9, 1) if v and v != "max" else None

def mem_avail_gb() -> float:
    """Memory a new process can take: the smaller of MemAvailable and, inside a container with a limit (cgroup v2), the limit minus what the container uses (inactive file cache not counted)."""
    if os.name == "nt": return _win_mem()[1]
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

VIRTUAL = ("lo", "docker", "br-", "veth", "virbr", "tun", "tap", "wg", "tailscale", "zt")   # interface names that are not a physical lane between machines

def _ethtool(name: str) -> dict:
    """How a wired link was negotiated, from ethtool (no root needed): the fastest mode this end supports and offers, the fastest the other end offered, whether auto-negotiation is on, and Energy-Efficient Ethernet. A link slower than both ends offer points at the driver or firmware; one that matches what this end offers, below what it supports, points at a setting here; one that matches the other end's offer points at the switch port or cable. {} where ethtool is missing or says nothing."""
    try: t = subprocess.run(["ethtool", name], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired): return {}
    got, key = {}, None
    for l in t.splitlines():
        k, sep, v = l.strip().partition(":")
        if sep and not k.endswith("base"): key, got[k] = k, v.strip()
        elif key: got[key] += " " + l.strip()   # a mode list continues on the next lines
    top = lambda k: max((int(x) for x in re.findall(r"(\d+)base", got.get(k, ""))), default=None)
    try: eee = next((l.split(":", 1)[1].strip() for l in subprocess.run(["ethtool", "--show-eee", name], capture_output=True, text=True, timeout=5).stdout.splitlines() if "EEE status" in l), None)
    except (OSError, subprocess.TimeoutExpired): eee = None
    return {k: v for k, v in {"supported_max": top("Supported link modes"), "offered_max": top("Advertised link modes"), "partner_max": top("Link partner advertised link modes"), "autoneg": got.get("Auto-negotiation"), "eee": eee}.items() if v is not None}

def _nm_link(name: str) -> dict:
    """A NetworkManager profile's own link settings for this interface (speed, duplex, auto-negotiate), when it sets them: a profile can hold a link to 10 Mbit whatever cable or port is used. {} without NetworkManager."""
    try:
        conn = subprocess.run(["nmcli", "-t", "-g", "GENERAL.CONNECTION", "device", "show", name], capture_output=True, text=True, timeout=5).stdout.strip()
        vals = subprocess.run(["nmcli", "-t", "-g", "802-3-ethernet.speed,802-3-ethernet.duplex,802-3-ethernet.auto-negotiate", "connection", "show", conn], capture_output=True, text=True, timeout=5).stdout.splitlines() if conn else []
    except (OSError, subprocess.TimeoutExpired): return {}
    return {k: v for k, v in zip(("nm_profile_speed", "nm_profile_duplex", "nm_profile_autoneg"), [x.strip() for x in vals]) if v and v not in ("0", "--")} | ({"nm_profile": conn} if conn else {})

def net() -> list:
    """This machine's network interfaces as lanes to other machines: name, kind, link speed (Mbit/s, as the driver negotiated it), duplex, state, MTU, its IPv4 addresses (with prefix length, so a portal can tell which addresses share a network), the driver, whether it carries the default route, the queue it sends through (a shaping queue such as tbf or htb limits what leaves), and error and drop counts since boot. A link that negotiated 10 or 100 Mbit, half duplex, or errors that keep growing are the usual causes of a slow path. Linux only; elsewhere empty."""
    out, addrs = [], {}
    try: addrs = {i["ifname"]: [f"""{a["local"]}/{a["prefixlen"]}""" for a in i.get("addr_info", []) if a.get("family") == "inet"] for i in json.loads(subprocess.run(["ip", "-j", "addr", "show"], capture_output=True, text=True, timeout=5).stdout or "[]")}
    except (OSError, ValueError, subprocess.TimeoutExpired): pass   # no ip tool (not Linux, or a minimal system): interfaces are listed without addresses
    try: qd = {q["dev"]: q.get("kind") for q in json.loads(subprocess.run(["tc", "-j", "qdisc", "show"], capture_output=True, text=True, timeout=5).stdout or "[]") if q.get("root")}
    except (OSError, ValueError, subprocess.TimeoutExpired): qd = {}   # no tc tool: the queue each interface sends through is not reported
    try: via = {r.get("dev") for r in json.loads(subprocess.run(["ip", "-j", "route", "show", "default"], capture_output=True, text=True, timeout=5).stdout or "[]")}
    except (OSError, ValueError, subprocess.TimeoutExpired): via = set()   # which interface carries traffic to everything not on a local network
    for d in sorted(Path("/sys/class/net").glob("*")) if Path("/sys/class/net").exists() else []:
        if d.name.startswith(VIRTUAL): continue
        r = lambda f: _read(d / f)
        drv = (d / "device" / "driver").resolve().name if (d / "device" / "driver").exists() else None
        kind = "wifi" if (d / "wireless").exists() or (d / "phy80211").exists() else "thunderbolt" if drv == "thunderbolt-net" else "usb" if "usb" in str((d / "device").resolve()) else "ethernet" if (d / "device").exists() else "other"
        sp = r("speed")   # reading it on a link that is down fails or gives -1
        out.append({"name": d.name, "kind": kind, "speed_mbit": int(sp) if sp.lstrip("-").isdigit() and int(sp) > 0 else None, "duplex": r("duplex") or None, "state": r("operstate"), "mtu": int(r("mtu") or 0) or None, "addrs": addrs.get(d.name, []), "driver": drv, "qdisc": qd.get(d.name), "default_route": d.name in via, **{k: int(r(f"statistics/{k}") or 0) for k in ("rx_errors", "tx_errors", "rx_dropped", "tx_dropped")}, **({**_ethtool(d.name), **_nm_link(d.name)} if kind == "ethernet" and r("operstate") == "up" else {})})
    return out

def power_w() -> dict:
    """What the machine's own sensors say it draws now, in watts, by sensor ({"amdgpu": 18.4}): AMD APUs report their package power through the graphics driver (power1_average or power1_input, in microwatts). Empty when no sensor reports power; an outlet meter (the whole machine, PSU included) is a separate, optional reading."""
    out = {}
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        f = next((Path(h) / n for n in ("power1_average", "power1_input") if (Path(h) / n).exists()), None)
        if f and (v := _read(str(f)).strip()).isdigit(): out[_read(f"{h}/name").strip() or Path(h).name] = round(int(v) / 1e6, 1)
    return out

def now() -> dict:
    m = {l.split(":")[0]: int(l.split()[1]) / 1e6 for l in _read("/proc/meminfo").splitlines() if l.startswith(("SwapTotal", "SwapFree"))}
    load = os.getloadavg() if hasattr(os, "getloadavg") else (None,)
    return {"ram_avail_gb": round(mem_avail_gb(), 1), "swap_used_gb": round(m.get("SwapTotal", 0) - m.get("SwapFree", 0), 1), "gpu_used_gb": gpu_used_gb(), "load1": round(load[0], 2) if load[0] is not None else None, "temp_c": _temp_c(), "power_w": power_w()}

def disk(path) -> dict:
    p = Path(path)
    while not p.exists() and p != p.parent: p = p.parent
    u = shutil.disk_usage(p)
    return {"free_gb": round(u.free / 1e9, 1), "total_gb": round(u.total / 1e9, 1), "dev": p.stat().st_dev}   # dev: which disk (places on one disk share it, so their free space is one figure)

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

def _mem_gb_s(threads: int = 1, seconds: float = 1.0, mb: int = 128) -> float:
    """Memory bandwidth as copies between buffers made and touched beforehand (so allocation is not measured), on threads at once: GB/s of traffic, read and written counted both (a copy of n bytes moves 2n, as STREAM counts it). ctypes.memmove runs without the interpreter lock, so the threads copy in parallel. Uses threads x 2 x mb of memory for about a second."""
    size = mb << 20
    bufs = [(bytearray(size), bytearray(size)) for _ in range(threads)]
    views = [((ctypes.c_char * size).from_buffer(d), (ctypes.c_char * size).from_buffer(s)) for d, s in bufs]
    for d, s in views: ctypes.memmove(d, s, size)
    moved, go = [0] * threads, threading.Barrier(threads + 1)
    def work(i):
        d, s = views[i]
        go.wait()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            ctypes.memmove(d, s, size)
            moved[i] += 2 * size
    ts = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
    for t in ts: t.start()
    go.wait()
    t0 = time.perf_counter()
    for t in ts: t.join()
    return sum(moved) / 1e9 / (time.perf_counter() - t0)

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

def speed(refresh: bool = False, during: dict = None) -> dict:
    """The machine's speed test: processor (SHA-256 hashing, MB/s of data hashed - a rough measure of a core's integer speed, not of the arithmetic models use), memory (copy rate), the models place's disk. Kept until the hardware changes; "during": what else ran while it measured."""
    p, sig = settings.STATE / "speed.json", hashlib.sha256(json.dumps({k: facts()[k] for k in ("cpu", "threads", "ram_gb", "arch")}).encode()).hexdigest()[:12]
    old = json.loads(p.read_text()) if p.exists() else {}
    if old.get("sig") == sig and not refresh: return old
    out = {"sig": sig, "t": round(time.time()), "cpu_mb_s_1": round(_cpu_mb_s(), 1), "cpu_mb_s_all": round(_threads_mb_s(facts()["threads"]), 1), "mem_gb_s_1": round(_mem_gb_s(), 1), "mem_gb_s": round(_mem_gb_s(min(facts()["threads"], 8)), 1), "disk_read_gb_s": (d := _disk_read_gb_s(settings.location("models"))) and round(d, 2), **({"during": during} if during else {})}
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
