"""core/autostart.py - starting the node with the machine, without a terminal kept open.
Linux with systemd: a service unit that restarts the node if it exits. Run as a user, it is that user's unit (~/.config/systemd/user) with lingering, so it starts at boot rather than at login; run as root, a system unit. Termux: a Termux:Boot script that runs run.sh. Containers already restart through docker-compose (restart: unless-stopped).
The unit reads optional KEY=VALUE lines from magpie.env beside magpie.py, for anything otherwise set in a shell (MAGPIE_ADMIN_KEY, NODE_ID...). Nothing here contacts any other machine."""
import os, pwd, shutil, subprocess
from pathlib import Path
from core import hardware, settings

HERE, NAME = Path(__file__).resolve().parent.parent, "magpie"
ROOT = os.geteuid() == 0
USER = pwd.getpwuid(os.geteuid()).pw_name

# --- Where and how ---

def kind() -> str:
    if hardware.facts()["container"]: return "container"
    if hardware.facts()["phone"]: return "termux"
    return "systemd" if shutil.which("systemctl") and Path("/run/systemd/system").exists() else "none"

def _unit_path() -> Path: return Path("/etc/systemd/system" if ROOT else Path.home() / ".config/systemd/user") / f"{NAME}.service"
def _boot_path() -> Path: return Path.home() / ".termux/boot" / NAME
def _ctl(*args) -> subprocess.CompletedProcess: return subprocess.run(["systemctl", *([] if ROOT else ["--user"]), *args], capture_output=True, text=True)
def _logs() -> str: return f"""tail -f {settings.location("logs") / "magpie.log"} (also the portal's node page; the system journal needs group membership on many systems)"""

def _unit() -> str:
    return f"""[Unit]
Description=Magpie node ({HERE})

[Service]
WorkingDirectory={HERE}
EnvironmentFile=-{HERE}/magpie.env
ExecStart="{HERE}/.venv/bin/python" "{HERE}/magpie.py"
Restart=always
RestartSec=5
TimeoutStopSec=60

[Install]
WantedBy={"multi-user.target" if ROOT else "default.target"}
"""

def _theirs(p: Path) -> bool:
    """True when p is another install's file (a second Magpie folder on this machine): it is left alone."""
    return p.exists() and str(HERE) not in p.read_text()

def _linger() -> bool: return ROOT or subprocess.run(["loginctl", "show-user", USER, "-p", "Linger", "--value"], capture_output=True, text=True).stdout.strip() == "yes"

# --- On, off, status ---

def on() -> list:
    """Starts the node with the machine from now on, and starts it now. -> lines to show."""
    k = kind()
    if k == "container": return ["In a container the node restarts with it: docker-compose.yml has restart: unless-stopped."]
    if k == "none": return ["No systemd and not Termux: start ./run.sh from the system's own startup (cron @reboot, an init script)."]
    p = _unit_path() if k == "systemd" else _boot_path()
    if _theirs(p): raise SystemExit(f"{p} belongs to another Magpie folder; turn it off there first (magpie.py autostart off), or edit it by hand.")
    p.parent.mkdir(parents=True, exist_ok=True)
    if k == "termux":
        p.write_text(f"""#!/data/data/com.termux/files/usr/bin/sh\nexec "{HERE}/run.sh"\n""")
        p.chmod(0o755)
        return [f"Wrote {p}. It runs at boot once the Termux:Boot app (F-Droid) is installed and opened once. Start it now with ./run.sh."]
    p.write_text(_unit())
    for args in (("daemon-reload",), ("enable", "--now", NAME)):
        r = _ctl(*args)
        if r.returncode: raise SystemExit(f"""systemctl {" ".join(args)}: {r.stderr.strip()}""")
    out = [f"Wrote {p}; enabled and started. Logs: {_logs()}", f"""Stop it with systemctl {"" if ROOT else "--user "}stop {NAME} (a node started by hand as well would find its port taken)."""]
    if not _linger() and subprocess.run(["loginctl", "enable-linger", USER], capture_output=True).returncode:
        out.append(f"It starts when {USER} logs in. To start at boot without a login: sudo loginctl enable-linger {USER}")
    return out

def off() -> list:
    """Stops starting with the machine and stops the node. Lingering is left as it is (other user services may rely on it)."""
    k = kind()
    p = _unit_path() if k == "systemd" else _boot_path() if k == "termux" else None
    if not p or not p.exists(): return ["Autostart is not on."]
    if _theirs(p): raise SystemExit(f"{p} belongs to another Magpie folder; it is left alone.")
    if k == "systemd": _ctl("disable", "--now", NAME)
    p.unlink()
    if k == "systemd": _ctl("daemon-reload")
    return [f"""Removed {p}; the node is stopped{"" if k == "systemd" else " at the next boot"}."""]

def status() -> list:
    k = kind()
    if k != "systemd": return [f"{k}: " + (f"""{"on" if _boot_path().exists() else "off"} ({_boot_path()})""" if k == "termux" else "see magpie.py autostart on")]
    p = _unit_path()
    if not p.exists(): return [f"off ({p} does not exist)"]
    return [f"""{_ctl("is-enabled", NAME).stdout.strip()}, {_ctl("is-active", NAME).stdout.strip()} ({p}{"" if not _theirs(p) else ", another Magpie folder's"})""", "starts at boot" if _linger() else f"starts when {USER} logs in (sudo loginctl enable-linger {USER} to start at boot)", f"Logs: {_logs()}"]