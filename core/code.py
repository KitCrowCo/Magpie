"""core/code.py - Magpie's own code, as a managing portal sees and changes it.
GET  /node/code           what this node's code is: a sha256 per file and one fingerprint over them, the rule that says which files are code (so a portal can apply the same rule to its copy), the earlier versions kept, and what would start the node again.
PUT  /node/code           {"files": {path: base64}, "remove": [path], "from": a label}: the changed files of a newer copy. Applied to a copy of the code first, which must compile and import; only then is the code changed. Every file replaced or removed is kept first (state/code/<time>/), and the node is started again where something supervises it.
POST /node/code/rollback  the last kept version back, the same way.
What is code: every file under the node's folder except what is the machine's own or made here (CODE_SKIP: data, environments, the owner's node.json and magpie.env, keys, caches). Nothing outside the node's folder is ever written. Admin only (the routes name no scope)."""
import os, sys, json, time, base64, shutil, signal, asyncio, hashlib, subprocess
from pathlib import Path
from starlette.exceptions import HTTPException
from core import settings, journal
from core.http import Router, json_body

router = Router()
CODE_SKIP = {"dirs": [".git", ".venv", "venv", "data", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "node_modules"], "names": ["node.json", "magpie.env", ".env", "first_admin.key"], "suffixes": [".pyc", ".pyo", ".save", ".bak", ".log", ".key"]}
def _keep() -> int: return max(1, int(settings.get("code.keep", 5) or 5))   # earlier versions kept; the oldest beyond this are removed when a new one is kept
CHECK_IMPORT = "import core.contract, core.http, core.settings, core.modules, core.api"   # what must import in the changed copy before it replaces this one

def _kept_dir() -> Path: return settings.STATE / "code"

def _is_code(rel: Path) -> bool: return not (set(rel.parts[:-1]) & set(CODE_SKIP["dirs"]) or rel.name in CODE_SKIP["names"] or rel.suffix in CODE_SKIP["suffixes"] or rel.parts[0] in CODE_SKIP["dirs"])

def files(root: Path = settings.HOME) -> dict:
    """{relative path: sha256} of every code file under root."""
    out = {}
    for f in sorted(root.rglob("*")):
        rel = f.relative_to(root)
        if f.is_file() and _is_code(rel) and not f.is_symlink(): out[rel.as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out

def fingerprint(fs: dict) -> str: return hashlib.sha256("\n".join(f"{p} {h}" for p, h in sorted(fs.items())).encode()).hexdigest()[:16]

def _safe(rel: str) -> Path:
    """A path a manager names, as one inside the node's folder that is code; anything else is refused."""
    p = (settings.HOME / rel).resolve()
    if settings.HOME not in p.parents or not _is_code(p.relative_to(settings.HOME)): raise HTTPException(status_code=400, detail=f"not a code file of this node: {rel}")
    return p

def _supervised() -> str: return "systemd" if os.getenv("INVOCATION_ID") else "run.sh" if os.getenv("MAGPIE_SUPERVISED") else "container" if Path("/.dockerenv").exists() else ""

def _kept() -> list: return [json.loads((d / "kept.json").read_text()) for d in sorted(_kept_dir().glob("*/"), reverse=True) if (d / "kept.json").exists()] if _kept_dir().exists() else []

# --- Checking a changed copy ---

def _check(stage: Path) -> list:
    """Problems of a changed copy of the code: Python files that do not compile, then the node's core failing to import. Empty when it is sound."""
    bad = []
    for f in stage.rglob("*.py"):
        rel = f.relative_to(stage)
        if not _is_code(rel): continue
        try: compile(f.read_bytes(), str(rel), "exec")   # the check only: nothing is written
        except (SyntaxError, ValueError) as e: bad.append(f"{rel}: {type(e).__name__}: {e}"[:300])
    if bad: return bad
    r = subprocess.run([sys.executable, "-c", CHECK_IMPORT], cwd=stage, capture_output=True, text=True, timeout=120, env={**os.environ, "MAGPIE_HOME": str(stage), "MAGPIE_CODE_CHECK": "1"})
    return [f"importing the changed core failed: {(r.stderr or r.stdout).strip()[-1200:]}"] if r.returncode else []

def _stage(changes: dict, remove: list) -> Path:
    """A copy of the code with the changes applied, in a fresh folder beside the kept versions."""
    stage = _kept_dir() / f"stage-{int(time.time() * 1000)}"
    for rel in files(): (stage / rel).parent.mkdir(parents=True, exist_ok=True) or shutil.copy2(settings.HOME / rel, stage / rel)
    for rel, data in changes.items(): (stage / rel).parent.mkdir(parents=True, exist_ok=True) or (stage / rel).write_bytes(data)
    for rel in remove: (stage / rel).unlink(missing_ok=True)
    return stage

# --- Changing the code ---

def _apply(changes: dict, remove: list, label: str) -> dict:
    """Keeps every file about to be replaced or removed, and the names of the new ones (so a rollback removes them), then writes. Answers what was done."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    keep, now = _kept_dir() / stamp, files()
    for rel in [r for r in [*changes, *remove] if r in now]: (keep / "files" / rel).parent.mkdir(parents=True, exist_ok=True) or shutil.copy2(settings.HOME / rel, keep / "files" / rel)
    keep.mkdir(parents=True, exist_ok=True)
    rec = {"id": stamp, "t": time.time(), "from": label, "before": fingerprint(now), "replaced": sorted(r for r in changes if r in now), "added": sorted(r for r in changes if r not in now), "removed": sorted(r for r in remove if r in now)}
    for rel, data in changes.items(): _safe(rel).parent.mkdir(parents=True, exist_ok=True) or _safe(rel).write_bytes(data)
    for rel in remove: _safe(rel).unlink(missing_ok=True)
    rec["after"] = fingerprint(files())
    (keep / "kept.json").write_text(json.dumps(rec, indent=1))
    for old in sorted(d for d in _kept_dir().glob("*/") if (d / "kept.json").exists())[:-_keep()]: shutil.rmtree(old)
    return rec

def _requirements(changed: bool) -> str:
    """When requirements.txt changed, the node's own environment takes it (pip, into the Python running this node). Answers pip's last lines."""
    if not changed: return ""
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "-r", str(settings.HOME / "requirements.txt")], capture_output=True, text=True, timeout=1800)
    return ("requirements: " + ("installed" if not r.returncode else "pip failed: " + (r.stderr or r.stdout).strip()[-600:]))

def _restart_later(by: str):
    if by: asyncio.get_running_loop().call_later(1.5, os.kill, os.getpid(), signal.SIGTERM)

# --- Routes ---

@router.get("/node/code")
async def code_view(request):
    fs = await asyncio.to_thread(files)
    return {"fingerprint": fingerprint(fs), "files": fs, "home": str(settings.HOME), "skip": CODE_SKIP, "kept": _kept(), "kept_bytes": sum(f.stat().st_size for f in _kept_dir().rglob("*") if f.is_file()) if _kept_dir().exists() else 0, "keep": _keep(), "supervised": _supervised(), "python": sys.version.split()[0]}

@router.put("/node/code")
async def code_put(request):
    b = await json_body(request)
    changes, remove = {str(k): base64.b64decode(v) for k, v in (b.get("files") or {}).items()}, [str(r) for r in b.get("remove") or []]
    for rel in [*changes, *remove]: _safe(rel)
    if not changes and not remove: return {"changed": False, "fingerprint": fingerprint(await asyncio.to_thread(files))}
    stage = await asyncio.to_thread(_stage, changes, remove)
    try: bad = await asyncio.to_thread(_check, stage)
    finally: shutil.rmtree(stage, ignore_errors=True)
    if bad: raise HTTPException(status_code=422, detail={"refused": "the changed code does not check out; nothing was changed", "problems": bad[:20]})
    rec = await asyncio.to_thread(_apply, changes, remove, str(b.get("from") or "a manager"))
    pip = await asyncio.to_thread(_requirements, "requirements.txt" in changes)
    by = _supervised()
    journal.event("node.code", id=rec["id"], before=rec["before"], after=rec["after"], replaced=len(rec["replaced"]), added=len(rec["added"]), removed=len(rec["removed"]), by=by)
    _restart_later(by)
    return {**rec, "requirements": pip, "restarting": bool(by), "by": by, "note": "" if by else "this node was started by hand: start it again for the new code to run"}

@router.post("/node/code/rollback")
async def code_rollback(request):
    """The last kept version back: replaced and removed files restored, files the update added removed. The kept version is used up (the next rollback goes one further back)."""
    kept = _kept()
    if not kept: raise HTTPException(status_code=404, detail="no earlier version is kept here")
    rec, d = kept[0], _kept_dir() / kept[0]["id"]
    for rel in rec["added"]: _safe(rel).unlink(missing_ok=True)
    for rel in [*rec["replaced"], *rec["removed"]]: _safe(rel).parent.mkdir(parents=True, exist_ok=True) or shutil.copy2(d / "files" / rel, _safe(rel))
    shutil.rmtree(d)
    pip = await asyncio.to_thread(_requirements, "requirements.txt" in rec["replaced"])
    by = _supervised()
    journal.event("node.code.rollback", id=rec["id"], to=rec["before"], by=by)
    _restart_later(by)
    return {"back_to": rec["before"], "fingerprint": fingerprint(await asyncio.to_thread(files)), "requirements": pip, "restarting": bool(by), "by": by}
