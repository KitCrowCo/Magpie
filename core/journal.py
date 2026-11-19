"""core/journal.py - what this node did: one event per line in <logs location>/journal/journal.<first id>.jsonl, {"id", "t", "kind", ...fields}. A reader pulls (modules/fleet: POST /fleet/events); nothing is pushed, so a node never needs to know who manages it.
Ids rise by one per event within one epoch (journal/epoch: a random id made when the journal starts empty), so a reader tells a continued journal from a reset one. A file rotates at fleet.journal.max_mb; files beyond fleet.journal.keep_files are deleted, oldest first.
A write never raises into its caller: a failure is counted (HOOK_ERRORS, shown in /node and /fleet/state). This is the one place errors are caught on purpose - the journal must never fail a user's request."""
import os, json, time, uuid, threading, collections
from pathlib import Path
from core import settings

def jdir() -> Path: return settings.location("logs") / "journal"
ON = os.getenv("FLEET", "1") != "0"
HOOK_ERRORS = {"n": 0, "last": ""}
J = {"id": 0, "epoch": "", "file": None, "fh": None}
RECENT = collections.deque(maxlen=5000)   # the newest events, so a reader that keeps up never touches the files
_LOCK = threading.Lock()                  # events come from the event loop and from threads
_RESERVED = ("id", "t", "kind")           # a field with one of these names is stored as id_, t_, kind_

def now() -> float: return round(time.time(), 3)
def files() -> list: return sorted(jdir().glob("journal.*.jsonl"))   # zero-padded first ids, so name order is id order
def first_id(f: Path) -> int: return int(f.name.split(".")[1])
def _cfg(k: str, d): return settings.get(f"fleet.journal.{k}", d)

def hook_error(e: BaseException):
    HOOK_ERRORS.update(n=HOOK_ERRORS["n"] + 1, last=f"""{time.strftime("%Y-%m-%d %H:%M:%S")} {type(e).__name__}: {e}"""[:300])
    print(f"""[JOURNAL] write failed ({HOOK_ERRORS["n"]} so far): {HOOK_ERRORS["last"]}""", flush=True)

def _last_id(f: Path) -> int:
    """The id of the last complete event in f; a line cut off by a crash is skipped."""
    with open(f, "rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 65536))
        lines = fh.read().decode("utf-8", "replace").splitlines()
    for line in reversed(lines):
        try: return int(json.loads(line)["id"])
        except (ValueError, KeyError, TypeError): continue
    return first_id(f) - 1

def start():
    if not ON: print("[JOURNAL] off (FLEET=0)", flush=True); return
    global RECENT
    jdir().mkdir(parents=True, exist_ok=True)
    fs, ep = files(), jdir() / "epoch"
    if not fs or not ep.exists(): ep.write_text(uuid.uuid4().hex[:12])   # an empty journal starts a new epoch: readers start over instead of waiting for ids they already have
    J["epoch"], J["id"] = ep.read_text().strip(), _last_id(fs[-1]) if fs else 0
    J["file"] = fs[-1] if fs else jdir() / f"journal.{1:012d}.jsonl"
    J["fh"] = open(J["file"], "a", buffering=1, encoding="utf-8")
    RECENT = collections.deque(maxlen=int(_cfg("memory_events", 5000)))
    RECENT.extend(scan(max(0, J["id"] - RECENT.maxlen)))

def stop():
    if J["fh"]: J["fh"].close(); J["fh"] = None

def reopen(move_to: Path = None):
    """For a move of the logs location: closed, moved by the caller's function, opened at the new place, with ids and epoch continuing."""
    with _LOCK:
        if J["fh"]: J["fh"].close()
        J["fh"] = None

def _rotate():
    J["fh"].close()
    J["file"] = jdir() / f"""journal.{J["id"]:012d}.jsonl"""   # named after the first id it holds (the one being written)
    J["fh"] = open(J["file"], "a", buffering=1, encoding="utf-8")
    for f in files()[:-int(_cfg("keep_files", 5))]: f.unlink(missing_ok=True)

def record(kind: str, data: dict):
    if not (ON and J["fh"]): return
    try:
        with _LOCK:
            J["id"] += 1
            rec = {"id": J["id"], "t": now(), "kind": kind, **{(f"{k}_" if k in _RESERVED else k): v for k, v in data.items()}}
            if J["fh"].tell() > float(_cfg("max_mb", 20)) * 1e6: _rotate()
            J["fh"].write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            RECENT.append(rec)
    except Exception as e: hook_error(e)

def event(kind: str, /, **data): record(kind, data)   # kind positional-only: a field may itself be called kind

def scan(after: int, limit: int = 10 ** 9) -> list:
    """Events after the id `after`, read from the files (a reader catching up); a line cut off by a crash is skipped."""
    fs, out = files(), []
    for f in fs[max([i for i, f in enumerate(fs) if first_id(f) <= after + 1], default=0):]:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try: rec = json.loads(line)
                except ValueError: continue
                if rec.get("id", 0) > after:
                    out.append(rec)
                    if len(out) >= limit: return out
    return out

def read(after: int, limit: int) -> list:
    with _LOCK: recent = list(RECENT)
    if recent and after >= recent[0]["id"] - 1: return [r for r in recent if r["id"] > after][:limit]
    return scan(after, limit)
