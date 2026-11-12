"""core/auth.py - API keys: every route but /health needs one. Keys are random and only their hashes are stored (state/keys.json).
First start with no key: one admin key is made and written once to state/first_admin.key (only the owner can read it) - give it to the portal connection's API Key, then delete the file. MAGPIE_ADMIN_KEY (from the shared .env) instead sets the admin key directly.
node.json "auth": {"open": true} turns checking off for a node reachable only from a trusted network; it is logged at every start and shown in /node. Keys travel in plain text unless the node runs TLS (node.json "bind": {"tls_cert", "tls_key"}) or sits behind a tunnel."""
import os, json, time, hashlib, secrets
from starlette.exceptions import HTTPException
from core import settings

SCOPES = {"read": 0, "infer": 1, "admin": 2}
_KEYS: dict = {}   # sha256 hex -> {"name", "scope", "created"}
_PATH = settings.STATE / "keys.json"
LOCAL_KEY = secrets.token_urlsafe(24)   # for this process's own calls to itself (tests, the passive worker): changes every start, never written anywhere

def _hash(k: str) -> str: return hashlib.sha256(k.encode()).hexdigest()
def is_open() -> bool: return bool(settings.owner("auth.open"))

def load():
    if _PATH.exists(): _KEYS.update(json.loads(_PATH.read_text()))
    if os.getenv("MAGPIE_ADMIN_KEY"): _KEYS[_hash(os.environ["MAGPIE_ADMIN_KEY"])] = {"name": "admin (MAGPIE_ADMIN_KEY)", "scope": "admin", "created": 0}
    _KEYS[_hash(LOCAL_KEY)] = {"name": "local", "scope": "admin", "created": 0, "volatile": True}
    if is_open(): print("[AUTH] OPEN: node.json auth.open is set - anything that can reach this port may do anything here", flush=True)
    elif not any(not v.get("volatile") for v in _KEYS.values()):
        k = new_key("admin (first start)", "admin")
        fd = os.open(settings.STATE / "first_admin.key", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f: f.write(k + "\n")
        print(f"""[AUTH] no keys yet - an admin key was written to {settings.STATE / "first_admin.key"} (put it in the portal connection's API Key, then delete the file)""", flush=True)

def new_key(name: str, scope: str) -> str:
    if scope not in SCOPES: raise ValueError(f"scope must be one of {list(SCOPES)}")
    k = f"mgp_{secrets.token_urlsafe(32)}"
    _KEYS[_hash(k)] = {"name": name, "scope": scope, "created": round(time.time())}
    _save_durable()
    return k

def _save_durable(): _PATH.write_text(json.dumps({h: v for h, v in _KEYS.items() if not v.get("volatile") and "MAGPIE_ADMIN_KEY" not in v["name"]}, indent=1))
def list_keys() -> list: return [{"id": h[:12], **{k: v for k, v in r.items() if k != "volatile"}} for h, r in _KEYS.items() if not r.get("volatile")]

def revoke(key_id: str) -> bool:
    hits = [h for h in _KEYS if h.startswith(key_id) and not _KEYS[h].get("volatile")]
    for h in hits: _KEYS.pop(h)
    _save_durable()
    return bool(hits)

def _presented(request) -> str:
    h = request.headers.get("authorization", "")
    return h[7:].strip() if h.lower().startswith("bearer ") else request.headers.get("x-api-key", "")

def check(request, scope: str) -> dict:
    """The key record the request carries, or 401 / 403. Compared by hash, so timing says nothing about a stored key."""
    if scope == "open" or is_open(): return {"name": "open", "scope": "admin"}
    k = _presented(request)
    rec = _KEYS.get(_hash(k)) if k else None
    if not rec: raise HTTPException(status_code=401, detail="missing or unknown API key (Authorization: Bearer <key>)")
    if SCOPES[rec["scope"]] < SCOPES[scope]: raise HTTPException(status_code=403, detail=f"""this key has scope {rec["scope"]}; this route needs {scope}""")
    return rec
