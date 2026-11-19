"""modules/text_encoder/listing.py - which model files this node can offer as the text encoder, found by what the file is (its GGUF header), not by its name.
Looked through: the Ollama store under "models" (manifests/ and blobs/, named as Ollama names them, "ollama/<name>"), GGUF files under models/gguf_local/**, and GGUF files at the top of image_models and under image_models/text_encoders/**.
A file fits when its header says the architecture and sizes the encoder needs (settings text_encoder.fits: architecture, hidden size, and at least the deepest layer read). Community builds, quantisations and name add-ons (MTP, abliterated, ...) do not matter; only the header does.
The owner decides past that: text_encoder.accept names files offered whatever their header says; text_encoder.offer "all" offers every GGUF found, each marked with what is known of its fit.
Used by the node (listing) and by the encoder service (a name back to its file). Standard library only."""
import re, json
from functools import lru_cache
from pathlib import Path

def _ollama(models: Path) -> list:
    """(name, blob path) of every model in an Ollama store: manifests/<host>/<namespace>/<repo>/<tag>, named as Ollama shows it (library models without host and namespace, :latest left off)."""
    out, mdir = [], models / "manifests"
    for mp in sorted(p for p in mdir.rglob("*") if p.is_file()) if mdir.exists() else []:
        parts = mp.relative_to(mdir).parts
        try: layer = next(l for l in json.loads(mp.read_text()).get("layers", []) if l.get("mediaType") == "application/vnd.ollama.image.model")
        except (ValueError, StopIteration, OSError): continue   # a manifest that is not a model's (a projector, a partial pull)
        blob = models / "blobs" / layer["digest"].replace(":", "-")
        host = [x for x in parts[:-2] if x not in ("registry.ollama.ai", "library")]
        if blob.exists(): out.append(("ollama/" + "/".join([*host, parts[-2]]) + ("" if parts[-1] == "latest" else f":{parts[-1]}"), blob))
    return out

def files(roots: dict) -> list:
    """(name, path) of every GGUF under the roots {"models": ..., "image_models": ...}, at any depth; whether one can encode is decided by its header (fit), not by its folder. Split files are taken by their first part."""
    out = _ollama(Path(roots["models"])) if roots.get("models") else []
    for root, globs in (("models", ["**/*.gguf"]), ("image_models", ["**/*.gguf"])):   # anywhere under the node's model places: where a file is kept is the owner's choice, and the header (fit) says what it is
        base = Path(roots.get(root) or "/nonexistent")
        out += [(f"{root}/{p.relative_to(base).as_posix()}", p) for g in globs for p in sorted(base.glob(g)) if p.is_file() and "mmproj" not in p.name.lower() and not re.search(r"-0*[2-9]\d*-of-\d+\.gguf$", p.name)]
    return out

def path_of(name: str, roots: dict) -> Path:
    """A listed name back to its file; a name this node does not list is refused."""
    found = dict(files(roots)).get(name)
    if not found: raise ValueError(f"not an encoder model this node lists: {name}")
    return found

@lru_cache(maxsize=256)
def header(path: str, mtime: float) -> dict:
    """What the GGUF header says, the parts the encoder cares about. mtime: a changed file is read again."""
    from modules.llama.gguf_lite import Reader   # the node's GGUF header reader (standard library only)
    try: r = Reader(path)
    except Exception as e: return {"error": f"{type(e).__name__}: {e}"[:200]}
    arch = r.fields.get("general.architecture")
    return {"architecture": arch, "hidden": r.fields.get(f"{arch}.embedding_length"), "layers": r.fields.get(f"{arch}.block_count"), "name": r.fields.get("general.name") or ""}

def fit(h: dict, want: dict) -> tuple:
    """("yes" | "no" | "unknown", why), against {"architecture": [...], "hidden": n, "min_layers": n}."""
    if h.get("error"): return "unknown", h["error"]
    miss = [f"""architecture {h["architecture"]}, not {"/".join(want["architecture"])}"""] * (h["architecture"] not in want["architecture"]) + [f"""hidden size {h["hidden"]}, not {want["hidden"]}"""] * (h["hidden"] != want["hidden"]) + [f"""{h["layers"]} layers, needs {want["min_layers"]}"""] * ((h["layers"] or 0) < want["min_layers"])
    return ("no", "; ".join(miss)) if miss else ("yes", f"""{h["architecture"]}, hidden {h["hidden"]}, {h["layers"]} layers""")

def models() -> list:
    """The node's listing for text_encode (called by the adapter's listing): what fits, what the owner accepts, or everything when the owner says so. Each says how its fit is known."""
    from core import settings
    roots, want = {"models": settings.location("text_models"), "image_models": settings.location("image_models")}, settings.get("text_encoder.fits", {}) or {}
    accept, offer = set(settings.get("text_encoder.accept", []) or []), settings.get("text_encoder.offer", "fitting")
    out = []
    for name, p in files(roots):
        st = p.stat()
        ok, why = fit(header(str(p), st.st_mtime), want)
        if ok == "yes" or name in accept or offer == "all": out.append({"name": name, "capabilities": ["text_encode"], "size": st.st_size, "format": "gguf", "mtime": st.st_mtime, "fits": "owner" if name in accept and ok != "yes" else ok, "why": why})
    return out
