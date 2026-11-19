"""listing.py - what the image program's video model folders offer, read from their own config files.
The node's process imports this to list models (module.json: "call"), so nothing here needs the worker's libraries or loads a model.
A folder under <image models>/transformers in the diffusers layout (<name>/transformer/config.json) is one video model; what it can do is read from that file, by the same rule the worker uses (video.py: _kind)."""
import re, json
from core import settings

CAPS = {"t2v": ["text_to_video"], "i2v": ["image_to_video"], "inp": ["text_to_video", "image_to_video", "start_end_to_video"]}   # by kind: plain, start frame only, Fun InP (start, or start and end)

def _kind(cfg: dict, name: str) -> str:
    c = cfg.get("out_channels", 16)
    ch = cfg.get("in_channels", c)
    return "inp" if ch == 2 * c + 1 else ("control" if re.search(r"control|pose", name, re.I) else "i2v") if ch == 2 * c else "t2v" if ch == c else "unknown"

def video() -> list:
    """[{"name": the folder's path below the image models folder, "capabilities", "size", "format": "folder", "mtime"}] for every video model folder this node can run; a control model, or one of an unknown shape, is left out (the worker lists those as "listed only")."""
    root = settings.location("image_models")
    found = [(d, CAPS.get(_kind(json.loads((d / "transformer" / "config.json").read_text()), d.name))) for d in sorted((root / "transformers").glob("*/")) if (d / "transformer" / "config.json").is_file()]
    return [{"name": str(d.relative_to(root)), "capabilities": caps, "size": sum(f.stat().st_size for f in d.rglob("*") if f.is_file()), "format": "folder", "mtime": d.stat().st_mtime} for d, caps in found if caps]
