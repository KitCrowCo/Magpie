"""modules/speech/listing.py - the wake words this machine has, for the speech program's list of what it offers (module.json "call"): one asset per openWakeWord model in audio_models/wakeword, leaving out the shared feature models every word uses."""
from core import settings

SHARED = {"melspectrogram", "embedding_model", "silero_vad"}

def wake_words() -> list:
    d = settings.location("audio_models") / "wakeword"
    return [{"name": p.stem, "capabilities": ["wake_detect"], "size": p.stat().st_size, "format": "onnx", "mtime": p.stat().st_mtime, "paths": [settings.where(p)]} for p in sorted(d.glob("*.onnx")) if p.stem not in SHARED] if d.exists() else []
