# Speech on PyTorch: text to speech in a cloned voice, the same voice in every language the model knows (Qwen3-TTS Base: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian).
# A voice is a clone folder under CLONES_DIR (shared with the CPU speech module): <name>/ref.wav (a few seconds of clear speech) + ref.txt (its exact words). The voice id is "qwen3:<name>".
# POST /api/clones makes a clone folder from a recording sent to it (any voice the person has the right to use: another engine's voice, their own); a name that exists is refused, never replaced.
# Everything loads from local files: the model is a full model folder (with its speech_tokenizer/ inside) under MODEL_DIR; HF_HUB_OFFLINE is set so nothing is fetched at run time.
import os
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import io, re, json, time, wave, base64, asyncio
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response

# --- Config ---

MODELS_DIR = Path(os.getenv("SPEECH_MODELS_DIR", "/models"))
CLONES_DIR = MODELS_DIR / "clones"
MODEL_DIR = MODELS_DIR / "qwen3-tts" / os.getenv("QWEN3_TTS_MODEL", "Qwen3-TTS-12Hz-0.6B-Base")
DEVICE = os.getenv("TTS_DEVICE", "auto")      # auto: the GPU when torch sees one, else the processor
DTYPE = os.getenv("TTS_DTYPE", "auto")        # auto: bfloat16 on a GPU, float32 on the processor (bfloat16 is not supported there)
PREFIX = "qwen3:"
LANGS = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean", "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese", "es": "Spanish", "it": "Italian"}   # ISO code -> the model's language name; anything else is "Auto" (the model decides from the text)
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

_state: dict = {}            # "model": the loaded model; "prompts": clone name -> (ref.wav mtime, prompt items)
_lock = asyncio.Lock()       # one synthesis at a time: one model, one device

# --- Model and voices ---

def _model():
    if "model" not in _state:
        import torch
        from qwen_tts import Qwen3TTSModel
        if not (MODEL_DIR / "speech_tokenizer" / "config.json").exists(): raise HTTPException(status_code=503, detail=f"no Qwen3-TTS model folder at {MODEL_DIR} (it needs the whole model folder, speech_tokenizer/ included) - install this module with the model option, or put the folder there")
        gpu = torch.cuda.is_available() if DEVICE == "auto" else DEVICE != "cpu"
        dev, dtype = ("cuda:0" if DEVICE == "auto" else DEVICE) if gpu else "cpu", getattr(torch, DTYPE) if DTYPE != "auto" else (torch.bfloat16 if gpu else torch.float32)
        _state["model"] = Qwen3TTSModel.from_pretrained(str(MODEL_DIR), device_map=dev, dtype=dtype, local_files_only=True)
        _state["device"] = f"""{dev} {str(dtype).removeprefix("torch.")}"""
    return _state["model"]

def _clone_dir(name: str) -> Path:
    d = CLONES_DIR / Path(name).name
    if not ((d / "ref.wav").exists() and (d / "ref.txt").exists()): raise HTTPException(status_code=404, detail=f"voice {PREFIX}{name} needs {d}/ref.wav (the recording) and {d}/ref.txt (its exact words)")
    return d

def _prompt(name: str):
    """The voice's clone prompt, built once from its recording and kept until the recording changes."""
    d = _clone_dir(name)
    at = (d / "ref.wav").stat().st_mtime
    kept = _state.setdefault("prompts", {}).get(name)
    if kept and kept[0] == at: return kept[1]
    words = (d / "ref.txt").read_text().strip()
    if not words: raise HTTPException(status_code=422, detail=f"{d}/ref.txt is empty: write the exact words of ref.wav (the model follows the recording through them)")
    items = _model().create_voice_clone_prompt(ref_audio=str(d / "ref.wav"), ref_text=words)
    _state["prompts"][name] = (at, items)
    return items

def voices() -> list: return [f"{PREFIX}{d.name}" for d in sorted(CLONES_DIR.glob("*")) if (d / "ref.wav").exists() and (d / "ref.txt").exists()] if CLONES_DIR.exists() else []

# --- Synthesis ---

def _wav(samples, sr: int) -> bytes:
    """float32 samples in [-1, 1] -> 16-bit mono WAV bytes."""
    import numpy as np
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes((np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0) * 32767).astype("<i2").tobytes())
    return buf.getvalue()

def _speak_sync(text: str, voice: str, language: str) -> dict:
    if not voice.startswith(PREFIX): raise HTTPException(status_code=400, detail=f"""voice {voice!r} is not one of this module's ({PREFIX}<clone name>): {", ".join(voices()) or "no clones yet - add one with POST /api/clones"}""")
    t0, lang = time.time(), LANGS.get((language or "").lower().split("-")[0], "Auto")
    wavs, sr = _model().generate_voice_clone(text=text, language=lang, voice_clone_prompt=_prompt(voice[len(PREFIX):]))
    return {"wav": _wav(wavs[0], sr), "sample_rate": sr, "duration_s": round(len(wavs[0]) / sr, 2), "elapsed_s": round(time.time() - t0, 2), "voice": voice, "language": lang}

# --- App ---

app = FastAPI(title="Speech on PyTorch - cloned voices in many languages")

@app.get("/health")
async def health(): return {"status": "ok", "model": MODEL_DIR.name, "model_present": (MODEL_DIR / "speech_tokenizer" / "config.json").exists(), "loaded": "model" in _state, "device": _state.get("device", ""), "voices": voices(), "capabilities": ["text_to_speech", "voice_clone_add"]}

@app.post("/api/speak")
async def api_speak(request: Request):
    """body: {text, voice (qwen3:<clone>; "model" is taken when voice is absent), language? (ISO code; absent: the model decides from the text)}. length_scale is not supported by this model and is refused rather than ignored. ?raw=1 returns audio/wav bytes."""
    b = await request.json()
    if b.get("length_scale") not in (None, "", 0, 1, 1.0): raise HTTPException(status_code=400, detail="this voice model has no speed setting: leave length_scale empty, or use another engine for slower or faster speech")
    if not str(b.get("text", "")).strip(): raise HTTPException(status_code=400, detail="nothing to say: text is empty")
    async with _lock: r = await asyncio.to_thread(_speak_sync, str(b["text"]), str(b.get("voice") or b.get("model") or ""), str(b.get("language") or ""))
    if request.query_params.get("raw"): return Response(r["wav"], media_type="audio/wav")
    return {"audio_b64": base64.b64encode(r.pop("wav")).decode(), "format": "wav", **r}

@app.get("/api/voices")
async def api_voices(): return {"default": next(iter(voices()), ""), "engines": ["qwen3"], "installed": voices(), "languages": sorted(LANGS), "how_to_add": "POST /api/clones {name, audio_b64 (wav), text: its exact words} - or a folder clones/<name>/ with ref.wav and ref.txt in the audio models location"}

@app.post("/api/clones")
async def api_clone_add(request: Request):
    """body: {name, audio_b64 (wav: a few seconds of clear speech in one voice), text (its exact words)}. Makes clones/<name>/; a name already there is refused (409), so no voice is ever replaced."""
    b = await request.json()
    name, audio, words = str(b.get("name", "")), base64.b64decode(str(b.get("audio_b64", "")).split(",", 1)[-1] or ""), str(b.get("text", "")).strip()
    if not NAME.match(name): raise HTTPException(status_code=400, detail="name: letters, digits, _ and - (up to 64), starting with a letter or digit")
    if not words: raise HTTPException(status_code=400, detail="text: the exact words spoken in the recording are needed")
    if audio[:4] != b"RIFF": raise HTTPException(status_code=400, detail="audio_b64 must be a WAV recording")
    d = CLONES_DIR / name
    if d.exists(): raise HTTPException(status_code=409, detail=f"a voice {name!r} exists already ({d}); choose another name - existing voices are never replaced")
    d.mkdir(parents=True)
    (d / "ref.wav").write_bytes(audio)
    (d / "ref.txt").write_text(words + "\n")
    with wave.open(io.BytesIO(audio), "rb") as r: seconds = r.getnframes() / max(r.getframerate(), 1)
    return {"voice": f"{PREFIX}{name}", "seconds": round(seconds, 2), "note": "3 to 15 seconds of clear speech clones best" if not 3 <= seconds <= 15 else ""}
