# Lyrebird - local speech node: STT (faster-whisper, CPU int8) + TTS (Piper, CPU/onnxruntime), FOSS end to end.
# JSON in / JSON out (audio as base64) so the portal's generic call_capability can drive it with no binary-transfer special case.
# Capabilities this node provides: speech_to_text (/api/transcribe), text_to_speech (/api/speak).
# /api/converse is a convenience loop (listen -> LLM -> speak) for the built-in test page; in the portal the same loop is a pipeline (transcribe node -> generate node -> speak node), so cadence/voice/sentence-streaming work later lands in pipelines, not here.
import os
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import io, re, json, time, wave, glob, base64, asyncio
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, Response
import httpx, uvicorn

# --- Config ---

MODELS_DIR = Path(os.getenv("SPEECH_MODELS_DIR", "/models"))
WHISPER_DIR, VOICES_DIR = MODELS_DIR / "whisper", MODELS_DIR / "piper"
STT_MODEL = os.getenv("STT_MODEL", "small")              # tiny / base / small / medium / large-v3 / distil-* or a local CTranslate2 dir
STT_DEVICE = os.getenv("STT_DEVICE", "cpu")
STT_COMPUTE = os.getenv("STT_COMPUTE_TYPE", "int8")     # int8 is the fast CPU path; float32 for max accuracy
STT_THREADS = int(os.getenv("STT_THREADS", "0"))        # 0 = library default
TTS_VOICE = os.getenv("TTS_VOICE", "")   # blank = the first voice installed
TTS_ENGINES = {e.strip() for e in os.getenv("TTS_ENGINES", "piper,kokoro").split(",") if e.strip()}   # engines this node serves - see the TTS section
KOKORO_DIR, CLONES_DIR, F5_DIR = MODELS_DIR / "kokoro", MODELS_DIR / "clones", MODELS_DIR / "f5"
KOKORO_MODEL, KOKORO_VOICES = os.getenv("KOKORO_MODEL", "kokoro-v1.0.onnx"), os.getenv("KOKORO_VOICES", "voices-v1.0.bin")
KOKORO_LANGS = {"a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi", "i": "it", "p": "pt-br", "j": "ja", "z": "cmn"}   # first letter of a Kokoro voice id is its language
F5_MODEL, F5_CKPT, F5_DEVICE = os.getenv("F5_MODEL", "F5TTS_v1_Base"), os.getenv("F5_CKPT", "F5TTS_v1_Base/model_1250000.safetensors"), os.getenv("F5_DEVICE") or None   # F5_DEVICE blank = F5-TTS picks (cuda on ROCm builds)
CLONE_MAX_REF_S = 12.0   # F5-TTS clips longer references internally
_ENGINES: dict = {}
LLM_URL = os.getenv("LLM_URL", "")                      # an Ollama-wire endpoint (llama_direct or Ollama) for /api/converse, e.g. http://192.168.1.193:8790
LLM_MODEL = os.getenv("LLM_MODEL", "")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8795))
_STT: dict = {}
_TTS: dict = {}
_stt_lock, _tts_lock = asyncio.Lock(), asyncio.Lock()   # one transcription / one synthesis at a time - CPU-bound, parallel runs only slow each other down

# --- Helpers ---

def _b64(data: bytes) -> str: return base64.b64encode(data).decode()

def _unb64(s: str) -> bytes:
    if not s: raise HTTPException(status_code=400, detail="audio_b64 is empty")
    return base64.b64decode(s.split(",", 1)[1] if s.startswith("data:") else s)

def speakable(text: str) -> str:
    """Strips what a voice should not read aloud: thinking blocks, code fences, markdown markers, link targets."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```.*?```", " (code omitted) ", text, flags=re.S)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_#`>|]+", "", text)
    return re.sub(r"\s+", " ", text).strip()

# --- STT ---

def stt_model(name: str):
    if name not in _STT:
        try: from faster_whisper import WhisperModel
        except ImportError: raise HTTPException(status_code=501, detail="this node has no speech-to-text (faster-whisper is not installed in this container) - point Speech-to-Text at the CPU speech node")
        WHISPER_DIR.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        _STT[name] = WhisperModel(name, device=STT_DEVICE, compute_type=STT_COMPUTE, download_root=str(WHISPER_DIR), cpu_threads=STT_THREADS)
        print(f"[STT] loaded {name} ({STT_DEVICE}/{STT_COMPUTE}) in {time.time() - t0:.1f}s", flush=True)
    return _STT[name]

def _transcribe_sync(audio: bytes, model: str, language, vad: bool, prompt) -> dict:
    t0 = time.time()
    segs, info = stt_model(model).transcribe(io.BytesIO(audio), language=language or None, vad_filter=vad, initial_prompt=prompt or None, beam_size=5)
    segs = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(), "no_speech_prob": round(s.no_speech_prob, 3), "avg_logprob": round(s.avg_logprob, 3)} for s in segs]
    el = time.time() - t0
    return {"text": " ".join(s["text"] for s in segs).strip(), "segments": segs, "language": info.language, "language_probability": round(info.language_probability, 3), "duration_s": round(info.duration, 2), "elapsed_s": round(el, 2), "rtf": round(el / info.duration, 3) if info.duration else None, "model": model}

async def transcribe(audio: bytes, model: str = "", language=None, vad: bool = True, prompt=None) -> dict:
    async with _stt_lock: return await asyncio.to_thread(_transcribe_sync, audio, model or STT_MODEL, language, vad, prompt)

# --- TTS ---
# A voice id is "<engine>:<name>". A bare name is a Piper voice, so ids saved before engines existed keep working.
#   piper   - .onnx voices under VOICES_DIR. CPU, fast. GPL code; per-voice licenses on the rhasspy/piper-voices cards.
#   kokoro  - kokoro-onnx (MIT) with the Kokoro-82M model (Apache-2.0) under KOKORO_DIR. CPU, ~2x faster than real time. 54 voices, e.g. kokoro:af_sky.
#   clone   - zero-shot clones with F5-TTS (code MIT, weights CC-BY-NC: personal/non-commercial use) from CLONES_DIR/<name>/ref.wav + ref.txt. Needs PyTorch; run it on a GPU node.
# TTS_ENGINES lists what this node serves; anything else is refused with a message, so one API can run as several instances (CPU node: piper,kokoro / GPU node: clone).
# Every engine loads from local files only - nothing here downloads; model files arrive by an explicit one-off command (see README).

def _wav(samples, sr: int) -> bytes:
    """float32 samples in [-1, 1] -> 16-bit mono WAV bytes."""
    import numpy as np
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(int(sr))
        wf.writeframes((np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0) * 32767).astype("<i2").tobytes())
    return buf.getvalue()

def _speed(length_scale) -> float: return 1.0 if length_scale in (None, "", 0) else 1.0 / float(length_scale)  # length_scale (>1 slower) is the API's pace unit; kokoro/F5 take speed (>1 faster)

def _voice_file(voice: str) -> Path:
    hits = glob.glob(str(VOICES_DIR / "**" / f"{voice}.onnx"), recursive=True)
    if not hits: raise HTTPException(status_code=404, detail=f"piper voice {voice!r} is not under {VOICES_DIR} - fetch it with the one-off download command in the README")
    return Path(hits[0])

def _voice_repo_path(voice: str) -> str:
    """A Piper voice id (language-name-quality, e.g. en_US-lessac-medium) -> its path in the rhasspy/piper-voices repository without the extension (en/en_US/lessac/medium/en_US-lessac-medium)."""
    lang, rest = voice.split("-", 1)
    name, quality = rest.rsplit("-", 1)
    return f"""{lang.split("_")[0]}/{lang}/{name}/{quality}/{voice}"""

def tts_voice(voice: str):
    if voice not in _TTS:
        from piper import PiperVoice
        _TTS[voice] = PiperVoice.load(str(_voice_file(voice)))
    return _TTS[voice]

def _synth(v, text: str, wf, length_scale):
    """piper-tts changed its API between 1.2 (synthesize(text, wav, length_scale=)) and 1.3 (synthesize_wav(text, wav, syn_config=SynthesisConfig(...))) - both handled here."""
    try:
        from piper import SynthesisConfig
        v.synthesize_wav(text, wf, syn_config=SynthesisConfig(length_scale=length_scale) if length_scale else None)
    except ImportError:
        v.synthesize(text, wf, **({"length_scale": length_scale} if length_scale else {}))

def _piper_wav(name: str, text: str, length_scale) -> bytes:
    buf, v = io.BytesIO(), tts_voice(name)  # resolve the voice first - a missing voice must surface as its own 404, not as a half-written wav error
    with wave.open(buf, "wb") as wf: _synth(v, text, wf, length_scale)
    return buf.getvalue()

def _kokoro():
    if "kokoro" not in _ENGINES:
        from kokoro_onnx import Kokoro
        model, voices = KOKORO_DIR / KOKORO_MODEL, KOKORO_DIR / KOKORO_VOICES
        if not (model.exists() and voices.exists()): raise HTTPException(status_code=503, detail=f"kokoro needs {model} and {voices} - fetch them with the one-off download command in the README")
        _ENGINES["kokoro"] = Kokoro(str(model), str(voices))
    return _ENGINES["kokoro"]

_G2P: dict = {}   # Kokoro language -> misaki G2P, for the languages espeak-ng phonemizes badly

def _kokoro_wav(name: str, text: str, length_scale) -> bytes:
    k = _kokoro()
    if name not in k.get_voices(): raise HTTPException(status_code=404, detail=f"""kokoro voice {name!r} not in {KOKORO_VOICES} - available: {", ".join(k.get_voices())}""")
    lang = KOKORO_LANGS.get(name[:1], "en-us")
    try:
        # Kokoro handles Japanese/Mandarin text directly; no G2P needed for languages where G2P is still needed, add them back later
        samples, sr = k.create(text, voice=name, speed=_speed(length_scale), lang=lang)
    except ValueError as e: raise HTTPException(status_code=400, detail=f"kokoro: {e}")
    return _wav(samples, sr)

def _f5():
    if "f5" not in _ENGINES:
        from f5_tts.api import F5TTS
        ckpt, vocos = F5_DIR / F5_CKPT, F5_DIR / "vocos"
        if not (ckpt.exists() and (vocos / "config.yaml").exists() and (vocos / "pytorch_model.bin").exists()): raise HTTPException(status_code=503, detail=f"clone engine needs {ckpt} and {vocos}/config.yaml + pytorch_model.bin - fetch them with the one-off download command in the README")
        _ENGINES["f5"] = F5TTS(model=F5_MODEL, ckpt_file=str(ckpt), vocoder_local_path=str(vocos), device=F5_DEVICE)   # explicit local files: F5TTS() would otherwise fetch the checkpoint and vocoder from Hugging Face on its own
    return _ENGINES["f5"]

def _clone_wav(name: str, text: str, length_scale) -> bytes:
    d = CLONES_DIR / Path(name).name
    ref, txt, cfg = d / "ref.wav", d / "ref.txt", d / "settings.json"
    if not (ref.exists() and txt.exists()): raise HTTPException(status_code=404, detail=f"clone {name!r} needs {ref} (the reference recording) and {txt} (its exact words)")
    ref_text = txt.read_text().strip()
    if not ref_text: raise HTTPException(status_code=422, detail=f"{txt} is empty - F5-TTS would download a Whisper model to transcribe the reference itself; write the exact words instead")
    import soundfile as sf
    dur = sf.info(str(ref)).duration
    if dur > CLONE_MAX_REF_S: raise HTTPException(status_code=422, detail=f"{ref} is {dur:.1f} s; F5-TTS clips references over {CLONE_MAX_REF_S:.0f} s without telling you, and the transcript then no longer matches the audio - trim it at a phrase boundary and cut {txt} to the same words")
    s = json.loads(cfg.read_text()) if cfg.exists() else {}
    samples, sr, _ = _f5().infer(str(ref), ref_text, text, show_info=lambda *a, **k: None, nfe_step=int(s.get("nfe_step", 32)), cfg_strength=float(s.get("cfg_strength", 2.0)), speed=float(s.get("speed", 1.0)) * _speed(length_scale), seed=s.get("seed"))
    return _wav(samples, sr)

def _espeak_wav(name: str, text: str, length_scale) -> bytes:
    """eSpeak-ng Tagalog synthesis. name is ignored (uses 'fil' only). Requires espeak-ng system binary."""
    import subprocess
    try:
        speed = int(150 * _speed(length_scale))  # espeak speed in wpm; default 150
        proc = subprocess.run(["espeak-ng", "-v", "fil", "-s", str(speed), "-w", "/dev/stdout", text], capture_output=True, check=True, timeout=300.0)
        return proc.stdout
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="espeak-ng system binary not found - install with: apt-get install espeak-ng")
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"espeak-ng: {e.stderr.decode()[:200]}")

_ENGINE_FNS = {"piper": _piper_wav, "kokoro": _kokoro_wav, "clone": _clone_wav, "espeak": _espeak_wav}

def _split_voice(voice: str) -> tuple:
    engine, sep, name = voice.partition(":")
    return (engine, name) if sep else ("piper", voice)

def _speak_sync(text: str, voice: str, length_scale) -> dict:
    engine, name = _split_voice(voice)
    if engine not in _ENGINE_FNS: raise HTTPException(status_code=400, detail=f"""unknown voice engine {engine!r} in {voice!r} - engines: {", ".join(_ENGINE_FNS)}""")
    if engine not in TTS_ENGINES: raise HTTPException(status_code=400, detail=f"""engine {engine!r} is not served by this node (TTS_ENGINES={",".join(sorted(TTS_ENGINES))}) - point this conversation's text-to-speech connection at the node that serves it""")
    t0 = time.time()
    data = _ENGINE_FNS[engine](name, text, length_scale)
    with wave.open(io.BytesIO(data), "rb") as r: sr, dur = r.getframerate(), r.getnframes() / max(r.getframerate(), 1)
    return {"wav": data, "sample_rate": sr, "duration_s": round(dur, 2), "elapsed_s": round(time.time() - t0, 2), "voice": voice}

async def speak(text: str, voice: str = "", length_scale=None) -> dict:
    text = speakable(text)
    if not text: raise HTTPException(status_code=400, detail="nothing speakable in text")
    async with _tts_lock: return await asyncio.to_thread(_speak_sync, text, voice or default_voice(), length_scale)

def default_voice() -> str: return TTS_VOICE or next(iter(installed_voices()), "")

def installed_voices() -> list:
    """Every voice id this node can speak right now, engine-prefixed (Piper ids stay bare)."""
    out = sorted(Path(p).stem for p in glob.glob(str(VOICES_DIR / "**" / "*.onnx"), recursive=True)) if "piper" in TTS_ENGINES else []
    if "kokoro" in TTS_ENGINES and (KOKORO_DIR / KOKORO_VOICES).exists():
        import numpy as np
        out += [f"kokoro:{v}" for v in sorted(np.load(str(KOKORO_DIR / KOKORO_VOICES)).keys())]
    if "espeak" in TTS_ENGINES: out += ["espeak:fil"]
    if "clone" in TTS_ENGINES: out += [f"clone:{d.name}" for d in sorted(CLONES_DIR.glob("*")) if (d / "ref.wav").exists() and (d / "ref.txt").exists()]
    return out


# --- App ---

app = FastAPI(title="Lyrebird - local STT/TTS node")

import speaker_id
speaker_id.register(app, WHISPER_DIR.parent / "speaker")

@app.get("/health")
async def health(): return {"status": "ok", "stt_model": STT_MODEL, "stt_loaded": sorted(_STT), "tts_voice": default_voice(), "tts_engines": sorted(TTS_ENGINES), "tts_loaded": sorted(_TTS) + sorted(_ENGINES), "voices_installed": installed_voices(), "llm_url": LLM_URL or None, "capabilities": ["speech_to_text", "text_to_speech"]}

@app.post("/api/transcribe")
async def api_transcribe(request: Request):
    """body: {audio_b64 (any ffmpeg-decodable format: webm/ogg/mp4/wav/mp3), language?, model?, vad_filter?, initial_prompt?}"""
    b = await request.json()
    return await transcribe(_unb64(b.get("audio_b64", "")), b.get("model", ""), b.get("language"), bool(b.get("vad_filter", True)), b.get("initial_prompt"))

@app.post("/api/speak")
async def api_speak(request: Request):
    """body: {text, voice?, length_scale? (>1 slower)}. ?raw=1 returns audio/wav bytes instead of JSON."""
    b = await request.json()
    r = await speak(b.get("text", ""), b.get("voice", ""), b.get("length_scale"))
    if request.query_params.get("raw"): return Response(r["wav"], media_type="audio/wav")
    return {"audio_b64": _b64(r.pop("wav")), "format": "wav", **r}

@app.get("/api/voices")
async def api_voices(): return {"default": default_voice(), "engines": sorted(TTS_ENGINES), "installed": installed_voices(), "how_to_add": {"piper": "one-off download from huggingface.co/rhasspy/piper-voices into piper/", "kokoro": "kokoro-v1.0.onnx + voices-v1.0.bin in kokoro/ (54 voices, ids listed in installed)", "espeak": "Tagalog via eSpeak-ng system binary (included in docker). Naturally lower quality than Piper/Kokoro but fully offline and free.", "clone": "clones/<name>/ref.wav (<= 12 s) + ref.txt (its exact words)"}}

@app.get("/api/languages")
async def api_languages():
    """Language codes this node's recognizer accepts (faster-whisper's own table), for the portal's language select."""
    from faster_whisper.tokenizer import _LANGUAGE_CODES   # private name: if a faster-whisper upgrade renames it, this route fails loudly rather than offering a stale list
    return {"languages": sorted(_LANGUAGE_CODES)}

@app.post("/api/voices/pull")
async def api_voices_pull(request: Request):
    from huggingface_hub import hf_hub_download
    voice = (await request.json()).get("voice") or default_voice()
    rp = _voice_repo_path(voice)
    files = [await asyncio.to_thread(hf_hub_download, repo_id="rhasspy/piper-voices", filename=f"{rp}{ext}", local_dir=str(VOICES_DIR)) for ext in (".onnx", ".onnx.json")]
    _TTS.pop(voice, None)
    return {"status": "success", "voice": voice, "files": files}

@app.get("/api/llm_models")
async def api_llm_models(llm_url: str = ""):
    url = llm_url or LLM_URL
    if not url: return {"models": [], "note": "no LLM_URL configured"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0)) as c: return {"models": [m["name"] for m in (await c.get(url.rstrip("/") + "/api/tags")).json().get("models", [])], "default": LLM_MODEL}

@app.post("/api/converse")
async def api_converse(request: Request):
    """One spoken turn: {audio_b64 | text, history:[{role,content}], system?, model?, llm_url?, voice?, think?, options?} -> {user_text, reply_text, thinking, audio_b64, timings}.
    History is carried by the caller here (test page); in the portal the conversation state lives server-side in the pipeline/Athena instead."""
    b = await request.json()
    url, model, timings = (b.get("llm_url") or LLM_URL).rstrip("/"), b.get("model") or LLM_MODEL, {}
    if not (url and model): raise HTTPException(status_code=400, detail="set LLM_URL/LLM_MODEL in the environment or pass llm_url/model")
    user_text = b.get("text") or ""
    if not user_text:
        st = await transcribe(_unb64(b.get("audio_b64", "")), language=b.get("language"))
        user_text, timings["stt_s"] = st["text"], st["elapsed_s"]
    if not user_text: return {"user_text": "", "reply_text": "", "note": "heard nothing"}
    msgs = ([{"role": "system", "content": b["system"]}] if b.get("system") else []) + list(b.get("history") or []) + [{"role": "user", "content": user_text}]
    t0 = time.time()
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=1800.0, write=30.0, pool=10.0)) as c:
        r = await c.post(f"{url}/api/chat", json={"model": model, "messages": msgs, "stream": False, "think": bool(b.get("think", False)), "options": b.get("options") or {}})
    if r.status_code != 200: raise HTTPException(status_code=502, detail=f"LLM {r.status_code}: {r.text[:300]}")
    m = r.json().get("message", {})
    reply, timings["llm_s"] = m.get("content") or "", round(time.time() - t0, 2)
    out = {"user_text": user_text, "reply_text": reply, "thinking": m.get("thinking", ""), "timings": timings}
    if reply.strip():
        sp = await speak(reply, b.get("voice", ""), b.get("length_scale"))
        out.update(audio_b64=_b64(sp.pop("wav")), tts=sp)
        timings["tts_s"] = sp["elapsed_s"]
    return out

# --- Voice Chat Test Page ---

PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lyrebird</title>
<style>
body{font:17px/1.5 system-ui,sans-serif;max-width:40rem;margin:0 auto;padding:1rem;background:#f4f6f3;color:#1f2a24}
#talk{display:block;width:100%;padding:1.6rem 1rem;font-size:1.3rem;border:0;border-radius:1rem;background:#2f6b4f;color:#fff;touch-action:none;user-select:none}
#talk.rec{background:#a8322d}#talk:focus-visible{outline:3px solid #e0a020;outline-offset:3px}
#log>div{margin:.7rem 0;padding:.5rem .7rem;border-radius:.6rem;background:#fff;white-space:pre-wrap}#log .u{background:#e3ece6}#log .meta{background:none;color:#6b7a70;font-size:.8rem;padding:0 .2rem}
details{margin:.8rem 0}input,select,textarea{font:inherit;width:100%;box-sizing:border-box;margin:.2rem 0}
.row{display:flex;gap:.5rem}.row>*{flex:1}
@media (prefers-reduced-motion:no-preference){#talk.rec{animation:p 1.2s ease-in-out infinite}@keyframes p{50%{opacity:.75}}}
</style>
<button id="talk" aria-live="polite">Hold to talk</button>
<div class="row"><input id="txt" placeholder="...or type and press Enter"><button id="stop">Stop audio</button></div>
<label style="font-size:.85rem"><input type="file" id="upl" accept="audio/*" capture style="width:auto"> send a recording instead (works where the microphone is blocked)</label>
<div id="log"></div>
<details><summary>Settings</summary>
<label>LLM endpoint<input id="url" placeholder="(server default)"></label>
<label>Model<select id="mdl"></select></label>
<label>Voice<select id="vce"></select></label>
<label>System prompt<textarea id="sys" rows="3">You are a warm, plain-spoken conversation partner. Answer in short spoken sentences, no lists or markdown.</textarea></label>
<label><input type="checkbox" id="thk" style="width:auto"> let the model think first (slower)</label>
<button id="clear">Clear conversation</button>
</details>
<script>
var H=[],rec=null,chunks=[],player=new Audio(),busy=false;
function el(c,t){var d=document.createElement('div');d.className=c;d.textContent=t;log.appendChild(d);d.scrollIntoView({block:'end'});return d}
async function j(u,o){var r=await fetch(u,o);var x=await r.json();if(!r.ok)throw new Error(x.detail||JSON.stringify(x));return x}
async function init(){try{var h=await j('/health');(h.voices_installed.length?h.voices_installed:[h.tts_voice]).forEach(v=>vce.add(new Option(v,v,v==h.tts_voice,v==h.tts_voice)));var m=await j('/api/llm_models'+(url.value?'?llm_url='+encodeURIComponent(url.value):''));m.models.forEach(x=>mdl.add(new Option(x,x,x==m.default,x==m.default)))}catch(e){el('meta','setup: '+e.message)}}
function b64(blob){return new Promise(r=>{var f=new FileReader();f.onload=()=>r(f.result.split(',')[1]);f.readAsDataURL(blob)})}
async function turn(p){
 if(busy)return; busy=true; talk.textContent='Thinking...'; var t0=Date.now();
 try{
  var x=await j('/api/converse',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(Object.assign({history:H,system:sys.value,model:mdl.value,voice:vce.value,think:thk.checked},url.value?{llm_url:url.value}:{},p))});
  if(!x.user_text){el('meta','(heard nothing)');return}
  el('u',x.user_text); el('a',x.reply_text||'(no reply)');
  H.push({role:'user',content:x.user_text},{role:'assistant',content:x.reply_text});
  el('meta',Object.entries(x.timings||{}).map(e=>e[0]+' '+e[1]+'s').join(', ')+', total '+((Date.now()-t0)/1000).toFixed(1)+'s');
  if(x.audio_b64){player.src='data:audio/wav;base64,'+x.audio_b64;player.play()}
 }catch(e){el('meta','error: '+e.message)}
 finally{busy=false;talk.textContent='Hold to talk'}
}
talk.addEventListener('pointerdown',async e=>{
 e.preventDefault(); if(busy||rec)return; player.pause();
 try{var s=await navigator.mediaDevices.getUserMedia({audio:true})}catch(err){el('meta','microphone blocked ('+err.name+') - browsers only allow it on https or localhost; use the recording button below');return}
 chunks=[]; rec=new MediaRecorder(s); rec.ondataavailable=ev=>chunks.push(ev.data);
 rec.onstop=async()=>{s.getTracks().forEach(t=>t.stop());var blob=new Blob(chunks,{type:rec.mimeType});rec=null;talk.classList.remove('rec');if(blob.size>2000)turn({audio_b64:await b64(blob)});else talk.textContent='Hold to talk'};
 rec.start(); talk.classList.add('rec'); talk.textContent='Listening... release to send';
});
['pointerup','pointerleave','pointercancel'].forEach(n=>talk.addEventListener(n,()=>{if(rec&&rec.state=='recording')rec.stop()}));
txt.addEventListener('keydown',e=>{if(e.key=='Enter'&&txt.value.trim()){turn({text:txt.value.trim()});txt.value=''}});
upl.addEventListener('change',async()=>{if(upl.files[0])turn({audio_b64:await b64(upl.files[0])});upl.value=''});
stop.addEventListener('pointerdown',()=>player.pause());
clear.addEventListener('pointerdown',()=>{H=[];log.innerHTML=''});
url.addEventListener('change',()=>{mdl.innerHTML='';init()});
init();
</script>"""

@app.get("/", response_class=HTMLResponse)
async def page(): return PAGE

if __name__ == "__main__": uvicorn.run(app, host=os.getenv("SERVER_HOST", "0.0.0.0"), port=SERVER_PORT)