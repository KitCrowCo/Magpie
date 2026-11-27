# speech_direct/speaker_id.py
"""
Speaker embeddings for the speech node - who is talking. Torch-free: numpy + onnxruntime, both already installed with faster-whisper.
  POST /api/speaker/embed {audio_b64}  ->  {embedding: [256 floats, unit length], duration_s, elapsed_s, model}
The node only measures; it stores nothing. Enrolled people and matching live in the portal, per login.
Model: WeSpeaker ResNet34-LM trained on VoxCeleb (ONNX, 26.5 MB, CC-BY-4.0), downloaded once from Hugging Face into model_dir.
With HF_HUB_OFFLINE=1 on the node the file must already be there (download it once with the flag off, or copy it in).
Features: Kaldi-compatible 80-bin log-mel fbank (25 ms / 10 ms frames, hamming, pre-emphasis 0.97, DC removal, 16-bit sample scale) with
per-utterance mean normalization - what WeSpeaker computes before its model.
Hook-up in main.py, after `app` is created:   import speaker_id; speaker_id.register(app, WHISPER_DIR.parent / "speaker")
"""
import os, base64, time, asyncio, threading
from pathlib import Path
import numpy as np
from fastapi import HTTPException

SPEAKER_REPO = os.getenv("SPEAKER_REPO", "Wespeaker/wespeaker-voxceleb-resnet34-LM")
SPEAKER_FILE = os.getenv("SPEAKER_FILE", "voxceleb_resnet34_LM.onnx")
SPEAKER_THREADS = int(os.getenv("SPEAKER_THREADS", "2"))
SR, MIN_S, FRAME, SHIFT, NFFT, BINS = 16000, 0.5, 400, 160, 512, 80
_S = {"sess": None, "dir": None}
_LOCK = threading.Lock()

# --- Features ---

def _mel_bank() -> np.ndarray:
    """Kaldi's triangular mel filters (20 Hz to Nyquist, mel = 1127 ln(1 + f/700)) over the 256 FFT bins below Nyquist."""
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)
    lo, hi = mel(20.0), mel(SR / 2)
    d, m = (hi - lo) / (BINS + 1), mel(np.arange(NFFT // 2) * SR / NFFT)
    left, center, right = (lo + (np.arange(BINS)[:, None] + k) * d for k in (0, 1, 2))
    return np.where((m > left) & (m < right), np.where(m <= center, (m - left) / (center - left), (right - m) / (right - center)), 0.0)

_BANK = _mel_bank()
_WIN = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(FRAME) / (FRAME - 1))

def fbank(x: np.ndarray) -> np.ndarray:
    """x: float mono at 16 kHz in [-1, 1]. Returns [frames, 80] log-mel energies, mean-normalized over the utterance."""
    x = x.astype(np.float64) * 32768.0
    n = 1 + (len(x) - FRAME) // SHIFT
    f = x[np.arange(FRAME)[None, :] + SHIFT * np.arange(n)[:, None]]
    f = f - f.mean(1, keepdims=True)
    f = np.concatenate([f[:, :1] * 0.03, f[:, 1:] - 0.97 * f[:, :-1]], 1) * _WIN   # kaldi pre-emphasis: the first sample is scaled by (1 - 0.97)
    e = np.log(np.maximum((np.abs(np.fft.rfft(f, NFFT)) ** 2)[:, :NFFT // 2] @ _BANK.T, np.finfo(np.float32).eps))
    return (e - e.mean(0)).astype(np.float32)

# --- Model ---

def _session():
    with _LOCK:
        if _S["sess"] is None:
            import onnxruntime as ort
            p = _S["dir"] / SPEAKER_FILE
            if not p.exists():
                from huggingface_hub import hf_hub_download
                _S["dir"].mkdir(parents=True, exist_ok=True)
                hf_hub_download(SPEAKER_REPO, SPEAKER_FILE, local_dir=str(_S["dir"]))
            o = ort.SessionOptions(); o.intra_op_num_threads = SPEAKER_THREADS
            _S["sess"] = ort.InferenceSession(str(p), o, providers=["CPUExecutionProvider"])
            print(f"[SPK] loaded {p}", flush=True)
        return _S["sess"]

def _decode(audio_b64: str) -> np.ndarray:
    """Any format the recognizer accepts (wav/webm/ogg/mp4/mp3), resampled to 16 kHz mono, through faster-whisper's own decoder."""
    try: from faster_whisper.audio import decode_audio
    except ImportError: raise HTTPException(status_code=501, detail="this node has no audio decoder (faster-whisper is not installed in this container) - point speaker identification at the CPU speech node")
    import io
    return decode_audio(io.BytesIO(base64.b64decode(audio_b64.split(",", 1)[-1])), sampling_rate=SR)

def embed_sync(audio_b64: str) -> dict:
    t0, x = time.time(), _decode(audio_b64)
    if len(x) < SR * MIN_S: raise HTTPException(status_code=422, detail=f"audio too short for a voiceprint ({len(x) / SR:.2f} s, needs {MIN_S} s)")
    sess = _session()
    v = sess.run(None, {sess.get_inputs()[0].name: fbank(x)[None]})[0].reshape(-1)
    return {"embedding": [round(float(a), 6) for a in v / (np.linalg.norm(v) or 1.0)], "duration_s": round(len(x) / SR, 2), "elapsed_s": round(time.time() - t0, 3), "model": SPEAKER_FILE}

# --- Routes ---

def register(app, model_dir):
    _S["dir"] = Path(model_dir)
    @app.post("/api/speaker/embed")
    async def speaker_embed(b: dict):
        if not b.get("audio_b64"): raise HTTPException(status_code=422, detail="audio_b64 is empty")
        return await asyncio.to_thread(embed_sync, b["audio_b64"])

