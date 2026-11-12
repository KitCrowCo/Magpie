"""modules/text_encoder/service.py - prompt in, embeddings file out, for Flux.2 Klein: the residual stream of chosen transformer blocks of a Qwen3 GGUF, concatenated per token. llama.cpp's public API exposes only the last layer, so a small tool (qwen3-layer-dump.cpp) reads them through its evaluation callback. No PyTorch, no GPU needed. Results land in the vault location, where the image worker reads them by job id."""
import os, json, time, hashlib, subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

VAULT = Path(os.environ["EMBED_VAULT_DIR"])
JOBS, SHARED = VAULT / "jobs", VAULT / "shared"
for d in (JOBS, SHARED): d.mkdir(parents=True, exist_ok=True)
TOOL, MODEL = os.environ["QWEN3_LAYER_DUMP_BIN"], os.environ["FLUX2_QWEN_MODEL"]
LAYERS = tuple(int(x) for x in os.getenv("TEXT_ENCODER_LAYERS", "9,18,27").split(","))
HIDDEN = int(os.getenv("TEXT_ENCODER_HIDDEN_SIZE", "4096"))
DEFAULT_MAX_SEQ = int(os.getenv("TEXT_ENCODER_MAX_SEQ_LEN", "512"))
HARD_TRUNCATE = os.getenv("TEXT_ENCODER_HARD_TRUNCATE", "true").lower() == "true"
N_CTX = int(os.getenv("QWEN3_N_CTX", "16384"))
CHARS_PER_TOKEN = 4   # good enough for a warning before tokenizing

def _tool_error() -> str:
    """What is wrong with the tool or the model, or empty. Asked per request, so a model moved into place needs no restart."""
    if not os.path.exists(TOOL): return f"the layer tool is not at {TOOL}"
    if not os.path.exists(MODEL): return f"the GGUF model is not at {MODEL}"
    names = subprocess.run([TOOL, "--model", MODEL, "--dump-tensor-names"], capture_output=True, text=True, timeout=300).stdout
    missing = [t for t in (f"l_out-{hf - 1}" for hf in LAYERS) if t not in names]
    return f"""this model's graph has no {missing}; first names: {names.splitlines()[:5]}""" if missing else ""

def _prompt(text: str) -> str: return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
def _job_id(prompt: str, max_len: int) -> str: return hashlib.sha256(f"{prompt}|{MODEL}|{LAYERS}|{max_len}".encode()).hexdigest()[:24]

def _dump(templated: str, job_dir: Path, max_len: int, truncate: bool) -> tuple:
    job_dir.mkdir(parents=True, exist_ok=True)
    pf, raw, est = job_dir / "templated_prompt.txt", job_dir / "raw_layers.bin", max(1, len(templated) // CHARS_PER_TOKEN)
    if est > N_CTX: truncate, max_len = True, min(max_len, N_CTX - 64)   # past the context the tool would crash
    pf.write_text(templated)
    t0 = time.time()
    r = subprocess.run([TOOL, "--model", MODEL, "--prompt-file", str(pf), "--out", str(raw), "--layers", ",".join(map(str, LAYERS)), "--n-ctx", str(N_CTX), "--batch-size", str(max_len), *(["--max-tokens", str(max_len)] if truncate else [])], capture_output=True, text=True, timeout=1800, env={**os.environ, **({"OMP_NUM_THREADS": os.environ["QWEN3_THREADS"]} if os.getenv("QWEN3_THREADS") else {})})
    if r.returncode: raise RuntimeError(f"the layer tool exited {r.returncode}: {r.stderr[-600:]}")
    with open(raw, "rb") as f:
        n_tok, n_lay, n_emb = (int(x) for x in np.frombuffer(f.read(12), dtype=np.int32))
        data = np.frombuffer(f.read(), dtype=np.float32)
    if data.size != n_tok * n_lay * n_emb: raise ValueError(f"raw_layers.bin holds {data.size} values, its header says {n_tok}x{n_lay}x{n_emb}")
    pf.unlink()
    return data.reshape(1, n_tok, n_lay * n_emb), {"tokenized_len": n_tok, "token_estimate": est, "concat_dim": n_lay * n_emb, "truncated": "(TRUNCATED)" in r.stderr, "elapsed_seconds": round(time.time() - t0, 2)}

def _check(a: np.ndarray) -> dict:
    s = {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()), "std": float(a.std()), "has_nan": bool(np.isnan(a).any()), "has_inf": bool(np.isinf(a).any())}
    if s["has_nan"] or s["has_inf"]: raise ValueError(f"encoder output has NaN or Inf: {s}")
    if s["std"] < 1e-6: raise ValueError(f"encoder output is degenerate: {s}")
    return s

def _save(d: Path, a: np.ndarray, meta: dict):
    np.save(d / "embeds.npy", a.astype(np.float32))
    (d / "meta.json").write_text(json.dumps(meta, indent=2))

def _negative(max_len: int, truncate: bool) -> Path:
    """The empty prompt's embeddings, kept: every classifier-free run needs the same one."""
    d = SHARED / f"negative_empty_prompt_{max_len}"
    if not (d / "embeds.npy").exists():
        a, info = _dump(_prompt(""), d, max_len, truncate)
        _save(d, a, {"prompt": "", "kind": "negative_cfg", "created_at": datetime.now(timezone.utc).isoformat(), "stats": _check(a), **info})
    return d

async def health(request): return JSONResponse({"model": MODEL, "tool": TOOL, "layers": LAYERS, "vault": str(VAULT), "n_ctx": N_CTX})

async def encode(request):
    b = await request.json()
    prompt, max_len, truncate = b["prompt"], int(b.get("max_sequence_length", DEFAULT_MAX_SEQ)), bool(b.get("hard_truncate", HARD_TRUNCATE))
    if err := _tool_error(): return JSONResponse({"detail": err}, status_code=503)
    jid = b.get("job_id") or _job_id(prompt, max_len)
    d, est = JOBS / jid, max(1, len(prompt) // CHARS_PER_TOKEN)
    cached = (d / "embeds.npy").exists() and not b.get("force_recompute")
    if not cached:
        try:
            a, info = _dump(_prompt(prompt), d, max_len, truncate)
            _save(d, a, {"prompt": prompt, "job_id": jid, "model_path": MODEL, "layers": list(LAYERS), "max_sequence_length": max_len, "hard_truncate": truncate, "created_at": datetime.now(timezone.utc).isoformat(), "stats": _check(a), **info})
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as e: return JSONResponse({"detail": f"encoding failed: {e}"}, status_code=500)
    out = {"status": "cached" if cached else "ok", "job_id": jid, "embeds_path": str(d / "embeds.npy"), "meta": json.loads((d / "meta.json").read_text()), "token_estimate": est, "will_truncate": est > max_len}
    if b.get("include_negative"): out["negative_embeds_dir"] = str(_negative(max_len, truncate))
    return JSONResponse(out)

app = Starlette(routes=[Route("/health", health), Route("/encode", encode, methods=["POST"])])
