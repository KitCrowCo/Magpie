"""modules/draft/module.py - feeds finished main-model replies to the draft slots when passive learning is on (header x-passive-learning: true, or PASSIVE_LEARNING_DEFAULT=true), here or on other nodes (DRAFT_ENDPOINTS)."""
import os, asyncio, httpx
from core import jobs
from modules.draft import draft
from modules.llama import module as llama, planner

QUEUE = asyncio.Queue(maxsize=200)
ENDPOINTS = [e.strip().rstrip("/") for e in os.getenv("DRAFT_ENDPOINTS", "").split(",") if e.strip()]

def _listen(job):
    """Takes one sample from each finished chat or completion on the main model, when passive learning is asked for."""
    if job.adapter.name != "llama" or job.capability not in ("chat", "generate") or job.outcome != "served" or not job.record.get("passive") or QUEUE.full(): return
    full, calls = job.record.get("full") or {}, job.record.get("calls") or []
    msg = {"role": "assistant", "content": full.get("content", ""), **({"reasoning_content": full["reasoning"]} if full.get("reasoning") else {}), **({"tool_calls": calls} if calls else {})}
    QUEUE.put_nowait({"prompt": job.body.get("messages") or job.body.get("prompt", ""), "target_output": {"choices": [{"message": msg, "finish_reason": "stop"}]}})

async def _worker():
    """Local slots directly; other nodes over HTTP, with the key in DRAFT_ENDPOINT_KEY."""
    async with httpx.AsyncClient(timeout=6000.0) as c:
        while True:
            s = await QUEUE.get()
            draft.take(s)
            for e in ENDPOINTS:
                try: await c.post(f"{e}/api/draft/adapt", json=s, headers={"Authorization": f"""Bearer {os.getenv("DRAFT_ENDPOINT_KEY", "")}"""})
                except httpx.HTTPError as x: print(f"[DRAFT] {e}: {x}", flush=True)

def setup(ctx):
    draft.MAIN_SPEED = llama.SPEED   # the llama module measures it
    ctx.add_router(draft.router)
    jobs.LISTENERS.append(_listen)
    jobs.spawn(_worker())
    cfg = ctx.cfg()
    cfg["train"]["bin"] = ctx.bin("trainer") or cfg["train"].get("bin", "")
    return draft.start(cfg, llama.find_model, llama.server_bin(), planner, main_busy=lambda: llama.A["adapter"].main.active > 0)
