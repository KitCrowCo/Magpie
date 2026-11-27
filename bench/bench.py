#!/usr/bin/env python3
"""bench.py - speed and fidelity measurements for any chat endpoint, Python standard library only (runs on any node, nothing to install).
Wires: ollama (/api/chat - llama_direct, Ollama) and openai (/v1/chat/completions - colibri, vLLM, llama-server; through llama_direct as .../engines/<name>).
  perf      time to first token, prompt (prefill) and decode speed at several prompt lengths; each prompt is sent --repeat times in a row,
            so run 1 is cold for that prompt and later runs show what a kept cache saves
  fidelity  greedy (temperature 0) answers to a fixed prompt set, each saved with a hash. A change that should not change the model (another node layout,
            a GPU tier, a mirror drive, a cluster split, a new engine build) must give byte-identical answers - compare two runs with compare
  compare   two fidelity runs: identical answers, and for each other one where the first difference is
Every measurement is one JSON line in results/<label>.jsonl: what was asked, what came back, the client machine, --note (cache state, what changed) and,
with --info-url, the server's own status at the start (e.g. llama_direct's /api/aux/status or /api/engines/status). Nothing leaves the network the URL names.
Examples:
  python3 bench.py perf --wire ollama --url http://192.168.0.230:8790 --model qwen3.5-9b-polaris-highiq-instruct-q4_k_m-imat --num-ctx 32768 --label knode-9b --note "after reboot"
  python3 bench.py fidelity --wire openai --url http://NODE:8790/engines/colibri --model deepseek-v4.1-flash --label v41-cpu
  python3 bench.py compare results/v41-cpu.jsonl results/v41-vulkan.jsonl"""
import os, json, time, hashlib, argparse, platform, urllib.request
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
NOTES = ["The river station measures water height every ten minutes and sends the reading over a slow radio link.",
         "A ledger lists who borrowed each tool, when it went out, and the condition it came back in.",
         "The bakery starts the first dough at four, so the ovens are hot when the market opens.",
         "Each node in the cluster reports free memory before it accepts a new job from the queue.",
         "The archive keeps one copy of every letter, sorted by the year it was received.",
         "Volunteers repaint the trail markers each spring after the snow has melted.",
         "The library's reading room closes an hour early on the first Monday of each month.",
         "A small solar panel charges the weather station's battery during the afternoon."]
FIDELITY = ["What is 17 * 23? Answer with the number only.",
            "Write a Python function is_prime(n) and nothing else.",
            "List the planets of the solar system in order from the sun, comma-separated.",
            "Explain in two sentences why the sky looks blue.",
            "Translate into Mandarin Chinese: The train leaves at seven tomorrow morning.",
            "A farmer has 15 sheep; all but 8 run away. How many are left? Show your reasoning in one short paragraph, then the answer.",
            "Give a JSON object with keys name, year, tags for the novel Frankenstein.",
            "Summarize the plot of Romeo and Juliet in exactly three sentences.",
            "What does the Unix command `ls -la` show? One paragraph.",
            "Write a haiku-length line (not a known poem) about mountain fog.",
            "Sort these numbers ascending: 42, 7, 19, 3, 88, 21.",
            "Name three differences between TCP and UDP as a bulleted list."]

# --- Endpoint ---

def _lines(url: str, body: dict, key: str, timeout: float, headers: dict = None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {}), **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r: yield raw.decode("utf-8", "replace").strip()

def _events(a, messages: list, max_tokens: int, temperature: float):
    """(kind, text, chunk) per streamed piece, kind = text | thinking | final (the last chunk, carrying usage / timings)."""
    if a.wire == "ollama":
        body = {"model": a.model, "messages": messages, "stream": True, "think": a.think not in ("", "off", "none"), "options": {"temperature": temperature, "num_predict": max_tokens, "seed": 0, **({"num_ctx": a.num_ctx} if a.num_ctx else {})}}
        for line in _lines(f"""{a.url.rstrip('/')}/api/chat""", body, a.key, a.timeout, getattr(a, "headers", None)):   # headers: e.g. X-Request-Id, set by fleet_tests.py
            if not line: continue
            c = json.loads(line)
            if c.get("error"): raise RuntimeError(c["error"])
            m = c.get("message") or {}
            if m.get("thinking"): yield "thinking", m["thinking"], c
            if m.get("content"): yield "text", m["content"], c
            if c.get("done"): yield "final", "", c
        return
    body = {"model": a.model, "messages": messages, "stream": True, "max_tokens": max_tokens, "temperature": temperature, "reasoning_effort": {"": "none", "on": "high"}.get(a.think, a.think), "stream_options": {"include_usage": True}}
    last = {}
    for line in _lines(f"""{a.url.rstrip('/')}/v1/chat/completions""", body, a.key, a.timeout, getattr(a, "headers", None)):
        if not line.startswith("data: ") or line == "data: [DONE]": continue
        c = json.loads(line[6:]); last = c if c.get("usage") else last
        if c.get("error"): raise RuntimeError(c["error"])
        d = ((c.get("choices") or [{}])[0].get("delta") or {})
        if d.get("reasoning_content"): yield "thinking", d["reasoning_content"], c
        if d.get("content"): yield "text", d["content"], c
    yield "final", "", last

def ask(a, prompt: str, max_tokens: int, temperature: float) -> dict:
    """One streamed request, timed on this side: ttft = first output of any kind; decode = later pieces over their span. Server counts and timings are kept when it sends them."""
    t0, first, last, n, text, thinking, final = time.time(), None, None, 0, "", "", {}
    for kind, piece, c in _events(a, [{"role": "user", "content": prompt}], max_tokens, temperature):
        if kind == "final": final = c; continue
        first, last, n = first or time.time(), time.time(), n + 1
        if kind == "text": text += piece
        else: thinking += piece
    u = final.get("usage") or {}
    pt, ct = final["prompt_eval_count"] if final.get("prompt_eval_count") is not None else u.get("prompt_tokens"), final.get("eval_count") or u.get("completion_tokens") or n   # a fully cached prompt reads 0 tokens: 0, not "unknown"
    out = {"ttft_s": round(first - t0, 3) if first else None, "total_s": round(time.time() - t0, 3), "prompt_tokens": pt, "completion_tokens": ct, "pieces": n, "text": text, "thinking": thinking}
    if first and last and last > first and ct and ct > 1: out["decode_tok_s"] = round((ct - 1) / (last - first), 2)
    if pt and first: out["prompt_tok_s_from_ttft"] = round(pt / (first - t0), 1)   # includes queueing and the first token's own decode - a lower bound on prefill speed
    if final.get("prompt_eval_duration") is not None: out["server"] = {"prompt_tok_s": round(pt / (final["prompt_eval_duration"] / 1e9), 1) if pt and final["prompt_eval_duration"] else None, "decode_tok_s": round(ct / (final["eval_duration"] / 1e9), 2) if final.get("eval_duration") else None, "cache_n": final.get("cache_n")}   # a fully cached prompt has duration 0: its cache_n is the point of a repeat run
    return out

# --- Records ---

def machine() -> dict:
    cpu = next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name")), platform.processor()) if Path("/proc/cpuinfo").exists() else platform.processor()
    mem = next((round(int(l.split()[1]) / 1e6, 1) for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemTotal")), None) if Path("/proc/meminfo").exists() else None
    return {"host": platform.node(), "os": platform.platform(), "cpu": cpu, "cores": os.cpu_count(), "ram_gb": mem, "python": platform.python_version()}

def server_info(url: str) -> dict:
    if not url: return {}
    try:
        with urllib.request.urlopen(url, timeout=15) as r: return json.loads(r.read())
    except Exception as e: return {"error": f"{type(e).__name__}: {e}"}

def record(a, rec: dict):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"{a.label}.jsonl", "a") as f: f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": a.label, "mode": a.cmd, "wire": a.wire, "url": a.url, "model": a.model, "think": a.think, "note": a.note, **rec}, ensure_ascii=False) + "\n")

def notes_prompt(tokens: int) -> str:
    """About tokens tokens (4 characters each) of numbered notes, then one fixed task - deterministic, so a repeated length is the same prompt."""
    paras, i = [], 0
    while sum(len(p) for p in paras) < tokens * 4 - 200: paras.append(f"Note {i + 1}: {NOTES[i % len(NOTES)]}"); i += 1
    return "\n".join(paras) + "\n\nWrite a numbered list with one short sentence per note above, in order, as far as you can."

# --- Modes ---

def perf(a):
    record(a, {"kind": "start", "machine": machine(), "server_info": server_info(a.info_url), "lengths": a.lengths, "repeat": a.repeat, "max_tokens": a.max_tokens})
    for n in [int(x) for x in a.lengths.split(",")]:
        for run in range(1, a.repeat + 1):
            r = ask(a, notes_prompt(n), a.max_tokens, 0.0)
            record(a, {"kind": "perf", "target_prompt_tokens": n, "run": run, **{k: v for k, v in r.items() if k not in ("text", "thinking")}, "text_head": r["text"][:200]})
            print(f"""prompt ~{n:>6} run {run}: ttft {r["ttft_s"]}s  prompt {r.get("prompt_tokens")} tok ({r.get("prompt_tok_s_from_ttft")} tok/s from ttft)  decode {r.get("decode_tok_s")} tok/s over {r["completion_tokens"]} tok  total {r["total_s"]}s""", flush=True)

def fidelity(a):
    prompts = json.loads(Path(a.prompts).read_text()) if a.prompts else FIDELITY
    record(a, {"kind": "start", "machine": machine(), "server_info": server_info(a.info_url), "prompts": len(prompts), "max_tokens": a.max_tokens})
    for i, p in enumerate(prompts):
        r = ask(a, p, a.max_tokens, 0.0)
        record(a, {"kind": "fidelity", "i": i, "prompt": p, "sha256": hashlib.sha256((r["thinking"] + "\x00" + r["text"]).encode()).hexdigest(), **r})
        print(f"""{i:>2} {hashlib.sha256((r["thinking"] + chr(0) + r["text"]).encode()).hexdigest()[:12]} {r["completion_tokens"]} tok {r["total_s"]}s  {r["text"][:70]!r}""", flush=True)

def compare(a):
    load = lambda p: {r["i"]: r for r in map(json.loads, Path(p).read_text().splitlines()) if r.get("kind") == "fidelity"}
    x, y = load(a.files[0]), load(a.files[1])
    same = [i for i in x if i in y and x[i]["sha256"] == y[i]["sha256"]]
    print(f"identical: {len(same)} of {len(set(x) & set(y))} prompts in both files")
    for i in sorted(set(x) & set(y) - set(same)):
        s, t = x[i]["thinking"] + x[i]["text"], y[i]["thinking"] + y[i]["text"]
        k = next((j for j, (c, d) in enumerate(zip(s, t)) if c != d), min(len(s), len(t)))
        print(f"  {i}: first difference at character {k}: {s[max(0, k - 30):k + 30]!r} vs {t[max(0, k - 30):k + 30]!r}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["perf", "fidelity", "compare"])
    p.add_argument("files", nargs="*", help="compare: two results/<label>.jsonl files")
    p.add_argument("--wire", choices=["ollama", "openai"], default="ollama")
    p.add_argument("--url", default="http://127.0.0.1:8790", help="base URL; openai wire through llama_direct: http://NODE:8790/engines/<name>")
    p.add_argument("--model", default="")
    p.add_argument("--key", default=os.getenv("BENCH_API_KEY", ""), help="sent as Authorization: Bearer (or set BENCH_API_KEY)")
    p.add_argument("--think", default="", help="off (default) / on / a level (low, medium, high, max)")
    p.add_argument("--num-ctx", type=int, default=0, help="ollama wire: options.num_ctx - keep it equal to what the node's other users ask for, or llama_direct restarts the model")
    p.add_argument("--lengths", default="128,2048,8192", help="perf: approximate prompt lengths in tokens")
    p.add_argument("--repeat", type=int, default=2, help="perf: runs per prompt (run 1 cold for that prompt, later runs warm)")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--prompts", default="", help="fidelity: a JSON list of prompts instead of the built-in twelve")
    p.add_argument("--label", default="run")
    p.add_argument("--note", default="", help="cache state and what differs from other runs, e.g. 'page cache dropped, --kv-slots 1, mirror on'")
    p.add_argument("--info-url", default="", help="a GET URL whose JSON describes the server, recorded at the start")
    p.add_argument("--timeout", type=float, default=36000)
    a = p.parse_args()
    if a.think in ("off", "none"): a.think = ""
    {"perf": perf, "fidelity": fidelity, "compare": compare}[a.cmd](a)

