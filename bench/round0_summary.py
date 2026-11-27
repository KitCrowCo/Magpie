#!/usr/bin/env python3
"""round0_summary.py - the numbers a Round 0 write-up needs, from bench.py result files (and power_log.py readings when present), as Markdown tables.

Give it result folders (as round0.sh writes them: results/round0/<node>/<date>/) or single .jsonl files. It prints:
  Fidelity: per run, the set hash (one hash over every prompt's answer, in order) and the answer count; the runs that share a set hash are byte-identical.
  Speed: per run and prompt length, cold (run 1) and warm (later runs): time to first token, prompt tokens per second from it, decode tokens per second.
  Energy (when the folder has power.jsonl): wall energy over each run's span, and per output token.
Standard library only."""
import sys, json, time, hashlib, argparse, statistics, collections
from pathlib import Path

def _t(s: str) -> float: return time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%S"))

def runs(paths: list) -> list:
    """One entry per result file: label, folder, records, and the power readings of its folder."""
    out = []
    for p in map(Path, paths):
        for f in (sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]):
            if f.name == "power.jsonl": continue
            recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            pw = f.parent / "power.jsonl"
            out.append({"file": f, "label": recs[0].get("label") if recs else f.stem, "recs": recs, "power": [json.loads(l) for l in pw.read_text(encoding="utf-8").splitlines() if l.strip()] if pw.exists() else []})
    return out

def energy(r: dict) -> dict:
    """Wall energy between the run's first and last record, every sensor in the folder summed, each reading held until the next."""
    if not r["power"] or len(r["recs"]) < 2: return {}
    a, b = _t(r["recs"][0]["t"]), _t(r["recs"][-1]["t"])
    by = collections.defaultdict(list)
    for x in r["power"]: by[x["sensor"]].append((x["t"], x["watts"]))
    j = 0.0
    for s in by.values():
        s.sort()
        for (t, w), (t2, _) in zip(s, s[1:] + [(float("inf"), 0)]):
            lo, hi = max(t, a), min(t2, b)
            if hi > lo: j += w * (hi - lo)
    out_tok = sum(x.get("completion_tokens") or 0 for x in r["recs"] if x.get("kind") in ("perf", "fidelity"))
    return {"wh": round(j / 3600, 2), "seconds": round(b - a), "mean_w": round(j / (b - a), 1) if b > a else None, "j_per_out_token": round(j / out_tok, 1) if out_tok else None}

def fidelity_table(rs: list) -> str:
    rows, sets = [], collections.defaultdict(list)
    for r in rs:
        fid = sorted((x for x in r["recs"] if x.get("kind") == "fidelity"), key=lambda x: x["i"])
        if not fid: continue
        h = hashlib.sha256("".join(x["sha256"] for x in fid).encode()).hexdigest()[:16]
        start = next((x for x in r["recs"] if x.get("kind") == "start"), {})
        sets[h].append(r["label"])
        rows.append(f"""| {r["label"]} | {start.get("model", "")} | {start.get("note", "")} | {len(fid)} | {start.get("max_tokens", "")} | `{h}` |""")
    if not rows: return ""
    same = [f"""{len(v)} runs share `{k}`: {", ".join(v)}""" for k, v in sets.items() if len(v) > 1]
    return "\n".join(["### Fidelity", "", "| Run | Model | Note | Answers | Max tokens | Set hash |", "|-|-|-|-|-|-|", *rows, "", *(f"- {s}" for s in same or ["no two runs share a set hash"]), ""])

def perf_table(rs: list) -> str:
    rows = []
    med = lambda xs: statistics.median(xs) if xs else None
    fmt = lambda v, f=".2f": "-" if v is None else format(v, f)
    for r in rs:
        pf = [x for x in r["recs"] if x.get("kind") == "perf"]
        if not pf: continue
        start = next((x for x in r["recs"] if x.get("kind") == "start"), {})
        e = energy(r)
        for n in sorted({x["target_prompt_tokens"] for x in pf}):
            cold = [x for x in pf if x["target_prompt_tokens"] == n and x["run"] == 1]
            warm = [x for x in pf if x["target_prompt_tokens"] == n and x["run"] > 1]
            g = lambda xs, k: med([x[k] for x in xs if x.get(k) is not None])
            rows.append(f"""| {r["label"]} | {start.get("model", "")} | {n} | {fmt(g(cold, "ttft_s"))} / {fmt(g(warm, "ttft_s"))} | {fmt(g(cold, "prompt_tok_s_from_ttft"), ".1f")} | {fmt(g(cold, "decode_tok_s"))} / {fmt(g(warm, "decode_tok_s"))} | {e.get("mean_w", "-")} | {e.get("j_per_out_token", "-")} |""")
    if not rows: return ""
    return "\n".join(["### Speed (median; cold = run 1 of each prompt, warm = later runs)", "", "| Run | Model | Prompt tokens (about) | Time to first token s, cold / warm | Prompt tok/s from TTFT, cold | Decode tok/s, cold / warm | Mean wall W (run) | J per output token (run) |", "|-|-|-|-|-|-|-|-|", *rows, "",
                      "Prompt tok/s from TTFT includes any queueing and the first token's own decode: a lower bound on prefill speed. Wall power covers every sensor in the run's folder.", ""])

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="result folders or .jsonl files")
    a = p.parse_args()
    rs = runs(a.paths)
    print(fidelity_table(rs) + "\n" + perf_table(rs))
