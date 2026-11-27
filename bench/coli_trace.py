#!/usr/bin/env python3
"""coli_trace.py - colibri's routing trace into Fleet's trace format, for AI Tools > Fleet > Experts.

colibri writes every routing decision when its engine runs with ROUTE_TRACE=<file> in its environment (in Magpie: the engine's "env" in engines.json).
Its format (colibri docs/routing-telemetry.md): one line per routing decision, "<call> <row> <layer> <expert>:<gate> <expert>:<gate> ...".
call counts moe() invocations (one per MoE layer per forward pass), row is the position in that pass's batch, the gates are the weights the layer applied.
Tracing never changes what the model computes, but colibri turns off its device-side router while tracing: keep trace runs apart from speed runs.

A forward pass starts again where the layer number stops rising from one call to the next. Each (pass, row) is one token, numbered in order.
Layers keep colibri's numbering, so a model's dense layers show as empty rows in the heat map.
Output: one JSON line per token and layer, {"tok", "layer", "experts"} - copy it to <portal>/data/ai_tools/fleet/traces/<name>.jsonl, then pick it in Fleet > Experts.
Standard library only."""
import sys, json, argparse

def passes(lines):
    """Yields one forward pass at a time: {row: {layer: [experts]}}."""
    cur, prev_call, prev_layer = {}, None, None
    for line in lines:
        f = line.split()
        if len(f) < 4 or not f[0].isdigit(): continue
        call, row, layer = int(f[0]), int(f[1]), int(f[2])
        if call != prev_call and prev_layer is not None and layer <= prev_layer and cur:
            yield cur
            cur = {}
        if call != prev_call: prev_call, prev_layer = call, layer
        cur.setdefault(row, {})[layer] = [int(x.split(":")[0]) for x in f[3:]]
    if cur: yield cur

def convert(src, out, decode_only: bool, max_tokens: int) -> dict:
    tok, layers, experts, n_passes = 0, set(), set(), 0
    for p in passes(src):
        n_passes += 1
        if decode_only and len(p) > 1: continue
        for row in sorted(p):
            for layer, ex in sorted(p[row].items()):
                out.write(json.dumps({"tok": tok, "layer": layer, "experts": ex}) + "\n")
                layers.add(layer)
                experts.update(ex)
            tok += 1
            if max_tokens and tok >= max_tokens: return {"tokens": tok, "passes": n_passes, "layers": len(layers), "max_expert": max(experts, default=-1), "stopped_at_max": True}
    return {"tokens": tok, "passes": n_passes, "layers": len(layers), "max_expert": max(experts, default=-1)}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace", help="colibri's ROUTE_TRACE file ('-' reads standard input)")
    p.add_argument("out", help="Fleet trace to write (.jsonl; '-' writes standard output)")
    p.add_argument("--decode-only", action="store_true", help="keep only passes of one token (decode), leaving out the batched prompt passes")
    p.add_argument("--max-tokens", type=int, default=0, help="stop after this many tokens (Fleet reads at most 20000 per trace)")
    a = p.parse_args()
    src = sys.stdin if a.trace == "-" else open(a.trace, encoding="utf-8")
    out = sys.stdout if a.out == "-" else open(a.out, "w", encoding="utf-8")
    r = convert(src, out, a.decode_only, a.max_tokens)
    print(json.dumps(r), file=sys.stderr)
