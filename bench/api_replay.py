#!/usr/bin/env python3
"""api_replay.py - what a stateless API would have billed for the requests a node actually served, next to what the node computed itself.

Reads Fleet journals: a node's MODELS_DIR/fleet/journal.*.jsonl files (copied, or read in place), or a node's /fleet/events over HTTP.
One job is one conversation: the part of the request id before the first dot (the portal sends <job>.<n>: one agent run, one ingest pass, one chat).

Local, per job: prompt tokens computed, prompt tokens reused from the node's cache (llama.cpp reports them; engines such as colibri do not, so their prompts count as computed), output tokens.
API, per job and per cache lifetime (--ttl): a request's prompt is cached input when the same job's previous request ended less than ttl seconds before
(the provider still holds the prefix: the previous prompt plus its answer), and uncached input otherwise; uncached input is written to the cache at the write price, when one is given.
This model is generous to the API: it assumes each request of a job extends the previous one exactly, as an agent loop's do, and that the provider caches every prompt.

Prices are dollars per million tokens, passed in with the date they were read: list prices change, and a replay is only as good as its price date.
Energy per job, when power_log.py output is given: wall watts integrated over the job's request spans on its node (the node's idle draw during those spans is included; say so when you quote it).
Standard library only."""
import json, sys, time, argparse, collections, urllib.request
from pathlib import Path

# --- Reading journals ---

def _events_from_url(url: str) -> list:
    out, after = [], 0
    while True:
        req = urllib.request.Request(f"""{url.rstrip("/")}/fleet/events""", data=json.dumps({"after": after, "limit": 10000}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r: d = json.loads(r.read())
        out += d["events"]
        if not d.get("more") or not d["events"]: return out
        after = d["events"][-1]["id"]

def _events_from_path(p: Path) -> list:
    files = sorted(p.glob("journal.*.jsonl")) if p.is_dir() else [p]
    return [json.loads(l) for f in files for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]

def load(sources: list) -> list:
    """sources: NAME=PATH_OR_URL or PATH_OR_URL (the name is then the path's or host's last part). Every event gets "node"."""
    evs = []
    for s in sources:
        name, _, src = s.partition("=") if "=" in s and not s.startswith(("http://", "https://")) else ("", "", s)
        name = name or Path(src.rstrip("/")).name.replace(":", "_")
        evs += [{**e, "node": name} for e in (_events_from_url(src) if src.startswith(("http://", "https://")) else _events_from_path(Path(src)))]
    return evs

# --- Requests and jobs ---

def requests(evs: list, tests: bool) -> list:
    """Served generation requests: llama.cpp (req.end) and engines (engine.req with gen). prompt = all prompt tokens (computed + reused)."""
    out = []
    for e in evs:
        if e["kind"] == "req.end": ok = e.get("outcome") == "served"
        elif e["kind"] == "engine.req" and e.get("gen"): ok = (e.get("status") or 500) < 400
        else: continue
        job = e.get("job") or (e.get("rid") or "").split(".", 1)[0]
        if not ok or (job.startswith("fleettest") and not tests) or e.get("prompt_n") is None: continue
        out.append({"node": e["node"], "job": job or "(no job id)", "solo": not job, "model": e.get("model") or e.get("name"), "start": e["t"] - (e.get("seconds") or 0), "end": e["t"], "computed": e["prompt_n"], "reused": e.get("cache_n") or 0, "out": e.get("gen_n") or 0})
    return sorted(out, key=lambda r: r["start"])

def bill(reqs: list, ttl: float) -> dict:
    """One job's requests under one cache lifetime: cached / uncached input tokens. Requests without a job id are not known to continue each other: each is billed as a new conversation."""
    prev_end, prev_ctx, cached, uncached = None, 0, 0, 0
    for r in reqs:
        prompt = r["computed"] + r["reused"]
        hit = min(prev_ctx, prompt) if prev_end is not None and not r["solo"] and r["start"] - prev_end < ttl else 0
        cached, uncached = cached + hit, uncached + prompt - hit
        prev_end, prev_ctx = r["end"], prompt + r["out"]
    return {"cached": cached, "uncached": uncached}

# --- Energy ---

def power_series(path: str, sensor_map: dict) -> dict:
    """node -> [(t, watts)] sorted; a sensor's readings belong to the node sensor_map names (default: the sensor's own name)."""
    s = collections.defaultdict(list)
    for l in Path(path).read_text(encoding="utf-8").splitlines():
        if not l.strip(): continue
        r = json.loads(l)
        if r.get("watts") is not None: s[sensor_map.get(r["sensor"], r["sensor"])].append((r["t"], float(r["watts"])))
    return {k: sorted(v) for k, v in s.items()}

def joules(series: list, a: float, b: float) -> float:
    """Energy between a and b, each reading held until the next one (the first reading before a counts from a)."""
    j = 0.0
    for (t, w), (t2, _) in zip(series, series[1:] + [(float("inf"), 0)]):
        lo, hi = max(t, a), min(t2, b)
        if hi > lo: j += w * (hi - lo)
    return j

# --- Report ---

def _ttls(spec: list, price_in: float) -> list:
    """--ttl seconds[:write price], repeated: [(seconds, write price per million)]; without a write price, uncached input costs the plain input price."""
    return [(float(s.split(":")[0]), float(s.split(":")[1]) if ":" in s else price_in) for s in spec]

def report(a) -> dict:
    t0 = time.mktime(time.strptime(a.since, "%Y-%m-%d")) if a.since else 0
    t1 = time.mktime(time.strptime(a.until, "%Y-%m-%d")) + 86400 if a.until else float("inf")
    want = set(a.jobs.split(",")) if a.jobs else None
    reqs = [r for r in requests(load(a.sources), a.tests) if t0 <= r["start"] < t1 and (want is None or r["job"] in want) and r["job"].startswith(a.job_prefix)]
    ttls, power = _ttls(a.ttl, a.price_in), power_series(a.power, dict(x.split("=", 1) for x in a.power_map)) if a.power else {}
    jobs = collections.defaultdict(list)
    for r in reqs: jobs[r["job"]].append(r)
    rows = []
    for job, rs in sorted(jobs.items(), key=lambda kv: kv[1][0]["start"]):
        row = {"job": job, "nodes": sorted({r["node"] for r in rs}), "models": sorted({str(r["model"]) for r in rs}), "requests": len(rs), "start": rs[0]["start"], "end": rs[-1]["end"],
               "local": {"computed": sum(r["computed"] for r in rs), "reused": sum(r["reused"] for r in rs), "out": sum(r["out"] for r in rs)}, "api": {}}
        for ttl, write in ttls:
            b = bill(rs, ttl)
            b["dollars"] = round((b["uncached"] * write + b["cached"] * a.price_cached + row["local"]["out"] * a.price_out) / 1e6, 4)
            row["api"][f"{ttl:g}s"] = b
        if power: row["energy_wh"] = round(sum(joules(power.get(r["node"], []), r["start"], r["end"]) for r in rs) / 3600, 3) if all(r["node"] in power for r in rs) else None
        rows.append(row)
    tot = {"jobs": len(rows), "requests": len(reqs), "local": {k: sum(r["local"][k] for r in rows) for k in ("computed", "reused", "out")},
           "api": {k: {f: round(sum(r["api"][k][f] for r in rows), 4) for f in ("cached", "uncached", "dollars")} for k in (rows[0]["api"] if rows else {})}}
    if power: tot["energy_wh"] = round(sum(r.get("energy_wh") or 0 for r in rows), 3)
    return {"prices": {"input": a.price_in, "cached_input": a.price_cached, "output": a.price_out, "write_by_ttl": {f"{t:g}s": w for t, w in ttls}, "read_on": a.price_date, "name": a.price_name},
            "assumptions": "API cache: a request reuses the same job's previous prompt plus answer when that request ended less than ttl seconds earlier; every prompt is cached; failed and test requests are left out.", "jobs": rows, "total": tot}

def print_report(r: dict):
    p = r["prices"]
    print(f"""Prices ($ per million tokens, read {p["read_on"] or "(date not given)"}{", " + p["name"] if p["name"] else ""}): input {p["input"]}, cached input {p["cached_input"]}, output {p["output"]}, cache write {p["write_by_ttl"]}""")
    print(r["assumptions"] + "\n")
    ttls = list(r["total"]["api"])
    print("| Job | Node | Requests | Local: computed / reused / out | " + " | ".join(f"API {t}: uncached / cached / $" for t in ttls) + (" | Wh |" if "energy_wh" in r["total"] else " |"))
    print("|-" * (4 + len(ttls) + ("energy_wh" in r["total"])) + "|")
    for j in r["jobs"] + [{"job": "**total**", "nodes": [], "requests": r["total"]["requests"], **r["total"]}]:
        l = j["local"]
        print(f"""| {j["job"]} | {", ".join(j["nodes"])} | {j["requests"]} | {l["computed"]} / {l["reused"]} / {l["out"]} | """ + " | ".join(f"""{j["api"][t]["uncached"]} / {j["api"][t]["cached"]} / {j["api"][t]["dollars"]}""" for t in ttls) + (f""" | {j.get("energy_wh")} |""" if "energy_wh" in r["total"] else " |"))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sources", nargs="+", help="NAME=PATH_OR_URL: a journal folder or file, or a node's base URL (http://NODE:8790); NAME labels the node")
    p.add_argument("--jobs", default="", help="only these job ids (comma-separated)")
    p.add_argument("--job-prefix", default="", help="only jobs whose id starts with this")
    p.add_argument("--since", default="", help="YYYY-MM-DD, local time")
    p.add_argument("--until", default="", help="YYYY-MM-DD, inclusive")
    p.add_argument("--tests", action="store_true", help="include Fleet's own test requests")
    p.add_argument("--ttl", action="append", default=[], help="cache lifetime in seconds, with an optional cache-write price: 300:3.75 (repeat for several; default 300 and 3600 at the plain input price)")
    p.add_argument("--price-in", type=float, required=True, help="uncached input, $ per million tokens")
    p.add_argument("--price-cached", type=float, required=True, help="cached input (cache read), $ per million tokens")
    p.add_argument("--price-out", type=float, required=True, help="output, $ per million tokens")
    p.add_argument("--price-date", default="", help="the date the prices were read (printed with the result)")
    p.add_argument("--price-name", default="", help="which provider and model the prices are for")
    p.add_argument("--power", default="", help="power_log.py output (JSONL) for energy per job")
    p.add_argument("--power-map", action="append", default=[], help="SENSOR=NODE when a sensor's name differs from the node's name")
    p.add_argument("--json", default="", help="also write the full result here")
    a = p.parse_args()
    a.ttl = a.ttl or ["300", "3600"]
    r = report(a)
    print_report(r)
    if a.json: Path(a.json).write_text(json.dumps(r, indent=1))
