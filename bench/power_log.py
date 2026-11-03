#!/usr/bin/env python3
"""power_log.py - wall power from network power meters, one JSON line per reading: {"t", "sensor", "watts"}, plus "kwh" when the meter reports its counter.

Sensors are given as NAME=URL. NAME should be the node the meter feeds (api_replay.py and round0_summary.py match readings to nodes by it).
The reading is found by the meter's known shape, or by a JSON path after '#': NAME=http://10.0.0.5/status#meters.0.power
Known shapes:
  Tasmota      http://<plug>/cm?cmnd=Status%2010      StatusSNS.ENERGY.Power (W), StatusSNS.ENERGY.Total (kWh)
  Shelly Gen2+ http://<plug>/rpc/Switch.GetStatus?id=0 apower (W), aenergy.total (Wh)
  Shelly Gen1  http://<plug>/status                   meters.0.power (W), meters.0.total (Wmin)
  ESPHome      http://<plug>/sensor/<power sensor id>  value (W)
  anything with a top-level "power", "watts" or "apower"
Reads over local HTTP only; keep the meters on a network without a route to the internet if their firmware has a cloud option.
Standard library only."""
import sys, json, time, argparse, urllib.request
from pathlib import Path

def _path(d, p: str):
    for k in p.split("."):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d

def _shape(d: dict) -> tuple:
    """(watts, kwh or None) from a known meter answer."""
    if "StatusSNS" in d: e = d["StatusSNS"]["ENERGY"]; return float(e["Power"]), float(e.get("Total")) if e.get("Total") is not None else None
    if "apower" in d: return float(d["apower"]), float(d["aenergy"]["total"]) / 1000 if isinstance(d.get("aenergy"), dict) else None
    if "meters" in d: m = d["meters"][0]; return float(m["power"]), float(m["total"]) / 60000 if m.get("total") is not None else None
    for k in ("power", "watts", "value"):
        if k in d: return float(d[k]), None
    raise ValueError(f"no known power field in {list(d)[:8]} - give its path after '#'")

def read(url: str, timeout: float = 5) -> tuple:
    base, _, p = url.partition("#")
    with urllib.request.urlopen(base, timeout=timeout) as r: d = json.loads(r.read())
    return (float(_path(d, p)), None) if p else _shape(d)

def run(sensors: dict, every: float, out, until: float):
    while time.time() < until:
        t = time.time()
        for name, url in sensors.items():
            try: w, kwh = read(url)
            except Exception as e: print(f"[power] {name}: {type(e).__name__}: {e}", file=sys.stderr, flush=True); continue
            out.write(json.dumps({"t": round(t, 3), "sensor": name, "watts": w, **({"kwh": kwh} if kwh is not None else {})}) + "\n")
        out.flush()
        time.sleep(max(0.0, every - (time.time() - t)))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sensors", nargs="+", help="NAME=URL[#json.path]")
    p.add_argument("--every", type=float, default=5, help="seconds between readings")
    p.add_argument("--out", default="-", help="JSONL file to append to ('-' = standard output)")
    p.add_argument("--for", dest="duration", type=float, default=0, help="stop after this many seconds (0 = until stopped)")
    p.add_argument("--once", action="store_true", help="print one reading per sensor and stop (to test a sensor)")
    a = p.parse_args()
    sensors = dict(s.split("=", 1) for s in a.sensors)
    if a.once:
        for n, u in sensors.items(): print(n, read(u))
        sys.exit(0)
    if a.out != "-": Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    try: run(sensors, a.every, sys.stdout if a.out == "-" else open(a.out, "a", encoding="utf-8"), time.time() + a.duration if a.duration else float("inf"))
    except KeyboardInterrupt: pass
