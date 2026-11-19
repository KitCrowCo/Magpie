# Round 0 - running the benchmark sessions

Everything here runs on the node itself, from Magpie's `bench/` folder, in `tmux` (a slow model takes hours; detach with Ctrl+b then d). Standard library Python only.

| Script | Does |
|-|-|
| `round0.sh` | one session: machine card, fidelity set, speed by prompt length, wall power while it runs; writes `results/round0/<node>/<date-time>/` and `summary.md` |
| `round0_summary.py` | the write-up's tables from any number of session folders: fidelity set hashes (identical runs share one), speed cold and warm, energy |
| `power_log.py` | wall power from network meters (Tasmota, Shelly, ESPHome, or any JSON with a path) |
| `api_replay.py` | from Fleet journals: tokens each job computed and reused locally, and what an API with a 5-minute or 1-hour prompt cache would bill for the same requests |
| `coli_trace.py` | colibri's routing trace (`ROUTE_TRACE`) into Fleet's trace format, for Fleet > Experts |
| `bench.py` | the measurements themselves (round0.sh calls it) |

**Keep results out of git:**
- `results/` holds machine cards (CPU, RAM part numbers, drives, hostnames), power readings and answers.
- Add `bench/results/` to Magpie's `.gitignore`, and to the `excludes` of the github target in sync.json, so it is never pushed.
- Serial numbers are already left out of the card.

## Settings (environment variables for round0.sh)

| Variable | Default | Meaning |
|-|-|-|
| `MODEL` | required | the model name as the server lists it: `/api/tags` for llama.cpp models, `/engines/<name>/v1/models` for an engine |
| `ENGINE` | empty | engine name in engines.json (colibri models); empty = Magpie's llama.cpp models |
| `URL` | `http://127.0.0.1:8790` | Magpie on this node |
| `NODE_NAME` | hostname | name in the results path; use the hardware name (GETORLI, MINISFORUM, ...) |
| `NUM_CTX` | empty | llama.cpp: the context the node's other users send, or the model reloads first |
| `LENGTHS` | `128,2048,8192` | prompt lengths for the speed run |
| `REPEAT` | `2` | runs per length: run 1 cold for that prompt, later runs warm |
| `FID_TOKENS` / `PERF_TOKENS` | `64` / `128` | answer lengths |
| `STEPS` | `card fidelity perf` | which parts, in order |
| `COLD` | `0` | `1`: drop the page cache before each part (needs root: `sudo -E bash round0.sh`) |
| `SENSORS` | empty | power meters, `NAME=URL` separated by spaces; NAME = the node the meter feeds |
| `NOTE` | empty | what differs from other runs: mirror on, kv-slots, BIOS GPU memory ... |

Test a meter first: `python3 power_log.py --once GETORLI=http://<meter>/cm?cmnd=Status%2010`.

## Sessions, in the order of the plan

### 1. Dry run of the suite (week 1, a fast model)
```
ENGINE=reap MODEL=deepseek-v4-colibri NODE_NAME=GETORLI LENGTHS=128,2048 FID_TOKENS=32 PERF_TOKENS=64 NOTE="dry run" bash round0.sh
```
Any fast model will do; for a llama.cpp model, use `MODEL=<name> NUM_CTX=32768` and leave out ENGINE. The point is a list of what breaks, not the numbers. Tier 2 (lm-eval, EvalPlus) is run separately: bench/README.md, "Tier 2".

### 2. V4.1 Flash fidelity, cold and warm (two different nights)
```
ENGINE=colibri MODEL=<id from /engines/colibri/v1/models> NODE_NAME=<node> STEPS="card fidelity" FID_TOKENS=64 COLD=1 NOTE="V4.1, one drive" sudo -E bash round0.sh
```
- The second night is the same command.
- `python3 round0_summary.py results/round0/<node>/` must show both runs sharing one set hash. That is the byte-identical claim.

### 3. V4.1 speed, one drive and with the mirror
```
ENGINE=colibri MODEL=<id> NODE_NAME=<node> STEPS="perf" LENGTHS=128,2048,8192 COLD=1 NOTE="one drive" sudo -E bash round0.sh
```
Then set `COLI_MODEL_MIRROR` in the engine's `env` (engines.json), restart Magpie, and run the same command with `NOTE="mirror on"`.

### 4. Total throughput with several conversations
`--kv-slots 4` in the engine's `cmd`, restart Magpie, then four clients at once:
```
for i in 1 2 3 4; do python3 bench.py perf --wire openai --url http://127.0.0.1:8790/engines/colibri --model <id> --lengths 2048 --repeat 1 --max-tokens 128 --label r0-<node>-slots4-$i --note "kv-slots 4, client $i" & done; wait
```
- Total decode = the sum of the four clients' decode tok/s over the same span.
- Compare it with one client on `--kv-slots 1`.
- Set `--kv-slots 1` back afterwards: it keeps DSpark drafting on.

### 5. Routing trace for Fleet > Experts (one night, apart from speed runs)
1. In engines.json, add `"ROUTE_TRACE": "/root/.ollama/models/fleet/v41.trace"` to the engine's `env`, then restart Magpie.
2. `ENGINE=colibri MODEL=<id> NODE_NAME=<node> STEPS="fidelity" NOTE="trace run" bash round0.sh`
3. Convert the trace:
   ```
   python3 coli_trace.py <models dir>/fleet/v41.trace v41_fidelity.jsonl --max-tokens 20000
   ```
4. Copy `v41_fidelity.jsonl` to `<portal>/data/ai_tools/fleet/traces/`, then use Fleet > Experts: pick the trace and score the placements for 2 and 4 nodes.
5. Take `ROUTE_TRACE` out of the env again.

Tracing does not change the answers, but colibri turns off its device-side router while it traces, so its speed is not representative.

### 6. The reproducible agent workload (R4) and the replay
After the ten fixed tasks have run (PLAN, R4), on any machine that can reach the node:
```
python3 api_replay.py GETORLI=http://<getorli>:8790 --since 2026-11-12 --until 2026-11-13 --price-in <$> --price-cached <$> --price-out <$> --ttl 300:<5-min write $> --ttl 3600:<1-h write $> --price-date <date read> --price-name "<provider, model>" --power results/round0/GETORLI/<session>/power.jsonl --json r4_replay.json
```
- Read the prices on the provider's pricing page on publication day, and keep the page date.
- Without a cache-write price, uncached input costs the plain input price.

## After the sessions
```
python3 round0_summary.py results/round0/*/ > round0_tables.md
```
The tables go into ROUND0_WRITEUP_TEMPLATE.md (project docs), in the order the template asks for them.
