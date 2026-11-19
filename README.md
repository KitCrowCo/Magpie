# Magpie

A universal AI interface and manager shell: it adds one machine, from a phone to a mini PC, to a Portal Server's pool of resources, in one consistent way. It may be the only thing running on the machine besides Linux.

The programs it runs (llama.cpp, colibri, Kokoro...) are the means available today, forced into shape until they get replaced. What the portal sees never depends on them. Each runs in a network of its own, so what it accepts, what it reaches, what it holds, where its files go and when it stops are Magpie's decisions. If you run it, you own it: no setting you cannot change, no outside contact you did not approve.

- `docs/ARCHITECTURE.md` - how it works and why, how to add an engine or a driver
- `docs/portal/PORTAL_TRANSITION.md` - how Portal Server moves to this; `magpie.json` and `PORTAL_STEPS.md` beside it

## One install

```bash
./install.sh                     # finds out what this machine is, installs what fits, sets up nothing it does not need
./install.sh --for text,speech   # what this node is for: text, moe, speech, image, knowledge
./install.sh --plan              # report only; install later, from here or from a portal
./install.sh --allow-system      # may install system packages (build tools, drivers). Termux and containers may already
./run.sh
```

The kernel needs three pure-Python packages. Each module has drivers, ways of running it, best first; the node picks the first this machine allows and puts what it needs in an environment of its own, so nothing one module needs can break another. A program already on the machine counts as a driver that needs nothing.

The first start writes an admin key to `data/state/first_admin.key` (or set `MAGPIE_ADMIN_KEY`). The rest is a portal's to manage: `GET /modules` shows every module's state, what blocks it, what must be decided before installing it, and `POST /modules/<name>/install` starts it from afar.

`node.json` is optional, and only for what the owner wants to keep from a manager:

```json
{"bind": {"port": 8790, "tls_cert": "cert.pem", "tls_key": "key.pem"}, "manager_may": ["*"], "setup": {"allow_system": false}, "auth": {"open": false}}
```

## What is where

```
magpie.py            starts the node; magpie.py setup is the installer behind install.sh
install.sh run.sh    one install; run it, restart it, hold a phone's wake lock
Dockerfile           optional: the same install in a container
core/                knows nothing about AI
  api.py             /node /run /info /act /jobs /control /services /modules /policy /settings /keys
  contract.py        one shape for every capability's answer; /interface
  contain.py netns.py  each program in its own network; the approval gate; /network
  channels.py        other ports, pass or shaped, as a manager decides
  files.py           /locations /files: move, rename, copy, send, fetch, delete
  drivers.py         what fits this machine; installing it
  modules.py         states, activation, capability registry
  adapters.py        the adapter interface; WorkerAdapter: an engine from data alone
  jobs.py            every piece of work: queue, scheduler, cancel, kill, statistics
  services.py        one way to run, watch, stop and kill any program
  budget.py          memory as one account across every service
  hardware.py        what the machine is, how fast, what blocks a driver
  settings.py        owner over manager over defaults; locations
  journal.py auth.py http.py
modules/             each optional; module.json, and module.py if data cannot say it
  llm/               OpenAI-shaped engines: messages, thinking, tool-call translation
  llama/             llama.cpp: chat, vision, completion, embeddings, decisions, LoRA, KV ring, planner
  colibri/           large MoE models; expert map
  draft/             draft models and live LoRA adaptation
  speech/ speech_clone/  STT, TTS, voiceprints; voice clones in their own environment
  image/ text_encoder/   Flux.2 and CogVideoX; the encoder that needs no PyTorch
  knowledge/         LightRAG instances, a reranker, behind an allowed-paths list
  fleet/             the portal Fleet tool's node side
  relay/             a program's raw protocol carried to another machine
  compat/            Ollama and OpenAI formats, for programs that speak them
bench/ tests/        benchmarks; python3 tests/test_node.py
```

## The interface

Every route but `/health` needs a key with at least the scope shown. Looking never starts anything.

| Route | Scope | Does |
|-|-|-|
| `GET /health` | none | alive |
| `GET /node` | read | identity, hardware, load, speed, locations, budget, modules, capabilities, what could be enabled, services, jobs |
| `GET /node/capabilities` | read | `capabilities` (now) and `could` (if installed, and what that takes) |
| `POST /node/speed` | admin | measures CPU, memory and disk |
| `GET /models?capability=` | read | every model of every active adapter |
| `GET /interface` | read | the contract, stream events, timings and shapes a portal can rely on |
| `POST /run/<capability>` | infer | any work, in this node's events or `?shape=ollama`, `openai`; `?background=1` answers a job id at once |
| `GET /info/<name>`, `POST /act/<name>` | read, infer | a program's status and controls, by what they are |
| `GET /jobs`, `/jobs/<id>`; `POST /jobs/<id>/cancel`, `/control/cancel` | read, infer | one job, or every job of one portal job |
| `POST /control/kill`, `GET /control/status` | admin, read | `requests`, `service` or `server`: needs nothing from the engine |
| `GET /services`, `/services/<name>/log`; `POST /services/<name>/<start, stop, restart, kill>` | read, admin | every process |
| `GET /modules`, `/modules/<name>`; `POST /modules/<name>/<install, activate, deactivate>`; `DELETE /modules/<name>` | read, admin | what could run here and bringing it about |
| `PATCH /settings`; `GET /policy`, `PUT /policy`, `POST /policy/rollback` | admin | any setting from afar, every version kept |
| `GET /locations`, `PUT /locations/<name>` | read, admin | where things live; `move` takes the files along |
| `GET /files/<location>[/<path>]`, `PUT` | read, infer | list, download, upload |
| `POST /files/move`, `copy`, `delete`, `send`, `fetch` | admin | rename, move, send to another node, fetch by URL |
| `GET /channels`; `PUT`, `DELETE /channels/<name>` | read, admin | other ports: `pass` straight to a program, `shape` this node's interface |
| `GET /network` | read | what is contained, what is approved, every outside contact asked for |
| `POST /relay/tunnel/<name>`, `GET /relay`, `POST /relay/link` | infer, read, admin | a program's raw protocol to another machine |
| `/fleet/*`, `/experts/<model>`, `/knowledge/<instance>/*` | read and up | Fleet; colibri's expert map; knowledge servers |
| `/api/*`, `/v1/*` | as the work | Ollama and OpenAI formats |

## Privacy and credit

- A program has no route out except the approval gate: nothing is approved by default, and every attempt is on record (`GET /network`). Where the kernel cannot contain it (Android, macOS), the node says so. This contains the network, not files.
- A fetch or a send contacts only the host named in the request or the `sources` table (default huggingface.co), by plain HTTP with no client library and no telemetry.
- An install contacts what that driver's steps name (PyPI, GitHub, PyTorch's wheel index, the release pages of model files). `--plan` and `GET /modules` show it without doing it.
- `PORTAL_SERVER`, when set, makes the llama module report this machine and its model list to that portal at startup. Unset, nothing is reported to anyone: a portal asks.
- Keys cross the network in the clear unless the node serves TLS or sits behind a tunnel.
- Every driver names who made what it installs (`credits`, shown under `/modules`). F5-TTS's weights are CC-BY-NC; Piper voices have per-voice licenses.
