# bench - measuring a node, an engine and a model

`bench.py` (standard library only, runs on any node) covers speed and fidelity. Quality comes from two established open harnesses, run against the same endpoints. Results go to `results/<label>.jsonl`, one line per measurement, with the client machine, the endpoint, the settings and your `--note`.

## The four questions, in order

| Tier | Question | Tool | Claim it supports |
|-|-|-|-|
| 0 Fidelity | Does this setup compute the same model as the reference setup? | `bench.py fidelity` + `compare`; for colibri also `coli bench` | "Moving experts to another node / a GPU tier / a mirror drive changes speed, not answers" (byte-identical greedy output) |
| 1 Speed | How fast, cold and warm, at which prompt lengths? | `bench.py perf` | Time to first token, prefill and decode speed, cache effect - for this hardware and configuration only |
| 2 Capability | Is quality where the model's published numbers say? | lm-evaluation-harness, EvalPlus | "No quality lost on this hardware": your score within the confidence interval of the vendor's, same task and settings |
| 3 Your work | Does it do the jobs you need? | Portal agent task set (see below) | Usefulness for this project - the most honest number for your own decisions |

Tier 0 comes first because a fast wrong engine looks exactly like a fast right one: a GPU kernel computing garbage reports no error and runs at full speed (colibri's benchmark notes record one such case on a gfx1151 HIP build).

## Start today: the 9B on KNODE

```
cd bench
python3 bench.py perf --url http://192.168.0.230:8790 --model qwen3.5-9b-polaris-highiq-instruct-q4_k_m-imat --num-ctx 32768 --lengths 128,2048,8192,24000 --repeat 2 --label knode-9b --note "LightRAG idle, embed slot up" --info-url http://192.168.0.230:8790/api/aux/status
python3 bench.py fidelity --url http://192.168.0.230:8790 --model qwen3.5-9b-polaris-highiq-instruct-q4_k_m-imat --num-ctx 32768 --label knode-9b-fid-a
```
- `--num-ctx` must equal what LightRAG and the agent use (32768), or llama_direct restarts the model and the first run measures a reload.
- Run the fidelity set again after any llama_direct, llama.cpp or rules change and `compare` the two files. A llama.cpp GPU build is not guaranteed bit-identical across versions (different kernels add in different orders); differences far into an answer are that, a difference in the first few words is a problem.
- Cold means cold on the server: `sync; echo 3 > /proc/sys/vm/drop_caches` (root, on the node) before run 1, and say so in `--note`. Without it, label the run warm.
- Take the median of three sessions on different occasions; one session is an anecdote.

## Tier 2: which sets, and why these

Chosen for: objective automatic scoring, short answers (each item costs minutes on a disk-streaming model), published reference scores to compare against, and open licences. Older sets are likely in every model's training data - that does not matter for a parity claim (the same weights should score the same on your hardware as on the vendor's), but it does for any claim that one model is better than another.

| Set | Items | What it tests | Command (lm-eval / EvalPlus) |
|-|-|-|-|
| IFEval | 541 | Following explicit format instructions; scored by rules, no judge model | `--tasks ifeval` |
| GSM8K (zero-shot chain of thought) | 1319 (use `--limit 200`) | Arithmetic word problems | `--tasks gsm8k_cot_zeroshot` |
| MMLU-Pro | 12k (use `--limit 20` per subject) | Knowledge across 14 subjects, 10 options | `--tasks mmlu_pro` |
| GPQA Diamond | 198 | Hard science questions - one of the few still separating frontier models | `--tasks gpqa_diamond_cot_zeroshot` (gated dataset: needs a Hugging Face account token) |
| HumanEval+ / MBPP+ | 164 / 378 | Writing code that passes many hidden tests | `evalplus.evaluate --dataset humaneval` |

lm-evaluation-harness (MIT) against any OpenAI-compatible endpoint - colibri through llama_direct shown:
```
pip install "lm-eval[api,ifeval]"   # ifeval extra: langdetect, immutabledict, nltk (nltk fetches its punkt data on first use)
export HF_HUB_DISABLE_TELEMETRY=1 OPENAI_API_KEY=unused
lm_eval --model local-chat-completions --model_args model=deepseek-v4.1-flash,base_url=http://NODE:8790/engines/colibri/v1/chat/completions,num_concurrent=1,max_retries=3,tokenized_requests=False \
        --tasks ifeval --apply_chat_template --limit 100 --log_samples --output_path results/lmeval
```
EvalPlus (Apache-2.0) - **it executes the code the model writes**: run it in a throwaway container or VM, never on a portal node:
```
pip install evalplus
OPENAI_API_KEY=unused evalplus.evaluate --model deepseek-v4.1-flash --dataset humaneval --base-url http://NODE:8790/engines/colibri/v1 --backend openai --greedy
```
- Both download their datasets once (Hugging Face, GitHub). After the first run set `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`; nothing else is sent anywhere. GPQA's token identifies your Hugging Face account to Hugging Face.
- llama_direct speaks the Ollama wire; lm-eval and EvalPlus need an OpenAI-compatible URL. For llama.cpp models that means llama_direct needs a `/v1/chat/completions` pass-through (a small follow-up in main.py) - until then run Tier 2 on models served by an OpenAI-compatible engine.
- Thinking: run each set once with thinking off and, where the vendor publishes thinking-mode scores, once at the same effort they used. Compare like with like or the comparison means nothing.

### How many items you need

The 95% interval on an accuracy p over n independent items is about 1.96 x sqrt(p(1-p)/n): 100 items gives about +-10 points at p = 0.5, 200 about +-7, 541 about +-4. A parity claim is "the vendor's score is inside my interval", stated with the interval. Fewer items than that are a smoke test.

### What it costs on a slow model

At about 2 tokens/s decode (a plausible warm figure for DeepSeek V4.1 Flash on APU nodes - measure yours with Tier 1 first): IFEval averages a few hundred output tokens, so 100 items is roughly 4 to 5 hours plus prefill; GSM8K with reasoning off about 2 to 3 hours per 100; GPQA with reasoning on is much longer. `coli serve --kv-slots 4` decodes four requests per step (one pass over the dense weights and over the union of their experts serves all four) - use it with `num_concurrent=4` for throughput runs, and `--kv-slots 1` (drafting on) for the latency numbers in Tier 1. Run Tier 2 on the mid-size model first: same tooling, a fraction of the time.

## colibri's own checks

- `coli bench` - MMLU / HellaSwag / ARC scored by reading answer probabilities, nothing generated: the cheapest quality check there is on a disk-streaming model. Run it once on the CPU path and once per changed setup; scores within noise of each other are the correctness line colibri's own benchmark protocol asks for next to every GPU number.
- colibri's benchmark protocol (docs/benchmarking.md in its repo) lists what every speed number must carry: cache state, request size and queue depth, expert hit rate, bytes read, the exact command. `--note` and `--info-url` are where those go here.
- Never use `--topp` or `--topk` for a parity or fidelity run: they change which experts are routed, which makes it a different model.

## Tier 3: a Portal task set

Twenty real tasks from the open-issues list, each with a written goal, the files it may touch, and a check that decides pass or fail without judgement (`check_files` clean, a test passes, a page renders an element). The agent runs them with each model and settings; you review the shadow-staged results. Pass rate, steps used and minutes per task are the numbers. Kept in `claude/AGENT_TASK_PLANS.md` next to the plans the local agent already takes.

