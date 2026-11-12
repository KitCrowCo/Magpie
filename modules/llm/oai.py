"""modules/llm/oai.py - talking to any engine with an OpenAI-shaped chat API, and turning what comes back into this node's events. What differs between engines is data (adapter "chat" settings), not code:
  path           default /v1/chat/completions
  options        {portal option: engine field}, e.g. {"num_predict": "max_tokens"}
  think          {"field": dotted request field, "map": {"false": ..., "true": ...}, "else": value}; with no "else", a level passes through when the map has no entry for it
  usage          true: ask for the usage chunk (stream_options.include_usage)
  tools          "native": sent as OpenAI tools; "text": rendered into the system prompt and parsed from the reply (modules/llm/tools.py)
Requests are always streamed from the engine, so a stall is seen and a cancel takes effect, whatever the client asked for."""
import json, time, asyncio
from typing import Any, Dict, List
import httpx
from modules.llm import tools as tooltext

# --- Thinking tags ---

class StreamThinkParser:
    """Statefully parses <think> tags split across pieces of a stream."""
    def __init__(self): self.is_thinking, self.buffer = False, ""

    def process(self, chunk: str):
        self.buffer += chunk
        out = {"r": "", "c": ""}
        while self.buffer:
            key, tag = ("r", "</think>") if self.is_thinking else ("c", "<think>")
            idx = self.buffer.find(tag)
            if idx != -1:
                out[key] += self.buffer[:idx]
                self.buffer, self.is_thinking = self.buffer[idx + len(tag):], not self.is_thinking
                continue
            lt = self.buffer.rfind("<")
            if lt != -1 and tag.startswith(self.buffer[lt:]):
                out[key] += self.buffer[:lt]
                self.buffer = self.buffer[lt:]
            else: out[key], self.buffer = out[key] + self.buffer, ""
            break
        return out["r"], out["c"]

    def flush(self):
        r, c = (self.buffer, "") if self.is_thinking else ("", self.buffer)
        self.buffer = ""
        return r, c

def think_on(v) -> bool: return bool(v) and str(v).lower() not in ("false", "off", "none", "0")

# --- Messages ---

def to_oai_msg(m: dict) -> dict:
    """Ollama-shaped message -> OpenAI-shaped: tool_calls arguments as JSON strings with ids; images (Ollama's base64 list) as image_url content parts."""
    out = {k: v for k, v in m.items() if k != "images"}
    if m.get("images"): out["content"] = [{"type": "text", "text": m.get("content") or ""}, *({"type": "image_url", "image_url": {"url": i if str(i).startswith("data:") else f"data:image/png;base64,{i}"}} for i in m["images"])]
    if m.get("tool_calls"): out["tool_calls"] = [{"id": c.get("id") or f"call_{i}", "type": "function", "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"] if isinstance(c["function"].get("arguments"), str) else json.dumps(c["function"].get("arguments") or {})}} for i, c in enumerate(m["tool_calls"])]
    return out

def coalesce(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merges sequential plain messages of the same role, which some chat templates would otherwise duplicate or refuse. Tool traffic (role tool, or assistant tool_calls) and image messages are never merged."""
    merged = []
    for msg in messages:
        plain = msg.get("role") != "tool" and not msg.get("tool_calls") and not msg.get("images") and not (merged and not isinstance(merged[-1].get("content"), str))
        if merged and plain and merged[-1].get("role") == msg.get("role") and not merged[-1].get("tool_calls") and merged[-1].get("role") != "tool": merged[-1]["content"] += "\n\n" + (msg.get("content") or "")
        else: merged.append(to_oai_msg(msg))
    return merged

def from_oai_calls(calls: list) -> list:
    """OpenAI tool_calls -> this node's shape: arguments parsed back to a dict when they are valid JSON."""
    out = []
    for c in calls or []:
        fn, args = c.get("function", {}), c.get("function", {}).get("arguments", "")
        try: args = json.loads(args) if isinstance(args, str) and args else (args or {})
        except json.JSONDecodeError: pass
        out.append({"id": c.get("id"), "function": {"name": fn.get("name", ""), "arguments": args}})
    return out

# --- One request ---

def _set(d: dict, path: str, v):
    *ps, last = path.split(".")
    for p in ps: d = d.setdefault(p, {})
    d[last] = v

def payload(body: dict, spec: dict, extra: dict = None) -> dict:
    """The engine's request from a job body. Options named by the adapter's table go under the engine's field; extra fields the rules add (per-model plan payload) come last."""
    o = body.get("options") if isinstance(body.get("options"), dict) else {}
    table = spec.get("options") or {}
    msgs = coalesce(body.get("messages", []))
    p = {"messages": msgs, "stream": True, **{table[k]: v for k, v in {**{k: body[k] for k in table if k in body}, **o}.items() if k in table and v is not None}}
    if spec.get("model_field", True) and body.get("_served_model"): p["model"] = body["_served_model"]
    if "think" in body and spec.get("think"):
        t = spec["think"]
        v = t.get("map", {}).get(str(body["think"]).lower(), t.get("else", body["think"]))
        if v is not None: _set(p, t["field"], v)
    if body.get("tools"):
        if spec.get("tools", "native") == "native": p.update({"tools": body["tools"], **({"tool_choice": body["tool_choice"]} if body.get("tool_choice") else {})})
        else: p["messages"] = tooltext.inject(p["messages"], body["tools"])
    if body.get("format"): p["response_format"] = {"type": "json_object"} if body["format"] == "json" else {"type": "json_schema", "json_schema": {"name": "response", "schema": body["format"]}}
    if spec.get("usage"): p["stream_options"] = {"include_usage": True}
    return {**p, **(extra or {})}

async def lines(client: httpx.AsyncClient, path: str, p: dict, stall=None):
    """Raw lines from the engine. stall: seconds with no line before this raises asyncio.TimeoutError."""
    async with client.stream("POST", path, json=p) as r:
        if r.status_code >= 400:
            from core.http import HTTPException
            text = (await r.aread()).decode(errors="replace")
            raise HTTPException(status_code=r.status_code, detail=text[:800])
        it = r.aiter_lines().__aiter__()
        while True:
            try: yield await (asyncio.wait_for(it.__anext__(), stall) if stall else it.__anext__())
            except StopAsyncIteration: return

def usage_event(timings: dict, usage: dict, elapsed: float, chunks: int) -> dict:
    """llama-server's timings, or an engine's usage object, in the journal's names; llama's names for draft counts vary by build."""
    t, u = timings or {}, usage or {}
    first = lambda *ks: next((t[k] for k in ks if t.get(k) is not None), None)
    d = {"prompt_n": t.get("prompt_n", u.get("prompt_tokens")), "prompt_ms": t.get("prompt_ms"), "cache_n": t.get("cache_n"), "gen_n": t.get("predicted_n", u.get("completion_tokens")), "gen_ms": t.get("predicted_ms"), "draft_n": first("draft_n", "spec_n"), "draft_ok": first("draft_n_accepted", "draft_accept", "spec_accept")}
    return {"type": "usage", **{k: v for k, v in d.items() if v is not None}}

async def chat_events(client: httpx.AsyncClient, body: dict, spec: dict, extra: dict = None, stall=None, on_raw=None):
    """Events for one chat request: thinking, text, assembled tool calls (once, at the end), usage, done. on_raw(line) lets a caller watch the raw stream. A dropped connection raises httpx errors to the caller, which knows whether the engine is worth retrying."""
    t0, chunks, finish, timings, usage, parser, calls = time.time(), 0, None, {}, {}, StreamThinkParser(), {}
    strip_lead = "think" in body and not think_on(body["think"])   # thinking off: the template's empty think block leaves only whitespace before the reply - dropped until the first visible character
    text_tools = tooltext.Parser() if body.get("tools") and spec.get("tools", "native") == "text" else None
    full = {"content": "", "reasoning": ""}
    async for line in lines(client, spec.get("path", "/v1/chat/completions"), payload(body, spec, extra), stall):
        if on_raw: on_raw(line)
        if not line.startswith("data: "): continue
        raw = line[6:].strip()
        if not raw or raw == "[DONE]": continue
        try: data = json.loads(raw)
        except json.JSONDecodeError: continue
        timings, usage = data.get("timings") or timings, data.get("usage") or usage
        choices = data.get("choices") or []
        if not choices: continue
        delta, finish = choices[0].get("delta", {}), choices[0].get("finish_reason") or finish
        for tc in delta.get("tool_calls") or []:   # streamed tool calls arrive as pieces keyed by index: assembled, sent once at the end
            slot = calls.setdefault(tc.get("index", len(calls) if tc.get("id") or not calls else max(calls)), {"id": tc.get("id"), "function": {"name": "", "arguments": ""}})
            slot["id"] = slot["id"] or tc.get("id")
            n, a = (tc.get("function") or {}).get("name") or "", (tc.get("function") or {}).get("arguments")
            slot["function"]["name"] += n if n != slot["function"]["name"] else ""
            slot["function"]["arguments"] += a if isinstance(a, str) else json.dumps(a) if a else ""
        raw_r, raw_c = delta.get("reasoning_content") or "", delta.get("content") or ""
        full["content"] += raw_c
        full["reasoning"] += raw_r
        r_out, c_out = (raw_r, raw_c) if raw_r else parser.process(raw_c)
        if text_tools: c_out = text_tools.feed(c_out)
        if strip_lead:
            r_out, c_out = (r_out if r_out.strip() else ""), c_out.lstrip()
            strip_lead = not c_out
        if r_out: yield {"type": "thinking", "text": r_out}
        if c_out:
            chunks += 1
            yield {"type": "text", "text": c_out}
    r_out, c_out = parser.flush()
    if text_tools: c_out = text_tools.feed(c_out) + text_tools.flush()
    if strip_lead: r_out, c_out = (r_out if r_out.strip() else ""), c_out.lstrip()
    if r_out: yield {"type": "thinking", "text": r_out}
    if c_out: yield {"type": "text", "text": c_out}
    out_calls = from_oai_calls([calls[i] for i in sorted(calls)]) + (text_tools.calls if text_tools else [])
    if out_calls: yield {"type": "tool_calls", "calls": out_calls}
    yield usage_event(timings, usage, time.time() - t0, chunks)
    yield {"type": "done", "reason": finish or "stop", "_full": full, "_calls": out_calls}
