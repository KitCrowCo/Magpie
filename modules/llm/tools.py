"""modules/llm/tools.py - tool calls for an engine or model with no working native tool format ("tools": "text" in its adapter settings): the tools are described in the system prompt, and calls are read back out of the reply, so the portal receives the same tool_calls either way."""
import re, json, uuid

OPEN, CLOSE = "<tool_call>", "</tool_call>"
PROMPT = 'You may call tools. To call one, reply with exactly <tool_call>{"name": "<tool name>", "arguments": {...}}</tool_call> and then stop. Tools:\n'

def inject(messages: list, tools: list) -> list:
    """The tool list added to the first system message, or as a new one; tool results and earlier calls turned into plain text."""
    listing = PROMPT + "\n".join(json.dumps(t.get("function", t)) for t in tools)
    out = []
    for m in messages:
        if m.get("role") == "tool": out.append({"role": "user", "content": f"""Tool result ({m.get("name") or m.get("tool_call_id") or "tool"}):\n{m.get("content")}"""})
        elif m.get("tool_calls"): out.append({"role": "assistant", "content": (m.get("content") or "") + "".join(f"""{OPEN}{json.dumps({"name": c["function"]["name"], "arguments": c["function"]["arguments"]})}{CLOSE}""" for c in m["tool_calls"])})
        else: out.append(m)
    if out and out[0].get("role") == "system": out[0] = {**out[0], "content": f"""{out[0].get("content") or ""}\n\n{listing}"""}
    else: out.insert(0, {"role": "system", "content": listing})
    return out

def _lenient(s: str):
    """The call's JSON; small models often add a trailing comma or wrap it in a code fence."""
    s = re.sub(r"^```\w*|```$", "", s.strip()).strip()
    try: return json.loads(s)
    except ValueError:
        try: return json.loads(re.sub(r",\s*([}\]])", r"\1", s))
        except ValueError: return None

class Parser:
    """feed() returns the text a client should see; finished calls collect in .calls. Text that may be the start of a call is held back until it is known."""
    def __init__(self): self.buf, self.calls = "", []

    def feed(self, text: str) -> str:
        self.buf += text
        out = ""
        while self.buf:
            i = self.buf.find(OPEN)
            if i == -1:
                keep = next((k for k in range(min(len(OPEN) - 1, len(self.buf)), 0, -1) if OPEN.startswith(self.buf[-k:])), 0)
                out, self.buf = out + self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
                break
            j = self.buf.find(CLOSE, i)
            if j == -1:
                out, self.buf = out + self.buf[:i], self.buf[i:]
                break
            c = _lenient(self.buf[i + len(OPEN):j])
            if isinstance(c, dict) and c.get("name"): self.calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", "function": {"name": c["name"], "arguments": c.get("arguments") or {}}})
            else: out += self.buf[i:j + len(CLOSE)]
            self.buf = self.buf[j + len(CLOSE):]
        return out

    def flush(self) -> str:
        """Anything still held back at the end: a call cut off by the output limit reaches the client as text, not silently lost."""
        out, self.buf = self.buf, ""
        return out
