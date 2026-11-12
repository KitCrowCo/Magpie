"""core/contract.py - what each capability answers on this node, whichever program does it. The contract belongs to Magpie: an adapter's "map" {contract field: dotted path in the program's answer} brings its program into shape, and an answer still missing a required field is an error naming that adapter, so a program's own shape never reaches a portal.
A capability not listed passes through as it is and shows as uncontracted under GET /interface. A module adds or extends entries with "contract" in its module.json."""
from core.http import HTTPException

CONTRACT = {
 "chat": {"stream": True, "request": ["model", "messages"], "optional": ["tools", "think", "options", "stream"]},
 "generate": {"stream": True, "request": ["model", "prompt"], "optional": ["options", "stream"]},
 "embed": {"request": ["model", "input"], "result": ["embeddings"]},
 "rerank": {"request": ["query", "documents"], "optional": ["model", "top_n"], "result": ["results"]},
 "decide": {"request": ["model", "input", "choices"], "optional": ["system"], "result": ["choice"]},
 "speech_to_text": {"request": ["audio_b64"], "optional": ["language", "model", "vad_filter", "initial_prompt"], "result": ["text"]},
 "text_to_speech": {"request": ["text"], "optional": ["voice", "length_scale"], "result": ["audio_b64", "format"]},
 "speaker_embed": {"request": ["audio_b64"], "result": ["embedding"]},
 "text_encode": {"request": ["prompt"], "optional": ["job_id", "max_sequence_length", "hard_truncate", "force_recompute", "include_negative"], "result": ["job_id"]},
 "image_generate": {"request": ["prompt"], "result": ["file_name"]},
 "image_sequence": {"request": ["prompt"], "result": ["file_name"]},
 "text_to_video": {"request": ["prompt"], "result": ["file_name"]},
 "image_to_video": {"request": ["prompt"], "result": ["file_name"]},
 "start_end_to_video": {"request": ["prompt"], "result": ["file_name"]},
 "video_join": {"result": ["file_name"]},
 "knowledge_query": {"request": ["model", "query"], "optional": ["mode", "options"], "result": ["text"]},
 "knowledge_add": {"request": ["model", "text"], "optional": ["name", "pieces", "again"], "result": ["status"]},
 "knowledge_upload": {"request": ["model", "name", "data"], "optional": ["again"], "result": ["status"]},
 "knowledge_status": {"request": ["model"], "result": ["counts", "busy"]},
 "knowledge_documents": {"request": ["model"], "optional": ["page", "page_size", "status"], "result": ["documents"]},
 "knowledge_remove": {"request": ["model", "ids"], "optional": ["delete_file"], "result": ["status"]},
 "knowledge_graph": {"request": ["model"], "optional": ["label", "depth", "max_nodes"], "result": ["nodes", "edges"]},
 "knowledge_labels": {"request": ["model"], "result": ["labels"]},
 "knowledge_cancel": {"request": ["model"], "result": []},
 "knowledge_retry": {"request": ["model"], "result": []},
}   # knowledge_*: "model" is the knowledge group; "options" are the program's own query options (what they are is listed with the group); "pieces": texts that are to stay one chunk each; knowledge_upload: "data" is the file, base64, and "name" its name - the program reads the kind of file itself (PDF, office, text). An add under a name already held replaces it; the same content again is "unchanged" unless "again". knowledge_documents without "page": every page; knowledge_graph: the part of the group's graph around label ("*": all of it), at most max_nodes, {nodes: [{id, labels, properties}], edges: [{source, target, properties}], is_truncated}; knowledge_cancel stops the group's work on what was added (what is half done is marked failed); knowledge_retry takes up what failed again; knowledge_status may also say what it is doing ("doing")
EVENTS = {"job": "first line: the job's id", "thinking": "text", "text": "text", "tool_calls": "calls: [{name, arguments}], assembled, whatever the program's tool format", "progress": "a long job's progress", "result": "data: the capability's result", "usage": "timings, always, last before done", "heartbeat": "nothing yet; the connection is alive", "done": "reason", "error": "message, status"}
TIMINGS = {"total_ms": "measured here", "wait_ms": "queue and load, measured here", "ttft_ms": "request to first output, measured here", "gen_ms": "the program's figure, else measured here", "gen_n": "the program's count, else output pieces counted here", "gen_n_source": "program or counted", "tps": "gen_n over gen_ms", "prompt_n": "when the program says", "prompt_ms": "when the program says"}

def add(capability: str, entry: dict): CONTRACT[capability] = {**CONTRACT.get(capability, {}), **entry}

def conform(capability: str, spec: dict, data, adapter: str):
    """The adapter's answer in the contract's shape, or a failure that says what is wrong and who to fix."""
    want = CONTRACT.get(capability, {}).get("result", [])
    if not want: return data
    if not isinstance(data, dict): raise HTTPException(status_code=502, detail=f"adapter {adapter} answered {capability} with {type(data).__name__}, not an object")
    if m := spec.get("map"): data = {**data, **{k: _dig(data, p) for k, p in m.items()}}
    if miss := [k for k in want if data.get(k) is None]: raise HTTPException(status_code=502, detail=f"""adapter {adapter} did not answer {capability} with {miss}: {str(data.get("detail") or data.get("error") or data.get("status") or data)[:300]}""")
    return data

def _dig(d, path: str):
    for k in path.split("."): d = d.get(k) if isinstance(d, dict) else None
    return d
