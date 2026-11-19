"""modules/web/module.py - the outside, read by this node: one page (web_fetch) or a search (web_search), each a job like any other, so it is queued, journaled, cancelled and counted the same way.
The node makes the call, never the portal and never a contained program: whoever asks gets text back, marked as outside content with where it came from.
What a page says is data. It can carry instructions meant for a model that reads it, so what is refused is decided here and not by the reader: an address that is not public (this machine, the local network, link-local, reserved) is refused, the host's every resolved address is checked, and each redirect is checked again before it is followed.
Limit, stated: the address is checked and then the page is asked for by name, so a name server that answers differently the second time could still point inward; a research node kept off the local network closes that."""
import re, html, socket, asyncio, ipaddress
from html.parser import HTMLParser
from urllib.parse import urlsplit, urljoin
import httpx
from core import settings, journal
from core.adapters import Adapter

NAME = "web"   # the one thing this module offers work with: a role is bound to it like to a model
def _cfg(key: str): return settings.get(f"web.{key}")
def _timeout() -> httpx.Timeout: return httpx.Timeout(connect=15.0, read=float(_cfg("timeout_s")), write=15.0, pool=15.0)

# --- What may be read ---

def _host_ok(host: str, lists: tuple) -> str:
    """Why a host is refused, or "" when it may be read."""
    allow, block = lists
    under = lambda h, names: any(h == n or h.endswith("." + n) for n in (str(x).lower().lstrip(".") for x in names))
    if under(host, block): return f"{host} is among the hosts never read (web.block_hosts)"
    if allow and not under(host, allow): return f"{host} is not among the hosts that may be read (web.allow_hosts)"
    return ""

async def _check(url: str) -> str:
    """Why this address is refused, or "". Every address the host's name resolves to must be a public one, unless local addresses are allowed."""
    u = urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname: return f"not a web address: {url[:200]}"
    if why := _host_ok(u.hostname.lower(), (_cfg("allow_hosts") or [], _cfg("block_hosts") or [])): return why
    if _cfg("allow_private"): return ""
    try: infos = await asyncio.get_running_loop().getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as e: return f"{u.hostname} does not resolve: {e}"
    inner = [i[4][0] for i in infos if not ipaddress.ip_address(i[4][0].split("%")[0]).is_global]
    return f"{u.hostname} is not a public address ({inner[0]}): pages on this machine or the local network are not read (web.allow_private)" if inner else ""

# --- A page as text ---

class _Text(HTMLParser):
    """A page's readable text: what scripts, styles and hidden parts hold is left out, blocks become lines, links keep their address."""
    SKIP, BLOCK = {"script", "style", "noscript", "template", "svg", "head"}, {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "blockquote", "pre", "table", "ul", "ol", "dt", "dd"}
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.title, self._skip, self._in_title, self._href = [], "", 0, False, None
    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP: self._skip += 1
        if tag == "title": self._in_title = True
        if tag in self.BLOCK: self.out.append("\n")
        if tag == "a": self._href = dict(attrs).get("href")
    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip: self._skip -= 1
        if tag == "title": self._in_title = False
        if tag == "a" and self._href and self._href.startswith("http"): self.out.append(f" ({self._href})")
        if tag == "a": self._href = None
        if tag in self.BLOCK: self.out.append("\n")
    def handle_data(self, data):
        if self._in_title: self.title += data
        elif not self._skip: self.out.append(data)

def _as_text(body: bytes, kind: str, charset: str) -> tuple:
    raw = body.decode(charset or "utf-8", errors="replace")
    if "html" not in kind and not raw.lstrip()[:200].lower().startswith(("<!doctype html", "<html")): return raw, ""
    p = _Text()
    p.feed(raw)
    return re.sub(r"\n\s*\n\s*", "\n\n", re.sub(r"[ \t\r\f\v]+", " ", "".join(p.out))).strip(), html.unescape(p.title).strip()

# --- The adapter ---

class Web(Adapter):
    capabilities = {"web_fetch": {}, "web_search": {}}
    owner = "web"
    def models(self) -> list: return [{"name": NAME, "capabilities": ["web_fetch", *(["web_search"] if _cfg("search_url") else [])], "size": 0, "format": "", "about": "pages read and searches made by this node"}]
    def needs_service(self, job) -> bool: return False   # nothing to start: the node itself asks
    def status(self) -> dict: return {"search": _cfg("search_url") or "none set", "local addresses": "allowed" if _cfg("allow_private") else "refused"}

    async def run(self, job, service):
        if job.capability == "web_search": data = await self._search(job)
        else: data = await self._fetch(job)
        yield {"type": "result", "data": data}
        yield {"type": "done", "reason": "stop"}

    async def _fetch(self, job) -> dict:
        url, seen = str(job.body.get("url") or "").strip(), []
        if "://" not in url: url = "https://" + url
        async with httpx.AsyncClient(timeout=_timeout(), trust_env=False, follow_redirects=False, headers={"User-Agent": _cfg("user_agent"), "Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.5"}) as c:
            for _ in range(int(_cfg("max_redirects")) + 1):
                if why := await _check(url):
                    journal.event("web.refused", job=job.id, url=url[:500], why=why)
                    raise RuntimeError(why)
                seen.append(url)
                async with c.stream("GET", url) as r:
                    if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                        url = urljoin(url, r.headers["location"])
                        continue
                    body, cut = b"", False
                    async for part in r.aiter_bytes():
                        body += part
                        if len(body) > int(_cfg("max_bytes")):
                            body, cut = body[:int(_cfg("max_bytes"))], True
                            break
                    kind = r.headers.get("content-type", "").split(";")[0].strip().lower()
                    journal.event("web.fetch", job=job.id, url=url[:500], status=r.status_code, bytes=len(body), kind=kind, via=seen[:-1])
                    if not (kind.startswith("text/") or kind in ("application/json", "application/xml", "application/xhtml+xml", "")): return {"url": url, "status": r.status_code, "kind": kind, "title": "", "text": "", "outside": True, "note": f"not text ({kind}, {len(body)} bytes): not read"}
                    text, title = _as_text(body, kind, r.charset_encoding or "")
                    most = int(job.body.get("max_chars") or 0)
                    return {"url": url, "status": r.status_code, "kind": kind, "title": title, "text": text[:most] if most else text, "chars": len(text), "outside": True, "via": seen[:-1], **({"note": "cut: the page was larger than this node reads (web.max_bytes)"} if cut else {"note": f"cut to {most} of {len(text)} characters"} if most and len(text) > most else {})}
        raise RuntimeError(f"more than {_cfg('max_redirects')} redirects from {seen[0]}")

    async def _search(self, job) -> dict:
        base = str(_cfg("search_url") or "").rstrip("/")
        if not base: raise RuntimeError("no search server is set on this node (web.search_url): a SearXNG you run, e.g. http://host:8888")
        async with httpx.AsyncClient(timeout=_timeout(), trust_env=False) as c: r = await c.get(base + "/search", params={"q": str(job.body["query"]), "format": "json", **({"language": job.body["language"]} if job.body.get("language") else {})})
        if r.status_code != 200: raise RuntimeError(f"the search server answered {r.status_code}: {r.text[:300]} (a SearXNG has to allow the json format in its settings)")
        got = r.json().get("results", [])[:int(job.body.get("n") or 10)]
        journal.event("web.search", job=job.id, query=str(job.body["query"])[:300], results=len(got))
        return {"query": job.body["query"], "outside": True, "results": [{"title": x.get("title", ""), "url": x.get("url", ""), "text": x.get("content", "")} for x in got]}

def standby() -> list: return Web.models(None)   # what this module serves, listed while it is off

def setup(ctx): ctx.add_adapter(Web("web"))
