"""core/http.py - routes with a required access scope, JSON answers, and NDJSON streams. Built on Starlette alone (no pydantic), so the kernel installs on phones and small boards without compiling anything.
A route names its scope: "open" (nothing to check), "read" (status and lists), "infer" (run work), "admin" (control, install, policy, delete). A route that names none is admin: forgetting a scope fails closed."""
import json
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse, HTMLResponse
from starlette.routing import Route
from starlette.exceptions import HTTPException
from core import auth

__all__ = ["Router", "Request", "Response", "StreamingResponse", "HTMLResponse", "HTTPException", "json_body", "ndjson", "sse"]

async def json_body(request: Request) -> dict:
    raw = await request.body()
    if not raw: return {}
    try: b = json.loads(raw)
    except ValueError: raise HTTPException(status_code=400, detail="request body is not JSON")
    if not isinstance(b, dict): raise HTTPException(status_code=400, detail="request body must be a JSON object")
    return b

def ndjson(gen, **kw) -> StreamingResponse: return StreamingResponse(gen, media_type="application/x-ndjson", **kw)
def sse(gen, **kw) -> StreamingResponse: return StreamingResponse(gen, media_type="text/event-stream", **kw)

class Router:
    def __init__(self): self.routes = []

    def add(self, path: str, fn, methods=("GET",), scope: str = "admin"):
        """Handler: async fn(request, **path_params) answering a dict or list (sent as JSON) or a Response. One path may be declared once per method set."""
        async def endpoint(request: Request):
            request.state.key = auth.check(request, scope)
            out = await fn(request, **request.path_params)
            return out if isinstance(out, Response) else JSONResponse(out)
        self.routes.append(Route(path, endpoint, methods=list(methods), name=f"{fn.__module__}.{fn.__name__}:{path}"))
        return fn

    def route(self, path: str, methods=("GET",), scope: str = "admin"): return lambda fn: self.add(path, fn, methods, scope)
    def get(self, path: str, scope: str = "read"): return self.route(path, ("GET",), scope)
    def post(self, path: str, scope: str = "admin"): return self.route(path, ("POST",), scope)
    def put(self, path: str, scope: str = "admin"): return self.route(path, ("PUT",), scope)
    def delete(self, path: str, scope: str = "admin"): return self.route(path, ("DELETE",), scope)
