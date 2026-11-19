"""core/netns.py - the shell around one program: it runs with its own network, where only loopback exists, so the only ways in or out are the ones Magpie made. Runs as a script ahead of the program, standard library only, and also holds the byte pipes every Magpie forwarder uses.
usage: netns.py <plan json> <program> <args...>   plan: {"sock": Unix socket Magpie reaches the program by, "port": the program's port, "reach": {inner port: Unix socket}, every other thing it may connect to}
netns.py --probe answers by exit code whether this kernel allows it."""
import os, sys, json, fcntl, ctypes, signal, socket, struct, asyncio

CLONE_NEWUSER, CLONE_NEWNET, SIOCGIFFLAGS, SIOCSIFFLAGS, IFF_UP, PR_SET_PDEATHSIG = 0x10000000, 0x40000000, 0x8913, 0x8914, 0x1, 1

# --- Byte pipes ---

async def pipe(r: asyncio.StreamReader, w: asyncio.StreamWriter, tap=None):
    """One direction until it ends. Passing on the end lets the other direction finish; a broken connection closes this side, which ends the other direction too."""
    try:
        while b := await r.read(1 << 16):
            if tap: tap(len(b))
            w.write(b)
            await w.drain()
        if w.can_write_eof(): w.write_eof()
    except OSError: w.close()

async def join(ar, aw, br, bw, tap_ab=None, tap_ba=None):
    """Both directions between two connections, then closes both."""
    await asyncio.gather(pipe(ar, bw, tap_ab), pipe(br, aw, tap_ba))
    for w in (aw, bw): w.close()

# --- The program's own network ---

def enter():
    """A new network namespace; without root, inside a user namespace mapping this same user, which needs no privilege where the kernel allows it. Only loopback exists there, so nothing else has a route."""
    uid, gid = os.geteuid(), os.getegid()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)   # nothing outlives the manager that started it
    if libc.unshare(CLONE_NEWNET | (CLONE_NEWUSER if uid else 0)): raise OSError(ctypes.get_errno(), "unshare")
    if uid:
        for f, v in (("setgroups", "deny"), ("uid_map", f"{uid} {uid} 1"), ("gid_map", f"{gid} {gid} 1")):
            with open(f"/proc/self/{f}", "w") as fh: fh.write(v)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        flags = struct.unpack("16sh", fcntl.ioctl(s, SIOCGIFFLAGS, struct.pack("16sh", b"lo", 0)))[1]
        fcntl.ioctl(s, SIOCSIFFLAGS, struct.pack("16sh", b"lo", flags | IFF_UP))

async def _carry(plan: dict, pid: int):
    async def inward(r, w): await join(r, w, *await asyncio.open_connection("127.0.0.1", plan["port"]))
    def outward(path):
        async def h(r, w): await join(r, w, *await asyncio.open_unix_connection(path))
        return h
    async def safe(fn, r, w):
        try: await fn(r, w)
        except OSError: w.close()
    if os.path.exists(plan["sock"]): os.unlink(plan["sock"])
    await asyncio.start_unix_server(lambda r, w: safe(inward, r, w), plan["sock"])
    for port, path in plan.get("reach", {}).items():
        h = outward(path)
        await asyncio.start_server(lambda r, w, h=h: safe(h, r, w), "127.0.0.1", int(port))
    _, status = await asyncio.get_running_loop().run_in_executor(None, os.waitpid, pid, 0)
    os._exit(os.waitstatus_to_exitcode(status) if os.waitstatus_to_exitcode(status) >= 0 else 128 - os.waitstatus_to_exitcode(status))

def main():
    if sys.argv[1] == "--probe": return enter()
    plan = json.loads(sys.argv[1])
    enter()
    pid = os.fork()
    if pid == 0:
        ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
        os.execvp(sys.argv[2], sys.argv[2:])
    signal.signal(signal.SIGTERM, signal.SIG_IGN)   # the program gets it too, as its group; this shell ends when the program has
    asyncio.run(_carry(plan, pid))

if __name__ == "__main__": main()
