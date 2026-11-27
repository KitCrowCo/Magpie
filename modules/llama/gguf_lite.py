"""modules/llama/gguf_lite.py - reads a GGUF file's metadata and tensor table with the standard library only, so model facts work on a phone with no numpy. Tensor data is never read: only the header, a few MB at most."""
import struct

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_KEEP_ARRAY = 4096   # longer arrays (tokenizer vocabularies) are skipped, not kept

class Reader:
    def __init__(self, path: str):
        with open(path, "rb") as f:
            self.f = f
            if f.read(4) != b"GGUF": raise ValueError(f"{path} is not a GGUF file")
            self.version = self._u32()
            n_tensors, n_kv = self._u64(), self._u64()
            self.fields = {}
            for _ in range(n_kv):
                k = self._str()
                self.fields[k] = self._value(self._u32())
            self.tensors = [self._tensor() for _ in range(n_tensors)]
        del self.f

    def _read(self, fmt: str): return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))[0]
    def _u32(self): return self._read("<I")
    def _u64(self): return self._read("<Q")
    def _str(self) -> str: return self.f.read(self._u64()).decode("utf-8", "ignore")

    def _value(self, t: int):
        if t == 8: return self._str()
        if t == 9:
            et, n = self._u32(), self._u64()
            if n > _KEEP_ARRAY:
                if et == 8:
                    for _ in range(n): self.f.seek(self._u64(), 1)
                else: self.f.seek(n * struct.calcsize(_SCALAR[et]), 1)
                return None
            return [self._value(et) for _ in range(n)]
        return self._read(_SCALAR[t])

    def _tensor(self) -> dict:
        name, dims = self._str(), self._u32()
        n = 1
        for _ in range(dims): n *= self._u64()
        self._u32(); self._u64()   # element type and data offset: not needed for model facts
        return {"name": name, "n_elements": n}
