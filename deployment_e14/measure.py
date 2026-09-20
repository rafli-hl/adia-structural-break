"""Local Windows measurement only; not shipped in the submission package."""
import ctypes
from ctypes import wintypes
import os
import threading


class Counters(ctypes.Structure):
    _fields_ = [("cb",wintypes.DWORD), ("PageFaultCount",wintypes.DWORD),
        *[(n,ctypes.c_size_t) for n in ("PeakWorkingSetSize","WorkingSetSize",
        "QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage",
        "QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage","PrivateUsage")]]


def snapshot():
    if os.name != "nt":
        raise RuntimeError("This local measurement helper requires the validated Windows host")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    api = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    api.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    api.restype = wintypes.BOOL
    value = Counters()
    value.cb = ctypes.sizeof(value)
    if not api(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(rss_bytes=int(value.WorkingSetSize), lifetime_peak_rss_bytes=int(value.PeakWorkingSetSize),
        private_bytes=int(value.PrivateUsage), lifetime_peak_commit_bytes=int(value.PeakPagefileUsage))


class MemoryProbe:
    def __enter__(self):
        self.before = snapshot()
        self.peak_rss, self.peak_private = self.before["rss_bytes"], self.before["private_bytes"]
        self.stop = threading.Event()
        self.errors = []
        def sample():
            while not self.stop.wait(.01):
                try:
                    v = snapshot()
                    self.peak_rss = max(self.peak_rss, v["rss_bytes"])
                    self.peak_private = max(self.peak_private, v["private_bytes"])
                except Exception as exc:
                    self.errors.append(str(exc))
                    break
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        after = snapshot()
        self.report = dict(before=self.before, after=after,
            sampled_peak_rss_bytes=max(self.peak_rss,after["rss_bytes"]),
            sampled_peak_private_bytes=max(self.peak_private,after["private_bytes"]),
            interval_seconds=.01, scope="whole local process including host, imports and loaded data; not isolated model memory",
            errors=self.errors)
        if self.errors:
            raise RuntimeError(self.errors)
