import datetime
import os
import platform
import socket
import sys

import torch


class EnvironmentSnapshot:
    def __init__(self):
        self.python_version = platform.python_version()
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.torch_version = torch.__version__
        self.cuda_available = torch.cuda.is_available()
        self.cuda_version = torch.version.cuda if torch.version.cuda is not None else None
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0
        self._device = None
        self._device_name = None
        self._capability = None
        self._memory = None
        try:
            import torch_geometric
            self.pyg_version = torch_geometric.__version__
        except Exception:
            self.pyg_version = None
        self.started_at = datetime.datetime.now().isoformat(timespec="seconds")

    def choose_device(self, requested=None):
        if requested:
            self._device = requested
        else:
            self._device = self._pick_default()
        self._collect_device_details()
        return self._device

    def _pick_default(self):
        if self.cuda_available:
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _collect_device_details(self):
        user_device = self._device
        if user_device == "cuda" and self.cuda_available:
            self._device_name = torch.cuda.get_device_name(0)
            self._capability = list(torch.cuda.get_device_capability(0))
            self._memory = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
        else:
            self._device_name = None
            self._capability = None
            self._memory = None

    def describe(self):
        return {
            "python": self.python_version,
            "hostname": self.hostname,
            "pid": self.pid,
            "torch": self.torch_version,
            "pyg": self.pyg_version,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "gpu_count": self.device_count,
            "device": self._device,
            "device_name": self._device_name,
            "capability": self._capability,
            "device_memory_gb": self._memory,
            "started_at": self.started_at,
        }

    def render(self, prefix):
        lines = []
        info = self.describe()
        lines.append("%s python=%s host=%s pid=%d" % (prefix, info["python"], info["hostname"], info["pid"]))
        lines.append("%s torch=%s pyg=%s cuda_available=%s" % (prefix, info["torch"], info["pyg"], info["cuda_available"]))
        lines.append("%s device=%s" % (prefix, info["device"]))
        if info["device_name"]:
            lines.append("%s gpu=%s capability=%s memory_gb=%s" % (prefix, info["device_name"], info["capability"], info["device_memory_gb"]))
        return lines


def probe_environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot()