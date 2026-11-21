"""Load an image base file and its overlay copy side by side under private module names."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

BASE_ROOT = Path("/usr/local/lib/python3.12/dist-packages")
OVERLAY_ROOT = Path(__file__).resolve().parents[1]
GUARD_FILE = "glm53_graph_state.py"
SIDECAR_FILE = "vllm/v1/kv_offload/cpu/recurrent_sidecar.py"
SCHEDULER_FILE = "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"


def _load(alias: str, path: Path) -> ModuleType:
    if alias in sys.modules:
        return sys.modules[alias]
    if str(OVERLAY_ROOT) not in sys.path:
        sys.path.append(str(OVERLAY_ROOT))  # after site-packages: only new top-level modules resolve from here
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}: run the tests inside the 1481bb99 container with the overlay mounted")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module  # dataclass annotations resolve through sys.modules
    spec.loader.exec_module(module)
    return module


def load_base(relative: str) -> ModuleType:
    return _load("base_" + relative.replace("/", "_").removesuffix(".py"), BASE_ROOT / relative)


def load_overlay(relative: str) -> ModuleType:
    return _load("overlay_" + relative.replace("/", "_").removesuffix(".py"), OVERLAY_ROOT / relative)
