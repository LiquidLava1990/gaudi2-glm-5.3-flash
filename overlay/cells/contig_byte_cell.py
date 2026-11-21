"""Step 3 / step 14 byte cell: the real RecurrentSidecarWorker on one card. NOT RUN by the build lane: needs a card.

Stores random device rows into `--slots` CPU blocks (D2H), loads each back into another row (H2D), compares bytes, and
measures what `/proc/self/maps` grew by per block against the state bytes. Accept (plan step 3): byte equal, and for
--mode contig growth about 1.03x, not 2x, over 50 slots; --mode native is the control that reproduces C603's 2x.

`--device cpu` rehearses the runner logic without a card; its verdict never counts as the cell (`card_verdict=false`).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

MIB = 1 << 20
RANK_STATE_BYTES = 36_904_960  # 147,619,840 B per boundary across 4 ranks [M /metrics]
STATE_TENSORS = 35  # about 1 MiB each [M C603]
MAX_CONTIG_RATIO = 1.10
MIN_CONTIG_RATIO = 0.90
ROWS = 4
SIDECAR = "vllm/v1/kv_offload/cpu/recurrent_sidecar.py"
MAPS_LINE = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) \S+ \S+ \S+ \S+\s*(.*)$")


def parse_maps(text: str) -> dict[str, int]:
    """Mapped bytes by backing: '/dev/zero', 'anon', 'other' (a mapping's backing name, deleted or not)."""
    sizes = {"/dev/zero": 0, "anon": 0, "other": 0}
    for line in text.splitlines():
        match = MAPS_LINE.match(line)
        if match is None:
            continue
        size = int(match.group(2), 16) - int(match.group(1), 16)
        path = match.group(3).strip()
        kind = "anon" if not path else "/dev/zero" if path.startswith("/dev/zero") else "other"
        sizes[kind] += size
    return sizes


def maps_growth(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {kind: after[kind] - before[kind] for kind in before}


def verdict(mode: str, device: str, byte_equal: bool, growth: dict[str, int], state_bytes: int) -> dict[str, Any]:
    """The pinned-memory ratio is /dev/zero + anon growth over the state stored; the file-backed part is not pinned."""
    ratio = (growth["/dev/zero"] + growth["anon"]) / max(1, state_bytes)
    within = MIN_CONTIG_RATIO <= ratio <= MAX_CONTIG_RATIO if mode == "contig" else True
    card = device == "hpu"
    return {"mode": mode, "device": device, "card_verdict": card, "byte_equal": byte_equal,
            "mapped_ratio": round(ratio, 4), "growth_mib": {kind: size // MIB for kind, size in growth.items()},
            "pass": bool(card and byte_equal and within)}


def load_sidecar(overlay: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("contig_cell_sidecar", overlay / SIDECAR)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {overlay / SIDECAR}: pass --overlay pointing at the overlay folder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def state_tensors(device: str, state_bytes: int, tensors: int, seed: int) -> dict[int, tuple[torch.Tensor, ...]]:
    """One compact group of ROWS x (about state_bytes / tensors) random tensors; the last one is bf16, the rest fp32."""
    generator = torch.Generator().manual_seed(seed)
    elements = state_bytes // tensors // 4
    made = []
    for index in range(tensors):
        dtype = torch.bfloat16 if index == tensors - 1 else torch.float32
        count = elements * 2 if dtype == torch.bfloat16 else elements
        raw = torch.randint(0, 255, (ROWS * count * torch.empty(0, dtype=dtype).element_size(),), generator=generator,
                            dtype=torch.uint8)
        made.append(raw.view(dtype).reshape(ROWS, count).clone().to(device))
    return {0: tuple(made)}


def as_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()


def build_worker(module: ModuleType, groups: dict[int, tuple[torch.Tensor, ...]], rows: dict[str, int]) -> Any:
    registry = module.RecurrentStateRegistry(
        groups, lambda req_id, group_idx: rows[req_id],
        sum(tensor[0].nbytes for tensors in groups.values() for tensor in tensors), "contig-cell", lambda req_id: rows[req_id])
    return module.RecurrentSidecarWorker(registry)


def round_trip(worker: Any, groups: dict[int, tuple[torch.Tensor, ...]], block_id: int) -> bool:
    """D2H row 1 into the block, clear row 2, H2D the block into row 2, compare with row 1 on the host."""
    tensors = groups[0]
    expected = [as_bytes(tensor[1]) for tensor in tensors]
    worker.submit_store(block_id * 2, "source", block_id)
    worker.wait({block_id * 2})
    stored = [as_bytes(tensor) for tensor in worker._slot(block_id)[0]]
    for tensor in tensors:
        tensor[2] = torch.zeros_like(tensor[2])
    worker.submit_load(block_id * 2 + 1, "target", block_id)
    worker.wait({block_id * 2 + 1})
    loaded = [as_bytes(tensor[2]) for tensor in tensors]
    return stored == expected and loaded == expected


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "hpu":
        import habana_frameworks.torch.core  # noqa: F401  registers the hpu device; fails here without a card
    os.environ["VLLM_RECURRENT_SIDECAR_CONTIG"] = "1" if args.mode == "contig" else "0"
    module = load_sidecar(args.overlay)
    groups = state_tensors(args.device, args.state_bytes, args.tensors, args.seed)
    worker = build_worker(module, groups, {"source": 1, "target": 2})
    per_block = sum(tensor[0].nbytes for tensor in groups[0])
    before = parse_maps(Path("/proc/self/maps").read_text())
    equal = all([round_trip(worker, groups, block_id) for block_id in range(1, args.slots + 1)])
    after = parse_maps(Path("/proc/self/maps").read_text())
    result = verdict(args.mode, args.device, equal, maps_growth(before, after), per_block * args.slots)
    result.update(slots=args.slots, per_block_bytes=per_block, tensors=args.tensors, pid=os.getpid())
    worker.shutdown()
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("contig", "native"), required=True)
    parser.add_argument("--device", choices=("hpu", "cpu"), required=True)
    parser.add_argument("--slots", type=int, default=50)
    parser.add_argument("--state-bytes", type=int, default=RANK_STATE_BYTES)
    parser.add_argument("--tensors", type=int, default=STATE_TENSORS)
    parser.add_argument("--seed", type=int, default=1106)
    parser.add_argument("--overlay", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    result = run(args)
    line = json.dumps(result, sort_keys=True)
    print(f"CONTIG_CELL {line}", flush=True)
    args.out.write_text(line + "\n")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
