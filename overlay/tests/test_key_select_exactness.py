"""Forward candidates 2 and 6 (plan 12.3c): the exact k512 member merge in prompt key captures and the decode selector in decode key captures.

The real kpool, R590 canonical sort and glm53_tpc_topk wrapper run; only the TPC kernels are replaced, by a reference written from their
documented contract (top 512 finite scores, ties to the lower column, columns ascending, -1 padding). A CPU cannot run the kernels themselves:
their own exactness is the card probe's (glm53_tpc_topk docstring; decodegap/decodesel-card.json) and the prompt key merge check's.
Every result is compared bitwise three ways: new route, legacy route, and a brute-force sort of the whole scan.
"""
from __future__ import annotations

import contextlib
import logging
import sys
import unittest
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, str(Path(__file__).resolve().parent))

import glm53_mixed_graph as graph  # noqa: E402
import key_stack  # noqa: E402
from overlay_testlib import BASE_ROOT, OVERLAY_ROOT  # noqa: E402

MERGE_FLAG = "HPU_GLM53_KEY_K512_MEMBER_MERGE"
DECODE_FLAG = "HPU_GLM53_MLA_DECODESEL"
KEY_ROWS, TOP_POOLS, PROMPT_CHUNK, POOLS_PER_BLOCK = 128, 512, 32768, 32
PROMPT_WIDTHS = (2048, 4096, 8192)
# (rows, table width, scorer chunk) of every lattice key that scans more than one chunk; the chunks are pinned by test_scan_table below.
DECODE_KEYS = ((1, 4096, 65536), (1, 8192, 65536), (2, 1024, 16384), (2, 4096, 16384), (2, 8192, 16384),
               (16, 1024, 16384), (16, 4096, 16384), (16, 8192, 16384))
MODES = ("random", "tied_levels", "constant", "signed_zero_nonfinite", "chunk_boundary_ties", "few_valid", "short_context")
BASE_ENV = {"VLLM_GAUDI_MIXED_PROMPT_KEYS": "1024:8192,1024:1024", "HPU_GLM53_R589_CAPTURE_MARKS": "1", "HPU_GLM53_MLA_TOPKFUSE": "1"}
FORBIDDEN_OPS = ("nonzero", "masked_select", "unique", "bincount")
HOST_READS = ("item", "tolist", "numpy", "__bool__", "__int__", "__index__", "__float__")

torch.set_num_threads(8)  # the host is serving two boots


@dataclass(frozen=True)
class Case:
    scores: torch.Tensor
    lengths: torch.Tensor


@dataclass
class Selection:
    scores: torch.Tensor
    ids: torch.Tensor
    logical: torch.Tensor
    physical: torch.Tensor
    native_calls: int
    legacy_merges: int
    fences: list[tuple[str, bool]]


def _few_valid(rows: int, scan: int, generator: torch.Generator) -> torch.Tensor:
    keep = torch.rand(rows, scan, generator=generator) < 100 / scan
    if rows > 1:
        keep[0] = False  # one row with nothing valid at all
    return torch.where(keep, torch.randint(0, 3, (rows, scan), generator=generator).float(), -torch.inf)


def _signed_zero_nonfinite(rows: int, scan: int, generator: torch.Generator) -> torch.Tensor:
    levels = torch.tensor([-0.0, 0.0, 1.0, 1.0, 1.0, 1.0, torch.nan, torch.inf, -torch.inf, -1.0])
    return levels[torch.randint(0, len(levels), (rows, scan), generator=generator)]


def _chunk_boundary_ties(rows: int, scan: int, generator: torch.Generator) -> torch.Tensor:
    """The top value sits in the first 600 columns of every 16,384, more than one chunk keeps: the winners tie inside and across chunks, and the lowest IDs must win."""
    scores = torch.randint(0, 2, (rows, scan), generator=generator).float()
    scores[:, torch.arange(scan) % 16384 < 600] = 5.0
    return scores


def case_for(mode: str, rows: int, scan: int, seed: int = 7) -> Case:
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.full((rows,), scan * key_stack.POOL)
    if mode == "random":
        scores = torch.randn(rows, scan, generator=generator)
    elif mode == "constant":
        scores = torch.ones(rows, scan)
    elif mode == "signed_zero_nonfinite":
        scores = _signed_zero_nonfinite(rows, scan, generator)
    elif mode == "chunk_boundary_ties":
        scores = _chunk_boundary_ties(rows, scan, generator)
    elif mode == "few_valid":
        scores = _few_valid(rows, scan, generator)
    elif mode in ("tied_levels", "short_context"):
        scores = torch.randint(0, 4, (rows, scan), generator=generator).float()
    else:
        raise ValueError(mode)
    if mode == "short_context":
        lengths = torch.randint(0, scan * key_stack.POOL, (rows,), generator=generator)
        if rows > 1:
            lengths[:2] = torch.tensor([0, 3])  # no complete pool: every chunk of these rows is all-invalid
    return Case(scores, lengths)


def oracle(case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    """Score descending, lower pool ID on ties, over the whole scan at once: no chunks, no kernel, no kpool code."""
    columns = torch.arange(case.scores.shape[1]).unsqueeze(0)
    valid = torch.isfinite(case.scores) & (columns < (case.lengths // key_stack.POOL).unsqueeze(1))
    finite = case.scores.masked_fill(~valid, -torch.inf)
    order = torch.argsort(finite, dim=1, descending=True, stable=True)[:, :TOP_POOLS]
    values = torch.gather(finite, 1, order)
    return values, torch.where(torch.isfinite(values), order, -1)


def reference_emit(scores: torch.Tensor) -> torch.Tensor:
    """The kernel's contract at count 512: chosen columns ascending in the first slots, -1 after them."""
    width = scores.shape[1]
    order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :TOP_POOLS]
    chosen = torch.isfinite(torch.gather(scores, 1, order))
    ascending = torch.sort(torch.where(chosen, order, width), dim=1).values
    return torch.where(ascending < width, ascending, -1)


def tie_high_emit(scores: torch.Tensor) -> torch.Tensor:
    """Mutant: ties go to the higher column."""
    width = scores.shape[1]
    flipped = reference_emit(scores.flip(1))
    return torch.sort(torch.where(flipped >= 0, width - 1 - flipped, width), dim=1).values.where(flipped >= 0, -1)


def short_emit(scores: torch.Tensor) -> torch.Tensor:
    """Mutant: the highest-column member is lost."""
    selected = reference_emit(scores)
    last = (selected >= 0).sum(dim=1, keepdim=True).clamp_min(1) - 1
    return selected.scatter(1, last, -1)


@dataclass
class Kernel:
    emit: Callable[[torch.Tensor], torch.Tensor] = reference_emit
    forgets_running: bool = False  # mutant: every chunk's merge drops the running top-k
    calls: int = 0
    in_kernel: bool = False

    def select(self, scores: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.in_kernel = True
        try:
            return self.emit(scores)
        finally:
            self.in_kernel = False

    def namespaces(self) -> tuple[SimpleNamespace, SimpleNamespace]:
        pair = SimpleNamespace(topk_threshold_k512=lambda scores: None, topk_emit_k512=lambda scores, _metadata: self.select(scores))
        return pair, SimpleNamespace(select=self.select)


def _exec_on_cpu(name: str, path: Path) -> ModuleType:
    source = path.read_text(encoding="utf-8").replace('"hpu"', '"cpu"').replace("'hpu'", "'cpu'")
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), vars(module))
    return module


@contextlib.contextmanager
def native_world(kernel: Kernel, **env: str) -> Iterator[SimpleNamespace]:
    """One sparse layer with the real k512 wrappers loaded and the reference kernel behind them; undone on exit."""
    pair, fuse = kernel.namespaces()
    with key_stack.layer_stack(OVERLAY_ROOT, {**BASE_ENV, **env}) as world, mock.patch.object(torch.ops, "load_library"), \
            mock.patch.object(torch.ops, "g6_tpc", pair, create=True), mock.patch.object(torch.ops, "mla_fuse", fuse, create=True):
        _exec_on_cpu("glm53_mla_topkfuse", OVERLAY_ROOT / "glm53_mla_topkfuse.py")
        world.kpool._tpc_topk = _exec_on_cpu("glm53_tpc_topk", BASE_ROOT / "glm53_tpc_topk.py")
        world.kpool._rc6_flags = SimpleNamespace(witness=lambda *args: 0, LOG=logging.getLogger("rc6"), process_rank=lambda: 0)
        world.kernel = kernel
        if kernel.forgets_running:
            real_merge = world.kpool._decode_member_merge
            world.kpool._decode_member_merge = lambda *args, first, **kwargs: real_merge(*args, first=True, **kwargs)
        graph.reset_witness()
        yield world


@contextlib.contextmanager
def key_capture() -> Iterator[None]:
    with graph.capture_context(), graph.key_capture_context():
        yield


def install_scores(world: SimpleNamespace, scores: torch.Tensor) -> None:
    cursor = [0]

    def score(q: torch.Tensor, weights: torch.Tensor, rows: torch.Tensor, physical: torch.Tensor, valid: torch.Tensor,
              *, query_shape: tuple, shared_block_table: bool) -> torch.Tensor:
        start, width = cursor[0], physical.shape[-1]
        cursor[0] = start + width
        return scores[:, start:start + width].masked_fill(~valid, -torch.inf)

    world.kpool._score_pool_chunk = score


def _count_legacy_merges(world: SimpleNamespace) -> list[int]:
    count, real = [0], world.kpool._partial_chunk_merge

    def counted(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        count[0] += 1
        return real(*args, **kwargs)

    world.kpool._partial_chunk_merge = counted
    return count


def candidates(world: SimpleNamespace, case: Case, chunk: int, shared: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """The real scan at this key's static bucket; no host read of a length."""
    rows, scan = case.scores.shape
    table = torch.arange(scan // POOLS_PER_BLOCK, dtype=torch.int32).expand(rows, -1)
    query = torch.zeros(rows, key_stack.HEADS, key_stack.INDEX_DIM)
    return world.kpool.streaming_pooled_topk_candidates(
        query, torch.zeros(rows, key_stack.HEADS), torch.zeros(1, 132, dtype=torch.uint8), table[:1] if shared else table, case.lengths,
        source_block_size=key_stack.BLOCK, pool_size=key_stack.POOL, topk_tokens=key_stack.TOPK, context_chunk_pools=chunk,
        max_complete_pools_bucket=scan, shared_block_table=shared)


def select(world: SimpleNamespace, case: Case, chunk: int, shared: bool) -> Selection:
    install_scores(world, case.scores)
    merges = _count_legacy_merges(world)
    scores, ids = candidates(world, case, chunk, shared)
    rows, scan = case.scores.shape
    table = torch.arange(scan // POOLS_PER_BLOCK, dtype=torch.int32).expand(rows, -1)
    logical, physical = world.kpool.finalize_pooled_topk(scores, ids, table, case.lengths, source_block_size=key_stack.BLOCK,
                                                         pool_size=key_stack.POOL, topk_tokens=key_stack.TOPK)
    return Selection(scores, ids, logical, physical, world.kernel.calls, merges[0], list(world.tally.calls))


def prompt_selection(case: Case, kernel: Kernel | None = None, chunk: int = PROMPT_CHUNK, **env: str) -> Selection:
    with native_world(kernel or Kernel(), **env) as world, key_capture():
        return select(world, case, chunk, shared=True)


def decode_selection(case: Case, key: tuple[int, int], chunk: int, kernel: Kernel | None = None, **env: str) -> Selection:
    with native_world(kernel or Kernel(), **env) as world, graph.decode_key_context(key):
        return select(world, case, chunk, shared=False)


def bits(values: torch.Tensor) -> torch.Tensor:
    return values.contiguous().view(torch.int32)


def assert_exact(test: unittest.TestCase, new: Selection, legacy: Selection, case: Case) -> None:
    """Membership, order and score bits against the legacy route and the brute-force sort; tokens against the legacy route."""
    values, ids = oracle(case)
    test.assertTrue(torch.equal(bits(new.scores), bits(legacy.scores)), "scores differ from the legacy route")
    test.assertTrue(torch.equal(new.ids, legacy.ids), "ids differ from the legacy route")
    test.assertTrue(torch.equal(new.ids, ids), "ids differ from the brute-force sort")
    test.assertTrue(torch.equal(bits(new.scores), bits(values)), "scores differ from the brute-force sort")
    test.assertTrue(torch.equal(new.logical, legacy.logical), "logical tokens differ")
    test.assertTrue(torch.equal(new.physical, legacy.physical), "physical tokens differ")


class PromptRoute(unittest.TestCase):
    """What the plan read from source, run: which route each prompt key width takes."""

    def run_route(self, width: int, **env: str) -> dict[str, int]:
        rows, scan = KEY_ROWS, width * POOLS_PER_BLOCK
        table = torch.arange(width, dtype=torch.int32).expand(rows, -1)
        case = case_for("random", rows, scan)
        with native_world(Kernel(), **env) as world, key_capture():
            install_scores(world, case.scores)
            merges = _count_legacy_merges(world)
            world.kpool.streaming_pooled_topk(
                torch.zeros(rows, key_stack.HEADS, key_stack.INDEX_DIM), torch.zeros(rows, key_stack.HEADS),
                torch.zeros(1, 132, dtype=torch.uint8), table, case.lengths, source_block_size=key_stack.BLOCK,
                pool_size=key_stack.POOL, topk_tokens=key_stack.TOPK, context_chunk_pools=PROMPT_CHUNK,
                max_complete_pools_bucket=scan, shared_block_table=True)
            return {"native": world.kernel.calls, "legacy": merges[0], "witness": graph._WITNESS["k512_topk"]}

    def test_width_1024_is_fused_k512_and_wider_keys_fall_to_the_legacy_merge(self) -> None:
        self.assertEqual(self.run_route(1024), {"native": 1, "legacy": 0, "witness": 1})
        for width in (2048, 8192):
            with self.subTest(width=width):
                self.assertEqual(self.run_route(width), {"native": 0, "legacy": width // 1024, "witness": 0})

    def test_the_flag_moves_every_chunk_of_a_wide_key_to_the_member_merge(self) -> None:
        for width in PROMPT_WIDTHS:
            with self.subTest(width=width):
                chunks = width // 1024
                self.assertEqual(self.run_route(width, **{MERGE_FLAG: "1"}), {"native": chunks, "legacy": 0, "witness": chunks})

    def test_the_flag_leaves_the_fused_width_1024_key_alone(self) -> None:
        self.assertEqual(self.run_route(1024, **{MERGE_FLAG: "1"}), {"native": 1, "legacy": 0, "witness": 1})

    def test_the_live_line_names_the_scan_once_per_key(self) -> None:
        with self.assertLogs("glm53.key_member_merge", "WARNING") as logs:
            self.run_route(2048, **{MERGE_FLAG: "1"})
        lines = [line for line in logs.output if "GLM53_KEY_MEMBER_MERGE_LIVE" in line]
        self.assertEqual(len(lines), 1, logs.output)
        self.assertIn("rows=128 chunk=32768 scan=65536", lines[0])


class ScanTable(unittest.TestCase):
    """The chunks every exactness case assumes, from the real scorer budget at r0's env."""

    @staticmethod
    def chunk(op: object, rows: int, width: int, shared: bool) -> tuple[int, int]:
        bucket = op._scan_pool_bucket(width, key_stack.BLOCK, width * key_stack.BLOCK, None)
        return bucket, op._scoring_chunk(batch=rows, index_heads=key_stack.HEADS, index_dim=key_stack.INDEX_DIM, packed_row_bytes=132,
                                         active_pool_bucket=bucket, shared_block_table=shared)

    def test_scan_table(self) -> None:
        with key_stack.layer_stack(OVERLAY_ROOT) as world:
            op = key_stack.indexer_op(world, key_stack.prompt_case(1, 8))
            for width in (1024, *PROMPT_WIDTHS):
                with self.subTest(prompt=width):
                    self.assertEqual(self.chunk(op, KEY_ROWS, width, True), (width * POOLS_PER_BLOCK, PROMPT_CHUNK))
            for rows, width, chunk in DECODE_KEYS:
                with self.subTest(rows=rows, width=width):
                    self.assertEqual(self.chunk(op, rows, width, False), (width * POOLS_PER_BLOCK, chunk))
            self.assertEqual(self.chunk(op, 1, 1024, False), (32768, 65536))  # one chunk: the fused route, no merge to replace


class PromptExactness(unittest.TestCase):
    def test_member_merge_equals_legacy_and_brute_force_at_every_width_and_mode(self) -> None:
        for width in PROMPT_WIDTHS:
            for mode in MODES:
                with self.subTest(width=width, mode=mode):
                    case = case_for(mode, KEY_ROWS, width * POOLS_PER_BLOCK)
                    new, legacy = prompt_selection(case, **{MERGE_FLAG: "1"}), prompt_selection(case)
                    self.assertEqual((new.native_calls, new.legacy_merges), (width // 1024, 0))
                    self.assertEqual((legacy.native_calls, legacy.legacy_merges), (0, width // 1024))
                    self.assertEqual(new.fences, legacy.fences, "the member merge moved a fence")
                    self.assertEqual(sum(sync for _, sync in new.fences), 0, "sync=True inside a key capture")
                    assert_exact(self, new, legacy, case)

    def test_several_seeds_on_the_tie_heavy_modes(self) -> None:
        for seed in (11, 23):
            for mode in ("tied_levels", "signed_zero_nonfinite", "short_context"):
                with self.subTest(seed=seed, mode=mode):
                    case = case_for(mode, KEY_ROWS, 2048 * POOLS_PER_BLOCK, seed)
                    assert_exact(self, prompt_selection(case, **{MERGE_FLAG: "1"}), prompt_selection(case), case)

    def test_an_unproven_chunk_width_keeps_the_v1_members_and_stays_exact(self) -> None:
        case = case_for("tied_levels", KEY_ROWS, 2048 * POOLS_PER_BLOCK)
        new = prompt_selection(case, chunk=8192, **{MERGE_FLAG: "1"})
        self.assertEqual((new.native_calls, new.legacy_merges), (0, 8))
        assert_exact(self, new, prompt_selection(case, chunk=8192), case)


class DecodeExactness(unittest.TestCase):
    def test_decode_selector_equals_legacy_and_brute_force_on_every_multi_chunk_key(self) -> None:
        for rows, width, chunk in DECODE_KEYS:
            for mode in MODES:
                with self.subTest(rows=rows, width=width, mode=mode):
                    case = case_for(mode, rows, width * POOLS_PER_BLOCK)
                    new = decode_selection(case, (rows, width), chunk, **{DECODE_FLAG: "1"})
                    legacy = decode_selection(case, (rows, width), chunk, **{DECODE_FLAG: "0"})
                    chunks = width * POOLS_PER_BLOCK // chunk
                    self.assertEqual(new.native_calls, chunks)
                    self.assertEqual(legacy.native_calls, 0)
                    self.assertEqual(new.fences, legacy.fences, "the selector moved a fence")
                    assert_exact(self, new, legacy, case)


class Mutations(unittest.TestCase):
    """Every mode of each exactness test can fail: a broken kernel or merge is caught, or the test proves nothing."""

    MUTANTS = {"random": Kernel(short_emit), "tied_levels": Kernel(tie_high_emit), "constant": Kernel(tie_high_emit),
               "signed_zero_nonfinite": Kernel(tie_high_emit), "chunk_boundary_ties": Kernel(tie_high_emit),
               "few_valid": Kernel(forgets_running=True), "short_context": Kernel(forgets_running=True)}

    def assert_caught(self, mode: str, new: Selection, legacy: Selection, case: Case) -> None:
        with self.assertRaises(AssertionError, msg=f"mode {mode}: this mutant passed the exactness check"):
            assert_exact(self, new, legacy, case)

    def test_each_prompt_mode_catches_its_mutant(self) -> None:
        for mode, mutant in self.MUTANTS.items():
            with self.subTest(mode=mode):
                case = case_for(mode, KEY_ROWS, 2048 * POOLS_PER_BLOCK)
                new = prompt_selection(case, mutant, **{MERGE_FLAG: "1"})
                self.assert_caught(mode, new, prompt_selection(case), case)

    def test_each_decode_mode_catches_its_mutant(self) -> None:
        rows, width, chunk = 16, 1024, 16384
        for mode, mutant in self.MUTANTS.items():
            with self.subTest(mode=mode):
                case = case_for(mode, rows, width * POOLS_PER_BLOCK)
                new = decode_selection(case, (rows, width), chunk, mutant, **{DECODE_FLAG: "1"})
                self.assert_caught(mode, new, decode_selection(case, (rows, width), chunk, **{DECODE_FLAG: "0"}), case)


class OpRecorder(TorchDispatchMode):
    """Each op's name and output shape, and the bytes of live non-view outputs. Kernel ops are the card's, so they are left out."""

    def __init__(self, kernel: Kernel) -> None:
        super().__init__()
        self.kernel = kernel
        self.ops: list[tuple[str, tuple[int, ...]]] = []
        self.total_bytes = self.live_bytes = self.peak_bytes = 0

    def _release(self, size: int) -> None:
        self.live_bytes -= size

    def _record(self, name: str, tensor: torch.Tensor) -> None:
        self.ops.append((name, tuple(tensor.shape)))
        if tensor._is_view():
            return
        size = tensor.numel() * tensor.element_size()
        self.total_bytes += size
        self.live_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.live_bytes)
        weakref.finalize(tensor, self._release, size)

    def __torch_dispatch__(self, func: Callable, types: object, args: tuple = (), kwargs: dict | None = None) -> object:
        output = func(*args, **(kwargs or {}))
        if not self.kernel.in_kernel:
            for tensor in output if isinstance(output, (tuple, list)) else (output,):
                if isinstance(tensor, torch.Tensor):
                    self._record(func.__name__, tensor)
        return output


class CapturePath(unittest.TestCase):
    """Legal in a captured graph: no host read, one op sequence whatever the data, no data-dependent shape, a smaller workspace."""

    def recorded(self, mode: str, **env: str) -> OpRecorder:
        case = case_for(mode, KEY_ROWS, 2048 * POOLS_PER_BLOCK)
        kernel = Kernel()
        with native_world(kernel, **env) as world, key_capture():
            install_scores(world, case.scores)
            recorder = OpRecorder(kernel)
            with recorder:
                candidates(world, case, PROMPT_CHUNK, True)
        return recorder

    def test_no_host_read_in_the_member_merge(self) -> None:
        case = case_for("tied_levels", KEY_ROWS, 2048 * POOLS_PER_BLOCK)
        refusal = AssertionError("a host read inside the capture")
        with native_world(Kernel(), **{MERGE_FLAG: "1"}) as world, key_capture():
            install_scores(world, case.scores)
            patches = [mock.patch.object(torch.Tensor, name, side_effect=refusal) for name in HOST_READS]
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                _, ids = candidates(world, case, PROMPT_CHUNK, True)
            self.assertEqual(world.kernel.calls, 2)
        self.assertEqual(tuple(ids.shape), (KEY_ROWS, TOP_POOLS))

    def test_one_op_sequence_of_static_shapes_whatever_the_data(self) -> None:
        reference = self.recorded("random", **{MERGE_FLAG: "1"}).ops
        for mode in ("few_valid", "constant", "short_context"):
            with self.subTest(mode=mode):
                self.assertEqual(self.recorded(mode, **{MERGE_FLAG: "1"}).ops, reference)
        self.assertFalse({name for name, _ in reference} & set(FORBIDDEN_OPS), reference)

    def test_the_member_merge_holds_less_workspace_than_the_legacy_merge(self) -> None:
        new, legacy = self.recorded("random", **{MERGE_FLAG: "1"}), self.recorded("random")
        mib = 2**20
        print(f"\nCAPTURE_WORKSPACE per 2-chunk tile, op outputs without the kernel's own: total legacy={legacy.total_bytes / mib:.1f} MiB "
              f"new={new.total_bytes / mib:.1f} MiB; peak live legacy={legacy.peak_bytes / mib:.1f} MiB new={new.peak_bytes / mib:.1f} MiB",
              file=sys.stderr)
        self.assertLess(new.total_bytes, legacy.total_bytes)
        self.assertLess(new.peak_bytes, legacy.peak_bytes)


class FlagOff(unittest.TestCase):
    def test_the_gate_is_closed_with_the_flag_unset(self) -> None:
        with native_world(Kernel()) as world, key_capture():
            self.assertFalse(world.kpool._key_member_merge_applies(
                batch=KEY_ROWS, width=PROMPT_CHUNK, top_pools=TOP_POOLS, tp_size=1, shared_block_table=True))

    def test_the_gate_opens_only_on_a_shared_128_row_tile_of_a_key_capture(self) -> None:
        good = {"batch": KEY_ROWS, "width": PROMPT_CHUNK, "top_pools": TOP_POOLS, "tp_size": 1, "shared_block_table": True}
        with native_world(Kernel(), **{MERGE_FLAG: "1"}) as world:
            applies = world.kpool._key_member_merge_applies
            self.assertFalse(applies(**good), "outside any capture")
            with graph.capture_context():
                self.assertFalse(applies(**good), "a dual capture")
            with graph.decode_key_context((16, 8192)):
                self.assertFalse(applies(**good), "a decode key capture")
            with key_capture():
                self.assertTrue(applies(**good))
                for bad in ({"batch": 64}, {"width": 256}, {"tp_size": 4}, {"shared_block_table": False}):
                    with self.subTest(bad=bad):
                        self.assertFalse(applies(**{**good, **bad}))

    def test_without_the_native_module_the_gate_stays_closed(self) -> None:
        with native_world(Kernel(), **{MERGE_FLAG: "1"}) as world, key_capture():
            world.kpool._tpc_topk = None
            self.assertFalse(world.kpool._key_member_merge_applies(
                batch=KEY_ROWS, width=PROMPT_CHUNK, top_pools=TOP_POOLS, tp_size=1, shared_block_table=True))


class DecodeReach(unittest.TestCase):
    """kpool's `not mixed_graph_capture_active()` does not close the selector in a decode key: only a mixed graph capture sets that context."""

    def test_a_decode_key_capture_is_not_a_mixed_graph_capture(self) -> None:
        seen = []
        runner = mock.MagicMock(attn_block_size=key_stack.BLOCK, max_model_len=1_048_576)
        runner._prepare_dummy_scenario.side_effect = lambda *args, **kwargs: seen.append(
            (graph.capture_active(), graph.key_capture_active(), graph.decode_key_capture_active()))
        with mock.patch.object(graph, "_free_hbm", return_value=0), mock.patch.object(graph, "_in_use_bytes", return_value=0), \
                mock.patch.object(graph, "_reset_peak_in_use"), mock.patch.object(graph, "_peak_in_use_bytes", return_value=0), \
                mock.patch.object(graph, "_graph_count", return_value=0):
            graph._run_decode_capture(runner, (16, 8192))
        self.assertEqual(seen, [(False, False, True)])

    def test_the_gate_is_open_in_a_decode_key_and_closed_in_a_dual_capture(self) -> None:
        good = {"batch": 16, "width": 16384, "top_pools": TOP_POOLS, "tp_size": 1, "shared_block_table": False}
        with native_world(Kernel(), **{DECODE_FLAG: "1"}) as world:
            applies = world.kpool._decode_select_applies
            with graph.decode_key_context((16, 8192)):
                self.assertTrue(applies(**good))
            with graph.capture_context():
                self.assertFalse(applies(**good))

    def live_lines(self, runs: tuple[tuple[int, int, int], ...]) -> list[str]:
        with self.assertLogs("glm53.mla_decodesel", "WARNING") as logs, native_world(Kernel(), **{DECODE_FLAG: "1"}) as world:
            for rows, width, chunk in runs:
                case = case_for("random", rows, width * POOLS_PER_BLOCK)
                with graph.decode_key_context((rows, width)):
                    install_scores(world, case.scores)
                    candidates(world, case, chunk, False)
        return [line for line in logs.output if "GLM53_MLA_DECODESEL_LIVE" in line]

    def test_one_live_line_per_decode_key_so_the_count_means_something(self) -> None:
        lines = self.live_lines(((16, 1024, 16384), (2, 1024, 16384), (16, 1024, 16384), (1, 4096, 65536)))
        self.assertEqual(len(lines), 3, lines)
        self.assertIn("rows=16 chunk=16384 scan=32768 native_k512=1 capture=decode_key", lines[0])
        self.assertIn("rows=2 chunk=16384 scan=32768 native_k512=1 capture=decode_key", lines[1])
        self.assertIn("rows=1 chunk=65536 scan=131072 native_k512=1 capture=decode_key", lines[2])

    def test_outside_a_key_capture_the_line_stays_once_per_process_and_unchanged(self) -> None:
        with self.assertLogs("glm53.mla_decodesel", "WARNING") as logs, native_world(Kernel(), **{DECODE_FLAG: "1"}) as world:
            for _ in range(2):
                case = case_for("random", 16, 1024 * POOLS_PER_BLOCK)
                install_scores(world, case.scores)
                candidates(world, case, 16384, False)
        lines = [line for line in logs.output if "GLM53_MLA_DECODESEL_LIVE" in line]
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].endswith("rows=16 chunk=16384 scan=32768 native_k512=1"), lines[0])


if __name__ == "__main__":
    unittest.main()
