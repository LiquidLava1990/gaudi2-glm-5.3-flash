"""Step 27: what the real layer does inside a key capture. A tally of mark_step in both arms, the static scan bound and the scorer's
arithmetic per key width, the levers that must stay off, the levers that must fire, and a scan of literal sync=True sites."""
from __future__ import annotations

import ast
import contextlib
import math
import os
import sys
import unittest
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import glm53_mixed_graph as graph  # noqa: E402
import key_stack  # noqa: E402
from overlay_testlib import OVERLAY_ROOT, load_overlay  # noqa: E402

KEYS = "VLLM_GAUDI_MIXED_PROMPT_KEYS"
MARKS = "HPU_GLM53_R589_CAPTURE_MARKS"
ARMS = {"A": "0", "B": "1"}
ROWS = 1024
WIDTHS = (1024, 8192)
TILE_LABELS = ("kda.tile", "moe.tile", "r590.indexer.tile")
KEPT_LEVERS = ("mla_unpack", "mla_topkfuse", "k512_topk", "fwht", "sinkhorn", "kda_bitwise_rows", "kda_block_inverse",
               "private_topk", "final_norm_mark")
# Every literal sync=True a key capture can reach is on a line this table allows: it runs only outside any capture, or it refuses.
SYNC_TRUE_ALLOWED = {
    "glm53_r589_bounded_prefill.py": {"_boundary": "returns first in a capture", "_cpu_synced": "raises first in a capture"},
    "glm53_kda_tile_graph.py": {"tile": "key capture takes the original kernel before this"},
    "vllm_gaudi/ops/hpu_glm5next_kda.py": {"_kda_v2_capture_window": "HPU_GLM53_KDA_V2_FOR unset in the key env",
                                           "_kda_v2_window_graph": "HPU_GLM53_KDA_V2_FOR unset in the key env"},
}
SCANNED = ("glm53_r589_bounded_prefill.py", "glm53_r590_long_history.py", "glm53_kda_tile_graph.py", "glm53_mla_topkfuse.py",
           "vllm_gaudi/attention/oot_mla.py", "vllm_gaudi/ops/hpu_glm5next_kpool.py",
           "vllm_gaudi/ops/hpu_glm5next_sparse_indexer.py", "vllm_gaudi/ops/hpu_glm5next_q4kv.py",
           "vllm_gaudi/ops/hpu_glm5next_kda.py")


@contextlib.contextmanager
def key_capture() -> Iterator[None]:
    with graph.capture_context(), graph.key_capture_context():
        yield


class StackCase(unittest.TestCase):
    def world(self, arm: str = "A", **env: str) -> contextlib.AbstractContextManager:
        return key_stack.layer_stack(OVERLAY_ROOT, {KEYS: "1024:8192,1024:1024", MARKS: ARMS[arm], **env})

    @staticmethod
    def layer(world: SimpleNamespace, width: int, op: object | None = None) -> torch.Tensor:
        world.kpool._score_pool_chunk = key_stack.fast_scores
        case = key_stack.prompt_case(ROWS, width)
        with key_capture():
            physical = key_stack.indexer_step(world, case, op)
            key_stack.q4_fetch_step(world, physical, width)
            key_stack.kda_step(world, ROWS)
        return physical


class Tally(StackCase):
    def test_no_sync_true_at_either_key_in_either_arm(self) -> None:
        for arm in ARMS:
            for width in WIDTHS:
                with self.subTest(arm=arm, width=width), self.world(arm) as world:
                    self.layer(world, width)
                    self.assertEqual(world.tally.syncs(), 0, world.tally.files())

    def test_arm_b_marks_the_tiles_without_sync_and_arm_a_does_not(self) -> None:
        counts = {}
        for arm in ARMS:
            with self.world(arm) as world:
                self.layer(world, 1024)
                counts[arm] = world.tally.files()[("glm53_r589_bounded_prefill.py", False)]
        self.assertEqual(counts["A"], 0)
        self.assertGreater(counts["B"], 0)

    def test_arm_b_marks_each_scan_chunk_of_the_widest_key(self) -> None:
        with self.world("B") as world:
            self.layer(world, 8192)
            self.assertGreater(world.tally.files()[("hpu_glm5next_kpool.py", False)], 0)

    def test_the_same_layer_outside_a_capture_still_syncs(self) -> None:
        """The fences are gated on the capture, not removed: the native route is untouched."""
        with self.world("A") as world:
            world.kpool._score_pool_chunk = key_stack.fast_scores
            key_stack.indexer_step(world, key_stack.prompt_case(ROWS, 8192))
            self.assertGreater(world.tally.syncs(), 0)


class BoundaryArms(StackCase):
    def boundary(self, world: SimpleNamespace, label: str) -> None:
        world.r589._boundary(label, torch.zeros(1), drain=False)

    def test_tile_labels_mark_unsynced_in_arm_b_only(self) -> None:
        for arm, expected in (("A", 0), ("B", 1)):
            for label in TILE_LABELS:
                with self.subTest(arm=arm, label=label), self.world(arm) as world, key_capture():
                    self.boundary(world, label)
                    self.assertEqual(len(world.tally.calls), expected)
                    self.assertEqual(world.tally.syncs(), 0)

    def test_other_labels_stay_silent_in_arm_b(self) -> None:
        with self.world("B") as world, key_capture():
            self.boundary(world, "kda.owned")
            self.assertEqual(world.tally.calls, [])

    def test_a_dual_capture_with_marks_on_stays_a_no_op(self) -> None:
        with self.world("B") as world, graph.capture_context():
            self.boundary(world, "kda.tile")
            self.assertEqual(world.tally.calls, [])


class ScanBound(StackCase):
    def scanned(self, width: int, **env: str) -> list[int]:
        buckets: list[int] = []
        with self.world("A", **env) as world:
            case = key_stack.prompt_case(ROWS, width)
            op = key_stack.indexer_op(world, case)
            real = op._scan_pool_bucket
            op._scan_pool_bucket = lambda *args: buckets.append(real(*args)) or buckets[-1]
            self.layer(world, width, op)
        return buckets

    def test_every_tile_scans_the_whole_key_width(self) -> None:
        for width in WIDTHS:
            with self.subTest(width=width):
                self.assertEqual(set(self.scanned(width)), {width * 32})

    def test_warmup_cap_never_applies_in_a_key_capture(self) -> None:
        env = {"HPU_GLM53_CAP_LEGACY_WARMUP": "1", "HPU_GLM53_WARMUP_MAX_KPOOLS": "64"}
        self.assertEqual(set(self.scanned(8192, **env)), {8192 * 32})

    def test_warmup_cap_is_untouched_outside_a_capture(self) -> None:
        buckets: list[int] = []
        env = {"HPU_GLM53_CAP_LEGACY_WARMUP": "1", "HPU_GLM53_WARMUP_MAX_KPOOLS": "64"}
        with self.world("A", **env) as world:
            world.kpool._score_pool_chunk = key_stack.fast_scores
            case = key_stack.prompt_case(ROWS, 1024)
            op = key_stack.indexer_op(world, case)
            real = op._scan_pool_bucket
            op._scan_pool_bucket = lambda *args: buckets.append(real(*args)) or buckets[-1]
            key_stack.indexer_step(world, case, op)
        self.assertIn(64, buckets)  # the scorer's call; the others are the adaptive plan's own probes, each capped by its caller

    def test_private_top_k_buffers_stay_writable_by_a_key_capture(self) -> None:
        with self.world("A") as world:
            physical = self.layer(world, 1024)
            self.assertEqual(physical.shape, (ROWS, key_stack.WIDTH))


class ScorerArithmetic(StackCase):
    """The C791 pattern to 262,144 pools: bucket, chunks and workspace of every key width, against the real functions."""

    WIDTHS_AND_POOLS = ((128, 4096), (640, 20480), (1024, 32768), (2048, 65536), (4096, 131072), (8192, 262144))

    def plan(self, world: SimpleNamespace, width: int) -> tuple[int, int]:
        case = key_stack.prompt_case(128, 1)
        op = key_stack.indexer_op(world, case)
        tokens = width * key_stack.BLOCK
        bucket = op._scan_pool_bucket(width, key_stack.BLOCK, tokens, tokens // key_stack.POOL)
        chunk = op._scoring_chunk(batch=128, index_heads=key_stack.HEADS, index_dim=key_stack.INDEX_DIM,
                                  packed_row_bytes=int(op.k_cache.kv_cache.shape[-1]), active_pool_bucket=bucket,
                                  shared_block_table=True)
        return bucket, chunk

    def test_bucket_is_exactly_the_key_pools(self) -> None:
        with self.world("A") as world:
            for width, pools in self.WIDTHS_AND_POOLS:
                with self.subTest(width=width):
                    self.assertEqual(self.plan(world, width)[0], pools)

    def test_chunks_cover_the_scan_within_the_budget(self) -> None:
        with self.world("A") as world:
            for width, pools in self.WIDTHS_AND_POOLS:
                with self.subTest(width=width):
                    bucket, chunk = self.plan(world, width)
                    self.assertGreater(chunk, 0)
                    self.assertLessEqual(chunk, pools)
                    self.assertLessEqual(math.ceil(bucket / chunk) * chunk - bucket, chunk)
                    self.assertLessEqual(math.ceil(bucket / chunk), 262144 // 512)

    def test_a_key_scan_outruns_the_warmup_cap_and_the_true_host_bound_would_not_apply(self) -> None:
        with self.world("A") as world:
            bucket = self.plan(world, 8192)[0]
            self.assertGreater(bucket, int(key_stack.LAYER_ENV["HPU_GLM53_WARMUP_MAX_KPOOLS"]))


class Witness(StackCase):
    def fired(self, run: object, width: int = 1024) -> dict[str, int]:
        with self.assertLogs(graph.LOG, "WARNING") as logs:
            graph.reset_witness()
            with key_capture():
                run()
            graph.log_witness((0, ROWS, width))
        fired = {}
        for line in logs.output:
            if "GLM53_KEY_WITNESS" in line:
                words = dict(word.split("=") for word in line.split("GLM53_KEY_WITNESS ")[1].split())
                self.assertEqual(words["key"], f"{ROWS}:{width}")
                fired[words["lever"]] = int(words["captures"])
        return fired

    def test_every_kept_lever_gets_a_line_even_at_zero(self) -> None:
        self.assertEqual(self.fired(lambda: None), dict.fromkeys(KEPT_LEVERS, 0))

    def test_only_inside_a_key_capture_counts(self) -> None:
        graph.reset_witness()
        graph.witness_lever("fwht")
        self.assertEqual(graph._WITNESS["fwht"], 0)

    def test_every_kept_lever_has_its_branch_in_the_overlay(self) -> None:
        sources = "".join((OVERLAY_ROOT / relative).read_text(encoding="utf-8") for relative in (*SCANNED, "glm53_mixed_graph.py",
                          "vllm/model_executor/kernels/mhc/torch.py"))
        for lever in KEPT_LEVERS:
            with self.subTest(lever):
                self.assertRegex(sources, rf"""witness_lever\(["']{lever}["']\)""")

    def test_unknown_lever_names_are_refused(self) -> None:
        with key_capture(), self.assertRaisesRegex(ValueError, "lever"):
            graph.witness_lever("not_a_lever")

    def test_the_real_fwht_branch_fires_when_the_native_kernel_applies(self) -> None:
        native = SimpleNamespace(applies=lambda value: True, transform=lambda value: value)
        with self.world("A") as world, mock.patch.dict(sys.modules, {"glm53_mla_fwht": native}):
            fired = self.fired(lambda: world.kpool.fwht128(torch.zeros(1, 128)))
        self.assertEqual(fired["fwht"], 1)

    def test_the_real_unpack_branch_fires_when_the_native_unpack_applies(self) -> None:
        native = SimpleNamespace(ENABLED=True, unpack=lambda packed: packed)
        with self.world("A") as world, mock.patch.dict(sys.modules, {"glm53_mla_unpack": native}), \
                mock.patch.object(world.q4kv, "_USE_BF16_UNPACK", True):
            fired = self.fired(lambda: world.q4kv._unpack_q4_mla_rows_impl(torch.zeros(1, 272, dtype=torch.uint8)))
        self.assertEqual(fired["mla_unpack"], 1)

    def test_the_k512_branch_fires_in_a_key_capture_and_not_in_a_dual_capture(self) -> None:
        native = SimpleNamespace(applies=lambda *args: True,
                                 member_columns=lambda logits, ids: (ids[:, :512], torch.ones(128, 512, dtype=torch.bool), logits))
        with self.world("A") as world:
            world.kpool._tpc_topk = native
            world.kpool._rc6_flags = SimpleNamespace(witness=lambda *args: 0)
            world.kpool._score_pool_chunk = key_stack.fast_scores
            run = self.single_chunk(world)
            fired = self.fired(run)
            graph.reset_witness()
            with graph.capture_context():
                run()
            self.assertEqual(fired["k512_topk"], 1)
            self.assertEqual(graph._WITNESS["k512_topk"], 0)

    @staticmethod
    def single_chunk(world: SimpleNamespace) -> object:
        table = torch.arange(1024, dtype=torch.int32).expand(128, -1)
        q, weights = torch.zeros(128, key_stack.HEADS, key_stack.INDEX_DIM), torch.zeros(128, key_stack.HEADS)
        lengths = torch.full((128,), 1024 * key_stack.BLOCK)
        cache = torch.zeros(1, 136, dtype=torch.uint8)
        return lambda: world.kpool._fused_single_chunk_pooled_topk(
            (q, weights, cache), table, lengths, (key_stack.BLOCK, key_stack.POOL, 512, 1024 * 32), True)


class IllegalLevers(StackCase):
    def test_the_kda_tile_graph_falls_to_the_kernel_in_a_key_capture(self) -> None:
        tile_graph = load_overlay("glm53_kda_tile_graph.py")
        hpu_like = mock.MagicMock()
        hpu_like.device.type = "hpu"
        hpu_like.shape = (1, 512, 16, 128)
        original = mock.Mock(return_value=("output", "state"))
        with mock.patch.dict(os.environ, {"HPU_GLM53_KDA_TILE_GRAPH": "1"}), key_capture():
            self.assertEqual(tile_graph.tile(original, hpu_like, 0, 0, 0, 0), ("output", "state"))
        original.assert_called_once()

    def test_topkfuse_tiles_mark_only_in_arm_b(self) -> None:
        with mock.patch.dict(os.environ, {"HPU_GLM53_MLA_TOPKFUSE": "0"}):
            topkfuse = load_overlay("glm53_mla_topkfuse.py")
        select = mock.Mock(side_effect=lambda tile: tile)
        for arm, marks in (("A", 0), ("B", 4)):
            mark_step = mock.Mock()
            core = SimpleNamespace(mark_step=mark_step)
            with self.subTest(arm=arm), mock.patch.dict(os.environ, {MARKS: ARMS[arm]}), \
                    mock.patch.dict(sys.modules, {"habana_frameworks": SimpleNamespace(), "habana_frameworks.torch": SimpleNamespace(),
                                                  "habana_frameworks.torch.core": core}), \
                    mock.patch.object(torch.ops, "mla_fuse", SimpleNamespace(select=select), create=True), key_capture():
                topkfuse.select(torch.zeros(256, 4))
                self.assertEqual(mark_step.call_count, marks)

    def test_oot_mla_fences_all_go_through_the_lazy_fence(self) -> None:
        source = (OVERLAY_ROOT / "vllm_gaudi/attention/oot_mla.py").read_text(encoding="utf-8")
        self.assertNotIn("mark_step(sync=True)", source)
        self.assertEqual(source.count("lazy_fence(htcore)"), 4)

    def test_lazy_fence_matrix(self) -> None:
        for arm, inside, expected in (("A", False, [True]), ("A", True, []), ("B", False, [True]), ("B", True, [False])):
            calls: list[bool] = []
            core = SimpleNamespace(mark_step=lambda sync=False: calls.append(sync))
            with self.subTest(arm=arm, inside=inside), mock.patch.dict(os.environ, {MARKS: ARMS[arm]}):
                with key_capture() if inside else contextlib.nullcontext():
                    graph.lazy_fence(core)
                self.assertEqual(calls, expected)


class SyncTrueScan(unittest.TestCase):
    def test_every_literal_sync_true_is_gated_or_named(self) -> None:
        for relative in SCANNED:
            tree = ast.parse((OVERLAY_ROOT / relative).read_text(encoding="utf-8"))
            allowed = SYNC_TRUE_ALLOWED.get(relative, {})
            for function in [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                literal = [call for call in ast.walk(function) if isinstance(call, ast.Call) and _is_sync_true(call)]
                if literal:
                    with self.subTest(file=relative, function=function.name):
                        self.assertIn(function.name, GATED_BY_CAPTURE | set(allowed))


GATED_BY_CAPTURE = {"_write_topk", "streaming_pooled_topk_candidates", "fetch_selected", "_all_gather_into",
                    "warm_adaptive_tiles_at_boot"}


def _is_sync_true(call: ast.Call) -> bool:
    name = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "")
    return name == "mark_step" and any(
        keyword.arg == "sync" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
        for keyword in call.keywords)


class BypassesStayOff(unittest.TestCase):
    def test_key_forward_never_reaches_the_adaptive_route(self) -> None:
        """Large-prefill bypass and segmented collectives live in adaptive_forward; a key graph calls the raw forward."""
        with mock.patch.dict(os.environ, {KEYS: "1024:1024", "VLLM_GAUDI_MIXED_GRAPHS": "1", "VLLM_GAUDI_MIXED_STEP": "1"}):
            graph = load_overlay("glm53_mixed_graph.py")
            adapter, raw = SimpleNamespace(), mock.sentinel.original_forward
            graph.install_entry(adapter, raw)
            self.assertIs(adapter._glm53_mixed_graph_forward, raw)

    def test_the_gather_commit_is_not_a_literal_fence_any_more(self) -> None:
        tree = ast.parse((OVERLAY_ROOT / "vllm_gaudi/ops/hpu_glm5next_q4kv.py").read_text(encoding="utf-8"))
        gather = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_all_gather_into")
        self.assertFalse([call for call in ast.walk(gather) if isinstance(call, ast.Call) and _is_sync_true(call)])


if __name__ == "__main__":
    unittest.main()
