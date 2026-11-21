"""Steps 29 + 21b: keyed boot order, no off-graph boot forward, the (decode rows, table width) lattice, its refusals, the width-padding
bitwise check, the boot witness, and the shared gather owner inside a decode capture. All on CPU; what only a card proves is in the plan."""
from __future__ import annotations

import ast
import collections
import contextlib
import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import key_stack  # noqa: E402
from overlay_testlib import OVERLAY_ROOT, load_overlay  # noqa: E402

RUNNER_FILE = OVERLAY_ROOT / "vllm_gaudi/v1/worker/hpu_model_runner.py"
Q4KV_FILE = "vllm_gaudi/ops/hpu_glm5next_q4kv.py"
BLOCK = 128
MIB = 1024 * 1024
PROMPT_KEYS = "1024:8192,1024:1024"
LATTICE_ENV = {
    "VLLM_GAUDI_MIXED_PROMPT_KEYS": PROMPT_KEYS, "VLLM_GAUDI_MIXED_GRAPHS": "1", "VLLM_GAUDI_MIXED_STEP": "1",
    "VLLM_GAUDI_MIXED_DECODE_WIDTHS": "1024,4096,8192", "HPU_GLM53_DECODE_WIDTH_PAD": "check",
    "HPU_GLM53_DECODE_PAD_CHECK_WIDTHS": "2048", "HPU_GLM53_MIXED_DECODE_REPLAY": "1", "VLLM_PREFILL_CONTEXT_CELL_BUDGET": "0",
    "VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB": "8452", "HPU_GLM53_Q4_SHARED_GATHER_OWNER": "1",
}
Trimmed = collections.namedtuple("Trimmed", "is_prompt glm53_index_block_table block_list block_usage block_groups slot_mapping")


def method_source(name: str) -> ast.FunctionDef:
    """The first method or function of this name in the overlay runner, decorators dropped."""
    tree = ast.parse(RUNNER_FILE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            node.decorator_list = []
            return node
    raise AssertionError(f"{name} is not in the overlay runner")


def method_of(name: str, namespace: dict) -> object:
    module = ast.fix_missing_locations(ast.Module(body=[method_source(name)], type_ignores=[]))
    exec(compile(module, str(RUNNER_FILE), "exec"), namespace)
    return namespace[name]


class GraphCase(unittest.TestCase):
    env = LATTICE_ENV

    def setUp(self) -> None:
        environ = mock.patch.dict(os.environ, self.env)
        environ.start()
        self.addCleanup(environ.stop)
        self.graph = load_overlay("glm53_mixed_graph.py")
        modules = mock.patch.dict(sys.modules, {"glm53_mixed_graph": self.graph})
        modules.start()
        self.addCleanup(modules.stop)
        self.graph.reset_decode_lattice()
        self.addCleanup(self.graph.reset_decode_lattice)

    def set_env(self, **values: str) -> None:
        patcher = mock.patch.dict(os.environ, values)
        patcher.start()
        self.addCleanup(patcher.stop)


class LatticeSpec(GraphCase):
    def test_keys_are_rows_by_width_widest_first_and_the_check_widths_at_one_row(self) -> None:
        keys = self.graph.decode_keys()
        self.assertEqual(keys[:5], ((16, 8192), (8, 8192), (4, 8192), (2, 8192), (1, 8192)))
        self.assertEqual(len(keys), 20)
        self.assertEqual(keys[-1], (1, 2048))

    def test_unset_means_no_lattice_and_every_selector_is_identity(self) -> None:
        self.set_env(VLLM_GAUDI_MIXED_DECODE_WIDTHS="", HPU_GLM53_DECODE_WIDTH_PAD="0", HPU_GLM53_DECODE_PAD_CHECK_WIDTHS="")
        self.assertEqual(self.graph.decode_keys(), ())
        self.assertEqual(self.graph.decode_rows_for(3, 7), 7)
        self.assertEqual(self.graph.decode_table_width(None, ["0"], 37, torch.zeros((1, 1))), 37)
        trimmed = Trimmed(False, torch.zeros((1, 2)), torch.zeros(3), torch.zeros(3), torch.zeros(3), None)
        self.assertIs(self.graph.decode_key_metadata(trimmed, 1, True, False), trimmed)

    def test_bad_specs_refuse_and_name_the_flag(self) -> None:
        for spec in ("1000", "abc", "1024,1024", "0"):
            with self.subTest(spec=spec), self.assertRaisesRegex(ValueError, "VLLM_GAUDI_MIXED_DECODE_WIDTHS"):
                self.set_env(VLLM_GAUDI_MIXED_DECODE_WIDTHS=spec)
                self.graph.decode_keys()

    def check_refusal(self, pattern: str, **values: str) -> None:
        self.set_env(**values)
        with self.assertRaisesRegex(ValueError, pattern):
            self.graph.check_prompt_key_flags()

    def test_the_lattice_needs_prompt_keys(self) -> None:
        self.check_refusal("VLLM_GAUDI_MIXED_PROMPT_KEYS", VLLM_GAUDI_MIXED_PROMPT_KEYS="")

    def test_the_lattice_needs_decode_replay_for_a_latched_route(self) -> None:
        self.check_refusal("HPU_GLM53_MIXED_DECODE_REPLAY=1", HPU_GLM53_MIXED_DECODE_REPLAY="0")

    def test_the_lattice_needs_cell_budget_zero(self) -> None:
        self.check_refusal("VLLM_PREFILL_CONTEXT_CELL_BUDGET=0", VLLM_PREFILL_CONTEXT_CELL_BUDGET="4096")

    def test_native_needs_every_power_of_two_up_to_the_top(self) -> None:
        self.check_refusal("every power of two", HPU_GLM53_DECODE_WIDTH_PAD="0", HPU_GLM53_DECODE_PAD_CHECK_WIDTHS="")

    def test_padding_without_a_proof_refuses(self) -> None:
        self.check_refusal("HPU_GLM53_DECODE_PAD_PROOF", HPU_GLM53_DECODE_WIDTH_PAD="1", HPU_GLM53_DECODE_PAD_CHECK_WIDTHS="")

    def test_check_widths_must_pad_up_to_a_listed_width(self) -> None:
        self.check_refusal("HPU_GLM53_DECODE_PAD_CHECK_WIDTHS", HPU_GLM53_DECODE_PAD_CHECK_WIDTHS="1024")

    def test_check_mode_needs_a_check_width(self) -> None:
        self.check_refusal("HPU_GLM53_DECODE_PAD_CHECK_WIDTHS", HPU_GLM53_DECODE_PAD_CHECK_WIDTHS="")

    def test_a_bad_pad_mode_refuses(self) -> None:
        self.check_refusal("HPU_GLM53_DECODE_WIDTH_PAD", HPU_GLM53_DECODE_WIDTH_PAD="yes")

    def test_the_good_env_passes(self) -> None:
        self.graph.check_prompt_key_flags()


class Selection(GraphCase):
    env = {**LATTICE_ENV, "HPU_GLM53_DECODE_WIDTH_PAD": "1", "HPU_GLM53_DECODE_PAD_PROOF": "k1-out/run", "HPU_GLM53_DECODE_PAD_CHECK_WIDTHS": ""}

    def test_every_decode_count_and_table_width_reaches_a_captured_key(self) -> None:
        """A superset of the 13 live r0 keys in section 3: every (requests, pow2 width) the runner can build at KV 31,264 and 1M."""
        keys = set(self.graph.decode_keys())
        for requests in range(1, 17):
            for power in range(14):
                with self.subTest(requests=requests, native=1 << power):
                    rows = self.graph.decode_rows_for(requests, 0)
                    width = self.graph.decode_table_width(None, ["0"] * requests, 1 << power, torch.zeros((rows, 1)))
                    self.assertIn((rows, width), keys)
                    self.assertGreaterEqual(rows, requests)
                    self.assertGreaterEqual(width, 1 << power)

    def test_rows_past_the_lattice_refuse_with_the_fix(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "max-num-seqs"):
            self.graph.decode_rows_for(17, 32)

    def test_a_table_past_the_top_width_refuses(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "VLLM_GAUDI_MIXED_DECODE_WIDTHS"):
            self.graph.decode_table_width(None, ["0"], 16384, torch.zeros((1, 1)))

    def test_native_mode_keeps_the_native_width(self) -> None:
        self.set_env(HPU_GLM53_DECODE_WIDTH_PAD="0")
        self.assertEqual(self.graph.decode_table_width(None, ["0"], 4096, torch.zeros((1, 1))), 4096)
        with self.assertRaisesRegex(RuntimeError, "native"):
            self.graph.decode_table_width(None, ["0"], 2048, torch.zeros((1, 1)))

    def test_a_key_capture_never_pads(self) -> None:
        with self.graph.decode_key_context((1, 2048)):
            self.assertEqual(self.graph.decode_table_width(None, ["0"], 2048, torch.zeros((1, 1))), 2048)

    def test_the_runner_table_is_minus_one_past_the_real_blocks(self) -> None:
        from test_c789_cases import configured, runner_class
        positions = [[BLOCK * 3 + 5]]
        narrow = configured(runner_class(RUNNER_FILE), BLOCK, 4)._make_glm53_sparse_metadata_cpu(["0"], [1], positions, 1, 1, is_decode=True)
        self.assertEqual(narrow[2].shape, (1, 1024))
        self.assertTrue(torch.equal(narrow[2][0, 4:], torch.full((1020,), -1, dtype=torch.int32)))
        self.assertTrue((narrow[2][0, :4] >= 0).all())


class Metadata(GraphCase):
    def trimmed(self, width: int = 8192) -> Trimmed:
        return Trimmed(False, torch.zeros((16, width)), torch.zeros(50), torch.zeros(50), torch.zeros(50), torch.zeros((16, 1)))

    def test_a_captured_key_drops_the_aggregate_dense_fields_so_one_graph_serves_any_lane_mix(self) -> None:
        self.graph.mark_decode_key_captured((16, 8192))
        result = self.graph.decode_key_metadata(self.trimmed(), 16, True, False)
        self.assertEqual((result.block_list, result.block_usage, result.block_groups), (None, None, None))
        self.assertIsNotNone(result.glm53_index_block_table)

    def test_a_prompt_is_untouched(self) -> None:
        prompt = self.trimmed()._replace(is_prompt=True)
        self.assertIs(self.graph.decode_key_metadata(prompt, 1, True, False), prompt)

    def test_serving_a_key_that_was_not_captured_refuses(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no captured decode key"):
            self.graph.decode_key_metadata(self.trimmed(), 16, True, False)

    def test_an_eager_decode_of_a_captured_key_refuses(self) -> None:
        self.graph.mark_decode_key_captured((16, 8192))
        with self.assertRaisesRegex(RuntimeError, "VLLM_HPU_RESTORED_DECODE_GRAPHS"):
            self.graph.decode_key_metadata(self.trimmed(), 16, False, False)

    def test_the_warmed_config_counts_the_table_width_so_a_lane_mix_never_reads_as_unwarmed(self) -> None:
        self.assertEqual(self.graph.decode_config_blocks(self.trimmed(), 131_072), 8192)
        self.assertEqual(self.graph.decode_config_blocks(self.trimmed()._replace(is_prompt=True), 77), 77)
        self.set_env(VLLM_GAUDI_MIXED_DECODE_WIDTHS="")
        self.assertEqual(self.graph.decode_config_blocks(self.trimmed(), 131_072), 131_072)

    def test_process_metadata_skips_the_dense_mapping_when_the_fields_are_gone(self) -> None:
        process = method_of("process_metadata", {})
        owner = mock.MagicMock(interleaved_sliding_window=False)
        trimmed = self.trimmed()._replace(block_usage=None)
        self.assertIs(process(owner, trimmed, 16, 1, "cpu", torch.bfloat16), trimmed)
        owner._set_block_mapping.assert_not_called()
        kept = self.trimmed()
        process(owner, kept, 16, 1, "cpu", torch.bfloat16)
        owner._set_block_mapping.assert_called_once()

    def test_the_warmup_scan_cap_stays_off_inside_a_decode_key_capture(self) -> None:
        source = method_of("_glm53_warmup_only_kpool_cap", {"os": os, "contextlib": contextlib})
        source = contextlib.contextmanager(source.__wrapped__) if hasattr(source, "__wrapped__") else contextlib.contextmanager(source)
        with self.graph.decode_key_context((1, 8192)), source(True):
            self.assertNotEqual(os.environ.get("HPU_GLM53_CAP_LEGACY_WARMUP"), "1")
        with source(True):
            self.assertEqual(os.environ.get("HPU_GLM53_CAP_LEGACY_WARMUP"), "1")


class DummyLengths(GraphCase):
    def scenario(self, decode_cfg: tuple, **kwargs: object) -> list:
        added = []
        runner = mock.MagicMock(use_contiguous_pa=False, attn_block_size=BLOCK, max_model_len=1_048_576)
        runner._add_dummy_request.side_effect = lambda _r, _s, **fields: added.append(fields["total_tokens"])
        runner._generate_seq_lengths.side_effect = lambda d, width, block: [width // d * block - 1] * d
        method_of("_prepare_dummy_scenario", {})(runner, None, decode_cfg, **kwargs)
        return added

    def test_one_long_row_and_short_rows(self) -> None:
        self.assertEqual(self.graph.decode_dummy_lengths(4, 8192, BLOCK, 1_048_576), [1_048_575, 127, 127, 127])
        self.assertEqual(self.graph.decode_dummy_lengths(1, 1024, BLOCK, 1_048_576), [131_071])

    def test_the_dummy_reaches_the_runner_as_given(self) -> None:
        self.assertEqual(self.scenario((2, 1, 8192), decode_seq_lengths=[1_048_575, 127]), [1_048_575, 127])

    def test_without_lengths_the_runner_keeps_its_even_split(self) -> None:
        self.assertEqual(self.scenario((2, 1, 8)), [511, 511])


class FakeRunner:
    attn_block_size = BLOCK
    block_size = BLOCK
    max_model_len = 1_048_576

    def __init__(self, graph: object, *, charged_mib: int = 2560, counts: dict | None = None) -> None:
        """charged_mib: the c1b prompt keys' GLM53_KEY_SEAL in_use_delta_MiB in diag-key128/runs/r1 base."""
        self.graph = graph
        self.input_batch = SimpleNamespace(max_num_reqs=16)
        cache = SimpleNamespace(sealed_bytes=charged_mib * MIB, counts=collections.Counter(counts or {}),
                                entries={((0, 1024, 8192), ()): 0, ((0, 1024, 1024), ()): 0})
        self.model = SimpleNamespace(_glm53_mixed_graph_cache=cache)
        self.calls: list[tuple] = []

    def _external_latch_applies(self, metadata: object) -> bool:
        return False

    def _prepare_dummy_scenario(self, prompt_cfg: object, decode_cfg: tuple, decode_seq_lengths: list | None = None) -> None:
        rows, _, width = decode_cfg
        self.calls.append((prompt_cfg, decode_cfg, decode_seq_lengths))
        trimmed = Trimmed(False, torch.zeros((rows, width)), None, None, None, None)
        self.graph.decode_key_metadata(trimmed, rows, True, True)


class LatticeBoot(GraphCase):
    def boot(self, runner: FakeRunner, deltas: int = 0, ballast: mock.Mock | None = None,
             graph_count: object = None, headroom: object = None, peak: object = None, reset_peak: object = None) -> list[str]:
        """Each decode key's dummy raises the allocator's in-use bytes by `deltas` MiB and the graph count by one; its peak is that too."""
        in_use = iter(0 if i % 2 == 0 else deltas * MIB for i in range(400))
        with mock.patch.object(self.graph, "_in_use_bytes", side_effect=lambda: next(in_use)), \
                mock.patch.object(self.graph, "_free_hbm", return_value=0), \
                mock.patch.object(self.graph, "_allocator_headroom", side_effect=headroom or (lambda: 100_000 * MIB)), \
                mock.patch.object(self.graph, "_reset_peak_in_use", side_effect=reset_peak or (lambda: None)), \
                mock.patch.object(self.graph, "_peak_in_use_bytes", side_effect=peak or (lambda: deltas * MIB)), \
                mock.patch.object(self.graph, "_graph_count", side_effect=graph_count or (lambda: len(runner.calls))), \
                mock.patch.object(self.graph, "_hold_ballast", ballast or mock.Mock(return_value=4096 * MIB)), \
                self.assertLogs(self.graph.LOG, "WARNING") as logs:
            self.graph.seal_boot_memory(runner, self.graph.warmup_decode_lattice(runner))
        return logs.output

    def test_every_key_once_widest_first_with_the_long_row(self) -> None:
        runner = FakeRunner(self.graph)
        self.boot(runner)
        self.assertEqual([call[1][:1] + call[1][2:] for call in runner.calls], list(self.graph.decode_keys()))
        self.assertEqual({call[0] for call in runner.calls}, {None})
        self.assertEqual(runner.calls[0][2][0], 1_048_575)

    def test_the_witness_lists_every_bucket_and_zero_fallbacks(self) -> None:
        output = self.boot(FakeRunner(self.graph))
        witness = [line for line in output if "GLM53_BOOT_WITNESS" in line]
        self.assertEqual(len(witness), 1)
        self.assertIn("prompt_keys=1024:8192,1024:1024", witness[0])
        for rows, width in self.graph.decode_keys():
            self.assertIn(f"{rows}x{width}", witness[0])
        self.assertIn("native_fallbacks=0", witness[0])
        self.assertIn("fallback_not_warmed=0", witness[0])
        self.assertIn("decode_pad=check", witness[0])

    def test_keys_without_a_lattice_still_get_a_witness(self) -> None:
        self.set_env(VLLM_GAUDI_MIXED_DECODE_WIDTHS="", HPU_GLM53_DECODE_WIDTH_PAD="0", HPU_GLM53_DECODE_PAD_CHECK_WIDTHS="")
        runner = FakeRunner(self.graph)
        output = self.boot(runner)
        self.assertEqual(runner.calls, [])
        self.assertTrue(any("decode_keys=none" in line for line in output))

    def test_no_keys_and_no_lattice_do_nothing(self) -> None:
        self.set_env(VLLM_GAUDI_MIXED_PROMPT_KEYS="", VLLM_GAUDI_MIXED_DECODE_WIDTHS="", HPU_GLM53_DECODE_WIDTH_PAD="0",
                     HPU_GLM53_DECODE_PAD_CHECK_WIDTHS="")
        runner = FakeRunner(self.graph)
        with mock.patch.object(self.graph, "LOG") as log:
            self.graph.seal_boot_memory(runner, self.graph.warmup_decode_lattice(runner))
        log.warning.assert_not_called()
        self.assertEqual(runner.calls, [])

    def test_a_fallback_before_serving_refuses_the_boot(self) -> None:
        for name in ("native_fallbacks", "fallback_not_warmed", "lazy_fallbacks"):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, name):
                self.boot(FakeRunner(self.graph, counts={name: 1}))

    def test_an_estimate_over_budget_refuses_before_any_capture(self) -> None:
        runner = FakeRunner(self.graph, charged_mib=8000)
        with self.assertRaisesRegex(RuntimeError, "VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB"):
            self.graph.warmup_decode_lattice(runner)
        self.assertEqual(runner.calls, [])

    def test_a_measured_key_over_budget_refuses_and_names_it(self) -> None:
        runner = FakeRunner(self.graph, charged_mib=4500)  # the lattice estimates fit 8452; a first key that took 4000 MiB does not
        with self.assertRaisesRegex(RuntimeError, "16x8192"):
            self.boot(runner, deltas=4000)
        self.assertEqual(len(runner.calls), 1)

    def test_a_failed_capture_refuses_and_arms_the_exit(self) -> None:
        runner = FakeRunner(self.graph)
        runner._prepare_dummy_scenario = mock.Mock(side_effect=RuntimeError("PT_DEVMEM"))
        with mock.patch.object(self.graph, "arm_failed_capture_exit") as arm, self.assertRaisesRegex(RuntimeError, "PT_DEVMEM"):
            self.boot(runner)
        arm.assert_called_once()

    def test_a_dummy_that_built_another_key_refuses(self) -> None:
        runner = FakeRunner(self.graph)
        real = runner._prepare_dummy_scenario
        runner._prepare_dummy_scenario = lambda p, d, decode_seq_lengths=None: real(p, (d[0], 1, d[2] // 2), decode_seq_lengths)
        with self.assertRaisesRegex(RuntimeError, "built"):
            self.boot(runner)

    def test_rows_past_the_request_capacity_refuse(self) -> None:
        runner = FakeRunner(self.graph)
        runner.input_batch.max_num_reqs = 8
        with self.assertRaisesRegex(RuntimeError, "max-num-seqs"):
            self.boot(runner)

    def test_a_top_width_short_of_max_model_len_refuses(self) -> None:
        self.set_env(VLLM_GAUDI_MIXED_DECODE_WIDTHS="1024,4096")
        with self.assertRaisesRegex(RuntimeError, "max_model_len"):
            self.boot(FakeRunner(self.graph))

    def test_a_restored_decode_that_would_run_eager_refuses(self) -> None:
        runner = FakeRunner(self.graph)
        runner._external_latch_applies = lambda metadata: True
        with self.assertRaisesRegex(RuntimeError, "VLLM_HPU_RESTORED_DECODE_GRAPHS"):
            self.boot(runner)

    def test_the_refusal_does_not_ask_for_lower_kv(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            self.graph.warmup_decode_lattice(FakeRunner(self.graph, charged_mib=8000))
        self.assertNotIn("num-gpu-blocks-override", str(caught.exception))


class PromptKeyOrder(GraphCase):
    def test_keys_capture_widest_first_whatever_the_written_order(self) -> None:
        self.set_env(VLLM_GAUDI_MIXED_PROMPT_KEYS="1024:1024,1024:8192", VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB="65536")
        runner = SimpleNamespace(attn_block_size=BLOCK, block_size=BLOCK, max_model_len=1_048_576)
        self.assertEqual(self.graph.validate_prompt_keys(runner), ((0, 1024, 8192), (0, 1024, 1024)))


class WarmupOrder(unittest.TestCase):
    """The real warmup_model, every collaborator a recorder: the boot order and what runs off graph."""

    def run_warmup(self, env: dict[str, str]) -> list:
        events: list = []
        compile_only = [0]

        @contextlib.contextmanager
        def env_setting(_name: str, _value: bool) -> object:
            compile_only[0] += 1
            try:
                yield
            finally:
                compile_only[0] -= 1

        def warmup_graphs(buckets: object, is_prompt: object, _kv: object) -> tuple:
            kind = {True: "prompt", False: "decode", "mixed": "mixed"}[is_prompt]
            events.append((kind, list(buckets), compile_only[0] > 0))
            return 0, 0, True

        runner = mock.MagicMock(is_pooling_model=False, skip_warmup=False, supports_mm_inputs=False, _has_heterogeneous_layers=True,
                                num_mamba_like_layers=0, mem_margin=1)
        runner.graphed_buckets = set()
        self.warmed_runner = runner
        runner.model_config.enforce_eager = False
        runner.bucketing_manager.prompt_buckets = [(1, 128, 0)]
        runner.bucketing_manager.decode_buckets = [(1, 1, 128)]
        runner.input_batch.max_num_reqs = 16
        runner.warmup_graphs.side_effect = warmup_graphs
        runner._read_profiling_cfg.return_value = (None, None)
        lattice = mock.Mock(side_effect=lambda _r: events.append(("lattice", [], compile_only[0] > 0)))
        seal = mock.Mock(side_effect=lambda _r, _charged: events.append(("seal", [], compile_only[0] > 0)))
        htorch = mock.MagicMock()
        htorch.utils.internal.is_lazy.return_value = True
        namespace = {"os": os, "time": time, "contextlib": contextlib, "functools": __import__("functools"), "htorch": htorch,
                     "bc": SimpleNamespace(env_setting=env_setting), "logger": mock.MagicMock(), "gaudi_envs": mock.MagicMock(),
                     "HabanaMemoryProfiler": mock.MagicMock(), "format_bytes": str, "OnlineDefragmenter": mock.MagicMock(),
                     "torch": mock.MagicMock(), "InputBatch": mock.MagicMock(), "build_logitsprocs": mock.MagicMock()}
        namespace["HabanaMemoryProfiler"].current_device_memory_usage.return_value = 0
        graph = SimpleNamespace(enabled=lambda: env.get("VLLM_GAUDI_MIXED_GRAPHS") == "1",
                                prompt_keys=lambda: tuple(filter(None, env.get("VLLM_GAUDI_MIXED_PROMPT_KEYS", "").split(","))),
                                warmup_decode_lattice=lattice, seal_boot_memory=seal)
        with mock.patch.dict(os.environ, env), mock.patch.dict(sys.modules, {"glm53_mixed_graph": graph}):
            method_of("warmup_model", namespace)(runner)
        return events

    def test_keys_first_then_the_lattice_and_nothing_off_graph(self) -> None:
        events = self.run_warmup({"VLLM_GAUDI_MIXED_GRAPHS": "1", "VLLM_GAUDI_MIXED_PROMPT_KEYS": PROMPT_KEYS,
                                  "VLLM_GAUDI_TARGETED_PROMPT_WARMUP_BUCKETS": "1,128,0", "VLLM_GAUDI_SKIP_DECODE_GRAPH_WARMUP": "0"})
        ran = [event for event in events if event[1] or event[0] in ("mixed", "lattice", "decode", "seal")]
        self.assertEqual([event[0] for event in ran], ["mixed", "lattice", "seal"])
        self.assertTrue(ran[1][2], "decode lattice keeps the base decode warmup's compile-only mode")
        self.assertFalse(ran[0][2], "key captures run for real, outside compile-only mode")
        self.assertFalse(ran[2][2], "the seal's ballast must really allocate, so it runs outside compile-only mode")

    def test_a_keyed_boot_is_marked_warmed_so_the_worker_does_not_run_warmup_again(self) -> None:
        """HPUWorker.compile_or_warm_up_model runs warmup whenever graphed_buckets is empty, and a keyed boot graphs nothing else
        (c1p-20261107T074505Z: the second run refused every decode key as captured no graph)."""
        self.run_warmup({"VLLM_GAUDI_MIXED_GRAPHS": "1", "VLLM_GAUDI_MIXED_PROMPT_KEYS": PROMPT_KEYS})
        self.assertTrue(self.warmed_runner.graphed_buckets)

    def test_flags_off_add_no_mark(self) -> None:
        self.run_warmup({"VLLM_GAUDI_MIXED_GRAPHS": "0", "VLLM_GAUDI_SKIP_DECODE_GRAPH_WARMUP": "0"})
        self.assertNotIn("mixed_keys", {bucket[0] for bucket in self.warmed_runner.graphed_buckets})

    def test_flags_off_keep_the_base_order(self) -> None:
        events = self.run_warmup({"VLLM_GAUDI_MIXED_GRAPHS": "0", "VLLM_GAUDI_SKIP_DECODE_GRAPH_WARMUP": "0"})
        self.assertEqual([event[0] for event in events[:2]], ["prompt", "decode"])
        self.assertEqual(events[0][1], [(1, 128, 0)])
        self.assertEqual(events[1][1], [(1, 1, 128)])
        self.assertNotIn("mixed", [event[0] for event in events])
        self.assertNotIn("lattice", [event[0] for event in events])


class PadCheck(GraphCase):
    def runner(self) -> SimpleNamespace:
        return SimpleNamespace(requests={"0": SimpleNamespace(prompt_token_ids=[1, 2, 3])})

    def step(self, position: int = 500) -> int:
        return self.graph.decode_table_width(self.runner(), ["0"], 2048, torch.tensor([[position]]))

    def test_first_run_native_repeat_padded_then_one_line_with_torch_equal(self) -> None:
        hidden = torch.arange(8.0).reshape(1, 1, 8)
        self.assertEqual(self.step(), 2048)
        self.graph.observe_decode_hidden(hidden)
        self.assertEqual(self.step(), 4096)
        with self.assertLogs(self.graph.LOG, "WARNING") as logs:
            self.graph.observe_decode_hidden(hidden.clone())
        self.assertIn("GLM53_DECODE_MERGE narrow=2048 wide=4096 rows=1 position=500 torch_equal=1", logs.output[0])
        self.assertEqual(self.step(), 2048)

    def test_a_difference_says_so_and_refuses(self) -> None:
        self.step()
        self.graph.observe_decode_hidden(torch.zeros((1, 1, 4)))
        self.step()
        with self.assertLogs(self.graph.LOG, "WARNING") as logs, self.assertRaisesRegex(RuntimeError, "HPU_GLM53_DECODE_WIDTH_PAD"):
            self.graph.observe_decode_hidden(torch.ones((1, 1, 4)))
        self.assertIn("torch_equal=0", logs.output[0])

    def test_only_a_real_row_of_the_first_padded_batch_is_compared(self) -> None:
        self.step()
        self.graph.observe_decode_hidden(torch.zeros((16, 1, 4)))
        stored = next(iter(self.graph._DECODE_CHECK.pending.values()))
        self.assertEqual(stored.shape[0], 1)

    def test_a_width_outside_the_check_runs_native_not_padded(self) -> None:
        self.set_env(HPU_GLM53_DECODE_PAD_CHECK_WIDTHS="512")
        self.assertEqual(self.graph.decode_table_width(self.runner(), ["0"], 2048, torch.tensor([[1]])), 2048)
        self.assertEqual(self.graph._DECODE_CHECK.pending, {})

    def test_ranks_that_disagree_all_stop(self) -> None:
        with mock.patch.object(self.graph, "rank_reports", return_value=[{"digest": "a", "pad": False}, {"digest": "a", "pad": True}]), \
                self.assertRaisesRegex(RuntimeError, "HPU_GLM53_DECODE_WIDTH_PAD"):
            self.step()

    def test_pending_is_bounded(self) -> None:
        for position in range(self.graph.DECODE_CHECK_MAX_PENDING + 5):
            self.step(position)
            self.graph.observe_decode_hidden(torch.zeros((1, 1, 2)))
        self.assertLessEqual(len(self.graph._DECODE_CHECK.pending), self.graph.DECODE_CHECK_MAX_PENDING)

    def test_a_key_capture_is_never_compared(self) -> None:
        with self.graph.decode_key_context((1, 2048)):
            self.assertEqual(self.step(), 2048)
        self.graph.observe_decode_hidden(torch.zeros((1, 1, 2)))
        self.assertEqual(self.graph._DECODE_CHECK.pending, {})


class StaticScanWidth(unittest.TestCase):
    """A decode graph replays one shape for any table, so the rank-local width cannot come from the dummy's `.item()`."""

    def scan_widths(self, rank: int, *, key: bool) -> list[int]:
        seen: list[int] = []
        with key_stack.layer_stack(OVERLAY_ROOT) as world:
            def score(q: torch.Tensor, weights: torch.Tensor, rows: torch.Tensor, physical: torch.Tensor, valid: torch.Tensor,
                      *, query_shape: tuple, shared_block_table: bool) -> torch.Tensor:
                seen.append(physical.shape[-1])
                return torch.ones(query_shape[0], physical.shape[-1]).masked_fill(~valid, -torch.inf)

            world.kpool._score_pool_chunk = score
            graph = sys.modules["glm53_mixed_graph"]
            width = 8
            table = torch.zeros((1, width), dtype=torch.int64)  # every block is block 0: rank 0 owns them all, the others none
            lengths = torch.tensor([width * BLOCK], dtype=torch.long)
            scope = graph.decode_key_context((1, width)) if key else contextlib.nullcontext()
            with scope:
                world.kpool.streaming_pooled_topk_candidates(
                    torch.ones((1, 8, 128)), torch.ones((1, 8)), torch.zeros((64, 16), dtype=torch.uint8), table, lengths,
                    source_block_size=BLOCK, pool_size=4, topk_tokens=16, context_chunk_pools=4096,
                    max_complete_pools_bucket=width * BLOCK // 4, tp_rank=rank, tp_size=4)
        return seen

    def test_a_rank_owning_no_dummy_block_still_captures_the_whole_scan(self) -> None:
        self.assertEqual(sum(self.scan_widths(1, key=True)), 8 * BLOCK // 4)

    def test_every_rank_captures_the_same_scan(self) -> None:
        self.assertEqual(self.scan_widths(0, key=True), self.scan_widths(3, key=True))

    def test_outside_a_key_the_width_stays_the_observed_one(self) -> None:
        self.assertEqual(sum(self.scan_widths(1, key=False)), 0)


class HpuRows(torch.Tensor):
    """A CPU tensor that reports device hpu, so the real gather paths take their lazy-mode branches."""

    @property
    def device(self) -> SimpleNamespace:  # type: ignore[override]
        return SimpleNamespace(type="hpu")


class OwnerInDecodeCapture(unittest.TestCase):
    """Step 21b. Rank 0 of TP4; the collectives are recorded; the owner only changes which tensor the gather writes."""

    FLAG = "HPU_GLM53_Q4_SHARED_GATHER_OWNER"
    ENV = {"HPU_GLM53_TP_SHARDED_Q4KV": "1", "HPU_GLM53_Q4_TOKEN_CYCLIC": "1"}

    def setUp(self) -> None:
        self.q4kv = load_overlay(Q4KV_FILE)
        self.q4kv._SHARED_OWNERS.clear()
        self.addCleanup(self.q4kv._SHARED_OWNERS.clear)

    def capturing(self, value: bool = True) -> object:
        return mock.patch("habana_frameworks.torch.hpu.graphs.is_current_stream_capturing", return_value=value)

    def owner_for(self, flag: str, *, capturing: bool, rows: torch.Tensor | None = None) -> torch.Tensor | None:
        local = rows if rows is not None else torch.zeros((2, 3, 4), dtype=torch.uint8).as_subclass(HpuRows)
        real_empty = torch.empty
        with mock.patch.dict(os.environ, {self.FLAG: flag}), self.capturing(capturing), \
                mock.patch("torch.empty", side_effect=lambda shape, dtype=None, device=None: real_empty(shape, dtype=dtype)):
            return self.q4kv.shared_gather_owner(local, 4)

    def test_a_stream_capture_uses_the_owner(self) -> None:
        self.assertIsNotNone(self.owner_for("1", capturing=True))

    def test_a_decode_capture_of_one_stripe_shape_reuses_the_prompt_owner(self) -> None:
        first = self.owner_for("1", capturing=True)
        self.assertIsNotNone(first)
        self.assertIs(first, self.owner_for("1", capturing=True))

    def test_a_lazy_step_outside_any_capture_keeps_a_fresh_owner(self) -> None:
        self.assertIsNone(self.owner_for("1", capturing=False))

    def test_flag_off_never_uses_the_owner(self) -> None:
        self.assertIsNone(self.owner_for("0", capturing=True))

    def test_a_cpu_rank_never_asks_the_stream(self) -> None:
        with mock.patch.dict(os.environ, {self.FLAG: "1"}), self.capturing(True) as asked:
            self.assertIsNone(self.q4kv.shared_gather_owner(torch.zeros((2, 3, 4), dtype=torch.uint8), 4))
        asked.assert_not_called()

    def fetch_events(self, flag: str) -> tuple[list[tuple], torch.Tensor]:
        """Every lazy call of the real fetch_selected in a decode capture, through the real communicator when the owner is off."""
        events: list[tuple] = []
        slots = torch.arange(3 * 8, dtype=torch.int64).reshape(3, 8) + 16
        cache = torch.arange(256 * 272, dtype=torch.int32).reshape(256, 272).remainder(251).to(torch.uint8)
        real_empty = torch.empty
        with mock.patch.dict(os.environ, {**self.ENV, self.FLAG: flag}):
            q4 = self.q4kv.HPUQ4MLAKVCache()
        q4._tp_coordinates = lambda: (0, 4)
        q4.dequant_output = lambda packed, dtype=torch.bfloat16: packed

        def stripes() -> torch.Tensor:
            return torch.cat([q4._owned_stripes(cache, slots, rank, 4) for rank in range(4)], dim=0)

        def gather_into(output: torch.Tensor, local: torch.Tensor, group: object = None) -> None:
            events.append(("gather",))
            output.copy_(stripes())

        from vllm_gaudi.distributed.device_communicators import hpu_communicator as communicator
        tp = SimpleNamespace(world_size=4, device_group=object())
        tp.all_gather = lambda rows, dim=-1: communicator.HpuCommunicator.all_gather(tp, rows, dim)
        mark = lambda *args, **kwargs: events.append(("mark_step", args, kwargs))  # noqa: E731
        finish = lambda: events.append(("finish",))  # noqa: E731
        stripe = q4._owned_stripes
        q4._owned_stripes = lambda *args: stripe(*args).as_subclass(HpuRows)
        with mock.patch.dict(os.environ, {**self.ENV, self.FLAG: flag}), self.capturing(True), \
                mock.patch("vllm.distributed.get_tp_group", return_value=tp), \
                mock.patch("habana_frameworks.torch.core.mark_step", side_effect=mark), \
                mock.patch.object(communicator, "_finish_segmented_lazy_collective", side_effect=finish), \
                mock.patch("torch.distributed.all_gather_into_tensor", side_effect=gather_into), \
                mock.patch("torch.empty", side_effect=lambda shape, dtype=None, device=None: real_empty(shape, dtype=dtype)):
            rows = q4.fetch_selected(cache, slots)
        return events, rows

    def test_the_mark_sequence_is_identical_on_and_off(self) -> None:
        off, _ = self.fetch_events("0")
        self.assertEqual(self.q4kv._SHARED_OWNERS, {})
        on, _ = self.fetch_events("1")
        self.assertEqual(len(self.q4kv._SHARED_OWNERS), 1, "the on run must really gather through the owner")
        self.assertEqual(on, off)
        self.assertEqual([event[0] for event in on], ["mark_step", "gather", "finish"])

    def test_the_grouped_path_keeps_its_commit(self) -> None:
        q4kv = self.q4kv
        source = Path(q4kv.__file__).read_text(encoding="utf-8")
        self.assertIn("return _all_gather_into(owner, local_rows.contiguous(), commit=True)", source)
        self.assertIn("return _all_gather_into(output, local_rows, commit=True)", source)

    def test_the_bytes_are_equal_on_and_off(self) -> None:
        on, off = self.fetch_events("1")[1], self.fetch_events("0")[1]
        self.assertTrue(torch.equal(on.as_subclass(torch.Tensor), off.as_subclass(torch.Tensor)))


class GuardRefusal(unittest.TestCase):
    def test_a_latched_decode_with_a_lattice_refuses_instead_of_running_eager(self) -> None:
        guard = load_overlay("glm53_graph_state.py")
        forward = guard.RoutedForward.__new__(guard.RoutedForward)
        forward.route = SimpleNamespace(latched=lambda: True, should_bypass=lambda *_: False, bypass_min_tokens=512, recapture_min_width=2,
                                        adaptive=mock.Mock())
        forward.mode = SimpleNamespace(routes=True, private=False, census=False)
        forward.state = SimpleNamespace(private=None)
        forward.hbm = None
        forward.mixed_decode_graph_forwards = 0
        kwargs = {"input_ids": torch.zeros((1, 1), dtype=torch.long), "attn_metadata": SimpleNamespace(is_prompt=False),
                  guard.EXTERNAL_KWARG: True}
        with mock.patch.dict(os.environ, {"VLLM_GAUDI_MIXED_DECODE_WIDTHS": "1024", "HPU_GLM53_MIXED_DECODE_REPLAY": "0"}), \
                self.assertRaisesRegex(guard.GraphStateError, "VLLM_GAUDI_MIXED_DECODE_WIDTHS"):
            forward(**kwargs)
        forward.route.adaptive.assert_not_called()


if __name__ == "__main__":
    unittest.main()
