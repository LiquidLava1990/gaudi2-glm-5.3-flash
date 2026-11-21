"""Step 5 Form B: VLLM_GAUDI_MIXED_PROMPT_KEYS. Parsing, the selector at every section 4 client target, key capture and replay of the
prompt region alone with decode left to the runner's own graph, the census fields K1 scores, and the width merge line."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from overlay_testlib import OVERLAY_ROOT, load_overlay  # noqa: E402

KEYS = "VLLM_GAUDI_MIXED_PROMPT_KEYS"
BLOCK = 128
# Section 4: (tokens, runner table width of that context, W with {1024,4096,8192}, W with {1024,8192})
CLIENT_TARGETS = ((98_304, 1024, 1024, 1024), (131_072, 1024, 1024, 1024), (150_000, 2048, 4096, 8192),
                  (214_000, 2048, 4096, 8192), (250_000, 2048, 4096, 8192), (393_216, 4096, 4096, 8192),
                  (400_000, 4096, 4096, 8192), (524_288, 4096, 4096, 8192), (786_432, 8192, 8192, 8192),
                  (1_048_576, 8192, 8192, 8192))


def pow2_width(tokens: int) -> int:
    """hpu_model_runner._glm53_sparse_block_bucket of a chunk ending at `tokens`."""
    blocks, width = (tokens + BLOCK - 1) // BLOCK, 1
    while width < blocks:
        width *= 2
    return width


@dataclass(frozen=True)
class FakeMetadata:
    glm53_index_block_table: torch.Tensor
    glm53_token_mask: torch.Tensor
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    num_actual_tokens: int
    num_decodes: int


class RowForward:
    """Hidden depends on the real rows only, so padding and width must not move any compact row."""

    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []
        self.tables: list[torch.Tensor] = []

    def __call__(self, *, input_ids: torch.Tensor, positions: torch.Tensor, attn_metadata: object) -> torch.Tensor:
        self.shapes.append(tuple(input_ids.shape))
        self.tables.append(attn_metadata.regions[0].metadata.glm53_index_block_table)
        ids = input_ids.reshape(-1).float()
        return torch.stack([ids, positions.reshape(-1).float()], dim=-1).unsqueeze(0)


LANE_DIR = Path(os.environ.get("LANE_DIR", "/nonexistent"))


class Backend:
    def __init__(self) -> None:
        self.prepared: list[tuple] = []
        self.replays = 0

    def prepare_capture(self, bucket: tuple, estimate: int, error: str) -> None:
        if error:
            raise RuntimeError(error)
        self.prepared.append((bucket, estimate))

    def capture(self, call: object) -> tuple:
        output = call()
        return (call, output), output

    def replay(self, handle: tuple) -> None:
        call, output = handle
        self.replays += 1
        output.copy_(call())


class KeyCase(unittest.TestCase):
    keys = "1024:8192,1024:1024"

    def setUp(self) -> None:
        environ = mock.patch.dict(os.environ, {KEYS: self.keys, "VLLM_GAUDI_MIXED_STEP": "1", "VLLM_GAUDI_MIXED_GRAPHS": "1",
                                               "VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB": "65536"})
        environ.start()
        self.addCleanup(environ.stop)
        self.graph = load_overlay("glm53_mixed_graph.py")
        modules = mock.patch.dict(sys.modules, {"glm53_mixed_graph": self.graph})
        modules.start()
        self.addCleanup(modules.stop)
        self.step = load_overlay("glm53_mixed_step.py")
        sys.modules["glm53_mixed_step"] = self.step

    def prompt_batch(self, rows: int, start: int, width: int, seed: int = 1, native_width: int | None = None) -> object:
        padded = native_width or rows
        table = torch.full((1, width), -1, dtype=torch.int32)
        used = max(1, (start + rows + BLOCK - 1) // BLOCK)
        table[0, :used] = torch.arange(used, dtype=torch.int32) + seed
        meta = FakeMetadata(table, torch.ones(padded), torch.arange(padded), torch.tensor([0, padded]), padded, 0)
        region = self.step.Region((0,), (rows,), (start,), padded, 1, tuple(range(rows)), meta, False)
        ids = torch.arange(padded).reshape(1, padded) + 100
        return self.step.PreparedBatch(region, ids, ids + start, (rows - 1,), id(meta))

    def boot_capture(self, cache: object, batch: object) -> torch.Tensor:
        with self.graph.boot_capture_context():
            return cache.run_key(batch)


class ParseKeys(KeyCase):
    def test_keys_parse_in_written_order(self) -> None:
        self.assertEqual(self.graph.prompt_keys(), ((1024, 8192), (1024, 1024)))

    def test_unset_is_no_keys(self) -> None:
        with mock.patch.dict(os.environ, {KEYS: ""}):
            self.assertEqual(self.graph.prompt_keys(), ())

    def test_bad_specs_refuse_and_name_the_flag(self) -> None:
        for spec in ("1024", "a:b", "1024:0", "0:1024", "1000:1024", "1024:1000", "1024:8192,1024:8192"):
            with self.subTest(spec), mock.patch.dict(os.environ, {KEYS: spec}), self.assertRaisesRegex(ValueError, KEYS):
                self.graph.prompt_keys()

    def test_keys_without_mixed_graphs_refuse_at_install(self) -> None:
        with mock.patch.dict(os.environ, {"VLLM_GAUDI_MIXED_GRAPHS": "0"}), self.assertRaisesRegex(ValueError, "MIXED_GRAPHS"):
            self.graph.install_entry(SimpleNamespace(), lambda: None)

    def test_keys_refuse_the_native_pin_and_tail_fuse(self) -> None:
        pin = {"VLLM_GAUDI_MIXED_SPLIT_FORWARD": "1", "VLLM_GAUDI_MIXED_EXACT_SHAPES": "1", "VLLM_GAUDI_MIXED_NATIVE_PIN": "1"}
        for extra, word in ((pin, "NATIVE_PIN"), ({"VLLM_RECURRENT_TAIL_FUSE": "1"}, "TAIL_FUSE")):
            with self.subTest(word), mock.patch.dict(os.environ, extra), self.assertRaisesRegex(ValueError, word):
                self.graph.install_entry(SimpleNamespace(), lambda: None)

    def test_marks_and_merge_check_without_keys_refuse(self) -> None:
        for flag in ("HPU_GLM53_R589_CAPTURE_MARKS", "HPU_GLM53_PROMPT_KEY_MERGE_CHECK"):
            with self.subTest(flag), mock.patch.dict(os.environ, {KEYS: "", flag: "1"}), \
                    self.assertRaisesRegex(ValueError, KEYS):
                self.graph.install_entry(SimpleNamespace(), lambda: None)

    def test_install_keeps_the_raw_forward_with_keys(self) -> None:
        adapter, raw = SimpleNamespace(), lambda: None
        self.graph.install_entry(adapter, raw)
        self.assertIs(adapter._glm53_mixed_graph_forward, raw)


class Selector(KeyCase):
    keys = "1024:8192,1024:4096,1024:1024"

    def select(self, rows: int, width: int, spec: str | None = None) -> tuple:
        with mock.patch.dict(os.environ, {KEYS: spec or self.keys}):
            return self.graph.select_prompt_key(rows, width)

    def test_every_client_target_reaches_a_key_in_both_key_sets(self) -> None:
        for tokens, table, three, two in CLIENT_TARGETS:
            with self.subTest(tokens=tokens):
                self.assertEqual(pow2_width(tokens), table)
                self.assertEqual(self.select(1024, table), ((1024, three), ""))
                self.assertEqual(self.select(1024, table, "1024:8192,1024:1024"), ((1024, two), ""))

    def test_rows_pad_up_to_r(self) -> None:
        for rows in (1, 17, 513, 1023, 1024):
            self.assertEqual(self.select(rows, 1024), ((1024, 1024), ""))

    def test_smallest_width_wins_then_smallest_rows(self) -> None:
        self.assertEqual(self.select(100, 1024, "512:2048,1024:1024,512:1024"), ((512, 1024), ""))

    def test_a_chunk_past_r_and_a_table_past_the_top_key_name_the_axis(self) -> None:
        self.assertEqual(self.select(1025, 1024), (None, "width_over"))
        self.assertEqual(self.select(1024, 16384), (None, "ctx_over"))


STEP_45_KEYS = "1024:8192,1024:4096,1024:2048,1024:1024"
# (tokens at the end of a chunk, key width): the first and last token of every power-of-two table tier the four keys serve
KEY_EDGES = ((1, 1024), (131_072, 1024), (131_073, 2048), (262_144, 2048), (262_145, 4096), (524_288, 4096), (524_289, 8192),
             (1_048_576, 8192))


class KeySetStep45(KeyCase):
    """Step 45: keys 1024:8192,1024:4096,1024:2048,1024:1024. 131,073, 262,145 and 524,289 tokens are the first to need the next key up."""

    keys = STEP_45_KEYS

    def runner(self) -> SimpleNamespace:
        return SimpleNamespace(attn_block_size=BLOCK, block_size=BLOCK, max_model_len=1_048_576,
                               input_batch=SimpleNamespace(max_num_reqs=16))

    def test_the_edges_of_every_tier_choose_their_key(self) -> None:
        for tokens, width in KEY_EDGES:
            with self.subTest(tokens=tokens):
                self.assertEqual(self.graph.select_prompt_key(1024, pow2_width(tokens)), ((1024, width), ""))

    def test_every_section_4_target_chooses_its_own_table_width(self) -> None:
        for tokens, table, _, _ in CLIENT_TARGETS:
            with self.subTest(tokens=tokens):
                self.assertEqual(self.graph.select_prompt_key(1024, pow2_width(tokens)), ((1024, table), ""))

    def test_one_token_past_the_widest_key_has_no_key(self) -> None:
        self.assertEqual(self.graph.select_prompt_key(1024, pow2_width(1_048_577)), (None, "ctx_over"))

    def test_boot_captures_widest_first_one_dummy_per_key(self) -> None:
        runner = self.runner()
        buckets = self.graph.validate_warmup(runner)
        self.assertEqual(buckets, tuple((0, 1024, width) for width in (8192, 4096, 2048, 1024)))
        dummies = [cfg[0][2] for cfg in self.graph.warmup_scenarios(runner, buckets)]
        self.assertEqual(dummies, [(width * BLOCK - 1024) // BLOCK for width in (8192, 4096, 2048, 1024)])

    def test_the_default_max_graphs_admits_the_four_keys_and_one_fewer_refuses(self) -> None:
        """The decode lattice's graphs are not GraphCache entries: MAX_GRAPHS counts the prompt keys only."""
        self.assertEqual(len(self.graph.validate_warmup(self.runner())), 4)
        with mock.patch.dict(os.environ, {"VLLM_GAUDI_MIXED_GRAPH_MAX_GRAPHS": "3"}), self.assertRaisesRegex(ValueError, "MAX_GRAPHS"):
            self.graph.validate_warmup(self.runner())

    def test_the_four_keys_validate_with_the_decode_lattice_on(self) -> None:
        lattice = {"VLLM_GAUDI_MIXED_DECODE_WIDTHS": "1024,4096,8192", "HPU_GLM53_DECODE_WIDTH_PAD": "check",
                   "HPU_GLM53_DECODE_PAD_CHECK_WIDTHS": "2048", "HPU_GLM53_MIXED_DECODE_REPLAY": "1", "VLLM_PREFILL_CONTEXT_CELL_BUDGET": "0"}
        with mock.patch.dict(os.environ, lattice):
            self.graph.check_prompt_key_flags()
            self.assertEqual(len(self.graph.validate_prompt_keys(self.runner())), 4)
            self.assertEqual(len(self.graph.decode_keys()), 20)

    def test_the_default_budget_does_not_judge_the_keys_by_estimate(self) -> None:
        """Four keys at the 2,846 MiB estimate come to 11,384 MiB; the estimate is not the measured residency, the seal's in-use delta is."""
        with mock.patch.dict(os.environ, {"VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB": "6144"}):
            self.assertEqual(len(self.graph.validate_warmup(self.runner())), 4)

    def test_a_replayed_chunk_at_each_edge_hits_its_key_with_no_capture(self) -> None:
        cache = self.graph.GraphCache(RowForward(), BLOCK, Backend())
        for width in (8192, 4096, 2048, 1024):
            self.boot_capture(cache, self.prompt_batch(1024, 0, width))
        captures, chosen, real = cache.counts["captures"], [], cache._replay
        cache._replay = lambda key, inputs, batches: chosen.append(key[0]) or real(key, inputs, batches)
        for tokens, _ in KEY_EDGES:
            rows = min(1024, tokens)
            self.assertIsNotNone(cache.run_key(self.prompt_batch(rows, tokens - rows, pow2_width(tokens))))
        self.assertEqual(chosen, [(0, 1024, width) for _, width in KEY_EDGES])
        self.assertEqual(cache.counts["captures"], captures)


class KeyedBatches(KeyCase):
    def batch_kinds(self, **kwargs: bool) -> list[bool]:
        kinds: list[bool] = []

        def fake_batch(_runner: object, _ids: list, _tokens: object, _positions: object, _metadata: object, _logits: object, _schedule: object,
                       decode: bool, _trim: object) -> SimpleNamespace:
            kinds.append(decode)
            return SimpleNamespace(key=len(kinds))

        tokens = torch.zeros((1, 4), dtype=torch.int64)
        prefills = SimpleNamespace(request_ids=[["p"]], token_ids=[tokens], position_ids=[tokens], attn_metadata=[None],
                                   logits_indices=[torch.zeros(1)], logits_requests=[["p"]])
        decode = SimpleNamespace(token_ids=tokens, position_ids=tokens, attn_metadata=None)
        with mock.patch.object(self.step, "_batch", side_effect=fake_batch):
            self.step._batches(None, SimpleNamespace(num_scheduled_tokens={"p": 4}), SimpleNamespace(decode_req_ids=["d"]), prefills, decode, **kwargs)
        return kinds

    def test_a_keyed_step_prepares_no_decode_region_it_would_discard(self) -> None:
        self.assertEqual(self.batch_kinds(), [True, False])
        self.assertEqual(self.batch_kinds(prompts_only=True), [False])

    def test_the_keyed_path_asks_for_prompts_only(self) -> None:
        import inspect
        self.assertIn("prompts_only=True", inspect.getsource(self.step.prepare))


class Capture(KeyCase):
    def setUp(self) -> None:
        super().setUp()
        self.forward, self.backend = RowForward(), Backend()
        self.cache = self.graph.GraphCache(self.forward, BLOCK, self.backend)

    def test_prompt_region_runs_alone_padded_to_the_key(self) -> None:
        batch = self.prompt_batch(700, 0, 1024)
        hidden = self.boot_capture(self.cache, batch)
        self.assertEqual(self.forward.shapes, [(1, 1024)])
        self.assertEqual(hidden.shape[0], 700)
        self.assertTrue(torch.equal(hidden[:, 0], batch.tokens[0, :700].float()))
        self.assertEqual(self.cache.counts["captures"], 1)
        self.assertEqual(self.backend.prepared[0][0], (0, 1024, 1024))

    def test_table_pads_with_minus_one_to_the_key_width(self) -> None:
        self.boot_capture(self.cache, self.prompt_batch(1024, 1024, 2048))
        table = self.forward.tables[0]
        self.assertEqual(table.shape, (1, 8192))
        self.assertTrue(torch.all(table[0, 2048:] == -1))

    def test_prompt_query_start_loc_holds_the_real_length_not_the_key_rows(self) -> None:
        inputs = self.graph.padded_inputs([self.prompt_batch(700, 0, 1024)], (0, 1024, 1024))
        self.assertEqual(inputs[0].regions[0].metadata.query_start_loc.tolist(), [0, 700])

    def test_a_native_chunk_wider_than_its_real_rows_still_fits(self) -> None:
        batch = self.prompt_batch(600, 0, 1024, native_width=2048)
        self.assertEqual(self.boot_capture(self.cache, batch).shape[0], 600)

    def test_second_chunk_of_a_key_replays_with_no_capture(self) -> None:
        self.boot_capture(self.cache, self.prompt_batch(1024, 0, 1024))
        hidden = self.cache.run_key(self.prompt_batch(512, 1024, 1024, seed=3))
        self.assertEqual(self.backend.replays, 1)
        self.assertEqual(self.cache.counts["captures"], 1)
        self.assertEqual(hidden.shape[0], 512)

    def test_misses_return_none_and_count_the_reason(self) -> None:
        self.assertIsNone(self.cache.run_key(self.prompt_batch(1025, 0, 1024, native_width=1025)))
        self.assertIsNone(self.cache.run_key(self.prompt_batch(1024, 0, 16384)))
        self.assertEqual((self.cache.counts["fallback_width_over"], self.cache.counts["fallback_ctx_over"]), (1, 1))
        self.assertEqual(self.cache.counts["native_fallbacks"], 2)

    def test_a_key_not_warmed_yet_runs_native_instead_of_capturing(self) -> None:
        """The runner's own prompt warmups run before the keys' and must not capture one outside BOOT."""
        self.assertIsNone(self.cache.run_key(self.prompt_batch(128, 0, 1024)))
        self.assertEqual((self.backend.prepared, self.cache.counts["fallback_not_warmed"]), ([], 1))

    def test_a_miss_at_boot_refuses_the_boot(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "width_over"):
            self.boot_capture(self.cache, self.prompt_batch(1025, 0, 1024, native_width=1025))

    def test_the_key_capture_context_is_only_on_inside_the_capture(self) -> None:
        seen = []
        real = self.backend.capture
        self.backend.capture = lambda call: seen.append(self.graph.key_capture_active()) or real(call)
        self.boot_capture(self.cache, self.prompt_batch(1024, 0, 1024))
        self.assertEqual(seen, [True])
        self.assertFalse(self.graph.key_capture_active())

    def test_capture_binds_the_private_top_k_when_the_guard_is_loaded(self) -> None:
        states = []
        guard = SimpleNamespace(private_key_bind=lambda: mock.MagicMock(
            __enter__=lambda _: states.append("bound") or 1, __exit__=lambda *_: states.append("released")))
        real = self.backend.capture
        self.backend.capture = lambda call: states.append("capture") or real(call)
        with mock.patch.dict(sys.modules, {"glm53_graph_state": guard}):
            self.boot_capture(self.cache, self.prompt_batch(1024, 0, 1024))
        self.assertEqual(states, ["bound", "capture", "released"])

    def test_first_hit_outside_boot_is_refused_by_the_backend(self) -> None:
        fake = SimpleNamespace(core=SimpleNamespace(mark_step=lambda: None),
                               hpu=SimpleNamespace(synchronize=lambda: None, mem_get_info=lambda: (10, 10)))
        with mock.patch.object(self.graph, "admit_capture", side_effect=RuntimeError("refused")) as admit, \
                mock.patch.dict(sys.modules, {"habana_frameworks": SimpleNamespace(torch=fake), "habana_frameworks.torch": fake}), \
                self.assertRaises(RuntimeError):
            self.graph.HpuBackend().prepare_capture((0, 1024, 1024), 1, "")
        self.assertIn("first-hit capture", admit.call_args.args[3])

    def test_estimate_and_requirement_follow_section_3(self) -> None:
        mib = self.graph.MIB
        self.assertEqual(self.graph.estimate_bytes((0, 1024, 8192), BLOCK), 2846 * mib)
        self.assertEqual(self.graph.estimate_bytes((0, 512, 1024), BLOCK), 1674 * mib)
        self.assertEqual(self.graph.estimate_bytes((0, 2048, 1024), BLOCK), 5190 * mib)
        self.assertEqual(self.graph.capture_requirement(2846 * mib) // mib, 512, "a keyed boot checks the reserve only")

    def test_flags_off_leave_the_requirement_alone(self) -> None:
        mib = self.graph.MIB
        with mock.patch.dict(os.environ, {KEYS: ""}):
            self.assertEqual(self.graph.capture_requirement(2846 * mib) // mib, 4269 + 2048)


class Warmup(KeyCase):
    def runner(self, max_len: int = 1_048_576) -> SimpleNamespace:
        return SimpleNamespace(attn_block_size=BLOCK, block_size=BLOCK, max_model_len=max_len,
                               input_batch=SimpleNamespace(max_num_reqs=16))

    def test_one_prompt_only_dummy_per_key_as_written(self) -> None:
        runner = self.runner()
        buckets = self.graph.validate_warmup(runner)
        self.assertEqual(buckets, ((0, 1024, 8192), (0, 1024, 1024)))
        self.assertEqual(self.graph.warmup_scenarios(runner, buckets),
                         [((1, 1024, (8192 * BLOCK - 1024) // BLOCK), None), ((1, 1024, (1024 * BLOCK - 1024) // BLOCK), None)])

    def test_a_boot_that_left_a_key_uncaptured_refuses(self) -> None:
        runner = self.runner()
        cache = self.graph.GraphCache(RowForward(), BLOCK, Backend())
        self.boot_capture(cache, self.prompt_batch(1024, 0, 1024))
        runner.model = SimpleNamespace(_glm53_mixed_graph_cache=cache)
        with self.assertRaisesRegex(RuntimeError, "1024:8192"):
            self.graph.check_keys_captured(runner, self.graph.validate_warmup(runner))
        self.boot_capture(cache, self.prompt_batch(1024, 0, 8192))
        self.graph.check_keys_captured(runner, self.graph.validate_warmup(runner))

    def test_a_key_past_max_model_len_refuses_naming_the_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "1024:8192"):
            self.graph.validate_warmup(self.runner(max_len=500_000))

    def test_max_graphs_counts_every_key(self) -> None:
        with mock.patch.dict(os.environ, {"VLLM_GAUDI_MIXED_GRAPH_MAX_GRAPHS": "1"}), self.assertRaisesRegex(ValueError, "MAX_GRAPHS"):
            self.graph.validate_warmup(self.runner())


class Dispatch(KeyCase):
    """prepare(): the prompt batch goes to its key when the runner calls for it, decode stays on the runner's own graph, in order."""

    def world(self) -> SimpleNamespace:
        self.order: list[str] = []
        forward = RowForward()
        cache = self.graph.GraphCache(forward, BLOCK, Backend())
        self.boot_capture(cache, self.prompt_batch(1024, 0, 1024))
        model = SimpleNamespace(_glm53_mixed_graph_cache=cache, _glm53_mixed_graph_forward=forward,
                                compute_logits=lambda rows: self.order.append("logits") or rows.sum(-1, keepdim=True))
        return SimpleNamespace(model=model, model_config=SimpleNamespace(enforce_eager=False), attn_block_size=BLOCK)

    def keyed(self, prompt: object) -> tuple:
        runner = self.world()
        generic = mock.Mock(side_effect=lambda *a, **k: self.order.append("generic") or "generic-result")
        runner._execute_model_generic = generic
        return self.step.KeyedForward(runner, {prompt.key: prompt}), generic

    def test_decode_runs_the_runner_forward_and_prompt_runs_the_key(self) -> None:
        prompt = self.prompt_batch(1024, 1024, 1024, seed=5)
        forward, generic = self.keyed(prompt)
        self.assertEqual(forward("tokens", "positions", object(), warmup_mode=False), "generic-result")
        generic.assert_called_once()
        self.assertEqual(generic.call_args.kwargs, {"warmup_mode": False})
        full, aux, sampled, logits = forward(None, None, prompt.region.metadata)
        self.assertEqual(full.shape[:2], (1, 1024))
        self.assertIsNone(aux)
        self.assertEqual((sampled.shape[0], logits.shape[0]), (1, 1))
        self.assertEqual(generic.call_count, 1)
        self.assertEqual(self.order, ["generic", "logits"])

    def test_a_key_miss_runs_the_runner_forward(self) -> None:
        prompt = self.prompt_batch(1024, 0, 16384)
        forward, generic = self.keyed(prompt)
        self.assertEqual(forward("t", "p", prompt.region.metadata), "generic-result")
        generic.assert_called_once()

    def test_non_final_chunk_logit_row_reads_the_last_real_row(self) -> None:
        prompt = self.prompt_batch(700, 0, 1024)
        prompt.logits = (-1,)
        self.assertEqual(self.step._key_logit_rows(prompt), (699,))

    def test_prepare_returns_the_dispatcher_only_with_keys(self) -> None:
        pd = SimpleNamespace(prompt_req_ids=["a"], decode_req_ids=[])
        batch = self.prompt_batch(1024, 0, 1024)
        runner = SimpleNamespace(_execute_model_generic="generic")
        with mock.patch.object(self.step, "_validate_runner"), mock.patch.object(self.step, "_batches", return_value=[batch]):
            self.assertIsInstance(self.step.prepare(runner, None, pd, None, None), self.step.KeyedForward)
            with mock.patch.dict(os.environ, {KEYS: ""}), mock.patch.object(self.step, "tail_fuse_enabled", return_value=False):
                self.assertEqual(self.step.prepare(runner, None, pd, None, None), "generic")

    def test_a_decode_only_step_stays_on_the_runner_with_keys(self) -> None:
        pd = SimpleNamespace(prompt_req_ids=[], decode_req_ids=["a"])
        runner = SimpleNamespace(_execute_model_generic="generic")
        with mock.patch.object(self.step, "_validate_runner"):
            self.assertEqual(self.step.prepare(runner, None, pd, None, None), "generic")


class FakeCore:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def mark_step(self, *args: object, **kwargs: object) -> None:
        self.calls.append(kwargs.get("sync", args[0] if args else False))


class FakeHpu:
    def __init__(self) -> None:
        self.in_use, self.peak = 1000, 1000
        self.syncs = 0

    def memory_stats(self) -> dict[str, int]:
        return {"InUse": self.in_use, "MaxInUse": self.peak, "Limit": 10 ** 9}

    def reset_peak_memory_stats(self) -> None:
        self.peak = self.in_use

    def synchronize(self) -> None:
        self.syncs += 1


class CensusFields(KeyCase):
    """The K1 scorer's contract: rows, context = key width, sync_true on captures, replay_ms on replays, all path=graph."""

    def setUp(self) -> None:
        super().setUp()
        import glm53_prompt_census as census

        self.census = census
        self.core, self.hpu = FakeCore(), FakeHpu()
        patches = [mock.patch.dict(os.environ, {"HPU_GLM53_PREFILL_HBM_CENSUS": "1"}),
                   mock.patch.object(census, "_runtime", lambda: (self.core, self.hpu))]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        census.GRAPH_LOG.clear()
        self.addCleanup(census.GRAPH_LOG.clear)
        self.forward, self.backend = RowForward(), Backend()
        self.cache = self.graph.GraphCache(self.forward, BLOCK, self.backend)

    def lines(self, run: object) -> list[dict]:
        with self.assertLogs("vllm.r589", "WARNING") as logs:
            run()
        return [json.loads(line.split("GLM53_PROMPT_CENSUS ", 1)[1]) for line in logs.output if "GLM53_PROMPT_CENSUS {" in line]

    def test_capture_line_carries_width_and_sync_true_count(self) -> None:
        real = self.backend.capture

        def capture(call: object) -> tuple:
            self.core.mark_step(sync=True)
            self.core.mark_step()
            return real(call)

        self.backend.capture = capture
        body = self.lines(lambda: self.boot_capture(self.cache, self.prompt_batch(1024, 0, 8192)))[0]
        self.assertEqual((body["path"], body["kind"], body["rows"], body["context"], body["sync_true"]),
                         ("graph", "capture", 1024, 8192, 1))

    def test_sync_true_counts_only_true_marks(self) -> None:
        census = self.census.MarkStepCensus(self.core, self.hpu)
        with census:
            self.core.mark_step(sync=True)
            self.core.mark_step(sync=False)
            self.core.mark_step()
            self.core.mark_step(True)
        self.assertEqual(census.report(1, 1)["sync_true"], 2)

    def test_every_second_replay_is_fenced_and_timed(self) -> None:
        self.lines(lambda: self.boot_capture(self.cache, self.prompt_batch(1024, 0, 1024)))
        bodies = self.lines(lambda: [self.cache.run_key(self.prompt_batch(1024, 1024, 1024, seed=2)) for _ in range(4)])
        self.assertEqual([body["kind"] for body in bodies], ["replay"] * 4)
        self.assertEqual(["replay_ms" in body for body in bodies], [False, True, False, True])
        self.assertEqual(self.hpu.syncs, 4)  # a timed replay drains the device before and after its clock: 2 timed replays
        self.assertTrue(all(body["context"] == 1024 and body["rows"] == 1024 for body in bodies))

    def test_the_k1_cells_repeats_score_five_timed_warmed_replays_per_width(self) -> None:
        """The cell's prompt, REPEATS times, merge check on: the log the scorer reads has >= 5 timed warmed replays of each key."""
        sys.path.insert(0, str(OVERLAY_ROOT / "cells"))
        self.addCleanup(sys.path.remove, str(OVERLAY_ROOT / "cells"))
        import k1_cell
        import k1_score

        self.boot_capture(self.cache, self.prompt_batch(1024, 1024, 8192))
        self.boot_capture(self.cache, self.prompt_batch(1024, 0, 1024))

        def requests() -> None:
            for _ in range(k1_cell.REPEATS):
                for chunk in range(k1_cell.PROMPT_TOKENS // 1024):
                    self.cache.run_key(self.prompt_batch(1024, chunk * 1024, 1024))

        with mock.patch.dict(os.environ, {"HPU_GLM53_PROMPT_KEY_MERGE_CHECK": "1"}),                 self.assertLogs(self.graph.LOG, "WARNING") as merge_logs:
            bodies = self.lines(requests)
        merges = [line.split(":", 2)[2] for line in merge_logs.output if "GLM53_K1_MERGE" in line]
        log = [f"(Worker_TP0_EP0 pid=1) WARNING GLM53_PROMPT_CENSUS {json.dumps(body)}" for body in bodies]
        log += [f"(Worker_TP0_EP0 pid=1) WARNING {message}" for message in merges]
        parsed = k1_score.parse_log(log)
        keys = parsed.ranks[0].keys
        wide, low = (k1_score.summarize_key(keys[key]) for key in (k1_score.WIDE_KEY, k1_score.LOW_KEY))
        self.assertGreaterEqual(wide.samples, k1_cell.TARGET_WARMED_SAMPLES)
        self.assertGreaterEqual(low.samples, k1_cell.TARGET_WARMED_SAMPLES)
        self.assertIsNotNone(wide.replay_ms)
        forced = k1_cell.REPEATS // k1_cell.MERGE_STRIDE * k1_cell.MERGE_CHUNKS_PER_REPEAT
        self.assertEqual(len([body for body in bodies if body["context"] == 8192]), min(forced, self.census.GRAPH_LOG_FIRST))
        self.assertEqual(sorted({start for start, _ in parsed.ranks[0].merge}), [0, 4096, 5120])
        row = k1_score.summarize_rank(0, parsed.ranks[0])
        self.assertEqual((row.merge, row.merge_late), (True, True))

    def test_census_off_writes_nothing_and_never_fences(self) -> None:
        with mock.patch.dict(os.environ, {"HPU_GLM53_PREFILL_HBM_CENSUS": "0"}):
            self.boot_capture(self.cache, self.prompt_batch(1024, 0, 1024))
            self.cache.run_key(self.prompt_batch(1024, 1024, 1024))
        self.assertEqual(self.hpu.syncs, 0)


class KeyedAdmission(KeyCase):
    """A keyed boot: free memory is not capacity, since capture transient reclaims under pressure. The reserve before, a real PT_DEVMEM during."""

    def device(self, free_mib: int, wrap: mock.Mock | None = None, headroom_mib: int | None = None) -> dict:
        """mem_get_info reads free_mib; the allocator's Limit - InUse reads headroom_mib (free_mib unless given)."""
        limit = 98_000 * self.graph.MIB
        in_use = limit - (free_mib if headroom_mib is None else headroom_mib) * self.graph.MIB
        hpu = SimpleNamespace(synchronize=lambda: None, mem_get_info=lambda: (free_mib * self.graph.MIB, 8452 * self.graph.MIB),
                              memory_allocated=lambda: 0, max_memory_allocated=lambda: 0, reset_peak_memory_stats=lambda: None,
                              memory_stats=lambda: {"Limit": limit, "InUse": in_use}, wrap_in_hpu_graph=wrap)
        fake = SimpleNamespace(core=SimpleNamespace(mark_step=lambda: None), hpu=hpu)
        return {"habana_frameworks": SimpleNamespace(torch=fake), "habana_frameworks.torch": fake}

    def prepare(self, bucket: tuple, estimate_mib: int, free_mib: int, headroom_mib: int | None = None) -> None:
        with mock.patch.dict(sys.modules, self.device(free_mib, headroom_mib=headroom_mib)), self.graph.boot_capture_context():
            self.graph.HpuBackend().prepare_capture(bucket, estimate_mib * self.graph.MIB, "")

    def test_a_key_under_the_reserve_below_the_allocator_limit_refuses_though_free_hbm_is_ample(self) -> None:
        """free HBM is the device's; the allocator aborts on PT_DEVMEM past its own Limit - InUse, which is what the reserve must fit in."""
        with self.assertRaises(RuntimeError) as refused:
            self.prepare((0, 1024, 2048), 2847, 6000, headroom_mib=100)
        message = str(refused.exception)
        for needle in ("prompt key 1024:2048", "100.0 MiB under the allocator limit", "512.0", "rejected on ALL ranks", KEYS):
            self.assertIn(needle, message)

    def test_an_allocator_that_cannot_be_read_refuses(self) -> None:
        modules = self.device(6000)
        modules["habana_frameworks.torch"].hpu.memory_stats = mock.Mock(side_effect=RuntimeError("device gone"))
        with mock.patch.dict(sys.modules, modules), self.graph.boot_capture_context(), self.assertRaisesRegex(RuntimeError, "device gone"):
            self.graph.HpuBackend().prepare_capture((0, 1024, 2048), 2847 * self.graph.MIB, "")

    def test_the_admitted_line_logs_both_readings(self) -> None:
        with self.assertLogs(self.graph.LOG, "WARNING") as logs:
            self.prepare((0, 1024, 2048), 2847, 6000, headroom_mib=1340)
        line = next(item for item in logs.output if "GLM53_MIXED_GRAPH admitted" in item)
        self.assertIn(f"headroom_bytes={1340 * self.graph.MIB}", line)
        self.assertIn(f"free_bytes={6000 * self.graph.MIB}", line)

    def capture_refusal(self, bucket: tuple, error: Exception, free_mib: int = 1340) -> tuple[str, mock.Mock]:
        wrap = mock.Mock(side_effect=error)
        backend = self.graph.HpuBackend()
        with mock.patch.dict(sys.modules, self.device(free_mib, wrap)), self.graph.boot_capture_context(), \
                mock.patch.object(self.graph.graphs_only, "arm_exit") as arm:
            backend.prepare_capture(bucket, 2847 * self.graph.MIB, "")
            with self.graph.key_capture_context() if bucket[0] == 0 else nullcontext(), self.assertRaises(RuntimeError) as refused:
                backend.capture(SimpleNamespace(tensor_inputs=()))
        wrap.assert_called_once()  # no retry
        return str(refused.exception), arm

    def test_the_real_boot_reading_is_admitted(self) -> None:
        """Key 3 estimated 2,847 MiB with about 1,340 free; a third capture ran in 1,326 MiB of peak and left more free than before."""
        self.prepare((0, 1024, 2048), 2847, 1340)

    def test_a_shared_keys_scaled_estimate_is_not_a_requirement(self) -> None:
        estimate = self.graph.estimate_bytes((0, 1024, 1024, 4), BLOCK)
        self.assertEqual(self.graph.capture_requirement(estimate), self.graph.KEY_RESERVE_DEFAULT_MIB * self.graph.MIB)

    def test_under_the_reserve_refuses_with_the_reading_and_the_setting(self) -> None:
        with self.assertRaises(RuntimeError) as refused:
            self.prepare((0, 1024, 2048), 2847, 100)
        message = str(refused.exception)
        for needle in ("100.0", "512.0", KEYS, "VLLM_GAUDI_MIXED_GRAPH_RESERVE_MIB"):
            self.assertIn(needle, message)
        self.assertNotIn("num-gpu-blocks-override", message)

    def test_the_reserve_is_a_setting(self) -> None:
        with mock.patch.dict(os.environ, {"VLLM_GAUDI_MIXED_GRAPH_RESERVE_MIB": "1"}):
            self.prepare((0, 1024, 2048), 2847, 2)

    def test_the_peak_percent_setting_is_gone(self) -> None:
        self.assertFalse([name for name in dir(self.graph) if "PEAK_PERCENT" in name])
        with mock.patch.dict(os.environ, {"VLLM_GAUDI_MIXED_KEY_PEAK_PERCENT": "176"}):
            self.assertEqual(self.graph.capture_requirement(100 * self.graph.MIB), 512 * self.graph.MIB)

    @unittest.skipUnless((LANE_DIR / "graphs-dual.env").is_file(), "the env files sit beside the overlay on the host (LANE_DIR)")
    def test_the_stage_env_files_do_not_set_the_removed_flag(self) -> None:
        for name in ("dual", "c1", "c1p", "c1m"):
            with self.subTest(name):
                self.assertNotIn("KEY_PEAK_PERCENT", (LANE_DIR / f"graphs-{name}.env").read_text())

    def test_a_real_pt_devmem_refuses_every_rank_naming_key_free_and_what_to_drop(self) -> None:
        message, arm = self.capture_refusal((0, 1024, 4096), RuntimeError("PT_DEVMEM: allocation of 4 GiB failed"))
        for needle in ("1024:4096", "1340.0 MiB free HBM", "PT_DEVMEM", f"Drop 1024:4096 or another key from {KEYS}", "rejected on ALL ranks"):
            self.assertIn(needle, message)
        self.assertNotIn("num-gpu-blocks-override", message)
        arm.assert_called_once()

    def test_a_shared_key_is_named_as_its_own_setting_lists_it(self) -> None:
        message, _ = self.capture_refusal((0, 1024, 1024, 4), RuntimeError("PT_DEVMEM"))
        self.assertIn("Drop 4:1024:1024 or another key from VLLM_GAUDI_MIXED_PROMPT_BATCH_KEYS", message)

    def test_a_failure_that_is_not_memory_tells_no_one_to_drop_a_key(self) -> None:
        message, _ = self.capture_refusal((0, 1024, 4096), RuntimeError("synapse said no"))
        self.assertIn("capture failed without retry: RuntimeError: synapse said no", message)
        self.assertNotIn("Drop 1024:4096", message)

    def test_the_unkeyed_path_keeps_its_requirement_and_its_advice(self) -> None:
        with mock.patch.dict(os.environ, {KEYS: ""}):
            self.assertEqual(self.graph.capture_requirement(100 * self.graph.MIB), (150 + 2048) * self.graph.MIB)
            message, arm = self.capture_refusal((4, 512, 256), RuntimeError("PT_DEVMEM"), free_mib=8000)
        self.assertIn("capture failed without retry: RuntimeError: PT_DEVMEM", message)
        self.assertIn("num-gpu-blocks-override", message)
        arm.assert_not_called()


class BallastEnv(KeyCase):
    def test_a_stride_of_1_holds_the_ballast_before_the_first_replay(self) -> None:
        allocated: list[int] = []
        with mock.patch.dict(os.environ, {self.graph.BALLAST_ENV: "1538", self.graph.BALLAST_EVERY_ENV: "1"}):
            ballast = self.graph.Ballast()
        with mock.patch.object(ballast, "_allocate", allocated.append), mock.patch.object(ballast, "_free_bytes", return_value=7):
            with self.assertLogs(self.graph.LOG, "WARNING") as logs:
                ballast.advance()
        self.assertEqual(allocated, [1538])
        self.assertIn("total_mib=1538 replays=1 free_bytes=7", logs.output[0])


class BallastSteps(KeyCase):
    def stepped(self, env: str, replays: int) -> tuple[list[int], list[str]]:
        allocated: list[int] = []
        with mock.patch.dict(os.environ, {self.graph.BALLAST_ENV: env}):
            ballast = self.graph.Ballast()
        with (mock.patch.object(ballast, "_allocate", allocated.append), mock.patch.object(ballast, "_free_bytes", return_value=7),
              self.assertLogs(self.graph.LOG, "WARNING") as logs):
            self.graph.LOG.warning("anchor")
            for _ in range(replays):
                ballast.advance()
        return allocated, [line for line in logs.output if "GLM53_K1_BALLAST" in line]

    def test_unset_never_allocates(self) -> None:
        self.assertEqual(self.stepped("", 100), ([], []))

    def test_it_grows_to_each_total_at_each_step(self) -> None:
        allocated, lines = self.stepped("1000,3000", self.graph.BALLAST_EVERY * 2)
        self.assertEqual(allocated, [1000, 2000])
        self.assertIn("total_mib=3000", lines[1])

    def test_no_step_before_the_stride_or_after_the_last_level(self) -> None:
        self.assertEqual(self.stepped("1000", self.graph.BALLAST_EVERY - 1)[0], [])
        self.assertEqual(self.stepped("1000", self.graph.BALLAST_EVERY * 3)[0], [1000])

    def test_a_cache_steps_only_on_replays(self) -> None:
        forward, backend = RowForward(), Backend()
        with mock.patch.dict(os.environ, {self.graph.BALLAST_ENV: "1000"}):
            cache = self.graph.GraphCache(forward, BLOCK, backend)
        self.boot_capture(cache, self.prompt_batch(1024, 0, 1024))
        self.assertEqual(cache.ballast.replays, 0)
        cache.run_key(self.prompt_batch(1024, 0, 1024))
        self.assertEqual(cache.ballast.replays, 1)


class MergeCheck(KeyCase):
    def setUp(self) -> None:
        super().setUp()
        self.forward, self.backend = RowForward(), Backend()
        self.cache = self.graph.GraphCache(self.forward, BLOCK, self.backend)
        self.boot_capture(self.cache, self.prompt_batch(1024, 1024, 8192))
        self.boot_capture(self.cache, self.prompt_batch(1024, 0, 1024))
        self.forward.tables.clear()
        patch = mock.patch.dict(os.environ, {"HPU_GLM53_PROMPT_KEY_MERGE_CHECK": "1"})
        patch.start()
        self.addCleanup(patch.stop)

    def request(self) -> list[str]:
        with self.assertLogs(self.graph.LOG, "WARNING") as logs:
            self.graph.LOG.warning("anchor")
            self.cache.run_key(self.prompt_batch(1024, 0, 1024))
        return [line for line in logs.output if "GLM53_K1_MERGE" in line]

    def test_repeats_alternate_narrow_then_wide_and_log_equal(self) -> None:
        self.assertEqual(self.request(), [])
        second = self.request()
        self.assertEqual(self.forward.tables[0].shape[1], 1024)
        self.assertEqual(self.forward.tables[1].shape[1], 8192)
        self.assertEqual(len(second), 1)
        self.assertRegex(second[0], r"GLM53_K1_MERGE narrow=1024 wide=8192 rows=1024 start=0 torch_equal=1")
        self.assertEqual(self.request(), [])
        self.assertEqual(self.forward.tables[2].shape[1], 1024)

    def test_a_different_prompt_never_compares(self) -> None:
        self.request()
        other = self.prompt_batch(1024, 0, 1024)
        other.tokens = other.tokens + 1
        self.cache.run_key(other)
        self.assertEqual(self.forward.tables[1].shape[1], 1024)

    def test_later_chunks_never_compare(self) -> None:
        self.cache.run_key(self.prompt_batch(1024, 1024, 1024))
        self.cache.run_key(self.prompt_batch(1024, 1024, 1024))
        self.assertEqual([table.shape[1] for table in self.forward.tables], [1024, 1024])

    def test_chunks_from_start_4096_compare_and_those_between_never_do(self) -> None:
        """Below start 4096 top-k keeps every token, so only start 0 and starts >= 4096 can show a width-dependent selection."""
        starts = (0, 1024, 2048, 3072, 4096, 5120)
        for repeat in range(2):
            with self.assertLogs(self.graph.LOG, "WARNING") as logs:
                self.graph.LOG.warning("anchor")
                for start in starts:
                    self.cache.run_key(self.prompt_batch(1024, start, 1024))
            merged = [line for line in logs.output if "GLM53_K1_MERGE" in line]
            self.assertEqual(len(merged), 3 * repeat)
        self.assertEqual([line.split("start=")[1].split()[0] for line in merged], ["0", "4096", "5120"])
        wide = [table.shape[1] == 8192 for table in self.forward.tables]
        self.assertEqual(wide, [False] * 6 + [True, False, False, False, True, True])

    def self_lines(self, hidden: torch.Tensor, chosen: tuple[int, int]) -> list[str]:
        batch = self.prompt_batch(1024, 0, 1024)
        with self.assertLogs(self.graph.LOG, "WARNING") as logs:
            self.graph.LOG.warning("anchor")
            self.cache.merge.observe(batch, (1024, 1024), chosen, hidden)
        return [line for line in logs.output if "GLM53_K1_SELF" in line]

    def test_a_single_key_boot_still_self_compares_and_saves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {self.graph.PROMPT_KEYS_FLAG: "1024:1024", "HPU_GLM53_K1_STALL_DIR": tmp}):
            self.assertEqual(self.self_lines(torch.ones(2), (1024, 1024)), [])
            same = self.self_lines(torch.ones(2), (1024, 1024))
            saved = sorted(path.name for path in Path(tmp).iterdir())
        self.assertRegex(same[0], r"GLM53_K1_SELF rows=1024 width=1024 start=0 torch_equal=1")
        self.assertEqual(saved, ["hidden-r0-1024-1024-0.pt"])

    def test_a_chunk_between_never_self_compares(self) -> None:
        batch = self.prompt_batch(1024, 1024, 1024)
        self.cache.merge.observe(batch, (1024, 1024), (1024, 1024), torch.ones(2))
        self.assertEqual(self.cache.merge.references, {})

    def test_a_replay_compares_with_its_own_first_output(self) -> None:
        self.assertEqual(self.self_lines(torch.ones(2), (1024, 1024)), [])
        same = self.self_lines(torch.ones(2), (1024, 1024))
        self.assertRegex(same[0], r"GLM53_K1_SELF rows=1024 width=1024 start=0 torch_equal=1")
        changed = self.self_lines(torch.zeros(2), (1024, 1024))
        self.assertRegex(changed[0], r"GLM53_K1_SELF rows=1024 width=1024 start=0 torch_equal=0")

    def test_each_key_keeps_its_own_reference(self) -> None:
        self.self_lines(torch.ones(2), (1024, 1024))
        self.assertEqual(self.self_lines(torch.zeros(2), (1024, 8192)), [])

    def test_a_width_dependent_hidden_logs_not_equal(self) -> None:
        def widthful(*, input_ids: torch.Tensor, positions: torch.Tensor, attn_metadata: object) -> torch.Tensor:
            width = attn_metadata.regions[0].metadata.glm53_index_block_table.shape[1]
            return (input_ids.reshape(-1).float() + width).unsqueeze(-1).unsqueeze(0)

        self.cache.entries = {}
        self.cache.raw_forward = widthful
        with mock.patch.dict(os.environ, {"HPU_GLM53_PROMPT_KEY_MERGE_CHECK": "0"}):
            self.boot_capture(self.cache, self.prompt_batch(1024, 1024, 8192))
            self.boot_capture(self.cache, self.prompt_batch(1024, 0, 1024))
        self.request()
        self.assertRegex(self.request()[0], r"torch_equal=0")

    def test_scorer_parses_the_line(self) -> None:
        sys.path.insert(0, str(OVERLAY_ROOT / "cells"))
        self.addCleanup(sys.path.remove, str(OVERLAY_ROOT / "cells"))
        import k1_score

        self.request()
        message = self.request()[0].split(":", 2)[2]
        parsed = k1_score.parse_log([f"(Worker_TP2_EP2 pid=9) WARNING 11-06 [x.py:1] {message}"])
        self.assertEqual(parsed.ranks[2].merge, [(0, True)])


class CensusHome(KeyCase):
    def test_mixed_graph_uses_the_census_home(self) -> None:
        source = (OVERLAY_ROOT / "glm53_mixed_graph.py").read_text(encoding="utf-8")
        self.assertNotIn("class MarkStepCensus", source)
        self.assertIn("from glm53_prompt_census import", source)


if __name__ == "__main__":
    unittest.main()
