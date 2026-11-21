"""Nativepin: a mixed graph step runs each region at the context width and logits rows native uses (C573 follow-up).

Fastexact (split forward + exact shapes) already pins each region's MoE path: the MoE picks it from x.shape[0]
alone (hpu_fp8.py:764-884, :1276) and each region keeps native's padded rows. Two inputs still differ from native:
  (a) width: the graph pads every region's glm53_index_block_table to the bucket's c blocks with fill 0
      (glm53_mixed_graph.pad_field), while native builds a power-of-two width of the batch's own blocks with fill -1
      (hpu_model_runner 308de401 _glm53_sparse_block_bucket / _make_glm53_sparse_metadata_cpu :2651-2701);
  (b) logits: the mixed step calls compute_logits once over the joined valid rows, while native calls it per batch
      on hidden[logits_indices]: D padded decode rows whose pads index row 0 (:4849-4869, :5557-5564) and the
      prompt's own host rows (logits_rounding = 1, :2592 / :4435).
VLLM_GAUDI_MIXED_NATIVE_PIN=1 (needs EXACT_SHAPES=1, which needs SPLIT_FORWARD=1) keeps the native table and computes
logits per region at native's rows. Because a captured graph has one static width, BOOT captures one graph per
(decode width, prompt width) pair in VLLM_GAUDI_MIXED_NATIVE_WIDTHS; a live step at an unwarmed width runs native.
Unset or 0 keeps the combined bytes. GRAPH_UNDER_TEST / STEP_UNDER_TEST point the suite at another build (RED: orig/).
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

OVERLAY = Path(__file__).resolve().parents[1]
LANE = OVERLAY / "tests" / "fixtures" / "nativepin"
BASE = Path("/usr/local/lib/python3.12/dist-packages")
GRAPH_UNDER_TEST = Path(os.environ.get("GRAPH_UNDER_TEST", OVERLAY / "glm53_mixed_graph.py"))
STEP_UNDER_TEST = Path(os.environ.get("STEP_UNDER_TEST", OVERLAY / "glm53_mixed_step.py"))
ORIG_GRAPH, ORIG_STEP = LANE / "orig" / "glm53_mixed_graph.py", LANE / "orig" / "glm53_mixed_step.py"
PIN, WIDTHS = "VLLM_GAUDI_MIXED_NATIVE_PIN", "VLLM_GAUDI_MIXED_NATIVE_WIDTHS"
EXACT, SPLIT = "VLLM_GAUDI_MIXED_EXACT_SHAPES", "VLLM_GAUDI_MIXED_SPLIT_FORWARD"
SWITCHES = (PIN, WIDTHS, EXACT, SPLIT, "VLLM_GAUDI_MIXED_GRAPH_ESTIMATE_MIB", "VLLM_GAUDI_MIXED_GRAPH_HBM_PROBE",
            "VLLM_GAUDI_MIXED_GRAPH_MAX_GRAPHS")
HIDDEN, BLOCK_SIZE, VOCAB = 4, 128, 3
DECODE_WIDTH, PROMPT_WIDTH = 32, 16    # c8 gate: ~3.7k-token prompts -> decode 32 blocks, a 1.5-2k chunk 16


@dataclass(frozen=True)
class FakeMetadata:
    glm53_index_block_table: torch.Tensor
    slot_mapping: torch.Tensor
    num_actual_tokens: int
    num_decodes: int


def _load(path: Path, name: str, monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}; is the lane tree complete (orig/, patched/, ref/)?")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _pair(monkeypatch: pytest.MonkeyPatch, graph_path: Path, step_path: Path) -> types.SimpleNamespace:
    """Load one (step, graph) build under the module names they import each other by."""
    _load(BASE / "mixed_recurrent.py", "mixed_recurrent", monkeypatch)
    step = _load(step_path, "glm53_mixed_step", monkeypatch)
    graph = _load(graph_path, "glm53_mixed_graph", monkeypatch)
    return types.SimpleNamespace(step=step, graph=graph)


@pytest.fixture()
def mods(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_GAUDI_MIXED_STEP", "1")
    monkeypatch.setenv("VLLM_GAUDI_MIXED_GRAPH_WARMUP", "8:512:512")
    monkeypatch.setenv("VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB", "65536")
    return _pair(monkeypatch, GRAPH_UNDER_TEST, STEP_UNDER_TEST)


def _pin_on(monkeypatch: pytest.MonkeyPatch, widths: str | None = None) -> None:
    for name in (SPLIT, EXACT, PIN):
        monkeypatch.setenv(name, "1")
    if widths is not None:
        monkeypatch.setenv(WIDTHS, widths)


def native_table(rows: int, live: int, width: int, seed: int) -> torch.Tensor:
    """hpu_model_runner._make_glm53_sparse_metadata_cpu: (target_bs, pow2 width), -1 past each request's blocks."""
    table = torch.full((rows, width), -1, dtype=torch.int32)
    for row in range(live):
        used = width // 2 + 1 + (row + seed) % (width // 2)
        table[row, :used] = torch.arange(used, dtype=torch.int32) * 7 + row + seed
    return table


def native_batches(step: types.ModuleType, *, decodes: int = 5, decode_rows: int = 8, prompt_tokens: int = 512,
                   prompt_width: int = 512, decode_width: int = DECODE_WIDTH, chunk_width: int = PROMPT_WIDTH,
                   seed: int = 1, prompt_logits: tuple[int, ...] | None = None) -> list:
    """One decode batch + one prompt chunk exactly as the runner prepares them natively."""
    decode_meta = FakeMetadata(native_table(decode_rows, decodes, decode_width, seed), torch.arange(decode_rows),
                               decode_rows, decode_rows)
    prompt_meta = FakeMetadata(native_table(1, 1, chunk_width, seed + 1), torch.arange(prompt_width), prompt_width, 0)
    decode = step.Region(tuple(range(decodes)), (1,) * decodes, (3000,) * decodes, 1, decode_rows,
                         tuple(range(decodes)), decode_meta, True)
    prompt = step.Region((decodes,), (prompt_tokens,), (1536,), prompt_width, 1, tuple(range(prompt_tokens)),
                         prompt_meta, False)
    tokens = torch.arange(decode_rows).reshape(decode_rows, 1) + 10 * seed
    prompt_ids = torch.arange(prompt_width).reshape(1, prompt_width) + 100 * seed
    logits = (prompt_tokens - 1,) if prompt_logits is None else prompt_logits
    return [step.PreparedBatch(decode, tokens, tokens + 3000, tuple(range(decode_rows)), 1),
            step.PreparedBatch(prompt, prompt_ids, prompt_ids + 1536, logits, 2)]


class RowForward:
    """Hidden row r = (token id, row index, ...) so every selection is visible in the output."""

    def __init__(self) -> None:
        self.rows: list[int] = []

    def __call__(self, *, input_ids: torch.Tensor, positions: torch.Tensor, attn_metadata: object) -> torch.Tensor:
        self.rows.append(input_ids.shape[-1])
        ids = input_ids.reshape(-1).double()
        return torch.stack([ids, positions.reshape(-1).double(), ids * 2, ids * 3], dim=-1).unsqueeze(0)


class ReplayBackend:
    def capture(self, call: object) -> tuple:
        output = call()
        return (call, output), output

    def replay(self, handle: tuple) -> None:
        call, output = handle
        output.copy_(call())


def _tables(inputs: tuple) -> list[torch.Tensor]:
    plan = inputs[0]
    return [region.metadata.glm53_index_block_table for region in plan.regions]


# ------------------------------------------------------------------ (a) native context width


def test_combined_pads_the_block_table_to_the_bucket_not_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mismatch nativepin removes, shown on the combined build: width 512 with fill 0 vs native 32 / 16 with -1."""
    monkeypatch.setenv("VLLM_GAUDI_MIXED_GRAPH_WARMUP", "8:512:512")
    orig = _pair(monkeypatch, ORIG_GRAPH, ORIG_STEP)
    batches = native_batches(orig.step)
    decode, prompt = _tables(orig.graph.padded_inputs(batches, (8, 512, 512)))
    assert decode.shape == (8, 512) and prompt.shape == (1, 512)
    assert int((decode[:, DECODE_WIDTH:] == 0).sum()) == 8 * (512 - DECODE_WIDTH)   # bucket padding, fill 0
    assert int((decode[5:] == 0).sum()) == 3 * 512                                     # padded rows, fill 0


def test_pin_keeps_each_region_at_its_native_block_table(mods: types.SimpleNamespace,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    _pin_on(monkeypatch)
    batches = native_batches(mods.step)
    decode, prompt = _tables(mods.graph.padded_inputs(batches, (8, 512, 512)))
    assert torch.equal(decode, batches[0].region.metadata.glm53_index_block_table)
    assert torch.equal(prompt, batches[1].region.metadata.glm53_index_block_table)
    assert decode.shape == (8, DECODE_WIDTH) and prompt.shape == (1, PROMPT_WIDTH)


def test_pin_still_pads_every_other_field_like_combined(mods: types.SimpleNamespace,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    _pin_on(monkeypatch)
    pinned = mods.graph.padded_inputs(native_batches(mods.step), (8, 512, 512))
    for name in (PIN, WIDTHS):
        monkeypatch.delenv(name, raising=False)
    plain = mods.graph.padded_inputs(native_batches(mods.step), (8, 512, 512))
    for a, b in zip(pinned[0].regions, plain[0].regions):
        assert torch.equal(a.metadata.slot_mapping, b.metadata.slot_mapping)
        assert a.metadata.num_actual_tokens == b.metadata.num_actual_tokens
    assert torch.equal(pinned[1], plain[1]) and torch.equal(pinned[2], plain[2])


def test_pin_without_exact_shapes_fails_loudly(mods: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SPLIT, "1")
    monkeypatch.setenv(PIN, "1")
    with pytest.raises(ValueError, match=f"{PIN}=1 needs {EXACT}=1"):
        mods.graph.native_pin_enabled()


@pytest.mark.parametrize("value", ["2", "true", " 1"])
def test_bad_pin_value_fails_loudly(mods: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    _pin_on(monkeypatch)
    monkeypatch.setenv(PIN, value)
    with pytest.raises(ValueError, match=f"{PIN}=.*must be 0 or 1"):
        mods.graph.native_pin_enabled()


# ------------------------------------------------------------------ width-keyed graphs


def test_pin_boot_captures_one_graph_per_width_pair_and_replays_each(mods: types.SimpleNamespace,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    _pin_on(monkeypatch, f"{DECODE_WIDTH}:{PROMPT_WIDTH},{DECODE_WIDTH}:{DECODE_WIDTH}")
    monkeypatch.setenv("VLLM_GAUDI_MIXED_GRAPH_MAX_GRAPHS", "2")
    cache = mods.graph.GraphCache(RowForward(), BLOCK_SIZE, backend=ReplayBackend())
    with mods.graph.boot_capture_context():
        assert cache.run(native_batches(mods.step, chunk_width=PROMPT_WIDTH)) is not None
        assert cache.run(native_batches(mods.step, chunk_width=DECODE_WIDTH)) is not None
    assert cache.counts["captures"] == 2 and len(cache.entries) == 2
    for width, seed in ((PROMPT_WIDTH, 7), (DECODE_WIDTH, 9)):
        batches = native_batches(mods.step, chunk_width=width, seed=seed)
        hidden = cache.run(batches)
        assert hidden is not None and hidden[0, 0].item() == 10 * seed     # replayed with this step's inputs
    assert cache.counts["graph_replays"] == 2 and cache.counts["native_fallbacks"] == 0


def test_pin_live_step_at_an_unwarmed_width_runs_native(mods: types.SimpleNamespace,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    _pin_on(monkeypatch, f"{DECODE_WIDTH}:{PROMPT_WIDTH}")
    forward = RowForward()
    cache = mods.graph.GraphCache(forward, BLOCK_SIZE, backend=ReplayBackend())
    with mods.graph.boot_capture_context():
        cache.run(native_batches(mods.step))
    assert cache.run(native_batches(mods.step, chunk_width=8)) is None
    assert cache.counts["native_fallbacks"] == 1 and cache.counts["fallback_signature"] == 1
    assert cache.counts["lazy_fallbacks"] == 0 and forward.rows == [8, 512] and len(cache.entries) == 1


def test_flag_unset_second_width_at_boot_is_still_one_graph(mods: types.SimpleNamespace) -> None:
    """Combined pads every width to c, so the second step replays the first graph; nothing new is captured."""
    cache = mods.graph.GraphCache(RowForward(), BLOCK_SIZE, backend=ReplayBackend())
    with mods.graph.boot_capture_context():
        cache.run(native_batches(mods.step, chunk_width=PROMPT_WIDTH))
        cache.run(native_batches(mods.step, chunk_width=DECODE_WIDTH))
    assert cache.counts["captures"] == 1 and cache.counts["graph_replays"] == 1


class DummyRunner:
    attn_block_size = block_size = BLOCK_SIZE
    max_model_len = 131072

    def __init__(self) -> None:
        self.input_batch = types.SimpleNamespace(max_num_reqs=16)
        self.scenarios: list[tuple] = []

    def _prepare_dummy_scenario(self, prompt_cfg: tuple, decode_cfg: tuple) -> None:
        self.scenarios.append((prompt_cfg, decode_cfg))


def _warm(mods: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch, runner: DummyRunner) -> None:
    monkeypatch.setattr(mods.graph, "admit_capture", lambda *args, **kwargs: None)
    long_history = types.ModuleType("glm53_r590_long_history")
    long_history.warm_adaptive_tiles_at_boot = lambda model: None
    monkeypatch.setitem(sys.modules, "glm53_r590_long_history", long_history)
    runner.model = None
    mods.graph.warmup(runner)


def test_pin_warmup_runs_one_dummy_per_width_pair_at_native_widths(mods: types.SimpleNamespace,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """Decode dummies sit at wd*128-1 tokens (width wd); the prompt chunk ends at wp*128 tokens (width wp)."""
    _pin_on(monkeypatch, "32:16,32:32")
    monkeypatch.setenv("VLLM_GAUDI_MIXED_GRAPHS", "1")
    monkeypatch.setenv("VLLM_GAUDI_MIXED_GRAPH_MAX_GRAPHS", "2")
    runner = DummyRunner()
    _warm(mods, monkeypatch, runner)
    assert runner.scenarios == [((1, 512, (16 * 128 - 512) // 128), (8, 1, 32)),
                                ((1, 512, (32 * 128 - 512) // 128), (8, 1, 32))]


def test_pin_warmup_default_is_the_bucket_width_for_both_regions(mods: types.SimpleNamespace,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    _pin_on(monkeypatch)
    monkeypatch.setenv("VLLM_GAUDI_MIXED_GRAPHS", "1")
    runner = DummyRunner()
    _warm(mods, monkeypatch, runner)
    assert runner.scenarios == [((1, 512, (512 * 128 - 512) // 128), (8, 1, 512))]


def test_flag_unset_warmup_is_one_dummy_per_bucket(mods: types.SimpleNamespace,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_GAUDI_MIXED_GRAPHS", "1")
    monkeypatch.setenv(WIDTHS, "32:16")                         # ignored without the pin
    runner = DummyRunner()
    _warm(mods, monkeypatch, runner)
    assert runner.scenarios == [((1, 512, (512 * 128 - 512) // 128), (8, 1, 512))]


@pytest.mark.parametrize(("widths", "message"), [
    ("32:24", "power of two"),
    ("1024:16", "exceeds the bucket's 512 context blocks"),
    ("32:2", "cannot hold the 512-token chunk"),
    ("32-16", "decode_width:prompt_width"),
    ("32:16,32:16", "duplicate"),
])
def test_pin_refuses_a_width_pair_native_cannot_produce(mods: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
                                                        widths: str, message: str) -> None:
    _pin_on(monkeypatch, widths)
    monkeypatch.setenv("VLLM_GAUDI_MIXED_GRAPH_MAX_GRAPHS", "4")
    with pytest.raises(ValueError, match=message):
        mods.graph.validate_warmup(DummyRunner())


def test_pin_counts_every_width_graph_against_max_graphs_and_budget(mods: types.SimpleNamespace,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    _pin_on(monkeypatch, "32:16,32:32")
    monkeypatch.setenv("VLLM_GAUDI_MIXED_GRAPH_ESTIMATE_MIB", "1800")
    monkeypatch.setenv("VLLM_GAUDI_MIXED_GRAPH_MAX_GRAPHS", "1")
    with pytest.raises(ValueError, match="2 graphs"):
        mods.graph.validate_warmup(DummyRunner())
    monkeypatch.setenv("VLLM_GAUDI_MIXED_GRAPH_MAX_GRAPHS", "2")
    monkeypatch.setenv("VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB", "3599")
    with pytest.raises(ValueError, match="3600.0 MiB"):
        mods.graph.validate_warmup(DummyRunner())
    monkeypatch.setenv("VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB", "4608")
    assert mods.graph.validate_warmup(DummyRunner()) == ((8, 512, 512),)


def test_pin_refuses_a_prompt_width_whose_dummy_context_misaligns(mods: types.SimpleNamespace,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """Same rule validate_warmup applies to the bucket: the dummy chunk's context must be whole KV pages."""
    _pin_on(monkeypatch, "8:4")
    monkeypatch.setenv("VLLM_GAUDI_MIXED_GRAPH_WARMUP", "8:256:512")
    runner = DummyRunner()
    runner.block_size = 768                       # bucket context 65,280 aligns; 4 * 128 - 256 = 256 does not
    with pytest.raises(ValueError, match="align to runner.block_size 768"):
        mods.graph.validate_warmup(runner)
    monkeypatch.setenv(WIDTHS, "8:8")             # 8 * 128 - 256 = 768 aligns
    assert mods.graph.validate_warmup(runner) == ((8, 256, 512),)


def test_pin_stray_boot_signature_past_max_graphs_stops_the_boot(mods: types.SimpleNamespace,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """A BOOT step at an unlisted width would be one graph too many: refused loudly, never captured."""
    _pin_on(monkeypatch, f"{DECODE_WIDTH}:{PROMPT_WIDTH}")
    monkeypatch.setenv("VLLM_GAUDI_MIXED_GRAPH_MAX_GRAPHS", "1")
    forward = RowForward()
    cache = mods.graph.GraphCache(forward, BLOCK_SIZE, backend=ReplayBackend())
    with mods.graph.boot_capture_context():
        cache.run(native_batches(mods.step))
        with pytest.raises(ValueError, match="budget exhausted before capture"):
            cache.run(native_batches(mods.step, chunk_width=8))
    assert len(cache.entries) == 1 and forward.rows == [8, 512]


def test_pin_logs_the_width_pair_on_capture_and_in_the_counters(mods: types.SimpleNamespace,
                                                                monkeypatch: pytest.MonkeyPatch,
                                                                caplog: pytest.LogCaptureFixture) -> None:
    _pin_on(monkeypatch, f"{DECODE_WIDTH}:{PROMPT_WIDTH}")
    monkeypatch.setenv("VLLM_GAUDI_MIXED_GRAPH_LOG_EVERY", "1")
    caplog.set_level("WARNING", logger="glm53_mixed_graph")
    cache = mods.graph.GraphCache(RowForward(), BLOCK_SIZE, backend=ReplayBackend())
    with mods.graph.boot_capture_context():
        cache.run(native_batches(mods.step))
    cache.run(native_batches(mods.step, seed=3))
    assert cache.counts["captures_w32x16"] == 1 and cache.counts["graph_replays_w32x16"] == 1
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "native_widths=(32, 16)" in text and "'graph_replays_w32x16': 1" in text


def test_flag_unset_logs_no_width_pair(mods: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
                                       caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("VLLM_GAUDI_MIXED_GRAPH_LOG_EVERY", "1")
    caplog.set_level("WARNING", logger="glm53_mixed_graph")
    cache = mods.graph.GraphCache(RowForward(), BLOCK_SIZE, backend=ReplayBackend())
    with mods.graph.boot_capture_context():
        cache.run(native_batches(mods.step))
    cache.run(native_batches(mods.step, seed=3))
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "native_widths" not in text and "_w32x" not in text


# ------------------------------------------------------------------ (b) native logits rows


class LogitsRunner:
    """compute_logits records the row count of every call; logits = the first VOCAB hidden columns."""

    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []
        self.input_batch = types.SimpleNamespace(req_ids=["p", "d0", "d1", "d2", "d3", "d4"])
        self.model = types.SimpleNamespace(compute_logits=self._logits)

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        self.calls.append(hidden.clone())
        return hidden[:, :VOCAB] * 1.0

    def _execute_model_generic(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("prepare must return this bound method, not call it")


def _prepare(mods: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch, runner: LogitsRunner,
             batches: list) -> object:
    monkeypatch.setattr(mods.step, "_validate_runner", lambda runner: None)
    monkeypatch.setattr(mods.step, "_batches", lambda *args, **kwargs: batches)
    forward = RowForward()

    def graph_forward(runner: object, batches: list) -> torch.Tensor:
        plan, tokens, positions = mods.graph.padded_inputs(batches)
        return mods.graph.compact_hidden(mods.graph.region_forwards(forward, plan, tokens, positions), batches)

    monkeypatch.setattr(mods.graph, "forward", graph_forward)
    pd = types.SimpleNamespace(prompt_req_ids=["p"], decode_req_ids=[f"d{i}" for i in range(5)])
    return mods.step.prepare(runner, None, pd, None, None)


def test_combined_computes_logits_once_over_the_joined_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mismatch nativepin removes, shown on the combined build: one call of 5 decode + 1 prompt rows."""
    monkeypatch.setenv("VLLM_GAUDI_MIXED_STEP", "1")
    orig = _pair(monkeypatch, ORIG_GRAPH, ORIG_STEP)
    runner = LogitsRunner()
    _prepare(orig, monkeypatch, runner, native_batches(orig.step))
    assert [call.shape[0] for call in runner.calls] == [5 + 1]


def test_pin_computes_logits_per_region_at_native_rows(mods: types.SimpleNamespace,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    _pin_on(monkeypatch)
    runner, batches = LogitsRunner(), native_batches(mods.step)
    replay = _prepare(mods, monkeypatch, runner, batches)
    assert [call.shape[0] for call in runner.calls] == [8, 1]               # D padded decode rows, then the prompt
    decode_ids = batches[0].tokens.reshape(-1).double()
    assert torch.equal(runner.calls[0][:, 0], decode_ids[[0, 1, 2, 3, 4, 0, 0, 0]])   # pads index row 0
    assert runner.calls[1][0, 0].item() == batches[1].tokens[0, 511].item()
    full, aux, samples, logits = replay.results[batches[0].key]
    assert aux is None and full.shape == (8, 1, HIDDEN)
    assert torch.equal(samples, runner.calls[0]) and torch.equal(logits, runner.calls[0][:, :VOCAB])
    _, _, prompt_samples, prompt_logits = replay.results[batches[1].key]
    assert prompt_samples.shape == (1, HIDDEN) and prompt_logits.shape == (1, VOCAB)


def test_pin_keeps_a_non_final_chunk_at_zero_logits_rows(mods: types.SimpleNamespace,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """An incomplete chunk has no logits rows natively; nativepin makes the same 0-row call native makes."""
    _pin_on(monkeypatch)
    runner, batches = LogitsRunner(), native_batches(mods.step, prompt_logits=())
    replay = _prepare(mods, monkeypatch, runner, batches)
    assert [call.shape[0] for call in runner.calls] == [8, 0]
    assert replay.results[batches[1].key][3].shape == (0, VOCAB)


def test_pin_runs_native_when_a_prompt_logit_row_is_padding(mods: types.SimpleNamespace,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """Native indexes hidden[-1] for a -1 pad (unreachable at logits_rounding = 1); never guess it in the graph."""
    _pin_on(monkeypatch)
    runner = LogitsRunner()
    assert _prepare(mods, monkeypatch, runner, native_batches(mods.step, prompt_logits=(511, -1))) == \
        runner._execute_model_generic
    assert runner.calls == []


# ------------------------------------------------------------------ flag off == combined


def _functions(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found[f"{node.name}.{item.name}"] = ast.dump(item)
            found[node.name] = ast.dump(ast.ClassDef(node.name, node.bases, node.keywords, [], node.decorator_list))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = ast.dump(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            found[f"assign:{ast.unparse(node).split('=')[0].strip()}"] = ast.dump(node)
    return found


# Every definition nativepin changes; each change is a branch on native_pin_enabled() (or code only it reaches).
CHANGED = {
    "glm53_mixed_graph.py": {"pad_field", "GraphCache.record", "GraphCache.run", "GraphCache.capture", "warmup",
                             "validate_warmup"},
    "glm53_mixed_step.py": {"prepare"},
}
ADDED = {
    "glm53_mixed_graph.py": {"assign:PIN_FLAG", "assign:WIDTHS_FLAG", "native_pin_enabled", "native_widths",
                             "check_native_widths", "check_width_pair", "warmup_scenarios", "native_width_pair"},
    "glm53_mixed_step.py": {"_native_logit_rows", "_native_results", "_prompt_logits_padded"},
}


@pytest.mark.parametrize("name", sorted(CHANGED))
def test_only_the_listed_definitions_change(name: str) -> None:
    orig, patched = _functions(LANE / "orig" / name), _functions(LANE / "patched" / name)
    assert set(patched) - set(orig) == ADDED[name] and set(orig) <= set(patched)
    assert {key for key in orig if orig[key] != patched[key]} == CHANGED[name]


def _differential(monkeypatch: pytest.MonkeyPatch, graph_path: Path, step_path: Path) -> dict[str, object]:
    """Everything a flag-off arm observes from one build, on native-shaped c8 inputs (split + exact on, pin off)."""
    monkeypatch.setenv(SPLIT, "1")
    monkeypatch.setenv(EXACT, "1")
    monkeypatch.setenv(WIDTHS, "32:16")
    build = _pair(monkeypatch, graph_path, step_path)
    inputs = build.graph.padded_inputs(native_batches(build.step), (8, 512, 512))
    cache = build.graph.GraphCache(RowForward(), BLOCK_SIZE, backend=ReplayBackend())
    with build.graph.boot_capture_context():
        captured = cache.run(native_batches(build.step))
    replayed = cache.run(native_batches(build.step, seed=4, chunk_width=8))
    runner = LogitsRunner()
    replay = _prepare(build, monkeypatch, runner, native_batches(build.step))
    warm = DummyRunner()
    monkeypatch.setenv("VLLM_GAUDI_MIXED_GRAPHS", "1")
    _warm(build, monkeypatch, warm)
    return {"signature": build.graph.signature(inputs), "tables": [t.tolist() for t in _tables(inputs)],
            "captured": captured.tolist(), "replayed": replayed.tolist(), "counts": dict(cache.counts),
            "logit_calls": [call.tolist() for call in runner.calls],
            "results": {key: [None if t is None else t.tolist() for t in value]
                        for key, value in replay.results.items()},
            "scenarios": warm.scenarios, "validate": build.graph.validate_warmup(warm)}


def test_flag_off_observables_equal_combined(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_GAUDI_MIXED_STEP", "1")
    monkeypatch.setenv("VLLM_GAUDI_MIXED_GRAPH_WARMUP", "8:512:512")
    monkeypatch.setenv("VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB", "65536")
    combined = _differential(monkeypatch, ORIG_GRAPH, ORIG_STEP)
    patched = _differential(monkeypatch, GRAPH_UNDER_TEST, STEP_UNDER_TEST)
    patched["signature"] = (patched["signature"][0].replace(", tail_fuse=None", ""), patched["signature"][1])  # 1481 added the field
    assert patched == combined


def test_long_lattice_is_opt_in_and_keeps_native_pin(monkeypatch, mods):
    key = (16, 1792, 1024)
    assert key not in mods.graph.lattice(128)
    monkeypatch.setenv('VLLM_GAUDI_MIXED_LONG_1792', '1')
    monkeypatch.setenv(PIN, '0')
    with pytest.raises(ValueError, match='native pinning'):
        mods.graph.lattice(128)
    for flag in (PIN, SPLIT, EXACT):
        monkeypatch.setenv(flag, '1')
    assert key in mods.graph.lattice(128)
    with pytest.raises(ValueError, match='128-token blocks'):
        mods.graph.lattice(64)


def test_pinned_production_flags_off_use_same_native_dispatch(monkeypatch, mods):
    for flag in ('VLLM_GAUDI_MIXED_STEP', 'VLLM_GAUDI_MIXED_GRAPHS',
                 'VLLM_GAUDI_MIXED_LONG_1792'):
        monkeypatch.delenv(flag, raising=False)
    native = LogitsRunner()
    assert mods.step.prepare(native, None, None, None, None) == native._execute_model_generic
    assert mods.graph.forward(native, []) is None
    old = _pair(monkeypatch, BASE / 'glm53_mixed_graph.py', BASE / 'glm53_mixed_step.py')
    assert old.step.prepare(native, None, None, None, None) == native._execute_model_generic
    assert old.graph.forward(native, []) is None
    assert old.graph.lattice(128) == mods.graph.lattice(128)


def test_tight_long_key_requires_both_native_tiers(monkeypatch, mods):
    key = (16, 1792, 640)
    monkeypatch.setenv('VLLM_GAUDI_MIXED_LONG_1792', '1')
    for flag in (PIN, SPLIT, EXACT):
        monkeypatch.setenv(flag, '1')
    monkeypatch.setenv('HPU_GLM53_SPARSE_DECODE_BLOCK_TIER_640', '1')
    assert key not in mods.graph.lattice(128)
    with pytest.raises(ValueError, match='power of two'):
        mods.graph.check_width_pair(key, (640, 640), 128, 128)
    monkeypatch.setenv('HPU_GLM53_SPARSE_PREFILL_BLOCK_TIER_640', '1')
    assert key in mods.graph.lattice(128)
    assert (16, 1792, 1024) not in mods.graph.lattice(128)
    mods.graph.check_width_pair(key, (640, 640), 128, 128)
    with pytest.raises(ValueError, match='power of two'):
        mods.graph.check_width_pair((16, 512, 640), (640, 640), 128, 128)
    assert mods.graph.warmup_scenarios(DummyRunner(), (key,)) == [((1, 1792, 626), (15, 1, 640))]
