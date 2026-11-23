"""glm53_hbm_margin: flag off is a no-op that imports nothing; flag on fenced-replays every captured key, prompt and decode, with the full
lattice resident and refuses under the margin or on zero readings."""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import glm53_hbm_margin as margin  # noqa: E402

MIB = 1 << 20
LIMIT = 96_895 * MIB
RESIDENT = 94_646 * MIB


class FakeRuntime:
    """Allocator readings: InUse fixed at the resident, MaxInUse raised by replays until the next reset."""

    def __init__(self, limit: int = LIMIT, in_use: int = RESIDENT, ballast_slow: float = 1.0, pin_over: int = 0) -> None:
        self.limit, self.in_use, self.max_in_use = limit, in_use, in_use
        self.fences = self.resets = 0
        self.now, self.holding, self.ballast_slow = 0.0, False, ballast_slow  # a scripted clock: each replay takes 1 s, x ballast_slow under a ballast
        self.pin_over = pin_over  # bytes a peak pinned at the Limit reads over it (arm d: 0.3-0.7 MiB, the allocator's granularity)

    def clock(self) -> float:
        return self.now

    def tick(self) -> None:
        self.now += self.ballast_slow if self.holding else 1.0

    def fence(self) -> None:
        self.fences += 1

    def reset_peak(self) -> None:
        self.resets += 1
        self.max_in_use = self.in_use

    def stats(self) -> dict[str, int]:
        return {"InUse": self.in_use, "MaxInUse": self.max_in_use, "Limit": self.limit}

    def peak(self, transient_mib: int, floor_mib: int | None = None) -> None:
        """A replay wanting `transient_mib`: fits when free allows; past free it runs pinned at the Limit when the runtime can hand workspace
        back down to `floor_mib` (the 128:8192 key: 2,085 natural, replayed at 1,744 free under the seal ballast), else it is the PT_DEVMEM kill."""
        wanted = self.in_use + transient_mib * MIB
        if wanted <= self.limit:
            self.max_in_use = max(self.max_in_use, wanted)
            return
        if floor_mib is not None and self.in_use + floor_mib * MIB <= self.limit:
            self.max_in_use = self.limit + self.pin_over
            return
        raise RuntimeError("PT_DEVMEM: out of device memory")

    def hold(self, nbytes: int, while_held) -> int:
        """A ballast that must fit in free memory, held across `while_held`; returns the free bytes with it held."""
        if self.in_use + nbytes > self.limit:
            raise RuntimeError(f"the {nbytes // MIB} MiB ballast does not fit")
        self.in_use += nbytes
        self.max_in_use = max(self.max_in_use, self.in_use)
        self.holding = True
        try:
            while_held()
        finally:
            self.in_use -= nbytes
            self.holding = False
        return self.limit - self.in_use - nbytes


@dataclass(frozen=True)
class FakeKey:
    """A key whose replay peaks at `transient` MiB (hand-back floor `floor`), takes a scripted second, and writes `outputs` (bit-drifting under a ballast)."""
    kind: str
    key: str
    transient: int
    floor: int | None = None
    outputs: tuple[bytearray, ...] | None = None
    drift_under_ballast: bool = False
    drift_every_replay: bool = False  # the output differs from one replay to the next, ballast or not (a nondeterministic key)


def _probe(runtime: FakeRuntime, fake: FakeKey) -> "margin.KeyProbe":
    buffers = list(fake.outputs) if fake.outputs is not None else [bytearray(b"same-bits")]
    replays = [0]

    def replay() -> None:
        runtime.peak(fake.transient, fake.floor)
        runtime.tick()
        replays[0] += 1
        if fake.drift_every_replay and buffers:
            buffers[0][:] = f"bits-{replays[0]}".encode()
        elif fake.drift_under_ballast and buffers:
            buffers[0][:] = b"diff-bits" if runtime.holding else b"same-bits"

    return margin.KeyProbe(fake.kind, fake.key, replay, lambda: list(buffers))


@dataclass(frozen=True)
class Lattice:
    """What the fake glm53_mixed_graph holds: its decode keys, each key's replay transient (MiB), the graph counts the seal observes, and the
    prompt key's hand-back floor (None: a prompt replay past free is the PT_DEVMEM kill)."""
    decode: tuple = ((16, 8192), (1, 1024))
    prompt_transient: int = 2135
    decode_transient: int = 410
    graphs: tuple[int, ...] = (24,)
    prompt_floor: int | None = None
    prompt_bits_drift: bool = False  # the prompt key's output differs when replayed under a ballast


def _mixed_graph(runtime: FakeRuntime, lattice: Lattice = Lattice()) -> tuple[types.ModuleType, types.SimpleNamespace, list]:
    """A fake glm53_mixed_graph with one prompt key in the cache and the lattice's decode keys replayed through the runner's dummy scenario."""
    admitted: list[tuple] = []
    counts = list(lattice.graphs)
    decode, prompt_transient, decode_transient, prompt_floor = lattice.decode, lattice.prompt_transient, lattice.decode_transient, lattice.prompt_floor
    module = types.ModuleType("glm53_mixed_graph")
    module.calls = []
    module.seal_boot_memory = lambda runner, charged: module.calls.append(("seal", charged))
    module.admit_capture = lambda bucket, free, required, error="", advice="": admitted.append((bucket, error, advice))
    module._key_setting = lambda bucket: (f"{bucket[1]}:{bucket[2]}", "VLLM_GAUDI_MIXED_PROMPT_KEYS")
    module.decode_keys = lambda: decode
    module.decode_dummy_lengths = lambda rows, width, block, max_len: [width * block] * rows
    module.decode_key_context = lambda key: _recording_context(module, key)
    module._graph_count = lambda: counts[0]
    module.active_keys = []
    module._DECODE_BUILT = ["stale"]

    prompt_out, decode_out = bytearray(b"prompt-out"), bytearray(b"decode-out")

    def replay_prompt(graph: str) -> None:
        module.calls.append(("prompt", graph))
        runtime.peak(prompt_transient, prompt_floor)
        runtime.tick()
        prompt_out[:] = b"prompt-DRIFT" if lattice.prompt_bits_drift and runtime.holding else b"prompt-out"

    def dummy(_request, shape, decode_seq_lengths) -> None:
        module.calls.append(("decode", shape, len(decode_seq_lengths), tuple(module.active_keys)))
        runtime.peak(decode_transient)
        runtime.tick()
        if len(counts) > 1:
            counts.pop(0)

    cache = types.SimpleNamespace(backend=types.SimpleNamespace(replay=replay_prompt),
                                  entries={((0, 1024, 8192), "x"): (None, "graph-1024:8192", prompt_out), ((16, 1, 8192), "x"): (None, "decode-graph", decode_out),
                                           ((1, 1, 1024), "x"): (None, "decode-graph-1", decode_out)})
    runner = types.SimpleNamespace(model=types.SimpleNamespace(_glm53_mixed_graph_cache=cache), attn_block_size=128, max_model_len=1_048_576,
                                   _prepare_dummy_scenario=dummy)
    return module, runner, admitted


class _recording_context:
    def __init__(self, module: types.ModuleType, key: tuple) -> None:
        self.module, self.key = module, key

    def __enter__(self) -> None:
        self.module.active_keys.append(self.key)

    def __exit__(self, *_: object) -> None:
        self.module.active_keys.remove(self.key)


def test_flag_unset_is_off_and_imports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(margin.FLAG, raising=False)
    monkeypatch.delitem(sys.modules, "glm53_mixed_graph", raising=False)
    assert margin.margin_mib() == 0 and margin.enabled() is False
    assert margin.install_from_env() is False
    assert "glm53_mixed_graph" not in sys.modules


@pytest.mark.parametrize("value", ["0", "", " 0 "])
def test_zero_or_blank_is_off(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(margin.FLAG, value)
    assert margin.enabled() is False


@pytest.mark.parametrize("value", ["-1", "abc", "1.5"])
def test_bad_value_refuses_with_the_setting_named(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(margin.FLAG, value)
    with pytest.raises(ValueError, match=margin.FLAG):
        margin.margin_mib()


def test_measure_key_reads_transient_headroom_and_retained_from_the_last_fenced_replay() -> None:
    runtime = FakeRuntime()
    reading = margin.measure_key(_probe(runtime, FakeKey("prompt", "1024:8192", 2136)), runtime)
    assert (reading.kind, reading.key) == ("prompt", "1024:8192")
    assert reading.transient == 2136 * MIB and reading.headroom == (96_895 - 94_646 - 2136) * MIB and reading.retained == 0
    assert runtime.resets == 1 and runtime.fences >= 2
    assert "kind=prompt" in reading.line() and "headroom_MiB=113" in reading.line()


def test_gate_passes_at_or_over_the_margin() -> None:
    reading = margin.KeyReading("decode", "16x8192", start_in_use=0, max_in_use=100 * MIB, end_in_use=0, limit=612 * MIB)
    assert margin.gate_error([reading], 512 * MIB) == ""


def test_gate_refuses_under_the_margin_naming_kind_key_numbers_and_levers() -> None:
    short = margin.KeyReading("decode", "16x8192", start_in_use=RESIDENT, max_in_use=RESIDENT + 2136 * MIB, end_in_use=RESIDENT, limit=LIMIT)
    fine = margin.KeyReading("prompt", "128:1024", start_in_use=RESIDENT, max_in_use=RESIDENT + 354 * MIB, end_in_use=RESIDENT, limit=LIMIT)
    error = margin.gate_error([fine, short], 512 * MIB)
    assert "decode key 16x8192" in error
    assert "113" in error and "512" in error and "399" in error and "2136" in error
    assert "HPU_GLM53_SHARED_KPOOL_CHUNK_BUDGET_MIB" in error and "context and KV stay" in error


def test_gate_refuses_zero_readings_with_the_lattice_settings_named() -> None:
    error = margin.gate_error([], 512 * MIB)
    assert "no captured key" in error
    assert "VLLM_GAUDI_MIXED_PROMPT_KEYS" in error and "VLLM_GAUDI_MIXED_DECODE_WIDTHS" in error and margin.FLAG in error


def test_key_replays_cover_every_prompt_and_decode_key_in_lattice_order() -> None:
    runtime = FakeRuntime()
    module, runner, _ = _mixed_graph(runtime)
    probes = margin.key_replays(module, runner)
    assert [(probe.kind, probe.key) for probe in probes] == [("prompt", "1024:8192"), ("decode", "16x8192"), ("decode", "1x1024")]
    probes[1].replay()
    assert module.calls[-1] == ("decode", (16, 1, 8192), 16, ((16, 8192),))  # the key's own context, rows x width, its dummy lengths
    assert probes[0].outputs() == [bytearray(b"prompt-out")] and probes[2].outputs() == [bytearray(b"decode-out")]  # the keys' static outputs


def test_seal_margin_replays_each_key_twice_and_logs_one_line_per_key() -> None:
    runtime = FakeRuntime()
    module, runner, _ = _mixed_graph(runtime)
    lines: list[str] = []
    readings = margin.seal_margin(margin.key_replays(module, runner), runtime, lines.append)
    assert [(reading.kind, reading.key) for reading in readings] == [("prompt", "1024:8192"), ("decode", "16x8192"), ("decode", "1x1024")]
    assert sum(1 for call in module.calls if call[0] == "prompt") == margin.REPLAYS_PER_KEY
    assert sum(1 for call in module.calls if call[0] == "decode") == 2 * margin.REPLAYS_PER_KEY
    assert len(lines) == 3 and all(line.startswith(margin.LINE) for line in lines)
    assert "kind=decode" in lines[1] and f"headroom_MiB={96_895 - 94_646 - 410}" in lines[1]


def test_install_wraps_the_seal_and_refuses_the_worst_key_through_admit_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime()
    module, runner, admitted = _mixed_graph(runtime, Lattice(prompt_transient=1000, decode_transient=2136))
    monkeypatch.setitem(sys.modules, "glm53_mixed_graph", module)
    monkeypatch.setenv(margin.FLAG, "512")
    monkeypatch.setattr(margin, "_runtime", lambda: runtime)
    monkeypatch.setattr(margin, "_INSTALLED", [])
    assert margin.install_from_env() is True and margin.install_from_env() is True
    module.seal_boot_memory(runner, 7)
    assert module.calls[0] == ("seal", 7)
    assert len(admitted) == 1 and admitted[0][0] == ("hbm_margin",)
    assert "decode key 16x8192" in admitted[0][1] and "113" in admitted[0][1] and margin.FLAG in admitted[0][2]
    assert module._DECODE_BUILT == []  # the dummy replays leave no capture bookkeeping behind


def test_install_refuses_when_a_decode_replay_captures_a_new_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime()
    module, runner, admitted = _mixed_graph(runtime, Lattice(decode=((16, 8192),), decode_transient=10, graphs=(24, 24, 25)))
    monkeypatch.setitem(sys.modules, "glm53_mixed_graph", module)
    monkeypatch.setenv(margin.FLAG, "512")
    monkeypatch.setattr(margin, "_runtime", lambda: runtime)
    monkeypatch.setattr(margin, "_INSTALLED", [])
    assert margin.install_from_env() is True
    module.seal_boot_memory(runner, 0)
    assert "captured" in admitted[0][1] and "16x8192" in admitted[0][1]


def test_install_refuses_an_empty_lattice(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime()
    module, runner, admitted = _mixed_graph(runtime, Lattice(decode=()))
    runner.model._glm53_mixed_graph_cache.entries = {}
    monkeypatch.setitem(sys.modules, "glm53_mixed_graph", module)
    monkeypatch.setenv(margin.FLAG, "512")
    monkeypatch.setattr(margin, "_runtime", lambda: runtime)
    monkeypatch.setattr(margin, "_INSTALLED", [])
    assert margin.install_from_env() is True
    module.seal_boot_memory(runner, 0)
    assert "no captured key" in admitted[0][1]


FREE_1900 = LIMIT - 1900 * MIB  # arm b's 128:8192 picture: 1,918 MiB free, a 2,085 MiB natural workspace, replay pinned at the Limit


def test_ballast_off_keeps_the_free_reading_as_the_headroom() -> None:
    runtime = FakeRuntime(in_use=FREE_1900)
    reading = margin.measure_key(_probe(runtime, FakeKey("prompt", "128:8192", 2085, 1200)), runtime)
    assert reading.headroom == 0 and reading.headroom_free == 0 and reading.under_ballast is None
    assert "ballast_MiB" not in reading.line() and "headroom_free_MiB=0" in reading.line() and "headroom_MiB=0" in reading.line()
    assert "128:8192" in margin.gate_error([reading], 512 * MIB)


def test_ballast_reading_proves_the_margin_on_a_key_whose_workspace_hands_back() -> None:
    runtime = FakeRuntime(in_use=FREE_1900)
    reading = margin.measure_key(_probe(runtime, FakeKey("prompt", "128:8192", 2085, 1200)), runtime, ballast=512 * MIB)
    assert reading.headroom_free == 0  # the free reading alone says -0, as in the gate's c1d-margin and arm b
    assert reading.under_ballast is not None and reading.under_ballast.ballast == 512 * MIB
    assert reading.transient == 1900 * MIB and reading.transient_ballast == 1388 * MIB and reading.compressible == 512 * MIB
    assert reading.headroom == 512 * MIB
    line = reading.line()
    assert "ballast_MiB=512" in line and "transient_ballast_MiB=1388.0" in line and "compressible_MiB=512.0" in line
    assert line.index("headroom_free_MiB=0") < line.index(f"headroom_MiB=512 headroom_B={512 * MIB} replays=")
    assert margin.gate_error([reading], 512 * MIB) == ""


def test_ballast_replay_that_cannot_hand_back_enough_refuses_naming_key_and_ballast() -> None:
    runtime = FakeRuntime(in_use=FREE_1900)
    with pytest.raises(RuntimeError, match=r"prompt key 128:8192 .*512 MiB ballast"):
        margin.measure_key(_probe(runtime, FakeKey("prompt", "128:8192", 2085, 1700)), runtime, ballast=512 * MIB)


def test_ballast_that_does_not_fit_refuses_before_any_replay() -> None:
    runtime = FakeRuntime(in_use=LIMIT - 300 * MIB)
    with pytest.raises(RuntimeError, match="512 MiB ballast"):
        margin.measure_key(_probe(runtime, FakeKey("decode", "16x8192", 10)), runtime, ballast=512 * MIB)


def test_ballast_gate_message_says_the_ballast_was_held() -> None:
    runtime = FakeRuntime(in_use=FREE_1900)
    reading = margin.measure_key(_probe(runtime, FakeKey("prompt", "128:8192", 2085, 1200)), runtime, ballast=256 * MIB)
    error = margin.gate_error([reading], 512 * MIB)
    assert "256 MiB ballast held" in error and "128:8192" in error and "256" in error


def test_install_with_ballast_flag_admits_the_elastic_key_and_logs_the_ballast(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    runtime = FakeRuntime(in_use=FREE_1900)
    module, runner, admitted = _mixed_graph(runtime, Lattice(prompt_transient=2085, prompt_floor=1200, decode_transient=400))
    monkeypatch.setitem(sys.modules, "glm53_mixed_graph", module)
    monkeypatch.setenv(margin.FLAG, "512")
    monkeypatch.setenv(margin.BALLAST_FLAG, "1")
    monkeypatch.setattr(margin, "_runtime", lambda: runtime)
    monkeypatch.setattr(margin, "_INSTALLED", [])
    assert margin.install_from_env() is True
    assert "ballast_MiB=512" in capsys.readouterr().out
    module.seal_boot_memory(runner, 0)
    assert admitted[0][1] == ""  # every key replays with 512 MiB less than free: the margin holds


def test_install_without_ballast_flag_refuses_the_same_lattice(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(in_use=FREE_1900)
    module, runner, admitted = _mixed_graph(runtime, Lattice(prompt_transient=2085, prompt_floor=1200, decode_transient=400))
    monkeypatch.setitem(sys.modules, "glm53_mixed_graph", module)
    monkeypatch.setenv(margin.FLAG, "512")
    monkeypatch.delenv(margin.BALLAST_FLAG, raising=False)
    monkeypatch.setattr(margin, "_runtime", lambda: runtime)
    monkeypatch.setattr(margin, "_INSTALLED", [])
    assert margin.install_from_env() is True
    module.seal_boot_memory(runner, 0)
    assert "prompt key 1024:8192" in admitted[0][1] and "512 MiB short" in admitted[0][1]


def test_ballast_reading_compares_output_bits_and_times_both_modes() -> None:
    runtime = FakeRuntime(in_use=FREE_1900, ballast_slow=1.03)
    reading = margin.measure_key(_probe(runtime, FakeKey("prompt", "128:8192", 2085, 1200)), runtime, ballast=512 * MIB)
    under = reading.under_ballast
    assert under is not None and under.bits_equal is True
    assert under.p50_free_s == 1.0 and round(under.p50_ballast_s, 3) == 1.03 and round(under.ratio, 3) == 1.03
    assert runtime.now >= 2 * margin.TIMED_REPLAYS  # at least N timed replays free and N under the ballast
    line = reading.line()
    assert "bits=equal" in line and "p50_free_ms=1000.0" in line and "p50_ballast_ms=1030.0" in line and "ratio=1.030" in line
    assert margin.gate_error([reading], 512 * MIB) == ""


def test_ballast_bits_that_differ_refuse_naming_the_key() -> None:
    runtime = FakeRuntime(in_use=FREE_1900)
    reading = margin.measure_key(_probe(runtime, FakeKey("prompt", "128:8192", 2085, 1200, drift_under_ballast=True)), runtime, ballast=512 * MIB)
    assert reading.under_ballast is not None and reading.under_ballast.bits_equal is False and "bits=differ" in reading.line()
    error = margin.gate_error([reading], 512 * MIB)
    assert "prompt key 128:8192" in error and "bit for bit" in error and "512 MiB ballast" in error


def test_ballast_slower_than_five_percent_refuses_with_both_times() -> None:
    runtime = FakeRuntime(in_use=FREE_1900, ballast_slow=1.08)
    reading = margin.measure_key(_probe(runtime, FakeKey("prompt", "128:8192", 2085, 1200)), runtime, ballast=512 * MIB)
    assert reading.compressed and "ratio=1.080" in reading.line()
    error = margin.gate_error([reading], 512 * MIB)
    assert "prompt key 128:8192" in error and "1000.0 ms" in error and "1080.0 ms" in error and "5%" in error


def test_ballast_within_five_percent_passes_and_logs_the_ratio() -> None:
    runtime = FakeRuntime(in_use=FREE_1900, ballast_slow=1.05)
    reading = margin.measure_key(_probe(runtime, FakeKey("decode", "16x8192", 400)), runtime, ballast=512 * MIB)
    assert "ratio=1.050" in reading.line() and margin.gate_error([reading], 512 * MIB) == ""


def test_compressed_key_without_output_tensors_refuses() -> None:
    runtime = FakeRuntime(in_use=FREE_1900)
    reading = margin.measure_key(_probe(runtime, FakeKey("prompt", "128:8192", 2085, 1200, outputs=())), runtime, ballast=512 * MIB)
    assert reading.compressed and reading.under_ballast is not None and reading.under_ballast.bits_equal is None and "bits=none" in reading.line()
    error = margin.gate_error([reading], 512 * MIB)
    assert "no output tensor" in error and "128:8192" in error


def test_a_key_that_fits_free_beside_the_ballast_never_ran_compressed_so_its_bits_are_information() -> None:
    """Arm d: every decode key had 2.2 GiB free and bits=none (no static output under restored decode graphs); nothing there to judge."""
    runtime = FakeRuntime(in_use=FREE_1900, ballast_slow=1.08)
    reading = margin.measure_key(_probe(runtime, FakeKey("decode", "2x2048", 60, outputs=())), runtime, ballast=512 * MIB)
    assert not reading.compressed and "bits=none" in reading.line() and "ratio=1.080" in reading.line()
    assert margin.gate_error([reading], 512 * MIB) == ""


def test_the_pinned_peak_reads_a_fraction_over_limit_so_headroom_rounds_to_511_yet_the_full_ballast_held() -> None:
    """Arm d, 128:8192: MaxInUse 96,895.9 against Limit 96,895.2 MiB with 512 MiB held, bits equal, ratio 1.000; refused by 1 MiB of rounding."""
    runtime = FakeRuntime(in_use=FREE_1900, pin_over=int(0.7 * MIB))
    reading = margin.measure_key(_probe(runtime, FakeKey("prompt", "128:8192", 2085, 1200)), runtime, ballast=512 * MIB)
    assert reading.headroom == 512 * MIB - int(0.7 * MIB) and reading.headroom_free == -int(0.7 * MIB)
    line = reading.line()
    assert "headroom_free_MiB=-1" in line and "headroom_MiB=511" in line and f"headroom_B={512 * MIB - int(0.7 * MIB)}" in line
    assert reading.holds(512 * MIB) and margin.gate_error([reading], 512 * MIB) == ""


def test_a_ballast_under_the_margin_proves_nothing_unless_the_free_peak_already_leaves_the_margin() -> None:
    runtime = FakeRuntime(in_use=FREE_1900)
    short = margin.measure_key(_probe(runtime, FakeKey("prompt", "1024:8192", 1600, 1200)), runtime, ballast=256 * MIB)  # 300 free at the peak
    assert not short.holds(512 * MIB) and "256 MiB ballast held (under HPU_GLM53_HBM_MARGIN_MIB=512)" in margin.gate_error([short], 512 * MIB)
    roomy = margin.measure_key(_probe(runtime, FakeKey("prompt", "1024:1024", 1300, 1200)), runtime, ballast=256 * MIB)  # 600 free at the peak
    assert roomy.holds(512 * MIB) and margin.gate_error([roomy], 512 * MIB) == ""


class _KdaLayer:
    """A fake HPUGlm5NextLinearAttention: its kv_cache is the (conv_state, recurrent_state) pools the prefill reads and writes in place."""

    def __init__(self, slots: int, width: int = 1) -> None:
        self.kv_cache = (torch.zeros((slots, 3)), torch.arange(float(slots * width)).reshape(slots * width, 1))


class _MlaLayer:
    def __init__(self) -> None:
        self.kv_cache = (torch.zeros((64, 2)), torch.zeros((64, 2)))


def _model(*layers: object) -> types.SimpleNamespace:
    return types.SimpleNamespace(modules=lambda: iter(layers))


def test_state_rows_map_the_padding_minus_one_the_way_the_kernel_does_and_widen_for_ladder_pools() -> None:
    indices = [torch.tensor([[3, -1, 3]]), None, torch.tensor([5])]  # 2-D cache-group vectors, a missing field, a flat one
    assert margin.state_rows(indices, pool_rows=8, base_rows=8).tolist() == [3, 5, 7]  # remainder(-1, 8) = 7: the padding rows write the last slot
    assert margin.state_rows(indices, pool_rows=16, base_rows=8).tolist() == [6, 7, 10, 11, 14, 15]  # a width-2 transition ladder


def test_kda_pools_are_the_linear_attention_layers_state_tuples_only() -> None:
    kda, mla = _KdaLayer(8), _MlaLayer()
    type(kda).__name__, type(mla).__name__ = "HPUGlm5NextLinearAttention", "HPUGlm5NextMLAAttention"
    pools = margin.kda_pools(_model(mla, kda))
    assert len(pools) == 2 and pools[0] is kda.kv_cache[0] and pools[1] is kda.kv_cache[1]


def test_kda_pools_accept_the_gate_parameters_as_the_layer_mark_and_a_per_engine_nesting() -> None:
    layer = types.SimpleNamespace(A_log=torch.zeros(4), kv_cache=[(torch.zeros((8, 3)), torch.zeros((8, 2)))])  # vLLM's [per virtual engine]
    assert len(margin.kda_pools(_model(layer))) == 2
    assert margin.kda_pools(types.SimpleNamespace()) == []  # a runner.model without modules()


def test_state_indices_are_read_from_the_plan_its_regions_and_their_metadata() -> None:
    """det-d of 16:42 booted the guard module and logged no state rows: glm53_mixed_step.MixedMetadata keeps the slot vectors at
    plan.regions[i].metadata, not on the plan, so the guard saw no indices and stayed silent."""
    region = types.SimpleNamespace(metadata=types.SimpleNamespace(load_indices_tensor=torch.tensor([2]), store_indices_tensor=None))
    plan = types.SimpleNamespace(regions=(region,), non_spec_state_indices_tensor=torch.tensor([5]))
    assert [tensor.tolist() for tensor in margin.state_indices(plan)] == [[5], [2]]
    guard = margin.state_guard([torch.zeros((8, 2))], plan)
    assert guard is not None and guard.rows[0].tolist() == [2, 5]


def test_an_unguarded_prompt_key_says_so_in_its_line() -> None:
    runtime = FakeRuntime(in_use=FREE_1900)
    module, runner, _ = _mixed_graph(runtime, Lattice(prompt_transient=2085, prompt_floor=1200))
    probe = margin.key_replays(module, runner)[0]  # the fake cache entry carries no plan and the fake model no KDA layer
    assert probe.guard is None and probe.unguarded == "state_guard=none pools=0 indices=0"
    assert "state_guard=none pools=0 indices=0" in margin.measure_key(probe, runtime, ballast=512 * MIB).line()


def test_the_guard_restores_the_rows_before_every_replay_so_identical_inputs_see_identical_state() -> None:
    layer = _KdaLayer(8)
    plan = types.SimpleNamespace(load_indices_tensor=torch.tensor([2, -1]), store_indices_tensor=torch.tensor([2, -1]), non_spec_state_indices_tensor=None)
    guard = margin.state_guard([layer.kv_cache[0], layer.kv_cache[1]], plan)
    assert guard is not None and guard.rows[0].tolist() == [2, 7]
    out = bytearray(8)

    def replay() -> None:  # S_new = S_old + 1 on the key's slots, output = the state it saw: the in-place recurrence a KDA prefill does
        rows = layer.kv_cache[1].index_select(0, guard.rows[1])
        layer.kv_cache[1].index_copy_(0, guard.rows[1], rows + 1)
        out[:] = (rows + 1).to(torch.float32).numpy().tobytes()[:8]

    replay(); first = bytes(out); replay(); second = bytes(out)
    assert first != second and guard.changed() == 1  # bare replays drift: the recurrent pool moved, the conv pool did not
    guard.restore(); replay(); restored = bytes(out)
    guard.restore(); replay()
    assert restored == first and bytes(out) == first and guard.changed() == 1


def test_a_plan_without_state_indices_or_a_model_without_kda_pools_has_no_guard() -> None:
    plan = types.SimpleNamespace(load_indices_tensor=None, store_indices_tensor=None, non_spec_state_indices_tensor=None)
    assert margin.state_guard([torch.zeros((8, 2))], plan) is None
    assert margin.state_guard([], types.SimpleNamespace(store_indices_tensor=torch.tensor([1]))) is None


def test_prompt_probes_restore_kda_state_before_each_replay_and_the_line_reports_the_rows() -> None:
    runtime = FakeRuntime(in_use=FREE_1900)
    module, runner, _ = _mixed_graph(runtime, Lattice(prompt_transient=2085, prompt_floor=1200))
    kda = _KdaLayer(8)
    type(kda).__name__ = "HPUGlm5NextLinearAttention"
    runner.model.modules = lambda: iter([kda])
    cache = runner.model._glm53_mixed_graph_cache
    plan = types.SimpleNamespace(load_indices_tensor=torch.tensor([1, -1]), store_indices_tensor=torch.tensor([1, -1]), non_spec_state_indices_tensor=None)
    cache.entries[((0, 1024, 8192), "x")] = ((plan, None, None), "graph-1024:8192", cache.entries[((0, 1024, 8192), "x")][2])
    replay_prompt = cache.backend.replay

    def drifting_replay(graph: str) -> None:  # the prompt output carries the recurrent state it started from
        replay_prompt(graph)
        rows = kda.kv_cache[1].index_select(0, torch.tensor([1, 7]))
        kda.kv_cache[1].index_copy_(0, torch.tensor([1, 7]), rows + 1)
        cache.entries[((0, 1024, 8192), "x")][2][:] = rows.to(torch.float32).numpy().tobytes()[:10]

    cache.backend.replay = drifting_replay
    probe = margin.key_replays(module, runner)[0]
    assert probe.guard is not None
    reading = margin.measure_key(probe, runtime, ballast=512 * MIB)
    assert reading.under_ballast is not None and reading.under_ballast.bits == "equal"  # every replay started from the snapshot
    assert "state_rows=2 state_pools=2 state_changed=1" in reading.line()


def test_free_replays_that_disagree_read_unstable_and_a_ballast_difference_proves_nothing() -> None:
    runtime = FakeRuntime(in_use=FREE_1900)
    reading = margin.measure_key(_probe(runtime, FakeKey("prompt", "1024:1024", 2085, 1200, drift_every_replay=True)), runtime, ballast=512 * MIB)
    under = reading.under_ballast
    assert under is not None and under.free_bits_stable is False and under.bits_equal is False and "bits=unstable" in reading.line()
    assert margin.gate_error([reading], 512 * MIB) == ""


def test_install_with_ballast_refuses_a_prompt_key_whose_bits_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(in_use=FREE_1900)
    module, runner, admitted = _mixed_graph(runtime, Lattice(prompt_transient=2085, prompt_floor=1200, decode_transient=400, prompt_bits_drift=True))
    monkeypatch.setitem(sys.modules, "glm53_mixed_graph", module)
    monkeypatch.setenv(margin.FLAG, "512")
    monkeypatch.setenv(margin.BALLAST_FLAG, "1")
    monkeypatch.setattr(margin, "_runtime", lambda: runtime)
    monkeypatch.setattr(margin, "_INSTALLED", [])
    assert margin.install_from_env() is True
    module.seal_boot_memory(runner, 0)
    assert "prompt key 1024:8192" in admitted[0][1] and "bit for bit" in admitted[0][1]
