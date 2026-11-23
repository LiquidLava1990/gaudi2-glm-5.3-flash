"""HBM margin seal for the GLM-5.3 dual-step boot (diag-hbm, 2026-11-08; every captured key since the c1d review).

HPU_GLM53_HBM_MARGIN_MIB=<MiB>: after glm53_mixed_graph.seal_boot_memory (every prompt key and decode key captured, the seal ballast released,
the boot witness accepted) every captured key is replayed REPLAYS_PER_KEY times with the whole lattice resident, device fenced before and
after, and the allocator's MaxInUse over the last replay is read: prompt keys on their capture inputs (the cache's entries), decode keys through
the runner's dummy scenario under the key's own context, the path that captured them. One GLM53_HBM_MARGIN line per key, with its kind: the
transient the replay allocates, what it keeps, and the headroom left under the allocator Limit at that peak. The boot is refused through
admit_capture (all ranks, graphs-only style) when a key's headroom is under the margin, when a decode replay captures a graph instead of
replaying one, or when no key was replayed at all; the message names the deficit and the levers. Context and KV stay as they are.

Why: run1-run3 (11-08) replayed the 1024:8192 key within -1..113 MiB of the Limit on every rank (resident 92.4-92.9 GiB plus a 1.7-2.3 GiB
replay transient), run1/run2 logged "defragmentation triggered more than 100 times" on every rank, and the c1d lattice relaxes its decode
peak rule only against a seal that measured the decode keys too.

HPU_GLM53_HBM_MARGIN_BALLAST=1 (arm d, 11-08): a key's free reading is the runtime's natural workspace measured against free, which pins at
the Limit whenever free is smaller (128:8192: a fixed 2,085 MiB when free allows, "headroom -0" at 1,730 and 1,918 free, replayed fine at
1,744 free under the seal's 512 MiB ballast) and so cannot say whether the key tolerates the margin. With the flag each key is replayed a
second time with the margin held as a device ballast: a replay that succeeds proves the key runs with that much less, headroom_MiB becomes
Limit - peak-under-ballast + ballast, and the line adds the ballast reading and the compressible workspace; a replay that fails under the
ballast refuses the boot naming the key. The ballast replay must also stay the key: its static output tensors are hashed per rank after
the free replays and after the ballast replays on the same inputs and must match bit for bit, and TIMED_REPLAYS replays are timed in each
mode (p50 logged for every key); a ballast p50 past SPEED_TOLERANCE of the free p50 refuses, naming the key and both times. No compute
changes: serving, graphs, context and KV are byte-identical.

HPU_GLM53_HBM_MARGIN_TIMED_REPLAYS=3|5 (r4g boot budget, 2026-11-09; unset = 5, the proven default): the timed replays per
mode. The seal replays a key 2 + 2 + 2 x TIMED_REPLAYS times (fenced free and ballast, then the timed pairs), 14 at 5: r4e-A2 spent
125 s of its 233 s warmup here (1024:8192 20 s at a 1.49 s p50). At 3 a key replays 10 times (-29%); each p50 is still a median of an
odd count, and every timed replay is still hashed against the free digest. Any other value refuses the boot, naming the setting.

Unset or 0: nothing is imported and the seal is untouched (byte-identical boot). glm53_hbm_margin.pth runs install_from_env() at start.

HPU_GLM53_POOL_LIMIT_MAX_MIB=<MiB> (card headroom plan, 2026-11-22): the PT pool acquires every card's HBM at init, so the only physical
floor is what PT_HPU_POOL_MEM_ACQUIRE_PERC keeps outside it. hpu_worker.initialize_from_config calls enforce_pool_floor() on every rank
before the KV cache is allocated: it prints GLM53_POOL_FLOOR and refuses the boot on all ranks (admit_capture) when the pool Limit is above
the setting, so an ignored or misspelled cap can never silently drop the floor. Unset: nothing is read or printed. The setting must reach
every rank (set it container-wide through the levers file): a rank without it skips the all-rank refusal its peers wait in.

HPU_GLM53_BREADCRUMB_FILE=<path> (crash plan 2026-11-22): breadcrumb() appends one O_DSYNC line per boot step that can take the host down
(pool floor, each prompt key's admission and capture, the capture ballast, the workspace over-ask, warmup end), so a host crash that loses
the docker log tail still leaves the last step on disk. Unset: nothing is opened.
"""
from __future__ import annotations

import datetime
import hashlib
import hmac
import logging
import os
import statistics
import sys
import time
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, NamedTuple, Protocol

if TYPE_CHECKING:  # the seal imports torch only when it runs (the flag set): a flag-off boot imports nothing
    import torch

FLAG = "HPU_GLM53_HBM_MARGIN_MIB"
BALLAST_FLAG = "HPU_GLM53_HBM_MARGIN_BALLAST"  # "1": replay every key a second time with the margin held as a device ballast
# glm53_mixed_graph.hbm_margin_seal_on() knows the installed wrapper by this attribute (True on it); images whose glm53_mixed_graph predates
# the marker read the wrapper's name instead, so the wrapper also carries the name those check: seal_boot_memory_with_margin.
SEAL_MARKER = "glm53_hbm_margin_seal"
SEAL_ALIAS_NAME = "seal_boot_memory_with_margin"
CONTRACT_BUCKET = ("hbm_margin_contract",)
CONTRACT_ADVICE = (f"Bake glm53_mixed_graph and glm53_hbm_margin from one image cut (the graph's HBM_MARGIN_MARKER must equal "
                   f"glm53_hbm_margin.SEAL_MARKER), or unset {FLAG} to boot without the margin seal; then restart all workers.")
LINE = "GLM53_HBM_MARGIN"
MIB = 1 << 20
BALLAST_CHUNK_MIB = 256
REPLAYS_PER_KEY = 2  # the first replay after the lattice can be the launch that hands back workspace; the second is the serving reading
TIMED_REPLAYS = 5  # replays timed in each mode (free, under the ballast) for the p50 comparison: the default of timed_replays()
TIMED_REPLAYS_FLAG = "HPU_GLM53_HBM_MARGIN_TIMED_REPLAYS"
TIMED_REPLAYS_CHOICES = ("3", "5")  # odd, so each p50 is a measured replay
SPEED_TOLERANCE = 1.05  # a ballast p50 past this multiple of the free p50 refuses: compressed workspace must not cost speed
PROMPT_KEYS_FLAG = "VLLM_GAUDI_MIXED_PROMPT_KEYS"
DECODE_WIDTHS_FLAG = "VLLM_GAUDI_MIXED_DECODE_WIDTHS"
# Port of diag-prefill dba517d3 (Fable 4, 2026-11-08): the sparse indexer's caches a prompt replay writes are restored with the KDA pools.
INDEXER_RESTORE_FLAG = "HPU_GLM53_HBM_MARGIN_RESTORE_INDEXER"  # "1" (default): restore the indexer tail and pooled rows before every replay; "0": KDA pools only
INDEXER_RESTORE_DEFAULT = "1"  # on: the guard only, the serving path never reads this flag
INDEXER_CACHES = (("k_cache", "glm53_index_slot_mapping"), ("tail_cache", "glm53_tail_slot_mapping"))  # each indexer cache module and its plan slot field
LEVERS = ("HPU_GLM53_SHARED_KPOOL_CHUNK_BUDGET_MIB bounds a prompt key's scan workspace (2048 in run3; the transient tracks it); "
          "HPU_GLM53_QUERY_SHARD_KEYS added 410 MiB to the 1024:8192 transient (run3 2,135 vs run1 1,725); "
          "HPU_GLM53_DECODE_WIDTH_PAD=1 drops the check-only 2048-width decode keys (597 MiB measured, run3); "
          "HPU_GLM53_DECODE_KPOOL_TILE_CEILING_FOR bounds a listed decode key's scan chunk")
ADVICE = (f"Shrink the key's replay transient or raise free HBM by one of: {LEVERS}; then restart all workers. "
          f"Lower {FLAG} only against a GLM53_HBM_MARGIN reading; context and KV stay as they are.")
POOL_LIMIT_FLAG = "HPU_GLM53_POOL_LIMIT_MAX_MIB"
POOL_ACQUIRE_FLAG = "PT_HPU_POOL_MEM_ACQUIRE_PERC"  # undocumented runtime env: the percent of card HBM the PT pool acquires at init
POOL_LINE = "GLM53_POOL_FLOOR"
POOL_BUCKET = ("pool_floor",)
POOL_ADVICE = (f"Set {POOL_ACQUIRE_FLAG} in the launch env of every worker (unset, the pool takes the whole card), or raise {POOL_LIMIT_FLAG} "
               "only after an hl-smi reading shows the HBM left outside the pool; then restart all workers.")
BREADCRUMB_FLAG = "HPU_GLM53_BREADCRUMB_FILE"  # a file on a persistent bind mount; unset = no breadcrumbs
BREADCRUMB_OPEN_FLAGS = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_DSYNC | os.O_NONBLOCK  # NONBLOCK: a FIFO path never stalls the boot
LOG = logging.getLogger("vllm.r589")
_INSTALLED: list[Callable[..., Any]] = []
_BREADCRUMB_WARNED: list[bool] = []  # one stderr warning per process for an unwritable breadcrumb file


@dataclass(frozen=True)
class KeyProbe:
    """One captured key: how to replay it on its capture inputs, where its static output tensors live after a replay, and the KDA state
    rows the replay reads and writes in place (restored before every replay so identical inputs see identical state)."""
    kind: str
    key: str
    replay: Callable[[], None]
    outputs: Callable[[], list[Any]]
    guard: StateGuard | None = None
    unguarded: str = ""  # why there is no guard ("state_guard=none pools=P indices=I"): a prompt key never replays silently unguarded
    extra_guard: StateGuard | None = None  # the indexer tail/pooled rows (indexer_guard), restored before every replay like the KDA pools


STATE_INDEX_FIELDS = ("load_indices_tensor", "store_indices_tensor", "non_spec_state_indices_tensor")  # the plan's KDA slot vectors
KDA_LAYER_NAME = "LinearAttention"  # HPUGlm5NextLinearAttention: kv_cache = (conv_state, recurrent_state[, transition_factors]) pools


def state_rows(indices: list[Any], pool_rows: int, base_rows: int) -> Any:
    """The rows of a state pool a prompt replay touches: every slot index mapped as the kernel maps it (remainder, so the -1 padding rows
    land on the last slot), widened for a pool of `pool_rows` = k x `base_rows` (a transition ladder holds k rows per slot)."""
    import torch

    flat = torch.cat([tensor.reshape(-1).to("cpu", torch.int64) for tensor in indices if tensor is not None])
    base = torch.unique(torch.remainder(flat, base_rows))
    width = max(1, pool_rows // base_rows)
    return torch.unique(torch.cat([base * width + offset for offset in range(width)]))


def kda_pools(model: Any) -> list[Any]:
    """Every KDA layer's state pools, in module order: the tensors a prefill replay reads and writes outside its graph inputs. A KDA layer
    is HPUGlm5NextLinearAttention (or any module carrying the KDA gate parameters); its kv_cache may nest one list per virtual engine."""
    pools: list[Any] = []
    for module in getattr(model, "modules", lambda: ())():
        cache = getattr(module, "kv_cache", None)
        if not isinstance(cache, (tuple, list)) or not (KDA_LAYER_NAME in type(module).__name__ or hasattr(module, "A_log")):
            continue
        flat = [item for entry in cache for item in (entry if isinstance(entry, (tuple, list)) else (entry,))]
        pools.extend(tensor for tensor in flat if hasattr(tensor, "index_select") and tensor.ndim >= 2 and tensor.shape[0] > 0)
    return pools


def state_indices(plan: Any) -> list[Any]:
    """The KDA slot vectors a padded plan carries: on the plan itself, on each region, or on each region's attention metadata
    (glm53_mixed_step.MixedMetadata holds them at plan.regions[i].metadata)."""
    holders = [plan]
    for region in getattr(plan, "regions", ()) or ():
        holders.extend((region, getattr(region, "metadata", None)))
    return [tensor for holder in holders if holder is not None for field in STATE_INDEX_FIELDS
            for tensor in (getattr(holder, field, None),) if tensor is not None]


@dataclass(frozen=True)
class StateGuard:
    """Host snapshots of the state rows a key's replay mutates: restore() puts them back before a replay, changed() counts pools that moved."""
    pools: tuple[Any, ...]
    rows: tuple[Any, ...]
    snapshots: tuple[Any, ...]

    def restore(self) -> None:
        for pool, rows, snapshot in zip(self.pools, self.rows, self.snapshots):
            pool.index_copy_(0, rows.to(pool.device), snapshot.to(pool.device))

    def changed(self) -> int:
        import torch

        return sum(not torch.equal(pool.index_select(0, rows.to(pool.device)).to("cpu"), snapshot)
                   for pool, rows, snapshot in zip(self.pools, self.rows, self.snapshots))

    def note(self) -> str:
        return f"state_rows={len(self.rows[0])} state_pools={len(self.pools)}"


def state_guard(pools: list[Any], plan: Any) -> StateGuard | None:
    """A guard over `pools` for the slots named by `plan`'s state index fields; None when the plan names none or there is no pool."""
    indices = state_indices(plan)
    if not pools or not indices:
        return None
    base_rows = min(int(pool.shape[0]) for pool in pools)
    rows = tuple(state_rows(indices, int(pool.shape[0]), base_rows) for pool in pools)
    snapshots = tuple(pool.index_select(0, pool_rows.to(pool.device)).to("cpu") for pool, pool_rows in zip(pools, rows))
    return StateGuard(tuple(pools), rows, snapshots)


class Runtime(Protocol):
    """The allocator reads the seal needs; HtorchRuntime is the device one, tests script their own."""

    def fence(self) -> None: ...
    def reset_peak(self) -> None: ...
    def stats(self) -> dict[str, int]: ...
    def hold(self, nbytes: int, while_held: Callable[[], None]) -> int: ...
    def clock(self) -> float: ...


@dataclass(frozen=True)
class BallastReading:
    """The same key replayed with `ballast` bytes held on the device: its peak and end InUse (ballast included), whether its output tensors
    match the free replay's bit for bit (None: the key exposes no output tensor), and the p50 replay time in each mode."""
    ballast: int
    max_in_use: int
    end_in_use: int
    bits_equal: bool | None
    p50_free_s: float
    p50_ballast_s: float
    free_bits_stable: bool = True  # the free timed replays agreed with each other bit for bit; when not, a ballast difference proves nothing

    @property
    def ratio(self) -> float:
        return self.p50_ballast_s / self.p50_free_s if self.p50_free_s > 0 else float("inf")

    @property
    def bits(self) -> str:
        if self.bits_equal is None:
            return "none"
        if not self.free_bits_stable:
            return "unstable"
        return "equal" if self.bits_equal else "differ"


@dataclass(frozen=True)
class KeyReading:
    kind: str
    key: str
    start_in_use: int
    max_in_use: int
    end_in_use: int
    limit: int
    under_ballast: BallastReading | None = None
    state_note: str = ""  # "state_rows=N state_pools=M state_changed=K": the KDA rows a replay moved, restored before every replay

    @property
    def transient(self) -> int:
        return max(0, self.max_in_use - self.start_in_use)

    @property
    def headroom_free(self) -> int:
        """Limit minus the free replay's peak: the natural workspace against free, pinned at 0 whenever free is smaller than it."""
        return self.limit - self.max_in_use

    @property
    def headroom(self) -> int:
        """Information: the free reading, or with a ballast held, Limit minus that peak plus the ballast. A peak pinned at Limit reads a few
        hundred KiB over it (arm d: 96,895.9 against 96,895.2 MiB), so under a ballast this is the ballast minus that slack, never a deficit."""
        if self.under_ballast is None:
            return self.headroom_free
        return self.limit - self.under_ballast.max_in_use + self.under_ballast.ballast

    @property
    def compressed(self) -> bool:
        """The key ran in compressed-workspace mode under the ballast: free alone could not have held the ballast beside its natural peak."""
        return self.under_ballast is not None and self.headroom_free < self.under_ballast.ballast

    def admit_value(self, margin: int) -> tuple[str, int]:
        """(basis, bytes) the gate compares against `margin`. Free: the peak's headroom under Limit. Under a ballast, when free alone falls
        short: the ballast the key replayed with held (the replay itself is the proof; bits and speed are judged apart, in
        _ballast_identity_error). Not headroom_MiB, which under a ballast is the ballast minus the few hundred KiB a pinned peak reads over Limit."""
        if self.headroom_free >= margin or self.under_ballast is None:
            return "free", self.headroom_free
        return "ballast", self.under_ballast.ballast

    def holds(self, margin: int) -> bool:
        """What the gate judges: the compared value (admit_value) is at least `margin`."""
        return self.admit_value(margin)[1] >= margin

    @property
    def transient_ballast(self) -> int:
        return 0 if self.under_ballast is None else max(0, self.under_ballast.max_in_use - self.start_in_use - self.under_ballast.ballast)

    @property
    def compressible(self) -> int:
        """Workspace the runtime handed back under the ballast: the free transient minus the one measured with the ballast held."""
        return 0 if self.under_ballast is None else max(0, self.transient - self.transient_ballast)

    @property
    def retained(self) -> int:
        return self.end_in_use - self.start_in_use

    @property
    def label(self) -> str:
        return f"{self.kind} key {self.key}"

    def line(self, margin: int | None = None) -> str:
        """The per-key log line; with `margin` it ends in the gate's own figures: the basis, the bytes compared, the margin, the verdict."""
        verdict = ""
        if margin is not None:
            basis, value = self.admit_value(margin)
            verdict = f" admit_by={basis} admit_B={value} margin_B={margin} admitted={int(value >= margin)}"
        ballast = ""
        if self.under_ballast is not None:
            under = self.under_ballast
            ballast = (f"ballast_MiB={under.ballast / MIB:.0f} transient_ballast_MiB={self.transient_ballast / MIB:.1f} "
                       f"compressible_MiB={self.compressible / MIB:.1f} bits={under.bits} p50_free_ms={under.p50_free_s * 1000:.1f} "
                       f"p50_ballast_ms={under.p50_ballast_s * 1000:.1f} ratio={under.ratio:.3f} ")
        return (f"{LINE} kind={self.kind} key={self.key} start_in_use_MiB={self.start_in_use / MIB:.1f} max_in_use_MiB={self.max_in_use / MIB:.1f} "
                f"end_in_use_MiB={self.end_in_use / MIB:.1f} limit_MiB={self.limit / MIB:.1f} transient_MiB={self.transient / MIB:.1f} "
                f"retained_MiB={self.retained / MIB:.1f} {self.state_note}{' ' if self.state_note else ''}"
                f"headroom_free_MiB={self.headroom_free / MIB:.0f} {ballast}"
                f"headroom_MiB={self.headroom / MIB:.0f} headroom_B={self.headroom} replays={REPLAYS_PER_KEY}{verdict}")


def margin_mib() -> int:
    """The margin in MiB; 0 when unset or blank. A value that is not a non-negative integer refuses the boot, naming the setting."""
    raw = os.environ.get(FLAG, "").strip()
    if raw == "":
        return 0
    if not raw.isdigit():
        raise ValueError(f"{FLAG}={raw!r}: set the MiB of allocator headroom every captured key must leave at its replay peak "
                         "(a non-negative integer), or 0 / unset for no margin seal")
    return int(raw)


def timed_replays() -> int:
    """TIMED_REPLAYS_FLAG's count, TIMED_REPLAYS when unset; any other value refuses the boot, naming the setting."""
    raw = os.environ.get(TIMED_REPLAYS_FLAG, "").strip()
    if raw == "":
        return TIMED_REPLAYS
    if raw not in TIMED_REPLAYS_CHOICES:
        raise ValueError(f"{TIMED_REPLAYS_FLAG}={raw!r}: set the timed replays per mode to one of {', '.join(TIMED_REPLAYS_CHOICES)} "
                         f"(an odd count, so each p50 is a measured replay), or unset it for {TIMED_REPLAYS}")
    return int(raw)


def enabled() -> bool:
    return margin_mib() > 0


def ballast_on() -> bool:
    return os.environ.get(BALLAST_FLAG, "").strip() == "1"


class HtorchRuntime:
    """Device fences and allocator stats through habana_frameworks.torch; imported only when the seal runs."""

    def __init__(self) -> None:
        import habana_frameworks.torch as htorch

        self.htorch = htorch

    def fence(self) -> None:
        self.htorch.core.mark_step()
        self.htorch.hpu.synchronize()

    def reset_peak(self) -> None:
        self.htorch.hpu.reset_peak_memory_stats()

    def stats(self) -> dict[str, int]:
        stats = self.htorch.hpu.memory_stats()
        return {name: int(stats[name]) for name in ("InUse", "MaxInUse", "Limit")}

    def clock(self) -> float:
        return time.perf_counter()

    def hold(self, nbytes: int, while_held: Callable[[], None]) -> int:
        """Materialize `nbytes` of ones on the device (a byte read back proves it), run `while_held` under them, free them; the free HBM held."""
        import torch

        mib = nbytes // MIB
        sizes = [BALLAST_CHUNK_MIB] * (mib // BALLAST_CHUNK_MIB) + [mib % BALLAST_CHUNK_MIB]
        held = [torch.ones(size * MIB, dtype=torch.uint8, device="hpu") for size in sizes if size]
        self.fence()
        if any(int(tensor[-1].item()) != 1 for tensor in held):
            raise RuntimeError(f"the {mib} MiB ballast does not read back, so it never reached the device; run the seal outside compile-only mode")
        free = int(self.htorch.hpu.mem_get_info()[0])
        try:
            while_held()
        finally:
            del held
            self.fence()
        return free


def _runtime() -> Runtime:
    return HtorchRuntime()


def _fenced_replay(replay: Callable[[], None], runtime: Runtime) -> tuple[int, int, int, int]:
    """Replay REPLAYS_PER_KEY times; (start InUse, MaxInUse, end InUse, Limit) of the last one, fenced on both sides so the device sets it."""
    for _ in range(REPLAYS_PER_KEY - 1):
        replay()
    runtime.fence()
    start = runtime.stats()
    runtime.reset_peak()
    replay()
    runtime.fence()
    end = runtime.stats()
    return start["InUse"], end["MaxInUse"], end["InUse"], end["Limit"]


def _tensor_bytes(tensor: Any) -> bytes:
    """The exact bytes of a device tensor (any dtype): copied to the host, contiguous, viewed as uint8."""
    import torch

    host = tensor.detach().to("cpu").contiguous()
    return host.view(torch.uint8).numpy().tobytes()


def digest(outputs: list[Any]) -> str | None:
    """sha256 over every output tensor's bytes in order; None when the key exposes no output tensor to compare."""
    if not outputs:
        return None
    sha = hashlib.sha256()
    for output in outputs:
        sha.update(bytes(output) if isinstance(output, (bytes, bytearray)) else _tensor_bytes(output))
    return sha.hexdigest()


class TimedReplays(NamedTuple):
    """TIMED_REPLAYS replays of one key in one mode: the median wall time and the output digest after each."""
    p50_s: float
    digests: list[str | None]

    @property
    def stable(self) -> bool:
        return len(set(self.digests)) <= 1


def _timed_replays(probe: KeyProbe, runtime: Runtime) -> TimedReplays:
    """Each replay fenced on both sides so the device, not the host queue, is what is timed; the outputs hashed after every one."""
    times = []
    digests: list[str | None] = []
    for _ in range(timed_replays()):
        runtime.fence()
        started = runtime.clock()
        probe.replay()
        runtime.fence()
        times.append(runtime.clock() - started)
        digests.append(digest(probe.outputs()))
    return TimedReplays(statistics.median(times), digests)


def _replay_under_ballast(probe: KeyProbe, runtime: Runtime, ballast: int, free: TimedReplays) -> BallastReading:
    """The key replayed with `ballast` bytes held: its peak, its timed replays and its output digest against the free replays'. A failure
    there is the key refusing to run with that much less free, and names it."""
    seen: list[tuple[int, int, TimedReplays]] = []

    def under_ballast() -> None:
        _, max_in_use, end_in_use, _ = _fenced_replay(probe.replay, runtime)
        seen.append((max_in_use, end_in_use, _timed_replays(probe, runtime)))

    try:
        runtime.hold(ballast, under_ballast)
    except Exception as exc:
        raise RuntimeError(f"{probe.kind} key {probe.key} did not replay with a {ballast // MIB} MiB ballast held ({type(exc).__name__}: {exc}): "
                           f"its launch needs more than free minus {FLAG}, so a {ballast // MIB} MiB allocation at serving would kill it. {ADVICE}") from exc
    max_in_use, end_in_use, held = seen[0]
    digest_free, digest_ballast = free.digests[0], held.digests[0]
    bits_equal = None if digest_free is None or digest_ballast is None else hmac.compare_digest(digest_free, digest_ballast)
    return BallastReading(ballast, max_in_use, end_in_use, bits_equal, free.p50_s, held.p50_s, free.stable)


def measure_key(probe: KeyProbe, runtime: Runtime, ballast: int = 0) -> KeyReading:
    """The free reading, then with `ballast` > 0 the same key timed, hashed and replayed under that much held: the proof it can spare the margin
    without changing its bits or its speed."""
    start, max_in_use, end_in_use, limit = _fenced_replay(probe.replay, runtime)
    note = f"{probe.guard.note()} state_changed={probe.guard.changed()}" if probe.guard is not None else probe.unguarded
    if probe.kind == "prompt":
        note = f"{note} {indexer_note(probe.extra_guard)}".strip()
    under = None
    if ballast > 0:
        under = _replay_under_ballast(probe, runtime, ballast, _timed_replays(probe, runtime))
    return KeyReading(probe.kind, probe.key, start, max_in_use, end_in_use, limit, under, note)


def gate_error(readings: list[KeyReading], margin: int) -> str:
    """'' when at least one key was measured and every key leaves `margin` bytes at its peak; else why not, with the numbers and the levers."""
    if not readings:
        return (f"no captured key was replayed under the margin seal, so {FLAG}={margin // MIB} has nothing to prove: the lattice holds no "
                f"prompt key ({PROMPT_KEYS_FLAG}) and no decode key ({DECODE_WIDTHS_FLAG}); list the keys the build serves, or unset {FLAG} "
                "on a boot that captures none")
    for reading in readings:
        error = _ballast_identity_error(reading)
        if error:
            return error
    short = [reading for reading in readings if not reading.holds(margin)]
    if not short:
        return ""
    worst = min(short, key=lambda reading: reading.headroom)
    deficit = margin - worst.headroom
    held = "" if worst.under_ballast is None else f" with a {worst.under_ballast.ballast // MIB} MiB ballast held (under {FLAG}={margin // MIB})"
    return (f"{worst.label} leaves {worst.headroom / MIB:.0f} MiB under the allocator Limit at its replay peak{held}, "
            f"{deficit / MIB:.0f} MiB short of {FLAG}={margin // MIB}: its replay allocates a {worst.transient / MIB:.0f} MiB transient over "
            f"{worst.start_in_use / MIB:.0f} MiB resident (Limit {worst.limit / MIB:.0f}); a serving step at that peak runs on workspace "
            f"hand-back and defragmentation, and any allocation past it kills the worker on PT_DEVMEM. {ADVICE}")


def _ballast_identity_error(reading: KeyReading) -> str:
    """'' unless the key ran compressed under the ballast and there changed its output bits or slowed past SPEED_TOLERANCE against free.
    A key whose free peak leaves the whole ballast under Limit never ran compressed (arm d: every decode key, 2.2 GiB free), so its bits
    and speed are information; free replays that disagree with each other (bits=unstable) leave nothing a ballast difference could prove."""
    under = reading.under_ballast
    if under is None or not reading.compressed:
        return ""
    held = f"{under.ballast // MIB} MiB ballast"
    if under.bits_equal is None:
        return (f"{reading.label} exposes no output tensor, so its replay under the {held} cannot be checked bit for bit against its free replay; "
                f"the graph cache entry for the key carries no static output. {ADVICE}")
    if not under.free_bits_stable:
        return ""
    if not under.bits_equal:
        return (f"{reading.label} did not reproduce its free replay bit for bit under the {held} on identical inputs (sha256 of its output "
                f"tensors differs): the compressed-workspace launch is not the same computation, so the margin cannot be taken from it. {ADVICE}")
    if round(under.ratio, 3) > SPEED_TOLERANCE:  # the logged three decimals decide, not float noise at the boundary
        return (f"{reading.label} replays {under.ratio:.3f}x slower under the {held} (p50 {under.p50_ballast_s * 1000:.1f} ms vs "
                f"{under.p50_free_s * 1000:.1f} ms free over {timed_replays()} replays each), past the {(SPEED_TOLERANCE - 1) * 100:.0f}% allowed: "
                f"the key would serve in that compressed-workspace mode under load. {ADVICE}")
    return ""


def _entry_outputs(cache: Any, matches: Callable[[tuple], bool]) -> list[Any]:
    """The static output tensors of every cache entry whose bucket `matches` (the graph writes the same tensor on every replay)."""
    return [output for (bucket, _), (_, _, output) in getattr(cache, "entries", {}).items() if matches(bucket) and output is not None]


def _prompt_probes(mixed_graph: Any, runner: Any, cache: Any) -> list[KeyProbe]:
    """Every prompt key (bucket[0] == 0) replayed on its capture inputs, in capture order; its output is the entry's static output. The KDA
    state rows its plan names are snapshotted here and restored before every replay: the prefill reads and writes them in place
    (glm5_next.py: conv_state through hpu_causal_conv1d_fn, recurrent_state index_select / index_copy_), so bare replays drift."""
    probes = []
    pools = kda_pools(getattr(runner, "model", None))
    for (bucket, _), (static, graph, _) in getattr(cache, "entries", {}).items():
        if bucket[0] != 0:
            continue
        plan = static[0] if isinstance(static, (tuple, list)) and static else None
        guard = state_guard(pools, plan) if plan is not None else None
        unguarded = "" if guard is not None else f"state_guard=none pools={len(pools)} indices={len(state_indices(plan)) if plan is not None else 0}"
        extra = indexer_guard(getattr(runner, "model", None), plan) if plan is not None and indexer_restore_enabled() else None
        probes.append(KeyProbe("prompt", mixed_graph._key_setting(bucket)[0],
                               _guarded(guard, _guarded(extra, lambda graph=graph: cache.backend.replay(graph))),
                               lambda bucket=bucket: _entry_outputs(cache, lambda other: other == bucket), guard, unguarded, extra))
    return probes


def cache_tensors(cache: object) -> list[torch.Tensor]:
    """Every non-empty tensor (ndim >= 2) inside a module's kv_cache: a tensor, a list per virtual engine, or tuples of pools."""
    if isinstance(cache, (tuple, list)):
        return [tensor for item in cache for tensor in cache_tensors(item)]
    return [cache] if hasattr(cache, "index_select") and cache.ndim >= 2 and cache.shape[0] > 0 else []


def _flat_view(tensor: Any) -> Any:
    """The cache as (rows, width) ([blocks, block_size, width] caches flatten to slot rows); None when it cannot be viewed that way."""
    try:
        return tensor.view(-1, int(tensor.shape[-1]))
    except RuntimeError:
        return None


def indexer_restore_enabled() -> bool:
    """HPU_GLM53_HBM_MARGIN_RESTORE_INDEXER: "1" (default) restores the indexer tail and pooled rows before every replay; "0" leaves them as
    the KDA-only guard did; anything else refuses the boot."""
    raw = os.environ.get(INDEXER_RESTORE_FLAG)
    value = INDEXER_RESTORE_DEFAULT if raw is None else raw.strip()
    if value not in ("0", "1"):
        raise ValueError(f"{INDEXER_RESTORE_FLAG}={value!r}: set 1 to restore the indexer tail and pooled rows before every replay, "
                         f"0 for the KDA pools only")
    return value == "1"


def indexer_field(name: str) -> str | None:
    """The plan metadata field addressing an indexer cache module by its name (...indexer.k_cache -> glm53_index_slot_mapping); None else."""
    return next((field for attr, field in INDEXER_CACHES if name.endswith(f".{attr}")), None)


def indexer_rows(plan: object, field: str, cache_rows: int) -> torch.Tensor | None:
    """The rows of an indexer cache a replay touches: the field's slots in the plan's regions that address it (the -1 fills dropped) plus
    the pad row (the last row, where non-completing and superseded tokens write zeros); None when no region carries the field."""
    import torch

    mappings = [getattr(getattr(region, "metadata", None), field, None) for region in getattr(plan, "regions", ()) or ()]
    flat = [mapping.reshape(-1).to("cpu", torch.int64) for mapping in mappings if mapping is not None]
    if not flat:
        return None
    slots = torch.cat(flat)
    slots = slots[(slots >= 0) & (slots < cache_rows)]
    return torch.unique(torch.cat([slots, torch.tensor([cache_rows - 1], dtype=torch.int64)]))


def indexer_guard(model: object, plan: object) -> StateGuard | None:
    """A StateGuard over the sparse indexer caches' touched rows, viewed as (slots, width): the pooled k_cache at glm53_index_slot_mapping
    and the raw tail_cache at glm53_tail_slot_mapping. hpu_glm5next_sparse_indexer._update_pooled_cache reads the tail rows (old_tail) it
    then overwrites with the chunk's trailing tokens, so a bare second replay of a pool-misaligned chunk compresses a different straddling
    pool and the key reads bits=unstable; restored before every replay, the replay is idempotent again. None when the model has no indexer
    cache module or the plan carries neither field."""
    pools, rows, snapshots = [], [], []
    for name, module in getattr(model, "named_modules", lambda: ())():
        field = indexer_field(name)
        for tensor in cache_tensors(getattr(module, "kv_cache", None)) if field is not None else ():
            view = _flat_view(tensor)
            touched = None if view is None else indexer_rows(plan, field, int(view.shape[0]))
            if touched is None:
                continue
            pools.append(view)
            rows.append(touched)
            snapshots.append(view.index_select(0, touched.to(view.device)).to("cpu"))
    return StateGuard(tuple(pools), tuple(rows), tuple(snapshots)) if pools else None


def indexer_note(guard: StateGuard | None) -> str:
    """The margin line's indexer note: the restore off, no cache to guard, or the caches and rows restored and how many caches the last
    replay moved (read after a replay: the evidence that the replay writes them)."""
    if not indexer_restore_enabled():
        return "indexer_restore=0"
    if guard is None:
        return "indexer_guard=none"
    return f"indexer_caches={len(guard.pools)} indexer_rows={sum(int(rows.numel()) for rows in guard.rows)} indexer_changed={guard.changed()}"


def _guarded(guard: StateGuard | None, replay: Callable[[], None]) -> Callable[[], None]:
    if guard is None:
        return replay

    def restored_replay() -> None:
        guard.restore()
        replay()

    return restored_replay


def _decode_probes(mixed_graph: Any, runner: Any, cache: Any) -> list[KeyProbe]:
    """Every decode key replayed the way it was captured: the runner's dummy scenario under the key's own context (glm53_mixed_graph._run_decode_capture);
    its outputs are the cache entries of rows x width."""
    probes: list[KeyProbe] = []
    for rows, width in mixed_graph.decode_keys():
        lengths = mixed_graph.decode_dummy_lengths(rows, width, runner.attn_block_size, runner.max_model_len)

        def replay(rows: int = rows, width: int = width, lengths: list[int] = lengths) -> None:
            with mixed_graph.decode_key_context((rows, width)):
                runner._prepare_dummy_scenario(None, (rows, 1, width), decode_seq_lengths=lengths)

        def outputs(rows: int = rows, width: int = width) -> list[Any]:
            return _entry_outputs(cache, lambda bucket: bucket[0] == rows and bucket[-1] == width)

        probes.append(KeyProbe("decode", f"{rows}x{width}", replay, outputs))
    return probes


def key_replays(mixed_graph: Any, runner: Any) -> list[KeyProbe]:
    """A probe per captured key: prompt keys first, then the decode lattice in its boot order."""
    cache = getattr(runner.model, "_glm53_mixed_graph_cache", None)
    return _prompt_probes(mixed_graph, runner, cache) + _decode_probes(mixed_graph, runner, cache)


def seal_margin(probes: list[KeyProbe], runtime: Runtime, log: Callable[[str], Any], ballast: int = 0,
                margin: int | None = None) -> list[KeyReading]:
    """Measure every key, log one line each (with `margin`, ending in the value the gate compares), return the readings. A replay that
    raises propagates: the seal refuses, not serves."""
    readings = []
    for probe in probes:
        reading = measure_key(probe, runtime, ballast)
        log(reading.line(margin))
        readings.append(reading)
    return readings


def _capture_during_seal_error(mixed_graph: Any, graphs_before: int | None, readings: list[KeyReading]) -> str:
    """A decode dummy that captured instead of replaying grew the graph count: that key was never a warmed graph, so the seal refuses."""
    graphs_after = mixed_graph._graph_count()
    if graphs_before is None or graphs_after is None or graphs_after == graphs_before:
        return ""
    keys = ", ".join(reading.key for reading in readings if reading.kind == "decode") or "none"
    return (f"the margin seal's decode replays captured {graphs_after - graphs_before} new graph(s) (graph count {graphs_before} -> {graphs_after}) "
            f"over decode keys {keys}; a key captured here was not warmed at boot, so serving would capture silently. Check GLM53_BOOT_WITNESS "
            f"and {DECODE_WIDTHS_FLAG}, and restart all workers.")


def _measure_lattice(mixed_graph: Any, runner: Any, margin: int, ballast: int) -> None:
    """After the original seal: replay every captured key under the margin (and the ballast when on) and admit or refuse the boot on every rank."""
    graphs_before = mixed_graph._graph_count()
    readings = seal_margin(key_replays(mixed_graph, runner), _runtime(), LOG.warning, ballast, margin=margin)
    getattr(mixed_graph, "_DECODE_BUILT", []).clear()  # the dummy replays' capture bookkeeping, meaningless after the lattice
    error = _capture_during_seal_error(mixed_graph, graphs_before, readings) or gate_error(readings, margin)
    mixed_graph.admit_capture(("hbm_margin",), 0, 0, error, ADVICE)


def seal_contract_error(mixed_graph: types.ModuleType) -> str:
    """Why glm53_mixed_graph cannot see the installed seal ('' when it can): its marker name differs from SEAL_MARKER, or its own
    hbm_margin_seal_on() reads False with the wrapper in place. A graph without that check (before c1d r3) has no contract to break."""
    graph_marker = getattr(mixed_graph, "HBM_MARGIN_MARKER", SEAL_MARKER)
    if graph_marker != SEAL_MARKER:
        return (f"glm53_mixed_graph.HBM_MARGIN_MARKER is {graph_marker!r} but glm53_hbm_margin marks its wrapper {SEAL_MARKER!r}, so the graph "
                f"cannot see the margin seal")
    if callable(getattr(mixed_graph, "hbm_margin_seal_on", None)) and mixed_graph.hbm_margin_seal_on() is not True:
        return (f"glm53_mixed_graph.hbm_margin_seal_on() reads False with the margin seal installed ({FLAG}={margin_mib()}), so its measured "
                f"charge and capture-margin peak check would stay off")
    return ""


def _wrap_seal(mixed_graph: Any, margin: int, ballast: int, contract: list[str]) -> Callable[..., Any]:
    """The seal plus the margin replays; a broken seal contract (``contract`` holds its error) refuses the boot after the original seal."""
    original = mixed_graph.seal_boot_memory

    def _seal_boot_memory_with_margin(runner: object, charged: int) -> None:
        original(runner, charged)
        if contract:
            mixed_graph.admit_capture(CONTRACT_BUCKET, 0, 0, contract[0], CONTRACT_ADVICE)
            return
        _measure_lattice(mixed_graph, runner, margin, ballast)

    setattr(_seal_boot_memory_with_margin, SEAL_MARKER, True)
    _seal_boot_memory_with_margin.__name__ = SEAL_ALIAS_NAME
    return _seal_boot_memory_with_margin


def pool_limit_max_mib(environ: Mapping[str, str]) -> int:
    """POOL_LIMIT_FLAG in MiB; 0 when unset or blank. Anything but a positive integer raises ValueError naming the setting."""
    raw = environ.get(POOL_LIMIT_FLAG, "").strip()
    if raw == "":
        return 0
    if not raw.isdecimal() or int(raw) == 0:
        raise ValueError(f"{POOL_LIMIT_FLAG}={raw!r}: set the highest PT pool Limit in MiB a worker may boot with (a positive integer), "
                         "or unset it to skip the pool floor check")
    return int(raw)


def check_pool_floor(stats: Mapping[str, int], environ: Mapping[str, str]) -> tuple[str, str]:
    """(GLM53_POOL_FLOOR line, refusal or '') for one rank's allocator stats against POOL_LIMIT_FLAG, which must be set."""
    limit_max, limit, in_use = pool_limit_max_mib(environ), int(stats["Limit"]) // MIB, int(stats["InUse"]) // MIB
    line = f"{POOL_LINE} limit_MiB={limit} max_MiB={limit_max} in_use_MiB={in_use} pid={os.getpid()}\n"
    if limit <= limit_max:
        return line, ""
    return line, (f"the PT pool Limit is {limit} MiB, above {POOL_LIMIT_FLAG}={limit_max} MiB, so the card keeps no HBM floor outside the pool "
                  f"({POOL_ACQUIRE_FLAG}={environ.get(POOL_ACQUIRE_FLAG, 'unset')} did not cap it)")


def enforce_pool_floor(environ: Mapping[str, str] | None = None) -> None:
    """hpu_worker.initialize_from_config, every rank, before the KV cache: refuse the boot on all ranks over the pool limit. Unset: no-op."""
    environ = os.environ if environ is None else environ
    if not environ.get(POOL_LIMIT_FLAG, "").strip():
        return
    limit_mib = "unknown"
    try:
        stats = _runtime().stats()
        limit_mib = str(int(stats["Limit"]) // MIB)
        line, error = check_pool_floor(stats, environ)
        sys.stdout.write(line)
    except Exception as exc:  # any failure refuses on all ranks below; a rank that raised alone would leave its peers in the gather
        error = f"{POOL_LIMIT_FLAG} is set but the pool floor check could not run: {type(exc).__name__}: {exc}"
    breadcrumb("pool_floor", limit_MiB=limit_mib, refused=int(bool(error)))
    import glm53_mixed_graph

    glm53_mixed_graph.admit_capture(POOL_BUCKET, 0, 0, error, POOL_ADVICE)


def _breadcrumb_rank() -> str:
    """The torch.distributed rank once the group is up, else the launcher's RANK, else '?'. Never imports torch and never raises."""
    distributed = getattr(sys.modules.get("torch"), "distributed", None)
    try:
        if distributed is not None and distributed.is_available() and distributed.is_initialized():
            return str(distributed.get_rank())
    except Exception:  # a rank label is a convenience: a breadcrumb never fails on it
        pass
    return os.environ.get("RANK", "?")


def breadcrumb_line(step: str, fields: Mapping[str, object]) -> str:
    """`<utc iso time> pid=<pid> rank=<rank> step=<step> k=v ...`: one line, values without spaces so every field splits on one."""
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds")
    pairs = "".join(f" {key}={str(value).replace(' ', '')}" for key, value in fields.items())
    return f"{stamp} pid={os.getpid()} rank={_breadcrumb_rank()} step={step}{pairs}\n"


def _warn_breadcrumb_once(path: str, exc: Exception) -> None:
    if _BREADCRUMB_WARNED:
        return
    _BREADCRUMB_WARNED.append(True)
    sys.stderr.write(f"GLM53_BREADCRUMB_UNWRITABLE {BREADCRUMB_FLAG}={path}: {type(exc).__name__}: {exc}; serving continues without breadcrumbs. "
                     f"Point {BREADCRUMB_FLAG} at a file on a writable bind-mounted host directory, or unset it. pid={os.getpid()}\n")


def breadcrumb(step: str, **fields: object) -> None:
    """Append one line to BREADCRUMB_FLAG's file, on disk when this returns (O_DSYNC, one write): a host crash loses the docker log tail still in
    page cache, never these. Unset: nothing is opened. A file that cannot be written warns once per process and never stops the caller."""
    path = os.environ.get(BREADCRUMB_FLAG, "").strip()
    if not path:
        return
    try:
        descriptor = os.open(path, BREADCRUMB_OPEN_FLAGS, 0o644)
        try:
            os.write(descriptor, breadcrumb_line(step, fields).encode())
        finally:
            os.close(descriptor)
    except (OSError, ValueError) as exc:  # ValueError: a path with a NUL byte
        _warn_breadcrumb_once(path, exc)


def install_from_env() -> bool:
    """Wrap glm53_mixed_graph.seal_boot_memory when the flag is set; False (and no import) when it is not. Idempotent."""
    margin = margin_mib()
    if margin <= 0:
        return False
    if _INSTALLED:
        return True
    import glm53_mixed_graph

    ballast = margin * MIB if ballast_on() else 0
    contract: list[str] = []
    wrapper = _wrap_seal(glm53_mixed_graph, margin * MIB, ballast, contract)
    glm53_mixed_graph.seal_boot_memory = wrapper
    _INSTALLED.append(wrapper)
    # Every rank runs this at interpreter start against the same image's two files, so every rank reads the same contract and refuses
    # together at admit_capture. A rank-level disagreement (one rank refusing while the others wait in the seal's collective) would need two
    # images in one boot, which a TP group cannot be.
    error = seal_contract_error(glm53_mixed_graph)
    if error:
        contract.append(error)
        sys.stdout.write(f"{LINE}_CONTRACT_BROKEN {error}; the boot is refused at the seal. {CONTRACT_ADVICE} pid={os.getpid()}\n")
    sys.stdout.write(f"{LINE}_INSTALL margin_MiB={margin} ballast_MiB={ballast // MIB} replays_per_key={REPLAYS_PER_KEY} timed_replays={timed_replays()} keys=prompt+decode "
                     f"pid={os.getpid()}\n")
    return True
