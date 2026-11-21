"""HPU_GLM53_GC_FREEZE_AT_SEAL (default off): freeze the worker's Python heap once, when the boot's warmup and seal are complete.

Why: Python's collector walks every tracked object in a full (generation 2) collection while it holds the GIL. A booted worker tracks about a
million long-lived objects (the model, the captured graph keys, the caches), so one full collection costs 300+ ms on this host, and a decode step
that overlaps one waits for it. gc.freeze() moves every object tracked now into the permanent generation, which no collection scans again. The
collector stays enabled, so garbage made after the freeze is still collected; nothing is disabled.

glm53_mixed_graph.seal_boot_memory calls freeze_at_seal after every key is captured, the seal ballast is held and the boot witness has accepted
the boot. Every rank reads the flag. A rank whose freeze fails refuses the boot on all ranks (admit_capture) and names the flag to unset. Off
(unset or 0) the hook returns before it touches the collector, the log or the ranks.
"""
from __future__ import annotations

import gc
import os
import time
from collections.abc import Callable, Mapping
from typing import NamedTuple, Protocol

FLAG = 'HPU_GLM53_GC_FREEZE_AT_SEAL'
WITNESS = 'GLM53_GC_FREEZE'
ADVICE = f'Unset {FLAG} to boot without the freeze (the collector then runs as it did), and restart all workers.'

Log = Callable[..., None]
Admit = Callable[..., None]


class CollectorLike(Protocol):
    """The slice of the gc module the freeze uses; there is deliberately no disable()."""

    def collect(self) -> int: ...

    def freeze(self) -> None: ...

    def get_freeze_count(self) -> int: ...

    def isenabled(self) -> bool: ...


class FreezeResult(NamedTuple):
    frozen: int          # objects in the permanent generation afterwards
    collected: int       # unreachable objects the collection before the freeze released
    collect_ms: float
    freeze_ms: float
    error: str           # empty when the freeze took


_DONE: list[FreezeResult] = []   # one entry once this worker has frozen: the freeze is once per worker


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    value = (os.environ if environ is None else environ).get(FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError(f'{FLAG}={value!r} must be 0 or 1; unset it (or 0) to keep it off')
    return value == '1'


def reset() -> None:
    """Forget that this worker froze (tests)."""
    _DONE.clear()


def skipped_without_keys(log: Log, environ: Mapping[str, str] | None = None) -> None:
    """The boot captured no keys, so it has no seal to freeze after. When the flag asked for a freeze, say that none happened."""
    if enabled(environ):
        log('GLM53_GC_FREEZE_SKIPPED reason=no_keys nothing_frozen=1 %s needs a keyed boot (VLLM_GAUDI_MIXED_PROMPT_KEYS or the decode lattice); '
            'the collector runs as it did', FLAG)


def _world_rank() -> int:
    import torch

    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _freeze(collector: CollectorLike, clock: Callable[[], float]) -> FreezeResult:
    """gc.collect() (so garbage is not frozen for good), then gc.freeze(). A failure comes back as text, so every rank can refuse together."""
    try:
        started = clock()
        collected = collector.collect()
        collected_at = clock()
        collector.freeze()
        frozen = collector.get_freeze_count()
        finished = clock()
    except Exception as exc:  # noqa: BLE001 - any failure refuses the boot with the flag named
        return FreezeResult(0, 0, 0.0, 0.0, f'{FLAG}=1 but the heap freeze failed ({type(exc).__name__}: {exc}); the collector would keep scanning it all')
    error = '' if frozen > 0 else (f'{FLAG}=1 but gc.freeze() left no object in the permanent generation, so the collector would keep scanning '
                                   'the whole heap')
    return FreezeResult(frozen, collected, (collected_at - started) * 1e3, (finished - collected_at) * 1e3, error)


def _log_result(log: Log, rank: int, result: FreezeResult, collector: CollectorLike) -> None:
    if result.error:
        log('%s_FAILED rank=%d error=%s', WITNESS, rank, result.error)
        return
    log('%s rank=%d frozen_objects=%d collected=%d collect_ms=%.1f freeze_ms=%.1f gc_enabled=%s', WITNESS, rank, result.frozen,
        result.collected, result.collect_ms, result.freeze_ms, collector.isenabled())


def freeze_at_seal(log: Log, admit: Admit, *, environ: Mapping[str, str] | None = None, collector: CollectorLike | None = None,
                   clock: Callable[[], float] = time.perf_counter, rank: int | None = None) -> FreezeResult | None:
    """None when the flag is off. Otherwise collect, freeze, log one witness line for this rank, and let `admit` (glm53_mixed_graph's
    admit_capture) raise on every rank if any rank's freeze failed. A second call in the same worker freezes nothing and says so."""
    if not enabled(environ):
        return None
    collector = gc if collector is None else collector
    rank = _world_rank() if rank is None else rank
    if _DONE:
        log('%s rank=%d already_frozen=1 frozen_objects=%d', WITNESS, rank, _DONE[0].frozen)
        return _DONE[0]
    result = _freeze(collector, clock)
    _log_result(log, rank, result, collector)
    admit(('gc_freeze',), 0, 0, result.error, ADVICE)
    _DONE.append(result)
    return result
