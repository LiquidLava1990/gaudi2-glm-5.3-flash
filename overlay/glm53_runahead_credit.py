"""Credit-bounded run-ahead (plan step 22, fix F1): R589 drains the device only when the lazy queue nears the Limit.

Plan rule: drain when this forward's growth exceeds free_at_forward_start - margin. Growth = free_at_start - free_now,
so that is free_now < margin (Limit - InUse < margin), whatever the forward started at. No forward-start state is needed,
which also means no hook at the forward's entry and no drift from a forward the hook did not see.

Only synchronize moves. The caller keeps every mark_step, because numerics depend on graph segmentation [M KB 11-05 10:03].

Sampling every N boundaries leaves N-1 boundaries unread, so a sampled decision also charges them at the largest growth
per boundary seen so far. N=1 charges nothing and is exactly the plan rule. cells/read_cost_cell.py picks N on a card.

HELD (plan step 22, 2026-11-06): this module and its tests stay on disk but are NOT in the image (.dockerignore), and the shipped
glm53_r589_bounded_prefill.py carries no hook. To lift the hold, put these three pieces back into R589 and drop the .dockerignore line:
    from glm53_runahead_credit import runahead_credit_from_env                              (with the other imports)
    RUNAHEAD = runahead_credit_from_env(os.environ, lambda: torch.hpu.memory_stats())      (after DRAIN_SITES)
    elif RUNAHEAD is not None and RUNAHEAD.should_drain(label):                             (after _boundary's `if drain and
        torch.hpu.synchronize()                                                              _site_drains(label)` branch)

Allocator reads are host-side (habana_frameworks/torch/hpu/memory.py): they neither step nor sync the device.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping

FLAG_ENV = "HPU_GLM53_RUNAHEAD_CREDIT"
MARGIN_ENV = "HPU_GLM53_RUNAHEAD_MARGIN_MIB"
SAMPLE_ENV = "HPU_GLM53_RUNAHEAD_SAMPLE_EVERY"
DEFAULT_MARGIN_MIB = 512
MIN_MARGIN_MIB = 256
DEFAULT_SAMPLE_EVERY = 1
MIB = 1 << 20
LOG_EVERY = 256  # a drain or read-error event logs its 1st, 256th, 512th, ... occurrence
STATS_EVERY = 4096  # boundaries between stats lines; about 866 boundaries is one forward [M census]
LOG = logging.getLogger("vllm.r589")

ReadStats = Callable[[], Mapping[str, int]]


class RunaheadCredit:
    """One worker's drain decision at R589 boundaries. Never raises: an unreadable allocator drains and logs."""

    def __init__(self, read_stats: ReadStats, margin_bytes: int, sample_every: int) -> None:
        self.read_stats = read_stats
        self.margin_bytes = margin_bytes
        self.sample_every = sample_every
        self.boundaries = 0
        self.reads = 0
        self.drains = 0
        self.read_errors = 0
        self.read_ns_total = 0
        self.unread = 0
        self.last_free: int | None = None
        self.largest_growth_per_boundary = 0.0

    def should_drain(self, label: str) -> bool:
        """Called after the boundary's mark_step, only at a boundary whose caller does not already drain."""
        self.boundaries += 1
        drain = self._decide(label)
        if self.boundaries % STATS_EVERY == 0:
            self._log_stats()
        return drain

    def _decide(self, label: str) -> bool:
        self.unread += 1
        if self.unread < self.sample_every:
            return False
        self.unread = 0
        try:
            free = self._read_free()
        except Exception as error:  # noqa: BLE001  an unreadable allocator must never lose a drain or kill a forward
            return self._read_failed(label, error)
        if self._projected_free(free) >= self.margin_bytes:
            return False
        self.drains += 1
        if self.drains == 1 or self.drains % LOG_EVERY == 0:
            LOG.warning("GLM53_RUNAHEAD event=drain site=%s free_mib=%d margin_mib=%d drains=%d boundaries=%d",
                        label, free // MIB, self.margin_bytes // MIB, self.drains, self.boundaries)
        return True

    def _read_free(self) -> int:
        started = time.perf_counter_ns()
        stats = self.read_stats()
        self.read_ns_total += time.perf_counter_ns() - started
        self.reads += 1
        limit = int(stats["Limit"])
        if limit <= 0:
            raise ValueError(f"allocator Limit={limit}")
        return limit - int(stats["InUse"])

    def _projected_free(self, free: int) -> float:
        """``free`` less the boundaries this sampled read leaves unread, charged at the largest growth seen."""
        if self.sample_every == 1:
            return free
        if self.last_free is not None:
            growth = (self.last_free - free) / self.sample_every
            self.largest_growth_per_boundary = max(self.largest_growth_per_boundary, growth)
        self.last_free = free
        return free - (self.sample_every - 1) * self.largest_growth_per_boundary

    def _read_failed(self, label: str, error: Exception) -> bool:
        self.read_errors += 1
        self.drains += 1
        if self.read_errors == 1 or self.read_errors % LOG_EVERY == 0:
            LOG.warning("GLM53_RUNAHEAD event=read_error site=%s read_errors=%d error=%r: draining this boundary; "
                        "check htorch.hpu.memory_stats() in this image, or unset %s to return to the configured drains",
                        label, self.read_errors, error, FLAG_ENV)
        return True

    def _log_stats(self) -> None:
        LOG.warning("GLM53_RUNAHEAD event=stats boundaries=%d reads=%d drains=%d read_errors=%d read_ns_total=%d "
                    "sample_every=%d margin_mib=%d", self.boundaries, self.reads, self.drains, self.read_errors,
                    self.read_ns_total, self.sample_every, self.margin_bytes // MIB)


def _whole_number(environ: Mapping[str, str], name: str, default: int, minimum: int, purpose: str) -> int:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    if not raw.isdigit() or int(raw) < minimum:
        raise ValueError(f"{name}={raw}: use a whole number, at least {minimum} ({purpose}; default {default}), "
                         f"or unset {FLAG_ENV} to turn run-ahead credit off")
    return int(raw)


def runahead_credit_from_env(environ: Mapping[str, str], read_stats: ReadStats) -> RunaheadCredit | None:
    """None unless the flag is 1, and then nothing else is parsed. A bad value raises, which refuses the boot."""
    flag = environ.get(FLAG_ENV, "0").strip()
    if flag not in ("", "0", "1"):
        raise ValueError(f"{FLAG_ENV}={flag!r}: set 1 to drain only near the allocator Limit, or 0 / unset for the "
                         "configured drains")
    if flag != "1":
        return None
    margin_mib = _whole_number(environ, MARGIN_ENV, DEFAULT_MARGIN_MIB, MIN_MARGIN_MIB, "MiB kept free")
    every = _whole_number(environ, SAMPLE_ENV, DEFAULT_SAMPLE_EVERY, 1, "boundaries per allocator read")
    LOG.warning("GLM53_RUNAHEAD event=on margin_mib=%d sample_every=%d", margin_mib, every)
    return RunaheadCredit(read_stats, margin_mib * MIB, every)
