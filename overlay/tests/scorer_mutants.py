"""Batch-dependent scorer faults, for the tests that ask whether a row split can change a selection.

A card's scoring kernels may round a column differently at 4 rows than at 16 (another matmul and head-sum shape). Each fault below changes the
scorer's output by one ULP or one column position, in a pattern that depends on the batch the scorer was given, so the whole key's scan (16, 8, 4
rows) and a rank's share of it (4, 2, 1 rows) never get the same pattern. ``CONTROL`` ignores the batch: a test that flags it cries wolf.

``installed`` puts a fault around kpool._score_pool_chunk, the one function the shard changes the shape of.
"""
from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import ModuleType

import torch

ADJACENT_FLIP_BATCHES = frozenset({1, 8, 16})  # one of every (whole, split) pair of the decode keys: 16/4, 8/2, 4/1


@dataclass(frozen=True)
class Fault:
    name: str
    apply: Callable[[torch.Tensor], torch.Tensor]  # [batch, width] float32 scores -> the scores the faulty kernel returns


def columns_at(scores: torch.Tensor, period: int, offset: int) -> torch.Tensor:
    return (torch.arange(scores.shape[1]) % period == offset).unsqueeze(0)


def one_ulp(scores: torch.Tensor, toward: float, columns: torch.Tensor) -> torch.Tensor:
    moved = torch.nextafter(scores, torch.full_like(scores, toward))
    return torch.where(columns & torch.isfinite(scores), moved, scores)


def _up_every_nth_column(scores: torch.Tensor) -> torch.Tensor:
    return one_ulp(scores, torch.inf, columns_at(scores, scores.shape[0] + 2, 0))


def _down_every_nth_column(scores: torch.Tensor) -> torch.Tensor:
    return one_ulp(scores, -torch.inf, columns_at(scores, scores.shape[0] + 3, 1))


def _adjacent_flip(scores: torch.Tensor) -> torch.Tensor:
    if scores.shape[0] not in ADJACENT_FLIP_BATCHES or scores.shape[1] < 2:
        return scores
    paired = scores.shape[1] // 2 * 2
    flipped = scores.clone()
    flipped[:, :paired] = scores[:, :paired].reshape(scores.shape[0], -1, 2).flip(-1).reshape(scores.shape[0], paired)
    return flipped


def _batch_blind(scores: torch.Tensor) -> torch.Tensor:
    """A function of each score alone: the same value gets the same answer at any batch, any chunk width, any column."""
    return one_ulp(scores, torch.inf, (scores.contiguous().view(torch.int32) & 7 == 0))


UP_EVERY_NTH_COLUMN = Fault("up one ULP on every (batch + 2)th column", _up_every_nth_column)
DOWN_EVERY_NTH_COLUMN = Fault("down one ULP on every (batch + 3)th column from 1", _down_every_nth_column)
ADJACENT_FLIP = Fault("adjacent columns swapped at batch 1, 8 and 16", _adjacent_flip)
FAULTS = (UP_EVERY_NTH_COLUMN, DOWN_EVERY_NTH_COLUMN, ADJACENT_FLIP)
CONTROL = Fault("up one ULP on every score whose low three bits are 0, whatever the batch", _batch_blind)


@contextlib.contextmanager
def installed(kpool: ModuleType, fault: Fault) -> Iterator[None]:
    """kpool._score_pool_chunk returns ``fault`` applied to the real scores, until the block ends."""
    real = kpool._score_pool_chunk

    def faulty(*args: object, **kwargs: object) -> torch.Tensor:
        return fault.apply(real(*args, **kwargs))

    kpool._score_pool_chunk = faulty
    try:
        yield
    finally:
        kpool._score_pool_chunk = real
