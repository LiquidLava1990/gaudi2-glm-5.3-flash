"""A real v1 Scheduler and real KV manager on CPU, driven step by step with a fake model that can only read rows in order.

The fake model is what makes the schedule-independence check honest: each request's state is a running hash of the rows the
runner would be handed for it, a row out of order raises, and the sampled token is a function of that state alone.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Callable, Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from overlay_testlib import load_base, load_overlay  # noqa: E402

import glm53_mixed_step as step_policy  # noqa: E402

from vllm.config import CacheConfig, ObservabilityConfig, ParallelConfig, SchedulerConfig  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec  # noqa: E402
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
from vllm.v1.request import Request  # noqa: E402

SCHEDULER_PATH = "vllm/v1/core/sched/scheduler.py"
BLOCK_SIZE = 128
ROWS = 1024
MAX_MODEL_LEN = 16_000_000
EOS_TOKEN = 2
init_none_hash(sha256)


def scheduler_module(*, base: bool = False) -> ModuleType:
    return load_base(SCHEDULER_PATH) if base else load_overlay(SCHEDULER_PATH)


class StandIn(SimpleNamespace):
    """A config with only the fields that matter set; every other switch reads as off."""

    def __getattr__(self, name: str) -> bool:
        return False


def _vllm_config(num_seqs: int, num_blocks: int) -> StandIn:
    scheduler_config = SchedulerConfig(
        max_model_len=MAX_MODEL_LEN, is_encoder_decoder=False, max_num_batched_tokens=ROWS * 2, max_num_scheduled_tokens=ROWS * 2,
        max_num_seqs=num_seqs, long_prefill_token_threshold=ROWS, enable_chunked_prefill=True)
    cache_config = CacheConfig()
    cache_config.num_gpu_blocks = num_blocks
    return StandIn(
        scheduler_config=scheduler_config, cache_config=cache_config, parallel_config=ParallelConfig(),
        observability_config=ObservabilityConfig(), lora_config=None, kv_events_config=None, kv_transfer_config=None,
        ec_transfer_config=None, speculative_config=None,
        ec_manager_config=SimpleNamespace(get_encoder_cache_manager_obj=lambda: None),
        model_config=StandIn(max_model_len=MAX_MODEL_LEN, runner_type="generate", architecture="GlmForCausalLM"),
        use_v2_model_runner=True, max_in_flight_tokens=0, num_lookahead_tokens=0, num_speculative_tokens=0)


def _kv_cache_config(num_blocks: int) -> KVCacheConfig:
    spec = FullAttentionSpec(BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32)
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)])


def make_request(request_id: str, tokens: list[int], max_tokens: int) -> Request:
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    params.update_from_generation_config({}, EOS_TOKEN)
    return Request(request_id=request_id, prompt_token_ids=tokens, sampling_params=params, pooling_params=None,
                   block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256))


def prompt_tokens(seed: int, count: int) -> list[int]:
    """Tokens no other seed shares, so prefix hits only come from a deliberately repeated seed."""
    return [10 + seed * 1000 + index % 500 for index in range(count)]


@dataclass
class Lane:
    """What the fake model holds for one request: rows it read, in order, and the tokens that state implies.

    The hash runs over rows, not chunks: co-batching or another chunk size must not change a request's output, and here cannot."""
    tokens: list[int]
    computed: int = 0
    hasher: "hashlib._Hash" = field(default_factory=hashlib.sha256)
    sampled: list[int] = field(default_factory=list)
    chunks: list[tuple[int, int]] = field(default_factory=list)
    restarts: int = 0

    @property
    def state(self) -> bytes:
        return self.hasher.digest()

    def read(self, start: int, rows: int) -> None:
        if start > self.computed:  # a prefix hit: those rows were computed by an earlier request, so hash them as read
            _hash_rows(self.hasher, self.tokens[self.computed:start])
        elif start != self.computed:
            raise AssertionError(f"row {start} read after {self.computed}: the prompt did not prefill in order")
        _hash_rows(self.hasher, self.tokens[start:start + rows])
        self.computed = start + rows
        self.chunks.append((start, rows))

    def preempted(self) -> None:
        """The scheduler freed this request's blocks, so it reads again from its restart row: the one legitimate re-read, which ``read``
        then sees as the prefix hit or cold start it is. Tokens it already sampled stay part of its sequence."""
        self.computed, self.hasher = 0, hashlib.sha256()
        self.restarts += 1

    def sample(self) -> int:
        token = 1000 + int.from_bytes(self.state[:2], "little") % 30000
        self.sampled.append(token)
        self.tokens.append(token)
        return token


def _hash_rows(hasher: "hashlib._Hash", tokens: Iterable[int]) -> None:
    hasher.update("".join(f"{token}," for token in tokens).encode())


@dataclass(frozen=True)
class StepTrace:
    """One step as the worker would see it: decode lanes, and the (request, start, rows) prompt chunks."""
    decodes: tuple[str, ...]
    chunks: tuple[tuple[str, int, int], ...]

    @property
    def kind(self) -> str:
        return "mixed" if self.chunks and self.decodes else "prompt" if self.chunks else "decode" if self.decodes else "idle"


class WallClock:
    """Real time as a run would see it, moved by each step's duration: the scheduler's ledger reads this, never the plan's table."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# Steps as a card might really run them, on purpose not the plan's table: aging must follow these, whatever glm53_mixed_step assumes.
CARD_DECODE_SECONDS = 0.15
CARD_CHUNK_SECONDS = ((1024, 1.5), (4096, 3.0), (8192, 4.6))
IDLE_SECONDS = 0.01


def card_seconds(step: StepTrace) -> float:
    """One step's wall time on the harness's card: the widest chunk it carried, else a decode, else a beat of idle."""
    if step.chunks:
        widest = max((start + rows + BLOCK_SIZE - 1) // BLOCK_SIZE for _, start, rows in step.chunks)
        return next((seconds for width, seconds in CARD_CHUNK_SECONDS if width >= widest), CARD_CHUNK_SECONDS[-1][1])
    return CARD_DECODE_SECONDS if step.decodes else IDLE_SECONDS


class Drive:
    """One scheduler plus the fake model; ``step`` is schedule() then update_from_output(), as EngineCore runs them."""

    def __init__(self, *, base: bool = False, num_blocks: int = 70_000, num_seqs: int = 32,
                 seconds: Callable[[StepTrace], float] = card_seconds) -> None:
        module = scheduler_module(base=base)
        self.scheduler = module.Scheduler(
            vllm_config=_vllm_config(num_seqs, num_blocks), kv_cache_config=_kv_cache_config(num_blocks), log_stats=False,
            block_size=BLOCK_SIZE, structured_output_manager=SimpleNamespace(should_advance=lambda *_, **__: False))
        self.lanes: dict[str, Lane] = {}
        self.finished_at: dict[str, int] = {}
        self.first_token_at: dict[str, int] = {}
        self.trace: list[StepTrace] = []
        self.clock, self.seconds = WallClock(), seconds
        self.started_at: list[float] = []  # wall clock at each step's start, by step index
        self.restarted: list[str] = []
        if not base and step_policy.slot_policy_active():
            self.scheduler._glm53_slot_ledger = step_policy.SlotLedger(clock=self.clock)

    def add(self, request_id: str, tokens: list[int], max_tokens: int = 1) -> None:
        self.lanes[request_id] = Lane(list(tokens))
        self.scheduler.add_request(make_request(request_id, list(tokens), max_tokens))

    def _starts(self, output: object) -> dict[str, int]:
        starts = {data.req_id: data.num_computed_tokens for data in output.scheduled_new_reqs}
        cached = output.scheduled_cached_reqs
        starts.update(zip(cached.req_ids, cached.num_computed_tokens))
        return starts

    def step(self) -> StepTrace:
        self.started_at.append(self.clock.now)
        output = self.scheduler.schedule()
        starts, counts = self._starts(output), output.num_scheduled_tokens
        for request_id in output.preempted_req_ids:
            self.lanes[request_id].preempted()
            self.restarted.append(request_id)
        decodes, chunks = [], []
        for request_id, rows in counts.items():
            lane = self.lanes[request_id]
            # The schedule output's num_computed_tokens is the first row of this step's work.
            lane_start = starts[request_id]
            if lane_start + rows > len(lane.tokens):
                raise AssertionError(f"{request_id}: rows past its tokens: {lane_start}+{rows} of {len(lane.tokens)}")
            is_decode = lane_start == len(lane.tokens) - 1 and rows == 1 and bool(lane.sampled)
            lane.read(lane_start, rows)
            (decodes.append(request_id) if is_decode else chunks.append((request_id, lane_start, rows)))
        sampled = {request_id: [self.lanes[request_id].sample()] for request_id in counts
                   if self.lanes[request_id].computed == len(self.lanes[request_id].tokens)}
        self._update(output, counts, sampled)
        trace = StepTrace(tuple(decodes), tuple(chunks))
        self.trace.append(trace)
        self.clock.now += self.seconds(trace)
        return trace

    def _update(self, output: object, counts: dict[str, int], sampled: dict[str, list[int]]) -> None:
        ids = list(counts)
        runner_output = ModelRunnerOutput(req_ids=ids, sampled_token_ids=[sampled.get(request_id, []) for request_id in ids],
                                          req_id_to_index={request_id: index for index, request_id in enumerate(ids)})
        step = len(self.trace)
        for request_id in sampled:
            self.first_token_at.setdefault(request_id, step)
        self.scheduler.update_from_output(output, runner_output)
        for request_id in ids:
            if request_id not in self.scheduler.requests:
                self.finished_at.setdefault(request_id, step)

    def run(self, steps: int) -> list[StepTrace]:
        return [self.step() for _ in range(steps)]

    def run_until_idle(self, limit: int) -> list[StepTrace]:
        idle = 0
        for _ in range(limit):
            if not self.scheduler.requests:
                return self.trace
            idle = idle + 1 if self.step().kind == "idle" else 0
            if idle > 1:  # one empty step is a requeued pick; two in a row with work left is the engine's "No progress"
                raise AssertionError(f"no progress at step {len(self.trace)} with {sorted(self.scheduler.requests)} unfinished")
        raise AssertionError(f"not idle after {limit} steps: {sorted(self.scheduler.requests)} unfinished")
