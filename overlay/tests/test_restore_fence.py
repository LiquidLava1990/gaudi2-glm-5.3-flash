"""Step 7 (KB 1984-1985): a CPU-restore copy is fenced, then its row replays on the landing step, never native.

The real _update_states, compact-slot allocator and forward paths run against a fake device with no clock: a copy sits in the lazy graph until
mark_step, then in flight until synchronize, and a replay reads whatever the row holds. So no change of timing can pass: a delay, a deferred
landing, or half a fence leaves the row stale (HarnessRejectsTiming proves it).
"""
from __future__ import annotations

import ast
import contextlib
import dataclasses
import os
import sys
import time
import unittest
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from overlay_testlib import OVERLAY_ROOT  # noqa: E402
from test_graphs_only import ONLY, GraphsOnlyCase  # noqa: E402
from test_prompt_keys import BLOCK, Backend, RowForward  # noqa: E402

RUNNER_FILE = OVERLAY_ROOT / "vllm_gaudi/v1/worker/hpu_model_runner.py"
RUNNER_METHODS = {"_update_states", "ensure_recurrent_sidecar_slot", "_external_latch_applies", "_execute_model_generic"}
STALE, RESTORED = "A-state", "B-restored"
FIRST_SLOT, TOP_OF_FREE_LIST = 5, 4


class LazyDevice:
    """Copy -> lazy graph; mark_step -> submitted; synchronize -> landed. A replay flushes neither queue and reads the row as it stands."""

    def __init__(self) -> None:
        self.rows: dict[int, str] = {}
        self.lazy: list[tuple[int, str]] = []
        self.submitted: list[tuple[int, str]] = []
        self.events: list[str] = []
        self.reads: list[str] = []
        self.watch = -1
        self.sync_error: Exception | None = None

    def copy(self, row: int, value: str) -> None:
        self.events.append("copy")
        self.lazy.append((row, value))

    def mark_step(self) -> None:
        self.events.append("mark_step")
        self.submitted.extend(self.lazy)
        self.lazy.clear()

    def synchronize(self) -> None:
        if self.sync_error is not None:
            raise self.sync_error
        self.events.append("synchronize")
        self.rows.update(self.submitted)
        self.submitted.clear()

    def replay(self) -> None:
        self.events.append("replay")
        self.reads.append(self.rows[self.watch])

    def module(self) -> SimpleNamespace:
        internal = SimpleNamespace(is_lazy=lambda: True)
        return SimpleNamespace(core=SimpleNamespace(mark_step=self.mark_step), hpu=SimpleNamespace(synchronize=self.synchronize),
                               utils=SimpleNamespace(internal=internal))


class FakeBatch:
    def __init__(self) -> None:
        self.req_ids: list[str] = []
        self.req_id_to_index: dict[str, int] = {}
        self.req_type: dict[str, str] = {}
        self.num_computed_tokens_cpu: dict[int, int] = {}
        self.num_prompt_tokens: dict[int, int] = {}

    def admit(self, req_id: str, computed: int = 1024, prompt: int = 1024) -> None:
        index = len(self.req_ids)
        self.req_ids.append(req_id)
        self.req_id_to_index[req_id] = index
        self.num_computed_tokens_cpu[index], self.num_prompt_tokens[index] = computed, prompt

    def remove_request(self, req_id: str) -> int | None:
        index = self.req_id_to_index.pop(req_id, None)
        if index is not None:
            self.req_ids[index] = "<removed>"
        return index

    def condense(self, _removed: list[int]) -> None:
        return

    def refresh_sampling_metadata(self) -> None:
        return


def real_runner_namespace(device: LazyDevice) -> dict[str, Any]:
    """The globals the extracted runner methods read; every value here stands in for an import of the full module."""
    logger = SimpleNamespace(warning_once=lambda *_: None, warning=lambda *_: None, debug=lambda *_: None)
    return {
        "os": os, "torch": torch, "contextlib": contextlib, "Mapping": Mapping, "logger": logger, "htorch": device.module(),
        "_r589_stage": lambda _name: (lambda function: function), "trim_attn_metadata": lambda metadata: metadata,
        "_glm53_warmup_only_kpool_cap": lambda _enabled: contextlib.nullcontext(), "RESTORED_DECODE_GRAPHS": True,
        "_graphs_preserve_external_state": lambda: True, "LoraMask": SimpleNamespace(setLoraMask=lambda _mask: None),
        "get_pp_group": lambda: SimpleNamespace(is_last_rank=True),
    }


def real_runner_class(device: LazyDevice) -> type:
    """The overlay's own methods on an otherwise empty class, so a test exercises what ships."""
    source = ast.parse(RUNNER_FILE.read_text(encoding="utf-8"))
    methods = [item for node in source.body if isinstance(node, ast.ClassDef) for item in node.body
               if isinstance(item, ast.FunctionDef) and item.name in RUNNER_METHODS]
    if {method.name for method in methods} != RUNNER_METHODS:
        raise AssertionError(f"runner methods missing from {RUNNER_FILE}: {RUNNER_METHODS - {method.name for method in methods}}")
    helper = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_external_load_req_ids")
    runner = ast.ClassDef(name="Runner", bases=[], keywords=[], body=methods, decorator_list=[])
    namespace = real_runner_namespace(device)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[helper, runner], type_ignores=[])), str(RUNNER_FILE), "exec"), namespace)
    return namespace["Runner"]


def scheduler_output(finished: tuple[str, ...] = (), loads: tuple[str, ...] = (), scheduled: tuple[str, ...] = ()) -> SimpleNamespace:
    jobs = {job_id: SimpleNamespace(req_id=req_id) for job_id, req_id in enumerate(loads)}
    return SimpleNamespace(finished_req_ids=set(finished), kv_connector_metadata=SimpleNamespace(load_jobs=jobs), free_encoder_mm_hashes=[],
                           num_scheduled_tokens=dict.fromkeys(scheduled, 1), scheduled_new_reqs=[], scheduled_spec_decode_tokens={},
                           scheduled_cached_reqs=SimpleNamespace(req_ids=[]))


class RestoreCase(GraphsOnlyCase):
    def setUp(self) -> None:
        super().setUp()
        # The fence runs in the extracted runner method, on fake metadata with no block table: the lattice's own hook has its own tests.
        os.environ.update({"VLLM_GAUDI_MIXED_DECODE_WIDTHS": "", "HPU_GLM53_DECODE_WIDTH_PAD": "0", "HPU_GLM53_DECODE_PAD_CHECK_WIDTHS": ""})
        self.device = LazyDevice()
        fake = self.device.module()
        modules = mock.patch.dict(sys.modules, {"habana_frameworks": SimpleNamespace(torch=fake), "habana_frameworks.torch": fake})
        modules.start()
        self.addCleanup(modules.stop)
        self.native: list[dict] = []
        self.backend = self.replaying_backend()
        self.cache = self.graph.GraphCache(RowForward(), BLOCK, self.backend)
        self.boot_capture(self.cache, self.prompt_batch(1024, 0, 1024))
        self.device.events.clear()
        self.runner_under_test = self.build_runner()

    def replaying_backend(self) -> Backend:
        backend, device = Backend(), self.device
        real_replay = backend.replay

        def replay(handle: tuple) -> None:
            device.replay()
            real_replay(handle)

        backend.replay = replay
        return backend

    def model_forward(self, **kwargs: Any) -> torch.Tensor:
        if kwargs.get("bypass_hpu_graphs"):
            self.native.append(kwargs)
        else:
            self.device.replay()
        return torch.zeros(1, 1, 4)

    def build_runner(self) -> Any:
        runner = real_runner_class(self.device)()
        batch = FakeBatch()
        model = SimpleNamespace(forward=self.model_forward, compute_logits=lambda rows: rows, _glm53_mixed_graph_cache=self.cache,
                                _glm53_mixed_graph_forward=RowForward())
        vars(runner).update(
            input_batch=batch, model=model, requests={}, reasoning_answer_reserve=None, encoder_cache={}, _glm53_aop=False,
            _compact_gdn_group_ids={0}, _gdn_req_to_base_slot={}, _gdn_active_column={}, _gdn_slot_free_list=[TOP_OF_FREE_LIST], _gdn_max_reqs=8,
            is_driver_worker=True, model_has_chunked_attention=False, use_aux_hidden_state_outputs=False, attn_block_size=BLOCK,
            profiler=SimpleNamespace(record_event=lambda *_: contextlib.nullcontext()), model_config=SimpleNamespace(enforce_eager=False),
            _seq_len=lambda _metadata: 1, _num_blocks=lambda _metadata: 1, _check_config=lambda *_: None, _use_graphs=lambda *_: True)
        batch.admit("A")
        runner.requests["A"] = SimpleNamespace()
        runner._gdn_req_to_base_slot["A"], runner._gdn_active_column["A"] = FIRST_SLOT, 0
        self.device.rows[FIRST_SLOT] = STALE
        return runner

    def lane(self, req_id: str, row: int, value: str) -> None:
        """A running lane on its own slot, whose state is already on the device."""
        runner = self.runner_under_test
        runner.input_batch.admit(req_id)
        runner._gdn_req_to_base_slot[req_id], runner._gdn_active_column[req_id] = row, 0
        self.device.rows[row] = value

    def restore_b(self, *others: str) -> int:
        """A finishes and frees its slot, B's load job takes the top of the LIFO free list and queues its copy. With no others: a transfer-only step."""
        runner = self.runner_under_test
        runner._update_states(scheduler_output(finished=("A",), loads=("B",), scheduled=others))
        row = runner.ensure_recurrent_sidecar_slot("B")
        self.device.copy(row, RESTORED)
        return row

    def land_b(self, *others: str) -> None:
        """The connector reported the load finished, so the next step schedules B."""
        runner = self.runner_under_test
        runner.input_batch.admit("B")
        runner._update_states(scheduler_output(scheduled=("B", *others)))

    def runner_forward(self, req_id: str, *, is_prompt: bool, **kwargs: Any) -> None:
        runner = self.runner_under_test
        self.device.watch = runner._gdn_req_to_base_slot[req_id]
        runner._execute_model_generic(torch.zeros(1, 1), torch.zeros(1, 1), SimpleNamespace(is_prompt=is_prompt), torch.tensor([0]), None, None, None,
                                      request_ids=[req_id], **kwargs)

    def decode(self, req_id: str) -> None:
        self.runner_forward(req_id, is_prompt=False)

    def prompt_generic(self, req_id: str, **kwargs: Any) -> None:
        self.runner_forward(req_id, is_prompt=True, **kwargs)

    def prompt_key(self, req_id: str) -> mock.Mock:
        """The keyed dispatch of glm53_mixed_step, on a key the boot captured."""
        runner = self.runner_under_test
        self.device.watch = runner._gdn_req_to_base_slot[req_id]
        batch = self.prompt_batch(1024, 1024, 1024, seed=5)
        batch.region = dataclasses.replace(batch.region, requests=(runner.input_batch.req_id_to_index[req_id],))
        generic = mock.Mock(side_effect=AssertionError("a restored prompt reached the runner's forward"))
        with mock.patch.object(runner, "_execute_model_generic", generic):
            self.step.KeyedForward(runner, {batch.key: batch})(None, None, batch.region.metadata)
        return generic

    def unfenced(self) -> set[str]:
        return getattr(self.runner_under_test, self.graph.RESTORE_LEDGER, set())

    def collectives(self) -> Any:
        stub = SimpleNamespace(segmented_lazy_collectives=contextlib.nullcontext)
        return mock.patch.dict(sys.modules, {"vllm_gaudi.distributed.device_communicators.hpu_communicator": stub})


class FenceOrdering(RestoreCase):
    def test_a_restored_decode_fences_the_copy_then_replays_it(self) -> None:
        self.restore_b()
        self.land_b()
        self.decode("B")
        self.assertEqual(self.device.events, ["copy", "mark_step", "synchronize", "replay"])
        self.assertEqual(self.device.reads, [RESTORED])
        self.assertEqual(self.native, [])

    def test_a_restored_prompt_chunk_fences_the_copy_then_replays_its_key(self) -> None:
        self.restore_b()
        self.land_b()
        generic = self.prompt_key("B")
        self.assertEqual(self.device.events, ["copy", "mark_step", "synchronize", "replay"])
        self.assertEqual(self.device.reads, [RESTORED])
        generic.assert_not_called()

    def test_the_load_and_the_landing_on_one_step_are_fenced_too(self) -> None:
        runner = self.runner_under_test
        runner.input_batch.admit("B", computed=1024, prompt=1024)
        runner._update_states(scheduler_output(finished=("A",), loads=("B",), scheduled=("B",)))
        self.device.copy(runner.ensure_recurrent_sidecar_slot("B"), RESTORED)
        self.decode("B")
        self.assertEqual(self.device.reads, [RESTORED])


class LifoSlotReuse(RestoreCase):
    def test_b_takes_the_slot_a_just_freed_and_never_reads_a_state(self) -> None:
        row = self.restore_b()
        self.assertEqual(row, FIRST_SLOT, "the free list is LIFO: the row A wrote is the very next one handed out")
        self.assertNotEqual(self.runner_under_test._gdn_slot_free_list[-1], FIRST_SLOT)
        self.assertEqual(self.device.rows[row], STALE, "before the fence the row still holds the previous owner's valid-looking state")
        self.land_b()
        self.decode("B")
        self.assertEqual(self.device.reads, [RESTORED])
        self.assertNotIn(STALE, self.device.reads)

    def test_a_second_owner_of_the_same_slot_gets_its_own_fence(self) -> None:
        row = self.restore_b()
        self.land_b()
        self.decode("B")
        runner = self.runner_under_test
        runner.input_batch.admit("C")
        runner._update_states(scheduler_output(finished=("B",), loads=("C",), scheduled=("C",)))
        self.assertEqual(runner.ensure_recurrent_sidecar_slot("C"), row)
        self.device.copy(row, "C-restored")
        self.decode("C")
        self.assertEqual(self.device.reads, [RESTORED, "C-restored"])


class FenceScope(RestoreCase):
    """The narrow fix: one fence per restore, on the landing step, not one per forward of the latch window (KB 1984's broad fix cost 60 vs 92 turns)."""

    def test_the_rest_of_the_latch_window_replays_with_no_fence(self) -> None:
        self.restore_b()
        self.land_b()
        self.decode("B")
        self.prompt_key("B")
        self.decode("B")
        self.assertEqual(self.device.events.count("synchronize"), 1)
        self.assertEqual(self.device.reads, [RESTORED] * 3)
        self.assertEqual(self.native, [])

    def test_another_lane_neither_fences_nor_spends_the_landing(self) -> None:
        self.lane("C", 2, "C-state")
        self.restore_b("C")
        self.decode("C")
        self.assertEqual(self.device.events, ["copy", "replay"])
        self.land_b("C")
        self.decode("B")
        self.assertEqual(self.device.reads, ["C-state", RESTORED])

    def test_a_new_load_while_still_latched_arms_a_new_fence(self) -> None:
        row = self.restore_b()
        self.land_b()
        self.decode("B")
        self.runner_under_test._update_states(scheduler_output(loads=("B",), scheduled=("B",)))
        self.device.copy(row, "B-reloaded")
        self.decode("B")
        self.assertEqual(self.device.events.count("synchronize"), 2)
        self.assertEqual(self.device.reads, [RESTORED, "B-reloaded"])

    def test_a_request_that_finishes_before_it_lands_leaves_nothing_armed(self) -> None:
        self.lane("C", 2, "C-state")
        self.restore_b("C")
        self.assertEqual(self.unfenced(), {"B"})
        self.runner_under_test._update_states(scheduler_output(finished=("B",), scheduled=("C",)))
        self.assertEqual(self.unfenced(), set())


class FenceLog(RestoreCase):
    def test_the_fence_logs_its_milliseconds(self) -> None:
        self.restore_b()
        self.land_b()
        with self.assertLogs(self.graph.LOG, "WARNING") as logs:
            self.decode("B")
        self.assertRegex(logs.output[0], r"GLM53_RESTORE_FENCE ms=\d+\.\d+ requests=1$")

    def test_a_forward_with_no_landing_logs_nothing(self) -> None:
        self.lane("C", 2, "C-state")
        with self.assertNoLogs(self.graph.LOG, "WARNING"):
            self.decode("C")


class NeverNative(RestoreCase):
    def test_a_latched_prompt_on_the_runner_forward_refuses_on_every_rank(self) -> None:
        for rank in (0, 1):
            with self.subTest(rank=rank):
                self.timer.reset_mock()
                self.native.clear()
                self.runner_under_test = self.build_runner()
                self.restore_b()
                self.land_b()
                with self.assertRaisesRegex(RuntimeError, r"would run native.*HPU_GLM53_GRAPHS_ONLY"):
                    self.prompt_generic("B")
                self.assertEqual(self.native, [])
                self.timer.assert_called_once()

    def test_a_refused_decode_names_the_flag_to_set(self) -> None:
        self.restore_b()
        self.land_b()
        self.runner_under_test._external_latch_applies = lambda _metadata: True
        with self.assertRaisesRegex(RuntimeError, "VLLM_HPU_RESTORED_DECODE_GRAPHS=1"):
            self.decode("B")
        self.assertEqual(self.native, [])
        self.assertEqual(self.device.reads, [])

    def test_a_fence_that_fails_stops_every_rank_replays_nothing_and_stays_armed(self) -> None:
        self.restore_b()
        self.land_b()
        self.device.sync_error = RuntimeError("device lost")
        with self.assertRaisesRegex(RuntimeError, r"restore fence failed.*device lost"):
            self.decode("B")
        self.timer.assert_called_once()
        self.assertEqual(self.device.reads, [])
        self.assertEqual(self.unfenced(), {"B"})

    def test_warmup_dummies_run_as_they_did(self) -> None:
        self.restore_b()
        self.land_b()
        with self.collectives():
            self.prompt_generic("B", warmup_mode=True)
        self.assertEqual(len(self.native), 1)
        self.timer.assert_not_called()


class FlagOff(RestoreCase):
    def setUp(self) -> None:
        super().setUp()
        os.environ[ONLY] = "0"
        self.runner_under_test = self.build_runner()

    def test_flag_off_the_runner_holds_no_ledger_and_touches_no_device(self) -> None:
        self.restore_b()
        self.land_b()
        self.decode("B")
        self.assertFalse(hasattr(self.runner_under_test, self.graph.RESTORE_LEDGER))
        self.assertEqual(self.device.events, ["copy", "replay"])

    def test_flag_off_a_latched_prompt_still_runs_native_as_base_does(self) -> None:
        self.restore_b()
        self.land_b()
        with self.collectives():
            self.prompt_generic("B")
        self.assertEqual(len(self.native), 1)
        self.timer.assert_not_called()


class HarnessRejectsTiming(RestoreCase):
    """KB 1985: a change of batch timing can hide a state race. Each stand-in below is a timing change, and each must leave the replay stale."""

    def landed_with(self, stand_in: Callable[..., None]) -> list[str]:
        self.restore_b()
        self.land_b()
        with mock.patch.object(self.graph, "fence_restored_rows", stand_in, create=True):
            self.decode("B")
        return self.device.reads

    def test_a_delay_does_not_land_the_copy(self) -> None:
        self.assertEqual(self.landed_with(lambda *_, **__: time.sleep(0.01)), [STALE])

    def test_mark_step_alone_does_not_land_the_copy(self) -> None:
        self.assertEqual(self.landed_with(lambda *_, **__: self.device.mark_step()), [STALE])

    def test_synchronize_alone_does_not_land_the_copy(self) -> None:
        self.assertEqual(self.landed_with(lambda *_, **__: self.device.synchronize()), [STALE])

    def test_deferring_the_landing_steps_does_not_land_the_copy(self) -> None:
        self.lane("C", 2, "C-state")
        self.restore_b("C")
        with mock.patch.object(self.graph, "fence_restored_rows", lambda *_, **__: None, create=True):
            for _ in range(8):
                self.decode("C")
            self.land_b("C")
            self.decode("B")
        self.assertEqual(self.device.reads[-1], STALE)


if __name__ == "__main__":
    unittest.main()
