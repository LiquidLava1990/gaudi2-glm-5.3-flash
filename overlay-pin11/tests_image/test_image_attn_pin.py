"""ATTN-PIN real-import tests, run INSIDE the r4d image on CPU (HABANA_VISIBLE_DEVICES=none, no card, --network none).

run_tests.sh copies the two overlay files into the container's own dist-packages (the image on disk is untouched), so the
.pth runs at interpreter start exactly as it will in r4e:
  * this interpreter has the flag unset -> the real classes carry no wrapper;
  * a child interpreter with HPU_GLM53_ATTN_PIN=1 -> the five real modules import wrapped at all twelve patch points;
  * a child interpreter with HPU_GLM53_ATTN_PIN=1 and HPU_GLM53_ATTENTION_ONLY_OFFLOAD=1 runs real_scheduler_cases.py:
    two turns through the real Scheduler waiting loop (adoption, held-plan exits, hybrid fail-closed);
  * the real classes still have the contracts the pin relies on (the fake stack of tests/fakes.py mirrors them);
  * the real BlockPool / KVCacheBlock primitives behave as the pin uses them.
"""
from __future__ import annotations

import importlib
import inspect
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

PINKIT = Path(os.environ.get("PINKIT", "/pinkit"))
sys.path.insert(0, str(PINKIT))

import glm53_attn_pin as pin  # noqa: E402  (the dist-packages copy when the .pth ran, else the kit's)

PROBE = Path(__file__).with_name("probe_wrapped.py")
REAL_CASES = Path(__file__).with_name("real_scheduler_cases.py")
ATTENTION_ONLY = "HPU_GLM53_ATTENTION_ONLY_OFFLOAD"


def patch_points() -> list[tuple[str, str, str]]:
    return [(module, cls, method) for module, points in pin.TARGET_MODULES.items() for cls, method in points]


class TestFlagOffIdentity(unittest.TestCase):
    def test_flag_unset_wraps_no_real_class(self) -> None:
        self.assertNotEqual(os.environ.get(pin.FLAG), "1", "this suite's own interpreter must run with the flag unset")
        self.assertFalse(any(isinstance(f, pin._PatchOnImport) for f in sys.meta_path))
        for module_name, cls_name, method in patch_points():
            cls = getattr(importlib.import_module(module_name), cls_name)
            self.assertNotIn(pin.ORIGINALS_ATTR, vars(cls), f"{cls_name}.{method} is wrapped with the flag unset")
            self.assertFalse(hasattr(getattr(cls, method), "__wrapped__"), f"{cls_name}.{method}")


def flag_on_env() -> dict[str, str]:
    return dict(os.environ, **{pin.FLAG: "1", ATTENTION_ONLY: "1", "PINKIT": str(PINKIT), "PYTHONDONTWRITEBYTECODE": "1"})


class TestFlagOnWrapsRealClasses(unittest.TestCase):
    def test_child_interpreter_with_flag_wraps_all_twelve(self) -> None:
        proc = subprocess.run([sys.executable, str(PROBE)], env=flag_on_env(), capture_output=True, text=True,
                              timeout=600)
        self.assertEqual(proc.returncode, 0, proc.stderr[-4000:])
        report = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertTrue(report["finder_installed"], report)
        self.assertEqual(report["module_file"].rsplit("/", 1)[0], "/usr/local/lib/python3.12/dist-packages",
                         "the .pth must have imported the dist-packages copy")
        missing = [point for point, wrapped in report["wrapped"].items() if not wrapped]
        self.assertEqual(missing, [], report)
        self.assertEqual(len(report["wrapped"]), 12)


class TestRealScheduler(unittest.TestCase):
    def test_two_turns_on_the_real_waiting_loop(self) -> None:
        """real_scheduler_cases.py in a flag-on child: adoption, the held-plan exits, the deferred-free pin, the idle
        age-out, the hybrid fail-closed path, reset_prefix_cache with a pin and a held plan."""
        proc = subprocess.run([sys.executable, str(REAL_CASES)], env=flag_on_env(), capture_output=True, text=True,
                              timeout=900)
        report = "\n".join(line for line in proc.stderr.splitlines()
                           if not line.startswith(("INFO", "WARNING", "DEBUG")))
        sys.stderr.write(report[-6000:] + "\n")
        self.assertEqual(proc.returncode, 0, report[-6000:])
        self.assertIn("Ran 8 tests", report)


class TestRealClassContracts(unittest.TestCase):
    """Every fact the pin relies on, checked against the real bytes (file:line refs in prebuild.md section 1)."""

    def setUp(self) -> None:
        self.single = importlib.import_module("vllm.v1.core.single_type_kv_cache_manager")
        self.coord = importlib.import_module("vllm.v1.core.kv_cache_coordinator")
        self.kvm = importlib.import_module("vllm.v1.core.kv_cache_manager")
        self.sched = importlib.import_module("vllm.v1.core.sched.scheduler")
        self.offl = importlib.import_module("vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler")

    def test_manager_classes_and_pinnable_predicate(self) -> None:
        full, tail = self.single.FullAttentionManager, self.single.KpoolTailManager
        self.assertTrue(issubclass(tail, full))
        plain = full.__new__(full)
        plain.block_size = 128
        self.assertTrue(pin.pinnable_manager(plain, 128))
        circular = tail.__new__(tail)
        circular.block_size = 128
        self.assertFalse(pin.pinnable_manager(circular, 128))
        # The pinned group must reach the wrapped base method; KpoolTail (single_type:1206) overrides it, which is fine
        # because it is never pinned (inject_after_local ignores managers outside the registry).
        self.assertNotIn("add_local_computed_blocks", vars(full), "FullAttentionManager overrides the wrapped base method")
        params = list(inspect.signature(self.single.SingleTypeKVCacheManager.add_local_computed_blocks).parameters)
        self.assertEqual(params[:2], ["self", "request_id"])

    def test_r4e_hybrid_specs_leave_exactly_the_mla_group_pinnable(self) -> None:
        """r4e-A refusal (03:03): the scheduler block size is KDA's 1048576; the pin must find MLA at 128 by class."""
        import probe_spec_managers

        names = probe_spec_managers.manager_names()
        self.assertEqual(names, {"mla": "FullAttentionManager", "tail": "KpoolTailManager", "kda": "MambaManager"})
        sizes = {name: spec.block_size for name, spec in probe_spec_managers.R4E_SPECS.items()}
        pinnable = [name for name, cls in names.items() if cls == pin.PINNABLE_CLASS]
        self.assertEqual([(name, sizes[name]) for name in pinnable], [("mla", 128)])

    def test_coordinator_and_manager_signatures(self) -> None:
        params = inspect.signature(self.coord.KVCacheCoordinator.get_num_blocks_to_allocate).parameters
        self.assertEqual(list(params)[:2], ["self", "request_id"])
        self.assertIn("apply_admission_cap", params)
        for cls in (self.coord.KVCacheCoordinatorNoPrefixCache, self.coord.UnitaryKVCacheCoordinator):
            self.assertNotIn("get_num_blocks_to_allocate", vars(cls), f"{cls.__name__} overrides the wrapped base method")
        alloc = inspect.signature(self.kvm.KVCacheManager.allocate_slots).parameters
        for name in ("num_external_computed_tokens", "delay_cache_blocks", "reserved_blocks"):
            self.assertIn(name, alloc)
        self.assertEqual(list(inspect.signature(self.kvm.KVCacheManager.free).parameters), ["self", "request"])

    def test_scheduler_gates_and_status(self) -> None:
        for name in ("_request_remaining_blocks", "_inflight_prefill_reserved_blocks", "_glm53_leaves_partials_room"):
            self.assertTrue(callable(getattr(self.sched.Scheduler, name)), name)
        status = importlib.import_module("vllm.v1.request").RequestStatus
        self.assertEqual(status.WAITING.name, "WAITING")

    def test_reset_prefix_cache_resets_the_kv_manager_before_the_connector(self) -> None:
        """The order the reset_prefix_cache wrapper exists for (scheduler.py:3849-3859), on the one class serving uses."""
        reset = self.sched.Scheduler.reset_prefix_cache
        self.assertEqual(list(inspect.signature(reset).parameters), ["self", "reset_running_requests", "reset_connector"])
        source = inspect.getsource(reset)
        self.assertLess(source.index("self.kv_cache_manager.reset_prefix_cache()"), source.index("reset_connector_cache()"))
        async_scheduler = importlib.import_module("vllm.v1.core.sched.async_scheduler").AsyncScheduler
        self.assertNotIn("reset_prefix_cache", vars(async_scheduler), "AsyncScheduler overrides the wrapped reset")

    def test_connector_local_split_and_status_dict(self) -> None:
        source = inspect.getsource(self.offl.OffloadingConnectorScheduler.update_state_after_alloc)
        self.assertIn("block.block_hash is None", source, "the hashed-leading-blocks-are-local split")
        self.assertIn("self._req_status[request.request_id]", source)
        self.assertIn("num_locally_computed_tokens + num_external_tokens", source)
        self.assertTrue(callable(self.offl.OffloadingConnectorScheduler.reset_cache))
        connector = importlib.import_module("vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector")
        self.assertIn("_offload_group_indices", inspect.getsource(connector.OffloadingConnector))


class TestRealBlockPoolPrimitives(unittest.TestCase):
    def test_marker_hash_and_lifo_free_on_real_blocks(self) -> None:
        block_pool_module = importlib.import_module("vllm.v1.core.block_pool")
        pool = block_pool_module.BlockPool(num_gpu_blocks=16, enable_caching=False, hash_block_size=128)
        free_before = pool.get_num_free_blocks()
        blocks = pool.get_new_blocks(4)
        self.assertTrue(all(b.ref_cnt == 1 and b.block_hash is None and not b.is_null for b in blocks))
        blocks[0]._block_hash = b"marker"
        blocks[0]._block_hash_num_tokens = 128
        self.assertEqual(blocks[0].block_hash, b"marker")
        blocks[0].reset_hash()
        self.assertIsNone(blocks[0].block_hash)
        pool.free_blocks(reversed(blocks))
        self.assertTrue(all(b.ref_cnt == 0 for b in blocks))
        self.assertEqual(pool.get_num_free_blocks(), free_before)
        # Unhashed frees go to the head of the free queue as a batch (r4d block_pool.py), so a freed block is reused first.
        self.assertIn(pool.get_new_blocks(1)[0].block_id, {b.block_id for b in blocks}, "freed blocks are reused first")


if __name__ == "__main__":
    unittest.main()
