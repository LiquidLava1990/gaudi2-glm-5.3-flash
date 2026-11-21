# SPDX-License-Identifier: Apache-2.0
"""WIDE-PROMPT-KERNEL-PLAN step 6: HPU_GLM53_EMIT_WORDSKIP in glm53_mla_topkfuse.py, the word-skip emit (GUID g6_topk_emit_wordskip_i32).

Flag off must run base's ops, marks and GUID chain in base's order; flag on must select the g6_ws lazy op and so the new GUID; flag on with
anything missing must refuse the worker at import with a message that says what to set or build. The kernel itself is proven on a card
(emit-wordskip/probe_emit_wordskip.py, 60 of 60 cells bit-exact); nothing here runs a kernel. Kernel and lazy op are adapted from
1CatAI/1Cat-vLLM-Gaudi (Apache-2.0)."""
from __future__ import annotations

import contextlib
import importlib.util
import itertools
import os
import sys
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import glm53_mixed_graph as graph  # noqa: E402
import key_stack  # noqa: E402
from overlay_testlib import BASE_ROOT, OVERLAY_ROOT  # noqa: E402
from test_key_select_exactness import (  # noqa: E402
    KEY_ROWS, MERGE_FLAG, MODES, POOLS_PER_BLOCK, PROMPT_CHUNK, Case, Kernel, Selection, assert_exact, case_for, key_capture, native_world,
    prompt_selection)
from test_key_select_exactness import _exec_on_cpu as exec_on_cpu  # noqa: E402
from test_key_select_exactness import select as select_keys  # noqa: E402

TOPKFUSE_FILE = "glm53_mla_topkfuse.py"
STOCK_LAZY_LIBRARY = "/opt/glm53/mla/libmla_topkfuse_lazy.so"
STOCK_PERFLIB = "/opt/glm53/rc6/libg6_k512_perflib.so"
STOCK_GUID, NEW_GUID = "g6_topk_emit_i32", "g6_topk_emit_wordskip_i32"
CHAIN_GUIDS = ("mla_topkkeys_i32", "mla_topkstate_i32")
FLAG, LAZY_FLAG = "HPU_GLM53_EMIT_WORDSKIP", "HPU_GLM53_EMIT_WORDSKIP_LAZY_LIB"
ON_ENV = {FLAG: "1", "HPU_GLM53_MLA_TOPKFUSE": "1", "HPU_GLM53_INDEXER_TPC_TOPK": "1"}
_UNIQUE = itertools.count()


def load_topkfuse(path: Path, env: dict[str, str]) -> tuple[ModuleType, list[str]]:
    """A fresh copy of the module under exactly `env`; the libraries it asks torch to load are recorded, not loaded."""
    loaded: list[str] = []
    spec = importlib.util.spec_from_file_location(f"topkfuse_under_test_{next(_UNIQUE)}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(torch.ops, "load_library", loaded.append):
        spec.loader.exec_module(module)
    return module, loaded


class Scores:
    """A float32 [rows, 8] input and the two lazy ops, each recording the tile it was given and returning two columns of it."""

    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.mark_step = lambda: self.events.append(("mark_step",))

    def op(self, name: str):
        def run(tile: torch.Tensor) -> torch.Tensor:
            self.events.append((name, tuple(tile.shape)))
            return tile[:, :2].clone()
        return run

    @contextlib.contextmanager
    def installed(self, *, g6_ws_allowed: bool) -> Iterator[None]:
        def forbidden(tile: torch.Tensor) -> torch.Tensor:
            raise AssertionError("flag off must never reach torch.ops.g6_ws")

        core = SimpleNamespace(mark_step=self.mark_step)
        with mock.patch.dict(sys.modules, {"habana_frameworks": SimpleNamespace(), "habana_frameworks.torch": SimpleNamespace(),
                                           "habana_frameworks.torch.core": core}), \
                mock.patch.object(torch.ops, "mla_fuse", SimpleNamespace(select=self.op("mla_fuse.select")), create=True), \
                mock.patch.object(torch.ops, "g6_ws", SimpleNamespace(select=self.op("g6_ws.select") if g6_ws_allowed else forbidden), create=True):
            yield

    @staticmethod
    def scores(rows: int) -> torch.Tensor:
        return torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8)


def run_select(module: ModuleType, rows: int, *, g6_ws_allowed: bool = False) -> tuple[list[tuple], torch.Tensor]:
    recorder = Scores()
    with recorder.installed(g6_ws_allowed=g6_ws_allowed):
        output = module.select(recorder.scores(rows))
    return recorder.events, output


@contextlib.contextmanager
def wordskip_world(kernel: Kernel, on_env: dict[str, str], **env: str) -> Iterator[SimpleNamespace]:
    """The key_select_exactness layer with the selector read again under the word-skip settings: torch.ops.g6_ws is the kernel, and torch.ops.mla_fuse
    refuses to run. `world.wordskip_calls` holds the row count of every g6_ws call."""
    def stock_select(scores: torch.Tensor) -> torch.Tensor:
        raise AssertionError("flag on must never reach torch.ops.mla_fuse")

    with native_world(kernel, **env) as world:
        world.wordskip_calls = []

        def wordskip_select(scores: torch.Tensor) -> torch.Tensor:
            world.wordskip_calls.append(scores.shape[0])
            return kernel.select(scores)

        with mock.patch.dict(os.environ, on_env):
            exec_on_cpu("glm53_mla_topkfuse", OVERLAY_ROOT / TOPKFUSE_FILE)  # the layer's copy was read with the flag unset
        with mock.patch.object(torch.ops, "g6_ws", SimpleNamespace(select=wordskip_select), create=True), \
                mock.patch.object(torch.ops, "mla_fuse", SimpleNamespace(select=stock_select), create=True):
            yield world


class Libraries(unittest.TestCase):
    """A directory with a lazy library stand-in and a perf-lib stand-in that carry the new GUID, as the built ones do."""

    @staticmethod
    def lazy_bytes(*guids: str) -> bytes:
        """What a built libg6_ws_lazy.so holds around its emit GUID: the node chain's names, then the GUID its AddNode passes to BuildOp."""
        return b"\x7fELF g6_ws::select\x00mla_topkkeys_i32\x00mla_topkstate_i32\x00" + b"".join(guid.encode() + b"\x00" for guid in guids)

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.lazy = self.directory / "libg6_ws_lazy.so"
        self.lazy.write_bytes(self.lazy_bytes(NEW_GUID))
        self.perflib = self.directory / "libg6_perflib.so"
        self.perflib.write_bytes(b"\x7fELF " + NEW_GUID.encode() + b"\x00" + STOCK_GUID.encode())
        self.on_env = {**ON_ENV, LAZY_FLAG: str(self.lazy), "G6_PERFLIB": str(self.perflib)}


class FlagOffEqualsBase(unittest.TestCase):
    def modules(self, env: dict[str, str]) -> tuple[ModuleType, list[str], ModuleType, list[str]]:
        base, base_loaded = load_topkfuse(BASE_ROOT / TOPKFUSE_FILE, env)
        overlay, overlay_loaded = load_topkfuse(OVERLAY_ROOT / TOPKFUSE_FILE, env)
        return base, base_loaded, overlay, overlay_loaded

    def test_every_row_count_runs_the_same_ops_and_marks_in_the_same_order(self) -> None:
        base, _, overlay, _ = self.modules({"HPU_GLM53_MLA_TOPKFUSE": "0"})
        for rows in (64, 128, 129, 256, 384, 640):
            with self.subTest(rows=rows):
                base_events, base_output = run_select(base, rows)
                overlay_events, overlay_output = run_select(overlay, rows)
                self.assertEqual(overlay_events, base_events)
                self.assertTrue(torch.equal(overlay_output, base_output))
                self.assertIn(("mla_fuse.select", (min(rows, 128), 8)), overlay_events)

    def test_the_same_libraries_load_with_the_selector_on_and_the_flag_unset(self) -> None:
        _, base_loaded, _, overlay_loaded = self.modules({"HPU_GLM53_MLA_TOPKFUSE": "1", "HPU_GLM53_INDEXER_TPC_TOPK": "1"})
        self.assertEqual(overlay_loaded, base_loaded)
        self.assertEqual(overlay_loaded, [STOCK_LAZY_LIBRARY])

    def test_the_same_shapes_pass_the_gate(self) -> None:
        base, _, overlay, _ = self.modules({"HPU_GLM53_MLA_TOPKFUSE": "1"})

        def scores(rows: int, width: int, device: str = "hpu", dtype: torch.dtype = torch.float32) -> SimpleNamespace:
            return SimpleNamespace(device=SimpleNamespace(type=device), dtype=dtype, ndim=2, shape=(rows, width))

        for case in (scores(128, 32768), scores(256, 20480), scores(128, 73728), scores(64, 32768), scores(128, 16384),
                     scores(128, 32768, "cpu"), scores(128, 32768, dtype=torch.bfloat16)):
            with self.subTest(shape=case.shape, device=case.device.type, dtype=case.dtype):
                self.assertEqual(overlay.applies(case), base.applies(case))

    def test_the_stock_library_carries_exactly_the_stock_chain_and_the_reader_says_so(self) -> None:
        _, _, overlay, _ = self.modules({})
        if not Path(STOCK_LAZY_LIBRARY).is_file():
            self.skipTest(f"{STOCK_LAZY_LIBRARY} is not in this image")
        stock = Path(STOCK_LAZY_LIBRARY).read_bytes()
        for guid in (*CHAIN_GUIDS, STOCK_GUID):
            self.assertIn(guid.encode(), stock)
        self.assertNotIn(NEW_GUID.encode(), stock)
        self.assertEqual(overlay.lazy_library_emit_guid(STOCK_LAZY_LIBRARY), STOCK_GUID)

    def test_the_flag_unset_and_zero_both_read_off_and_log_no_live_line(self) -> None:
        for env in ({}, {FLAG: "0"}):
            with self.subTest(env=env):
                overlay, loaded = load_topkfuse(OVERLAY_ROOT / TOPKFUSE_FILE, env)
                self.assertFalse(overlay.EMIT_WORDSKIP)
                self.assertEqual(loaded, [])
                self.assertEqual(overlay.LAZY_EMIT_GUID, "")
                with self.assertNoLogs("vllm.rc6", "WARNING"):
                    run_select(overlay, 128)


class FlagOn(Libraries):
    def test_the_select_runs_the_word_skip_op_on_every_tile_with_base_s_marks(self) -> None:
        base, _ = load_topkfuse(BASE_ROOT / TOPKFUSE_FILE, {"HPU_GLM53_MLA_TOPKFUSE": "0"})
        overlay, _ = load_topkfuse(OVERLAY_ROOT / TOPKFUSE_FILE, self.on_env)
        for rows in (128, 256, 384):
            with self.subTest(rows=rows):
                base_events, base_output = run_select(base, rows)
                events, output = run_select(overlay, rows, g6_ws_allowed=True)
                self.assertEqual(events, [(("g6_ws.select" if event[0] == "mla_fuse.select" else event[0]), *event[1:]) for event in base_events])
                self.assertNotIn("mla_fuse.select", [event[0] for event in events])
                self.assertTrue(torch.equal(output, base_output))

    def test_the_guid_the_module_reports_is_read_from_the_lazy_library_it_loads(self) -> None:
        overlay, _ = load_topkfuse(OVERLAY_ROOT / TOPKFUSE_FILE, self.on_env)
        self.assertEqual(overlay.LAZY_EMIT_GUID, NEW_GUID)
        self.assertEqual(overlay.lazy_library_emit_guid(str(self.lazy)), NEW_GUID)

    def test_the_reader_tells_a_stock_a_word_skip_a_both_and_a_neither_library_apart(self) -> None:
        overlay, _ = load_topkfuse(OVERLAY_ROOT / TOPKFUSE_FILE, self.on_env)
        cases = {"stock": ((STOCK_GUID,), STOCK_GUID), "word-skip": ((NEW_GUID,), NEW_GUID), "both": ((STOCK_GUID, NEW_GUID), None), "neither": ((), None)}
        for name, (guids, expected) in cases.items():
            with self.subTest(library=name):
                path = self.directory / f"{name}.so"
                path.write_bytes(self.lazy_bytes(*guids))
                self.assertEqual(overlay.lazy_library_emit_guid(str(path)), expected)
        self.assertIsNone(overlay.lazy_library_emit_guid(str(self.directory / "missing.so")))

    def test_the_stock_library_loads_first_then_the_word_skip_library(self) -> None:
        _, loaded = load_topkfuse(OVERLAY_ROOT / TOPKFUSE_FILE, self.on_env)
        self.assertEqual(loaded, [STOCK_LAZY_LIBRARY, str(self.lazy)])

    def test_the_default_library_path_is_the_one_the_run_file_mounts(self) -> None:
        overlay, _ = load_topkfuse(OVERLAY_ROOT / TOPKFUSE_FILE, self.on_env)
        self.assertEqual(overlay.wordskip_lazy_library({}), "/opt/glm53/emit/libg6_ws_lazy.so")
        self.assertEqual(overlay.STOCK_PERFLIB, STOCK_PERFLIB)

    def test_the_live_line_is_logged_once_per_process_with_rank_rows_width_and_guid(self) -> None:
        overlay, _ = load_topkfuse(OVERLAY_ROOT / TOPKFUSE_FILE, self.on_env)
        with mock.patch.dict(os.environ, {"RANK": "3"}), self.assertLogs("vllm.rc6", "WARNING") as logged:
            run_select(overlay, 256, g6_ws_allowed=True)
            run_select(overlay, 256, g6_ws_allowed=True)
        lines = [line for line in logged.output if "FLASH_EMIT_WORDSKIP_LIVE" in line]
        self.assertEqual(len(lines), 1)
        self.assertIn(f"FLASH_EMIT_WORDSKIP_LIVE rank=3 rows=256 width=8 guid={NEW_GUID}", lines[0])

    def test_the_live_line_reports_what_the_loaded_library_says_not_a_constant(self) -> None:
        overlay, _ = load_topkfuse(OVERLAY_ROOT / TOPKFUSE_FILE, self.on_env)
        overlay.LAZY_EMIT_GUID = "guid_read_from_the_library"
        with self.assertLogs("vllm.rc6", "WARNING") as logged:
            run_select(overlay, 128, g6_ws_allowed=True)
        self.assertIn("guid=guid_read_from_the_library", "\n".join(logged.output))

    def test_a_1024_row_key_runs_eight_tiles_with_sixteen_marks_in_arm_b_and_none_in_arm_a(self) -> None:
        overlay, _ = load_topkfuse(OVERLAY_ROOT / TOPKFUSE_FILE, self.on_env)
        for marks, expected in (("0", 0), ("1", 16)):
            with self.subTest(R589_CAPTURE_MARKS=marks), mock.patch.dict(os.environ, {"HPU_GLM53_R589_CAPTURE_MARKS": marks}), \
                    graph.capture_context(), graph.key_capture_context():
                events, output = run_select(overlay, 1024, g6_ws_allowed=True)
                self.assertEqual(events.count(("mark_step",)), expected)
                self.assertEqual([event for event in events if event[0] != "mark_step"], [("g6_ws.select", (128, 8))] * 8)
                self.assertEqual(tuple(output.shape), (1024, 2))

    def test_the_kept_lever_witness_still_fires_once_whatever_the_tile_count(self) -> None:
        overlay, _ = load_topkfuse(OVERLAY_ROOT / TOPKFUSE_FILE, self.on_env)
        graph.reset_witness()
        with graph.capture_context(), graph.key_capture_context():
            run_select(overlay, 256, g6_ws_allowed=True)
        self.assertEqual(graph._WITNESS["mla_topkfuse"], 1)

    def test_tile_boundaries_mark_only_in_arm_b_as_base_does(self) -> None:
        overlay, _ = load_topkfuse(OVERLAY_ROOT / TOPKFUSE_FILE, self.on_env)
        for marks, expected in (("0", 0), ("1", 4)):
            with self.subTest(R589_CAPTURE_MARKS=marks), mock.patch.dict(os.environ, {"HPU_GLM53_R589_CAPTURE_MARKS": marks}), \
                    graph.capture_context(), graph.key_capture_context():
                events, _ = run_select(overlay, 256, g6_ws_allowed=True)
                self.assertEqual(events.count(("mark_step",)), expected)


class FlagOnRefusals(Libraries):
    def refusal(self, drop: tuple[str, ...] = (), **changes: str) -> str:
        env = {name: value for name, value in {**self.on_env, **changes}.items() if name not in drop}
        with self.assertRaises(RuntimeError) as raised:
            load_topkfuse(OVERLAY_ROOT / TOPKFUSE_FILE, env)
        return str(raised.exception)

    def test_a_missing_lazy_library_names_the_path_the_setting_and_the_build(self) -> None:
        missing = str(self.directory / "absent" / "libg6_ws_lazy.so")
        message = self.refusal(**{LAZY_FLAG: missing})
        for expected in ("HPU_GLM53_EMIT_WORDSKIP=1 cannot boot", missing, LAZY_FLAG, "libg6_ws_lazy.so", "build_emit_wordskip.sh", "RUN.txt"):
            self.assertIn(expected, message)

    def test_the_default_lazy_path_is_what_the_refusal_names_when_nothing_is_set(self) -> None:
        if Path("/opt/glm53/emit/libg6_ws_lazy.so").exists():
            self.skipTest("this container has the library mounted")
        self.assertIn("/opt/glm53/emit/libg6_ws_lazy.so is not a readable file", self.refusal(drop=(LAZY_FLAG,)))

    def test_a_lazy_library_that_still_selects_the_stock_emit_is_refused_and_the_perflib_is_not_blamed(self) -> None:
        stale = self.directory / "stale" / "libg6_ws_lazy.so"
        stale.parent.mkdir()
        stale.write_bytes(self.lazy_bytes(STOCK_GUID))
        message = self.refusal(**{LAZY_FLAG: str(stale)})
        for expected in ("HPU_GLM53_EMIT_WORDSKIP=1 cannot boot", str(stale), f"does not select {NEW_GUID}", f"it names {STOCK_GUID}",
                         "rebuild libg6_ws_lazy.so", "build_emit_wordskip.sh"):
            self.assertIn(expected, message)
        self.assertNotIn("G6_PERFLIB=", message)

    def test_a_lazy_library_that_names_both_emit_guids_or_neither_is_refused(self) -> None:
        for name, guids, said in (("both", (STOCK_GUID, NEW_GUID), f"it names {STOCK_GUID}, {NEW_GUID}"), ("neither", (), "it names no emit GUID")):
            with self.subTest(library=name):
                path = self.directory / f"{name}.so"
                path.write_bytes(self.lazy_bytes(*guids))
                message = self.refusal(**{LAZY_FLAG: str(path)})
                self.assertIn(f"{path} does not select {NEW_GUID}", message)
                self.assertIn(said, message)

    def test_a_missing_lazy_library_is_one_problem_not_two(self) -> None:
        missing = str(self.directory / "absent.so")
        self.assertNotIn("does not select", self.refusal(**{LAZY_FLAG: missing}))

    def test_a_stale_lazy_library_and_a_stale_perflib_are_both_in_the_one_message(self) -> None:
        stale = self.directory / "stale.so"
        stale.write_bytes(self.lazy_bytes(STOCK_GUID))
        message = self.refusal(G6_PERFLIB=str(self.directory / "nope.so"), **{LAZY_FLAG: str(stale)})
        self.assertIn(f"{stale} does not select {NEW_GUID}", message)
        self.assertIn("is not a perf-lib that carries", message)

    def test_an_unset_perflib_names_g6_perflib_the_guid_and_the_path_to_mount_over(self) -> None:
        message = self.refusal(drop=("G6_PERFLIB",))
        for expected in ("G6_PERFLIB=(unset)", NEW_GUID, STOCK_PERFLIB, "mount libg6_perflib.so over that path", "glm53_rc6_flags",
                         "build_emit_wordskip.sh"):
            self.assertIn(expected, message)

    def test_the_image_s_own_stock_perflib_is_refused_so_a_stock_chain_cannot_pass_as_word_skip(self) -> None:
        if not Path(STOCK_PERFLIB).is_file():
            self.skipTest(f"{STOCK_PERFLIB} is not in this image")
        self.assertNotIn(NEW_GUID.encode(), Path(STOCK_PERFLIB).read_bytes())
        self.assertIn(f"G6_PERFLIB={STOCK_PERFLIB} is not a perf-lib that carries {NEW_GUID}", self.refusal(G6_PERFLIB=STOCK_PERFLIB))

    def test_a_perflib_without_the_guid_is_refused(self) -> None:
        stock = self.directory / "stock.so"
        stock.write_bytes(b"\x7fELF " + STOCK_GUID.encode())
        message = self.refusal(G6_PERFLIB=str(stock))
        self.assertIn(f"G6_PERFLIB={stock} is not a perf-lib that carries {NEW_GUID}", message)

    def test_a_perflib_path_that_is_missing_is_refused(self) -> None:
        self.assertIn("is not a perf-lib that carries", self.refusal(G6_PERFLIB=str(self.directory / "nope.so")))

    def test_the_selector_flag_off_is_refused_because_the_flag_would_do_nothing(self) -> None:
        self.assertIn("set HPU_GLM53_MLA_TOPKFUSE=1", self.refusal(HPU_GLM53_MLA_TOPKFUSE="0"))

    def test_the_tpc_topk_flag_off_or_unset_is_refused_for_the_same_reason(self) -> None:
        self.assertIn("set HPU_GLM53_INDEXER_TPC_TOPK=1", self.refusal(HPU_GLM53_INDEXER_TPC_TOPK="0"))
        self.assertIn("set HPU_GLM53_INDEXER_TPC_TOPK=1", self.refusal(drop=("HPU_GLM53_INDEXER_TPC_TOPK",)))

    def test_any_value_but_0_and_1_is_a_refusal_not_a_silent_off(self) -> None:
        for value in ("true", "2", "", " 1", "on"):
            with self.subTest(value=value):
                message = self.refusal(**{FLAG: value})
                self.assertIn(f"{FLAG}={value!r} is not 0 or 1", message)

    def test_every_problem_is_in_the_one_message(self) -> None:
        gone = str(self.directory / "gone.so")
        message = self.refusal(drop=("G6_PERFLIB", "HPU_GLM53_INDEXER_TPC_TOPK"), HPU_GLM53_MLA_TOPKFUSE="0", **{LAZY_FLAG: gone})
        for expected in ("set HPU_GLM53_MLA_TOPKFUSE=1", "set HPU_GLM53_INDEXER_TPC_TOPK=1", gone, "G6_PERFLIB=(unset)"):
            self.assertIn(expected, message)

    def test_nothing_loads_when_the_boot_is_refused(self) -> None:
        loaded: list[str] = []
        env = {**self.on_env, "G6_PERFLIB": ""}
        spec = importlib.util.spec_from_file_location("topkfuse_refused", OVERLAY_ROOT / TOPKFUSE_FILE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(torch.ops, "load_library", loaded.append), \
                self.assertRaises(RuntimeError):
            spec.loader.exec_module(module)
        self.assertEqual(loaded, [])

    def test_a_ready_boot_has_no_problems(self) -> None:
        overlay, _ = load_topkfuse(OVERLAY_ROOT / TOPKFUSE_FILE, self.on_env)
        self.assertEqual(overlay.wordskip_boot_problems(self.on_env, True), [])


class EightChunkKey(Libraries):
    """A 1024:8192 prompt key scans 8 chunks of 32768 pools. The real kpool member merge and glm53_tpc_topk wrapper run on the CPU with the selector
    read under the word-skip flag; the reference emit stands behind torch.ops.g6_ws (a CPU cannot run the kernel: the card cells prove it)."""

    SCAN = 8192 * POOLS_PER_BLOCK
    CHUNKS = SCAN // PROMPT_CHUNK

    def word_skip_selection(self, case: Case) -> tuple[Selection, list[int]]:
        with wordskip_world(Kernel(), self.on_env, **{MERGE_FLAG: "1"}) as world, key_capture():
            return select_keys(world, case, PROMPT_CHUNK, shared=True), list(world.wordskip_calls)

    def test_every_chunk_goes_to_the_word_skip_op_and_each_mode_is_exact(self) -> None:
        for mode in MODES:
            with self.subTest(mode=mode):
                case = case_for(mode, KEY_ROWS, self.SCAN)
                new, wordskip_calls = self.word_skip_selection(case)
                legacy = prompt_selection(case)
                self.assertEqual(wordskip_calls, [KEY_ROWS] * self.CHUNKS)
                self.assertEqual((new.native_calls, new.legacy_merges), (self.CHUNKS, 0))
                assert_exact(self, new, legacy, case)

    def test_the_fences_are_the_stock_member_merge_s(self) -> None:
        case = case_for("random", KEY_ROWS, self.SCAN)
        new, _ = self.word_skip_selection(case)
        self.assertEqual(new.fences, prompt_selection(case, **{MERGE_FLAG: "1"}).fences, "the word-skip select moved a fence")


class Rc6PinsThePerflibPath(unittest.TestCase):
    """Why the word-skip perf-lib is mounted over the stock path and G6_PERFLIB is left alone: the image's own flag registry, parsed at the worker's
    first import, accepts exactly one G6_PERFLIB while the tpc top-k is on, and a boot with any other stops there, before this module is imported."""

    def chain(self, rc6: ModuleType, perflib: str) -> dict[str, str]:
        return {"HPU_GLM53_INDEXER_TPC_TOPK": "1", "GC_KERNEL_PATH": "/opt/glm53/mla/libmla_filter.so", **dict(rc6.TPC_LIBS[1:]),
                "MLA_BASELINE_LIB": "/opt/glm53/rc6/libg6_k512_filter.so", "MLA_PERFLIB": "/opt/glm53/mla/libmla_perflib.so", "G6_PERFLIB": perflib}

    def test_the_stock_path_boots_and_a_new_path_is_refused(self) -> None:
        import glm53_rc6_flags as rc6

        readable = lambda path: True  # noqa: E731  (the libraries are not in a unit-test container)
        self.assertIn("tpc_topk", rc6.Rc6Flags.from_env(self.chain(rc6, STOCK_PERFLIB), readable=readable).enabled)
        with self.assertRaisesRegex(rc6.Rc6FlagError, "G6_PERFLIB='/opt/glm53/emit/libg6_perflib.so'"):
            rc6.Rc6Flags.from_env(self.chain(rc6, "/opt/glm53/emit/libg6_perflib.so"), readable=readable)


class KpoolStopsTheWorkerAtImport(unittest.TestCase):
    """The selector is imported lazily by glm53_tpc_topk 18 minutes into a boot; kpool imports it at worker start when the flag is set."""

    @contextlib.contextmanager
    def layer(self, env: dict[str, str]) -> Iterator[SimpleNamespace]:
        """The overlay's own selector must be the one kpool finds, whatever the image has in site-packages."""
        with mock.patch.object(sys, "path", [str(OVERLAY_ROOT), *sys.path]), key_stack.layer_stack(OVERLAY_ROOT, env) as world:
            yield world

    def test_the_flag_on_with_nothing_mounted_stops_kpool_s_import(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "HPU_GLM53_EMIT_WORDSKIP=1 cannot boot"), self.layer({FLAG: "1", "G6_PERFLIB": ""}):
            pass

    def test_the_flag_unset_does_not_import_the_selector_at_all(self) -> None:
        with self.layer({}):
            self.assertNotIn("glm53_mla_topkfuse", sys.modules)

    def test_the_gate_is_one_guarded_import(self) -> None:
        source = (OVERLAY_ROOT / "vllm_gaudi/ops/hpu_glm5next_kpool.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("glm53_mla_topkfuse"), 1)
        self.assertIn('if os.getenv("HPU_GLM53_EMIT_WORDSKIP", "0") != "0":\n    import glm53_mla_topkfuse', source)


if __name__ == "__main__":
    unittest.main()
