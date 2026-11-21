"""Prove every env flag the overlay adds reads OFF with no environment set. Runs inside the built image (no network, no cards).

    docker run --rm --network none -e HABANA_VISIBLE_DEVICES=none -v <lane>/check_flags_off.py:/check/check_flags_off.py:ro \
        --entrypoint python3 <image> -B /check/check_flags_off.py

A flag is new when its quoted name appears in an installed file but not in the base copy that file replaced (build state under
/opt/sustain-overlay/base). Each new flag needs a probe below that calls the real reader with an empty environment. Exit 0 when
no probed flag is on; flags with no probe are listed as UNPROBED (add a probe when another lane adds a flag).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Callable
from unittest import mock

PKG = Path("/usr/local/lib/python3.12/dist-packages")
STATE = Path("/opt/sustain-overlay")
FLAG_NAME = re.compile(r"""["']((?:VLLM|HPU)_[A-Z0-9_]+)["']""")
LANE_BLOCK_NAME = "VLLM_OFFLOAD_MISS_ATTRIB_LANE_BLOCK"
PREVIOUS_HASH_NAME = "VLLM_RECURRENT_OFFLOAD_PREVIOUS_HASH_CHECKPOINT"
SHARED_OWNER_NAME = "HPU_GLM53_Q4_SHARED_GATHER_OWNER"
# Set by base image 1481bb99's own ENV, so no overlay turned them on. Only these values pass; any other value, and any other flag in the image's env, fails.
BASE_IMAGE_ENV = {"HPU_GLM53_MIXED_DECODE_REPLAY": "1", "HPU_GLM53_PRIVATE_PROMPT_TOPK": "1", "HPU_GRAPH_DRY_RUN": "0",
                  "VLLM_PREFILL_MAX_REQUESTS_PER_STEP": "0"}


def _tripwire_v2_off() -> bool:
    import glm53_graph_state

    return glm53_graph_state.tripwire_v2({}) is False


def _contig_off() -> bool:
    from vllm.v1.kv_offload.cpu import recurrent_sidecar

    return recurrent_sidecar.contig_enabled() is False


def _miss_attrib_off() -> bool:
    import glm53_miss_attrib

    return glm53_miss_attrib.from_env(128, print, {}) is None


def _lane_block_inert_when_flag_off() -> bool:
    import glm53_miss_attrib

    return glm53_miss_attrib.from_env(128, print, {LANE_BLOCK_NAME: "600"}) is None


def _prompt_keys_off() -> bool:
    import glm53_mixed_graph

    return glm53_mixed_graph.prompt_keys() == ()


def _capture_marks_off() -> bool:
    import glm53_mixed_graph

    return glm53_mixed_graph.capture_marks_enabled() is False


def _merge_check_off() -> bool:
    import glm53_mixed_graph

    return glm53_mixed_graph.merge_check_enabled() is False


def _split_forward_off() -> bool:
    import glm53_mixed_graph

    return glm53_mixed_graph.split_forward_enabled() is False


def _exact_shapes_off() -> bool:
    import glm53_mixed_graph

    return glm53_mixed_graph.exact_shapes_enabled() is False


def _native_pin_off() -> bool:
    import glm53_mixed_graph

    return glm53_mixed_graph.native_pin_enabled() is False


def _native_widths_off() -> bool:
    import glm53_mixed_graph

    return glm53_mixed_graph.native_widths(8) == ((8, 8),)


def _small_key_merge_off() -> bool:
    from vllm_gaudi.ops import hpu_glm5next_kpool

    return hpu_glm5next_kpool._USE_SMALL_KEY_PARTIAL_MERGE is False


def _final_norm_mark_off() -> bool:
    from vllm_gaudi.v1.worker import hpu_model_runner

    return hpu_model_runner._glm53_final_norm_mark_enabled() is False


def _tight_long_off() -> bool:
    import glm53_mixed_graph

    return glm53_mixed_graph.tight_long_context_enabled() is False


def _measured_estimate_off() -> bool:
    import glm53_mixed_graph

    return glm53_mixed_graph.measured_estimate_bytes() is None


def _hbm_probe_off() -> bool:
    import glm53_mixed_graph

    return glm53_mixed_graph.hbm_probe_enabled() is False


def _previous_hash_checkpoint_off() -> bool:
    """The flag is read inline in a scheduler method, so call that method on a stand-in at an exact reusable boundary.

    The stand-in answers True when the flag is on, so a False with the real environment is the flag reading off. If a control run
    with the flag set does not answer True the probe is blind (the method changed) and raises instead of reporting off.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler

    chunk = SimpleNamespace(tokens_per_chunk=64)
    scheduler = SimpleNamespace(_recurrent_manager=object(), recurrent_sidecar_ready=True,
                                config=SimpleNamespace(tokens_per_hash=64, kv_group_configs=[chunk]))
    status = SimpleNamespace(req=SimpleNamespace(num_prompt_tokens=256))
    ask = OffloadingConnectorScheduler._is_previous_hash_checkpoint
    answer = ask(scheduler, status, 192, 192)
    with mock.patch.dict(os.environ, {PREVIOUS_HASH_NAME: "1"}):
        control = ask(scheduler, status, 192, 192)
    if control is not True:
        raise RuntimeError(f"{PREVIOUS_HASH_NAME} probe is blind: the stand-in did not reach True with the flag set")
    return answer is False


def _shared_owner_off() -> bool:
    from vllm_gaudi.ops import hpu_glm5next_q4kv

    return hpu_glm5next_q4kv.shared_owner_enabled() is False


def _tail_fuse_off() -> bool:
    import glm53_mixed_tail_fuse

    return glm53_mixed_tail_fuse.enabled() is False


def _graphs_only_off() -> bool:
    import glm53_graph_state
    import glm53_mixed_graph

    return glm53_mixed_graph.graphs_only_enabled() is False and glm53_graph_state.graphs_only({}) is False


def _query_shard_off() -> bool:
    import glm53_mixed_graph

    return glm53_mixed_graph.query_shard_keys() == ()


def _query_shard_decode_off() -> bool:
    import glm53_mixed_graph

    return glm53_mixed_graph.decode_query_shard_keys() == ()


def _rank_check_off() -> bool:
    import glm53_mixed_graph

    return glm53_mixed_graph.rank_check_mode() == ""


def _sharded_kpool_off() -> bool:
    """glm53_mixed_graph only refuses a boot on it; the indexer's own reader is the base's, needing a live model to call."""
    return os.environ.get("HPU_GLM53_TP_SHARDED_KPOOL", "0").strip().lower() not in ("1", "true")


def _seal_ballast_off() -> bool:
    import glm53_mixed_graph

    glm53_mixed_graph.seal_boot_memory(None, glm53_mixed_graph.warmup_decode_lattice(None))  # no keys and no lattice: returns before it reads the runner or holds a ballast
    return glm53_mixed_graph.prompt_keys() == () and glm53_mixed_graph.decode_keys() == ()


def _lazy_namespaces_reached(selector: ModuleType) -> list[str]:
    """Which lazy op namespaces selector.select calls on a 128-row input, a recorder standing in for each (a card runs the real ops)."""
    import torch

    reached: list[str] = []

    def recorder(namespace: str) -> SimpleNamespace:
        return SimpleNamespace(select=lambda tile: (reached.append(namespace), tile[:, :2])[1])

    with mock.patch.object(torch.ops, "mla_fuse", recorder("mla_fuse"), create=True), \
            mock.patch.object(torch.ops, "g6_ws", recorder("g6_ws"), create=True), mock.patch.object(selector, "_announce_live"):
        selector.select(torch.zeros(128, 8))
    return reached


def _emit_wordskip_off() -> bool:
    """The module as the image imports it with the flag unset, asked to select: it must reach the stock mla_fuse op and never the g6_ws one.

    The control sets the module's own switch and must reach g6_ws, so a select that ignored the switch raises as a blind probe instead of
    reporting off. HPU_GLM53_EMIT_WORDSKIP_LAZY_LIB and HPU_GLM53_INDEXER_TPC_TOPK are read only by the word-skip boot check and share this probe.
    """
    with mock.patch.dict(os.environ, {"HPU_GLM53_MLA_TOPKFUSE": "0"}):  # the stock lazy library needs a card to load
        import glm53_mla_topkfuse as selector

    answer = _lazy_namespaces_reached(selector)
    with mock.patch.object(selector, "EMIT_WORDSKIP", True):
        control = _lazy_namespaces_reached(selector)
    if control != ["g6_ws"]:
        raise RuntimeError(f"HPU_GLM53_EMIT_WORDSKIP probe is blind: the selector reached {control} with its switch on, not ['g6_ws']")
    return answer == ["mla_fuse"] and selector.EMIT_WORDSKIP is False


def _gc_freeze_off() -> bool:
    """Flag unset: the seal hook returns before it touches the collector, the log or the ranks. A control with the flag set must reach freeze on a
    stand-in collector (the real heap of this check is never frozen), or the probe is blind and raises instead of reporting off."""
    import glm53_gc_freeze

    def calls_with(environ: dict[str, str]) -> list[str]:
        calls: list[str] = []
        collector = SimpleNamespace(collect=lambda: calls.append("collect") or 0, freeze=lambda: calls.append("freeze"),
                                    get_freeze_count=lambda: 1, isenabled=lambda: True)
        glm53_gc_freeze.reset()
        glm53_gc_freeze.freeze_at_seal(lambda *args: calls.append("log"), lambda *args: calls.append("admit"), environ=environ,
                                       collector=collector, rank=0)
        glm53_gc_freeze.reset()
        return calls

    answer, control = calls_with({}), calls_with({glm53_gc_freeze.FLAG: "1"})
    if "freeze" not in control:
        raise RuntimeError(f"{glm53_gc_freeze.FLAG} probe is blind: the hook did not reach freeze with the flag set (reached {control})")
    return answer == [] and glm53_gc_freeze.enabled({}) is False


PROBES: dict[str, Callable[[], bool]] = {
    "VLLM_GAUDI_MIXED_PROMPT_KEYS": _prompt_keys_off,
    "HPU_GLM53_R589_CAPTURE_MARKS": _capture_marks_off,
    "HPU_GLM53_PROMPT_KEY_MERGE_CHECK": _merge_check_off,
    "VLLM_GAUDI_MIXED_SPLIT_FORWARD": _split_forward_off,
    "VLLM_GAUDI_MIXED_EXACT_SHAPES": _exact_shapes_off,
    "VLLM_GAUDI_MIXED_NATIVE_PIN": _native_pin_off,
    "VLLM_GAUDI_MIXED_NATIVE_WIDTHS": _native_widths_off,
    "VLLM_GAUDI_MIXED_LONG_1792": _tight_long_off,
    "HPU_GLM53_SPARSE_DECODE_BLOCK_TIER_640": _tight_long_off,
    "HPU_GLM53_SPARSE_PREFILL_BLOCK_TIER_640": _tight_long_off,
    "VLLM_GAUDI_MIXED_GRAPH_ESTIMATE_MIB": _measured_estimate_off,
    "VLLM_GAUDI_MIXED_GRAPH_HBM_PROBE": _hbm_probe_off,
    "VLLM_RECURRENT_TAIL_FUSE": _tail_fuse_off,
    "HPU_GLM53_GS_TRIPWIRE_V2": _tripwire_v2_off,
    "VLLM_RECURRENT_SIDECAR_CONTIG": _contig_off,
    "VLLM_OFFLOAD_MISS_ATTRIB": _miss_attrib_off,
    LANE_BLOCK_NAME: _lane_block_inert_when_flag_off,
    PREVIOUS_HASH_NAME: _previous_hash_checkpoint_off,
    SHARED_OWNER_NAME: _shared_owner_off,
    "HPU_GLM53_GRAPHS_ONLY": _graphs_only_off,
    "VLLM_GAUDI_MIXED_SEAL_BALLAST_MIB": _seal_ballast_off,
    "HPU_GLM53_QUERY_SHARD_KEYS": _query_shard_off,
    "HPU_GLM53_QUERY_SHARD_DECODE_KEYS": _query_shard_decode_off,
    "HPU_GLM53_INDEXER_RANK_CHECK": _rank_check_off,
    "HPU_GLM53_TP_SHARDED_KPOOL": _sharded_kpool_off,
    "HPU_GLM53_FINAL_NORM_MARK": _final_norm_mark_off,
    "HPU_GLM53_SMALL_KEY_PARTIAL_MERGE": _small_key_merge_off,
    "HPU_GLM53_EMIT_WORDSKIP": _emit_wordskip_off,
    "HPU_GLM53_EMIT_WORDSKIP_LAZY_LIB": _emit_wordskip_off,
    "HPU_GLM53_INDEXER_TPC_TOPK": _emit_wordskip_off,
    "HPU_GLM53_GC_FREEZE_AT_SEAL": _gc_freeze_off,
}


def flag_names(text: str) -> set[str]:
    return set(FLAG_NAME.findall(text))


def new_flags() -> dict[str, list[str]]:
    """flag name -> installed files that introduce it."""
    found: dict[str, list[str]] = {}
    manifest = (STATE / "INSTALLED_SHA256.txt").read_text().splitlines()
    for line in manifest:
        relative = line.split("  ", 1)[1]
        if not relative.endswith(".py"):
            continue
        base_file = STATE / "base" / relative
        base_names = flag_names(base_file.read_text(encoding="utf-8")) if base_file.exists() else set()
        for name in flag_names((PKG / relative).read_text(encoding="utf-8")) - base_names:
            found.setdefault(name, []).append(relative)
    return found


def judge(name: str) -> str:
    """'off', 'ON', 'UNPROBED', 'base-image-env' or 'SET-IN-IMAGE-ENV' for one new flag."""
    if name in os.environ:
        return "base-image-env" if BASE_IMAGE_ENV.get(name) == os.environ[name] else "SET-IN-IMAGE-ENV"
    if name not in PROBES:
        return "UNPROBED"
    return "off" if PROBES[name]() else "ON"


def main() -> int:
    introduced = new_flags()
    verdicts = {name: judge(name) for name in sorted(introduced)}
    for name, verdict in verdicts.items():
        print(f"{verdict:>16}  {name}  <- {', '.join(introduced[name])}")
    bad = [name for name, verdict in verdicts.items() if verdict in ("ON", "SET-IN-IMAGE-ENV")]
    unprobed = [name for name, verdict in verdicts.items() if verdict == "UNPROBED"]
    allowed = [name for name, verdict in verdicts.items() if verdict == "base-image-env"]
    print(f"FLAGS: {len(verdicts) - len(bad) - len(unprobed) - len(allowed)} off, {len(allowed)} at base image env, {len(bad)} on, {len(unprobed)} unprobed {unprobed}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
