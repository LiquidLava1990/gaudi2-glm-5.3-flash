# SPDX-License-Identifier: Apache-2.0
"""Exact k512 selector with bounded 128-row recipe workspace.

HPU_GLM53_EMIT_WORDSKIP=1 (default off; WIDE-PROMPT-KERNEL-PLAN step 6) ends the selector in the word-skip emit kernel, GUID
g6_topk_emit_wordskip_i32, instead of g6_topk_emit_i32. The ids are bit-identical (60 of 60 card cells); the scalar loads that gave a dead
or tie-free slice its answer are not issued. The node chain is keys -> state -> emit either way and only the last GUID changes, so
torch.ops.g6_ws.select (libg6_ws_lazy.so) stands in for torch.ops.mla_fuse.select and nothing else moves.

On, it needs the settings below, and the worker stops at import, before any library loads, naming each one that is missing:
  HPU_GLM53_MLA_TOPKFUSE=1 and HPU_GLM53_INDEXER_TPC_TOPK=1   the selector only runs under both
  HPU_GLM53_EMIT_WORDSKIP_LAZY_LIB                           the lazy op library, default /opt/glm53/emit/libg6_ws_lazy.so; it must name
                                                             the word-skip GUID and not the stock one (a stale build is refused)
  G6_PERFLIB                                                 the perf-lib the kernel library chain loads (libmla_filter.so ->
                                                             libg6_k512_filter.so -> G6_PERFLIB) must carry the GUID. glm53_rc6_flags pins
                                                             G6_PERFLIB to /opt/glm53/rc6/libg6_k512_perflib.so while the tpc top-k is on,
                                                             so libg6_perflib.so is mounted over that path, in flag-on runs only
                                                             (diag-emit/RUN.txt); this module reads the file the setting names.
The kernel and its lazy op are adapted from 1CatAI/1Cat-vLLM-Gaudi (Apache-2.0, commit e2266e0e); LICENSE-1Cat-vLLM-Gaudi and
NOTICE-1Cat-deepseek_v4 ship beside the libraries (/opt/glm53/emit/ and /opt/glm53/rc6/licenses/).
"""
import logging
import os
from collections.abc import Callable, Mapping

import torch
from glm53_mixed_graph import capture_tile_marks, key_capture_active, witness_lever

ENABLED = os.environ.get('HPU_GLM53_MLA_TOPKFUSE', '0') == '1'
WIDTHS = frozenset({20480, 32768, 73728})

EMIT_WORDSKIP_FLAG = 'HPU_GLM53_EMIT_WORDSKIP'
WORDSKIP_LIB_FLAG = 'HPU_GLM53_EMIT_WORDSKIP_LAZY_LIB'
TPC_TOPK_FLAG = 'HPU_GLM53_INDEXER_TPC_TOPK'
TOPKFUSE_FLAG = 'HPU_GLM53_MLA_TOPKFUSE'
PERFLIB_ENV = 'G6_PERFLIB'
EMIT_DIR = '/opt/glm53/emit'
DEFAULT_WORDSKIP_LIB = EMIT_DIR + '/libg6_ws_lazy.so'
STOCK_PERFLIB = '/opt/glm53/rc6/libg6_k512_perflib.so'  # where glm53_rc6_flags pins G6_PERFLIB; the word-skip perf-lib is mounted over it
STOCK_EMIT_GUID = 'g6_topk_emit_i32'
WORDSKIP_EMIT_GUID = 'g6_topk_emit_wordskip_i32'
BUILD_HINT = 'build both with emit-wordskip/build_emit_wordskip.sh and mount them as diag-emit/RUN.txt shows'
LOG = logging.getLogger('vllm.rc6')  # the logger FLASH_K512_LIVE uses
_LIVE_LOGGED: list[bool] = []


def emit_wordskip_enabled(environ: Mapping[str, str] = os.environ) -> bool:
    """HPU_GLM53_EMIT_WORDSKIP: unset or 0 keeps the stock emit, 1 runs the word-skip emit; any other value stops the boot, not a silent off."""
    value = environ.get(EMIT_WORDSKIP_FLAG, '0')
    if value not in ('0', '1'):
        raise RuntimeError(f'{EMIT_WORDSKIP_FLAG}={value!r} is not 0 or 1: set it to 1 to run the word-skip emit, or unset it for the stock emit')
    return value == '1'


def wordskip_lazy_library(environ: Mapping[str, str] = os.environ) -> str:
    return environ.get(WORDSKIP_LIB_FLAG, DEFAULT_WORDSKIP_LIB)


def _readable_file(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.R_OK)


def _emit_guids_named(path: str) -> list[str]:
    """The emit GUIDs, stock and word-skip, whose names a readable file holds; empty for a file that cannot be read."""
    if not _readable_file(path):
        return []
    with open(path, 'rb') as library:
        content = library.read()
    return [guid for guid in (STOCK_EMIT_GUID, WORDSKIP_EMIT_GUID) if guid.encode() in content]


def lazy_library_emit_guid(path: str) -> str | None:
    """The one emit GUID a lazy op library's node chain ends in (its AddNode hands it to BuildOp); None when the file is unreadable or names both or neither."""
    named = _emit_guids_named(path)
    return named[0] if len(named) == 1 else None


def _lazy_library_problem(lazy: str) -> str | None:
    if not _readable_file(lazy):
        return f'{lazy} is not a readable file: mount libg6_ws_lazy.so there or point {WORDSKIP_LIB_FLAG} at it ({BUILD_HINT})'
    if lazy_library_emit_guid(lazy) == WORDSKIP_EMIT_GUID:
        return None
    named = ', '.join(_emit_guids_named(lazy)) or 'no emit GUID'
    return f'{lazy} does not select {WORDSKIP_EMIT_GUID} (it names {named}): rebuild libg6_ws_lazy.so ({BUILD_HINT})'


def _perflib_problem(perflib: str) -> str | None:
    if WORDSKIP_EMIT_GUID in _emit_guids_named(perflib):
        return None
    return (f'{PERFLIB_ENV}={perflib or "(unset)"} is not a perf-lib that carries {WORDSKIP_EMIT_GUID}: {TPC_TOPK_FLAG}=1 pins '
            f'{PERFLIB_ENV} to {STOCK_PERFLIB} (glm53_rc6_flags), so mount libg6_perflib.so over that path instead of changing '
            f'the setting ({BUILD_HINT})')


def wordskip_boot_problems(environ: Mapping[str, str] = os.environ, topkfuse_enabled: bool = ENABLED) -> list[str]:
    """What keeps the word-skip emit from running, each with the setting or build that fixes it; empty when it can run."""
    problems = []
    if not topkfuse_enabled:
        problems.append(f'set {TOPKFUSE_FLAG}=1: the word-skip emit is the last node of the topkfuse select, so with it off the flag would do nothing')
    if environ.get(TPC_TOPK_FLAG, '0') == '0':
        problems.append(f'set {TPC_TOPK_FLAG}=1: glm53_tpc_topk is the only caller of the topkfuse select, so with it off the flag would do nothing')
    library_problems = (_lazy_library_problem(wordskip_lazy_library(environ)), _perflib_problem(environ.get(PERFLIB_ENV, '')))
    return problems + [problem for problem in library_problems if problem]


def require_wordskip_ready(environ: Mapping[str, str] = os.environ, topkfuse_enabled: bool = ENABLED) -> None:
    problems = wordskip_boot_problems(environ, topkfuse_enabled)
    if problems:
        raise RuntimeError(f'{EMIT_WORDSKIP_FLAG}=1 cannot boot: ' + '; '.join(problems))


EMIT_WORDSKIP = emit_wordskip_enabled()
LAZY_EMIT_GUID = ''  # the GUID the loaded lazy library names, read from the file; empty with the flag off
if EMIT_WORDSKIP:
    require_wordskip_ready(os.environ, ENABLED)
    LAZY_EMIT_GUID = lazy_library_emit_guid(wordskip_lazy_library()) or ''
if ENABLED:
    torch.ops.load_library('/opt/glm53/mla/libmla_topkfuse_lazy.so')
if EMIT_WORDSKIP:
    torch.ops.load_library(wordskip_lazy_library())


def applies(scores: torch.Tensor) -> bool:
    return (ENABLED and scores.device.type == 'hpu' and scores.dtype == torch.float32
            and scores.ndim == 2 and scores.shape[0] >= 128 and scores.shape[1] in WIDTHS)


def _select_op() -> Callable[[torch.Tensor], torch.Tensor]:
    return torch.ops.g6_ws.select if EMIT_WORDSKIP else torch.ops.mla_fuse.select


def _announce_live(scores: torch.Tensor) -> None:
    """One FLASH_EMIT_WORDSKIP_LIVE line per process, on the first select (a capture): proof the word-skip path ran, as FLASH_K512_LIVE is for k512."""
    if _LIVE_LOGGED:
        return
    _LIVE_LOGGED.append(True)
    import glm53_rc6_flags as rc6_flags  # stdlib only, and already imported by kpool before this module
    LOG.warning('FLASH_EMIT_WORDSKIP_LIVE rank=%d rows=%d width=%d guid=%s',
                rc6_flags.process_rank(), scores.shape[0], scores.shape[1], LAZY_EMIT_GUID)


def _tile_mark(mark_step: Callable[[], None]) -> None:
    if capture_tile_marks() or not key_capture_active():
        mark_step()


def select(scores: torch.Tensor) -> torch.Tensor:
    witness_lever('mla_topkfuse')
    if EMIT_WORDSKIP:
        _announce_live(scores)
    select_op = _select_op()
    if scores.shape[0] <= 128:
        return select_op(scores)
    from habana_frameworks.torch.core import mark_step
    parts = []
    for start in range(0, scores.shape[0], 128):
        tile = scores[start:start + 128]
        _tile_mark(mark_step)
        parts.append(select_op(tile))
        _tile_mark(mark_step)
    return torch.cat(parts, dim=0)
