"""HPU_GLM53_GRAPHS_ONLY, the parts steps 23 (guard witness mode) and 30 (graphs-only enforcement) share.

Standard library only, so the HPU graph guard (loaded at interpreter start) and the mixed-graph module (loaded with torch) import it alike.

Under the flag every serving forward replays a captured graph. A forward that cannot is refused, and a refusal must reach every TP rank:
a rank that raises while its peers wait in a graph's collective would hang them, so a refusal or a real allocation failure arms a bounded
exit. The engine normally stops every worker first; the timer is the backstop that turns one stuck rank into a stop of all of them.
"""
from __future__ import annotations

import contextlib
import logging
import os
import threading
from typing import Iterator

ENV = 'HPU_GLM53_GRAPHS_ONLY'
EXIT_AFTER_S = 120.0  # how long a refusing rank waits for the engine to stop its peers before it exits itself
EXIT_CODE = 3
ALLOCATION_MARKERS = ('PT_DEVMEM', 'out of memory', 'OOM', 'allocation failed')
LOG = logging.getLogger(__name__)


def _exit_process(label: str) -> None:
    LOG.error('GLM53_GRAPHS_ONLY_EXIT %s: no all-rank answer in %.0f s, so a peer may be stuck on the device; exiting so the executor '
              'stops every worker', label, EXIT_AFTER_S)
    os._exit(EXIT_CODE)


def arm_exit(label: str) -> threading.Timer:
    """Exit this process in EXIT_AFTER_S unless it is already gone: the engine stopping the workers ends the timer with the process."""
    timer = threading.Timer(EXIT_AFTER_S, _exit_process, args=(label,))
    timer.daemon = True
    timer.start()
    return timer


def is_allocation_failure(error: BaseException) -> bool:
    text = f'{type(error).__name__}: {error}'
    return any(marker in text for marker in ALLOCATION_MARKERS)


@contextlib.contextmanager
def fail_fast_on_allocation_error(where: str) -> Iterator[None]:
    """A real allocation failure inside ``where`` re-raises after arming the bounded exit; any other error passes through untouched."""
    try:
        yield
    except Exception as error:
        if is_allocation_failure(error):
            LOG.error('GLM53_GRAPHS_ONLY allocation failure in %s (%s): failing fast on every rank; no retry, no eager fallback',
                      where, type(error).__name__)
            arm_exit(where)
        raise
