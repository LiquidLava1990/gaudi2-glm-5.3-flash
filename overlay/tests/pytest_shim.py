"""The small pytest surface pin_cases_c789.py uses, so dual-4m's cases run in the image (it has no pytest; no pip install)."""
from __future__ import annotations

import contextlib
import logging
import os
import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any


class Mark:
    @staticmethod
    def parametrize(names: str | Sequence[str], values: list) -> Callable:
        listed = [names] if isinstance(names, str) else list(names)

        def decorate(function: Callable) -> Callable:
            function.parameters = (listed, values)
            return function

        return decorate


mark = Mark()


def fixture(*_args: object, **_kwargs: object) -> Callable:
    return lambda function: function


@contextlib.contextmanager
def raises(kind: type[BaseException], match: str | None = None) -> Iterator[None]:
    try:
        yield
    except kind as error:
        if match is not None and not re.search(match, str(error)):
            raise AssertionError(f"{error!s} does not match {match!r}") from error
        return
    raise AssertionError(f"{kind.__name__} was not raised")


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


class Patcher:
    """monkeypatch: every change is undone in reverse order by undo()."""

    def __init__(self) -> None:
        self.undos: list[Callable[[], object]] = []

    def setenv(self, name: str, value: str) -> None:
        previous = os.environ.get(name)
        self.undos.append(lambda: _restore_env(name, previous))
        os.environ[name] = value

    def delenv(self, name: str, raising: bool = True) -> None:
        previous = os.environ.get(name)
        self.undos.append(lambda: _restore_env(name, previous))
        os.environ.pop(name, None)

    def setattr(self, owner: Any, name: str, value: Any, raising: bool = True) -> None:
        previous = getattr(owner, name)
        self.undos.append(lambda: setattr(owner, name, previous))
        setattr(owner, name, value)

    def setitem(self, mapping: Any, key: Any, value: Any) -> None:
        missing = key not in mapping
        previous = mapping.get(key)
        self.undos.append(lambda: mapping.pop(key, None) if missing else mapping.__setitem__(key, previous))
        mapping[key] = value

    def undo(self) -> None:
        while self.undos:
            self.undos.pop()()


class CapLog(logging.Handler):
    """caplog: records of the named logger."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []
        self.watched: list[logging.Logger] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    @property
    def text(self) -> str:
        return "\n".join(record.getMessage() for record in self.records)

    def set_level(self, level: str | int, logger: str = "") -> None:
        target = logging.getLogger(logger)
        target.setLevel(level)
        target.addHandler(self)
        self.watched.append(target)

    def close_all(self) -> None:
        for target in self.watched:
            target.removeHandler(self)
