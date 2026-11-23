"""Craft gate for the overlay: every function <= 50 lines, nesting <= 3, type hints on every signature. Exit 1 on a miss."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

MAX_LINES = 50
MAX_DEPTH = 3
NESTING = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.AsyncFor, ast.AsyncWith)


def depth_of(node: ast.AST, depth: int = 0) -> int:
    deepest = depth
    for child in ast.iter_child_nodes(node):
        nested = depth + 1 if isinstance(child, NESTING) else depth
        deepest = max(deepest, depth_of(child, nested))
    return deepest


def misses_for(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    misses: list[str] = []
    length = (function.end_lineno or function.lineno) - function.lineno + 1
    if length > MAX_LINES:
        misses.append(f"{function.name}: {length} lines > {MAX_LINES}")
    depth = depth_of(function)
    if depth > MAX_DEPTH:
        misses.append(f"{function.name}: nesting {depth} > {MAX_DEPTH}")
    implicit = {"self", "cls"}  # the receiver of a method / classmethod carries no annotation
    unhinted = [a.arg for a in function.args.args + function.args.kwonlyargs
                if a.annotation is None and a.arg not in implicit]
    if unhinted or (function.returns is None and function.name != "__init__"):
        misses.append(f"{function.name}: missing type hints {unhinted or 'on the return'}")
    return misses


def main(paths: list[str]) -> int:
    misses: list[str] = []
    for path in paths:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                misses.extend(f"{path}:{node.lineno} {miss}" for miss in misses_for(node))
    print("\n".join(misses) if misses else f"craft OK: {len(paths)} file(s)")
    return 1 if misses else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
