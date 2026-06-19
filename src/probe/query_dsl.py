# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Safe evaluator for the occquery predicate DSL.

queries.yaml lives in a PUBLIC repo, so a query expression is untrusted input — a PR could
add a malicious one. Python's eval() with a stripped __builtins__ is NOT a sandbox; it is
bypassable (e.g. via ``().__class__.__bases__``...). Instead we parse the expression to an
AST and evaluate only a whitelist of node types: boolean ops, ``not``, comparisons,
literals, the bare names the caller provides, and calls to caller-supplied functions. Any
other syntax raises UnsafeExpression — the wrong thing is unrepresentable, not filtered.
"""
from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any

__all__ = ["safe_eval", "UnsafeExpression"]

_COMPARATORS: dict[type, Callable[[Any, Any], bool]] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


class UnsafeExpression(ValueError):
    """Raised when an expression uses syntax outside the predicate DSL whitelist."""


def safe_eval(expr: str, names: dict[str, Any]) -> Any:
    """Evaluate a whitelisted boolean/comparison expression.

    `names` maps every allowed identifier (predicate functions, `scene`, `t`) to its value.
    Raises NameError for an unknown identifier (callers read that as "predicate not in the
    v0 core"), UnsafeExpression for disallowed syntax, SyntaxError if it does not parse.
    """
    return _eval(ast.parse(expr, mode="eval").body, names)


def _eval(node: ast.AST, names: dict[str, Any]) -> Any:
    if isinstance(node, ast.BoolOp):
        values = [_eval(v, names) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
        raise UnsafeExpression(f"boolean operator {type(node.op).__name__}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval(node.operand, names)
    if isinstance(node, ast.Compare):
        left = _eval(node.left, names)
        for op, comparator in zip(node.ops, node.comparators):
            fn = _COMPARATORS.get(type(op))
            if fn is None:
                raise UnsafeExpression(f"comparison operator {type(op).__name__}")
            right = _eval(comparator, names)
            if not fn(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise UnsafeExpression("only direct calls to named functions are allowed")
        func = _lookup(node.func.id, names)
        args = [_eval(a, names) for a in node.args]
        kwargs: dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise UnsafeExpression("dict unpacking is not allowed")
            kwargs[kw.arg] = _eval(kw.value, names)
        return func(*args, **kwargs)
    if isinstance(node, ast.Name):
        return _lookup(node.id, names)
    if isinstance(node, ast.Constant):
        return node.value
    raise UnsafeExpression(f"disallowed syntax: {type(node).__name__}")


def _lookup(name: str, names: dict[str, Any]) -> Any:
    if name not in names:
        raise NameError(name)
    return names[name]
