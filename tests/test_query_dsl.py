# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""probe.query_dsl.safe_eval — correct on the predicate DSL, and rejects anything outside
the whitelist (it evaluates untrusted, public-repo query strings)."""
import pytest

from probe.query_dsl import UnsafeExpression, safe_eval


def _names() -> dict:
    return {
        "scene": "S",
        "t": 0,
        "clearance": lambda sc, t: 0.4,
        "speed": lambda sc, t: 12.0,
        "width": lambda sc: 1.85,
    }


def test_evaluates_and_of_comparisons():
    assert safe_eval("clearance(scene, t) < 0.5 and speed(scene, t) > 8.33", _names()) is True


def test_and_is_false_when_a_clause_fails():
    assert safe_eval("clearance(scene, t) < 0.5 and speed(scene, t) > 100", _names()) is False


def test_not_and_keyword_arguments():
    names = _names()
    names["freepath"] = lambda sc, t, horizon: horizon > 1.0
    assert safe_eval("not freepath(scene, t, horizon=0.0) and freepath(scene, t, horizon=2.0)", names) is True


def test_unknown_identifier_raises_nameerror():
    with pytest.raises(NameError):
        safe_eval("missing(scene, t) < 1.0", _names())


@pytest.mark.parametrize(
    "expr",
    [
        "().__class__.__bases__",        # attribute access -> sandbox-escape shape
        "__import__('os').system('x')",  # attribute call on a call
        "1 + 1",                          # arithmetic is not whitelisted
        "[x for x in scene]",            # comprehension
        "lambda: 1",                      # lambda
    ],
)
def test_rejects_non_whitelisted_syntax(expr):
    with pytest.raises((UnsafeExpression, NameError)):
        safe_eval(expr, _names())


def test_syntax_error_propagates():
    with pytest.raises(SyntaxError):
        safe_eval("speed(scene, class='v')", _names())  # 'class' is a Python keyword


def test_object_class_keyword_parses_and_evaluates():
    # the fix for the reserved-word problem: object_class= is a valid identifier, so the
    # tracking-baseline query parses and runs instead of raising SyntaxError
    names = _names()
    names["dist"] = lambda sc, t, object_class=None: 1.5
    assert safe_eval("dist(scene, t, object_class='vehicle') < 2.0", names) is True
