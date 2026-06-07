"""
Regression tests for src.metrics helpers.

Covers the docstring-stripping in `_normalize_code`:
removing a docstring that is the sole statement of a class/function must not
leave an empty (invalid) body.
"""

import ast

import pytest

from src.metrics import _normalize_code


class TestNormalizeCodeDocstrings:
    @pytest.mark.parametrize("code", [
        'class Empty:\n    """Only a docstring."""\n',
        'def stub():\n    """Only a docstring."""\n',
        'class A:\n    def m(self):\n        """only doc"""\n',
        'async def afn():\n    """only doc"""\n',
    ])
    def test_normalize_keeps_body_valid(self, code):
        normalized = _normalize_code(code)
        ast.parse(normalized)           # must stay parseable (no empty body)
        assert "pass" in normalized     # emptied body replaced by `pass`

    def test_normalize_still_strips_docstring_when_body_remains(self):
        normalized = _normalize_code(
            'def f(x):\n    """prose to remove"""\n    return x + 1\n'
        )
        assert "prose" not in normalized
        assert "return x + 1" in normalized

    def test_unparseable_code_returned_unchanged(self):
        broken = "def f(:\n    pass"
        assert _normalize_code(broken) == broken
