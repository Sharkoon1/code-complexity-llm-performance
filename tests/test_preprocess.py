"""Tests for the shared source preprocessing (src/preprocess.py)."""

import shutil

import pytest

from src.preprocess import (
    _remove_comments_and_docstrings,
    _format_python_code,
    normalize,
)

HAS_BLACK = shutil.which("black") is not None


class TestRemoveCommentsDocstrings:
    def test_removes_comments(self):
        out = _remove_comments_and_docstrings("x = 1  # inline\n# full\ny = 2\n")
        assert "#" not in out
        assert "x = 1" in out and "y = 2" in out

    def test_removes_docstring(self):
        out = _remove_comments_and_docstrings(
            'def f():\n    """doc prose"""\n    return 1\n'
        )
        assert "prose" not in out
        assert "return 1" in out

    def test_keeps_string_literals(self):
        out = _remove_comments_and_docstrings('x = "not a # comment"\n')
        assert "not a # comment" in out


@pytest.mark.skipif(not HAS_BLACK, reason="black CLI not on PATH")
class TestFormatAndNormalize:
    def test_black_formats(self):
        out = _format_python_code("def f( x ):\n  return  x+1\n")
        assert out is not None
        assert "def f(x):" in out
        assert "return x + 1" in out

    def test_strips_decorators(self):
        out = _format_python_code("@deco\ndef f():\n    return 1\n")
        assert "@deco" not in out

    def test_normalize_pipeline(self):
        out = normalize('def f(x):\n    """doc"""\n    return x+1  # c\n')
        assert out is not None
        assert "doc" not in out and "# c" not in out
        assert "return x + 1" in out
