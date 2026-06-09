"""Tests for the semantic block-tree construction (src/block_tree.py).
"""

import pytest

from src.block_tree import get_code_with_boundaries, CodeBlockProcessor
from src.tree_sitter_config import py_parser
from src.lm_cc import _get_lmcc, _get_total_branch, _get_max_depth


class TestTreeSitter:
    def test_parser_parses(self):
        tree = py_parser.parse(b"def f():\n    if x:\n        return 1\n")
        assert tree.root_node.type == "module"


class TestSegmentationHierarchy:
    def _build(self, toks, ents):
        cb, _, set_ = get_code_with_boundaries(toks, ents, threshold=0.67)
        return CodeBlockProcessor().parse_code_blocks(cb, toks, set_)

    def test_control_block_keeps_body_nested(self):
        # "if x:\n    y=1\nz=2" with a high-entropy token on each line.
        toks = ["▁if", "▁x", ":", "<0x0A>", "▁", "▁y", "=", "1", "<0x0A>", "▁z", "=", "2"]
        ents = [0.0, 0.9, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1]
        bt = self._build(toks, ents)
        # the if-block absorbs the interior boundary and keeps y=1 nested
        # (root=1, if=2, y=3); z=2 stays a sibling of the if-block
        assert _get_max_depth(bt) == 3
        assert _get_total_branch(bt) == 3   # root, if, y, z -> N-1

    def test_no_boundaries_single_block(self):
        toks = ["▁a", "▁=", "▁1"]
        ents = [0.0, 0.0, 0.0]
        bt = self._build(toks, ents)
        assert _get_total_branch(bt) == 0
        assert _get_lmcc(bt) == pytest.approx(0.2)   # root-only tree (depth 1)
