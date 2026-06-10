"""Tests for the LM-CC feature definitions + the Proposition B.1 structural claim."""

import os

import numpy as np
import pytest

from src.lm_cc import (
    _get_lmcc,
    _get_total_branch,
    _get_depth_sum,
    _get_max_depth,
    _get_avg_depth,
    _get_max_width,
    _get_avg_children,
)


def _chain(n: int) -> dict:
    """Nested tree of n nodes (root -> a -> b -> ...), like deeply nested code."""
    root = node = {"children": []}
    for _ in range(n - 1):
        child = {"children": []}
        node["children"].append(child)
        node = child
    return root


def _star(n: int) -> dict:
    """Flat tree: root with n leaf children, like sequential top-level code."""
    return {"children": [{"children": []} for _ in range(n)]}


class TestFeatureDefinitions:
    # tree (N=4): root -> [A -> [c], B]; depths (root=1): 1, 2, 3, 2
    TREE = {"children": [{"children": [{"children": []}]}, {"children": []}]}

    def test_total_branch(self):
        assert _get_total_branch(self.TREE) == 3  # N - 1

    def test_depth_sum(self):
        assert _get_depth_sum(self.TREE) == 1 + 2 + 3 + 2  # 8

    def test_lmcc(self):
        assert _get_lmcc(self.TREE) == pytest.approx(0.8 * 3 + 0.2 * 8)

    def test_max_depth(self):
        assert _get_max_depth(self.TREE) == 3

    def test_avg_depth(self):
        assert _get_avg_depth(self.TREE) == pytest.approx(8 / 4)

    def test_max_width(self):
        assert _get_max_width(self.TREE) == 2  # level 2 holds A and B

    def test_avg_children(self):
        assert _get_avg_children(self.TREE) == pytest.approx(3 / 2)

    def test_empty_root(self):
        assert _get_lmcc({"children": []}) == pytest.approx(0.2)


class TestPropositionB1:
    """Nested-shaped trees -> Theta(n^2) depth_sum; flat -> Theta(n)."""

    NS = np.array([2, 4, 6, 8, 10])

    def test_nested_quadratic(self):
        vals = np.array([_get_depth_sum(_chain(n)) for n in self.NS])
        assert np.polyfit(self.NS, vals, 2)[0] > 0

    def test_flat_linear(self):
        vals = np.array([_get_depth_sum(_star(n)) for n in self.NS])
        assert abs(np.polyfit(self.NS, vals, 2)[0]) < 1e-3
