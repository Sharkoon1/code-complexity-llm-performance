"""Tests for the call graph and program dependence graph."""

import graphviz
import networkx as nx
import pytest

from src.graphs import (
    _control_metrics,
    _data_metrics,
    _longest_chain,
    _metrics_from_call_graph,
    _n_cycles,
    program_dependence_graph,
)
from scripts.render_graphs import _function_subgraph, render_call_graph, render_pdg


def _chain(n: int) -> nx.DiGraph:
    g = nx.DiGraph()
    nx.add_path(g, range(n))
    return g


class TestGraphHelpers:
    def test_longest_chain(self):
        assert _longest_chain(_chain(4)) == 3

    def test_longest_chain_empty(self):
        assert _longest_chain(nx.DiGraph()) == 0

    def test_cycles_selfloop(self):
        assert _n_cycles(nx.DiGraph([(1, 1)])) == 1

    def test_cycles_mutual(self):
        assert _n_cycles(nx.DiGraph([(1, 2), (2, 1)])) == 1

    def test_no_cycles(self):
        assert _n_cycles(_chain(3)) == 0


class TestCallGraphMetrics:
    GRAPH = nx.DiGraph([("f", "g"), ("f", "h"), ("g", "g")])

    def test_zero_on_empty(self):
        assert _metrics_from_call_graph(nx.DiGraph()) == {
            "cg_n_nodes": 0,
            "cg_n_edges": 0,
            "cg_max_out_degree": 0,
            "cg_longest_chain": 0,
            "cg_density": 0.0,
            "cg_n_cycles": 0,
        }

    def test_metrics(self):
        m = _metrics_from_call_graph(self.GRAPH)
        assert m["cg_n_nodes"] == 3
        assert m["cg_n_edges"] == 3
        assert m["cg_max_out_degree"] == 2
        assert m["cg_n_cycles"] == 1
        assert m["cg_density"] == pytest.approx(3 / 6)


class TestDataMetrics:
    EDGES = [(1, 2), (2, 3)]

    def test_zero_on_empty(self):
        m = _data_metrics([])
        assert m["pdg_longest_chain"] == 0
        assert m["pdg_density"] == 0.0

    def test_metrics(self):
        m = _data_metrics(self.EDGES)
        assert m["pdg_longest_chain"] == 2
        assert m["pdg_max_defuse_distance"] == 1
        assert m["pdg_density"] == pytest.approx(2 / 3)
        assert m["pdg_max_fan_in"] == 1
        assert m["pdg_max_fan_out"] == 1
        assert m["pdg_mean_degree"] == pytest.approx(4 / 3)
        assert m["pdg_slice_max"] == pytest.approx(2 / 3)
        assert m["pdg_slice_mean"] == pytest.approx((0 + 1 / 3 + 2 / 3) / 3)
        assert m["pdg_n_cycles"] == 0


class TestControlMetrics:
    NODES = [1, 2, 3, 4]
    CONTROL = [(2, 3), (2, 4)]

    def test_zero_on_empty(self):
        m = _control_metrics([], [], 0)
        assert m["pdg_control_data_ratio"] == 0.0
        assert m["pdg_unconditional_fraction"] == 0.0

    def test_metrics(self):
        m = _control_metrics(self.NODES, self.CONTROL, n_data_edges=2)
        assert m["pdg_control_data_ratio"] == pytest.approx(1.0)
        assert m["pdg_unconditional_fraction"] == pytest.approx(0.5)
        assert m["pdg_n_control_regions"] == 2
        assert m["pdg_control_span_max"] == 1
        assert m["pdg_control_span_mean"] == pytest.approx(1.0)
        assert m["pdg_control_betweenness_max"] == pytest.approx(0.0)


class TestRealPdg:
    CODE = (
        "def f(x):\n"
        "    a = x + 1\n"
        "    if a > 0:\n"
        "        b = a * 2\n"
        "    else:\n"
        "        b = a - 2\n"
        "    return b\n"
    )

    def test_has_control_and_data_edges(self):
        graph = program_dependence_graph(self.CODE)
        kinds = {d.get("kind") for _, _, d in graph.edges(data=True)}
        assert "control" in kinds
        assert "data" in kinds


class TestRender:
    CODE = TestRealPdg.CODE

    def test_call_graph_is_digraph(self):
        dot = render_call_graph("def a():\n    return b()\n\ndef b():\n    return 1\n")
        assert isinstance(dot, graphviz.Digraph)

    def test_pdg_shows_both_edge_colors(self):
        dot = render_pdg(self.CODE)
        assert isinstance(dot, graphviz.Digraph)
        assert "red" in dot.source
        assert "blue" in dot.source

    def test_function_subgraph(self):
        graph = nx.DiGraph([("m.f", "m.g"), ("m.g", "m.h"), ("m.x", "m.y")])
        sub = _function_subgraph(graph, "f")
        assert set(sub.nodes()) == {"m.f", "m.g", "m.h"}
