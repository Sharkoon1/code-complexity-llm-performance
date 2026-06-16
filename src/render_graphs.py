"""Render call graphs and program dependence graphs as figures."""

import logging

import graphviz
import networkx as nx
import pandas as pd

from src.config import PATHS
from src.graphs import call_graph, program_dependence_graph
from src.preprocess import normalize
from src.shared import cache_path

logger = logging.getLogger(__name__)

_EDGE_STYLE = {
    "control": {"color": "red", "style": "dashed"},
    "data": {"color": "blue", "style": "solid"},
    "both": {"color": "purple", "style": "solid"},
}


def _render(dot: graphviz.Digraph, out_path: str, fmt: str) -> graphviz.Digraph:
    if out_path is not None:
        path = dot.render(out_path, format=fmt, cleanup=True)
        logger.info(f"wrote {path}")
    return dot


def _function_subgraph(graph: nx.DiGraph, function: str) -> nx.DiGraph:
    roots = [n for n in graph if n == function or n.rsplit(".", 1)[-1] == function]
    keep = set(roots)
    for root in roots:
        keep |= nx.descendants(graph, root)
    return graph.subgraph(keep).copy()


def render_call_graph(
    code: str, out_path: str = None, function: str = None, fmt: str = "pdf"
):
    graph = call_graph(code)
    if function:
        graph = _function_subgraph(graph, function)

    dot = graphviz.Digraph("call_graph")
    dot.attr(rankdir="LR")
    dot.attr("node", shape="box", fontname="Helvetica", fontsize="10")
    for node in graph.nodes():
        dot.node(str(node), node.rsplit(".", 1)[-1])
    for u, v in graph.edges():
        dot.edge(str(u), str(v))
    return _render(dot, out_path, fmt)


def _pdg_label(line: int, lines: list) -> str:
    text = lines[line - 1].strip() if 0 < line <= len(lines) else ""
    if len(text) > 50:
        text = text[:47] + "..."
    return f"{line}: {text}"


def render_pdg(code: str, out_path: str = None, function: str = None, fmt: str = "pdf"):
    graph = program_dependence_graph(code, function)
    lines = code.splitlines()

    dot = graphviz.Digraph("pdg")
    dot.attr(
        rankdir="TB", label="control: red dashed   data: blue solid", fontsize="10"
    )
    dot.attr("node", shape="box", fontname="Helvetica", fontsize="10")
    for node in graph.nodes():
        dot.node(str(node), _pdg_label(node, lines))
    for u, v, data in graph.edges(data=True):
        dot.edge(str(u), str(v), **_EDGE_STYLE.get(data.get("kind"), {}))
    return _render(dot, out_path, fmt)


def render_instance(
    instance_id: str,
    kind: str = "call",
    *,
    dataset=PATHS.VERIFIED_FILTERED_DATASET,
    out_path: str = None,
    function: str = None,
    fmt: str = "pdf",
):
    df = pd.read_parquet(dataset)
    rows = df[df["instance_id"] == instance_id]
    if rows.empty:
        raise ValueError(f"{instance_id} not found in {dataset}")
    row = rows.iloc[0]

    raw_code = cache_path(
        row["repo"], row["base_commit"], row["python_files"][0]
    ).read_text()
    code = normalize(raw_code)
    if code is None:
        raise ValueError(f"preprocessing (black) failed for {instance_id}")

    if kind == "call":
        return render_call_graph(code, out_path=out_path, function=function, fmt=fmt)
    if kind == "pdg":
        return render_pdg(code, out_path=out_path, function=function, fmt=fmt)
    raise ValueError(f"unknown kind {kind!r} (use 'call' or 'pdg')")
