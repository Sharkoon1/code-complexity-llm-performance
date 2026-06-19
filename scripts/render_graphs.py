"""Render call graphs and program dependence graphs as figures."""

import logging

import graphviz
import networkx as nx
import pandas as pd

from src.config import PATHS
from src.graphs import (
    _metrics_from_call_graph,
    _metrics_from_pdg,
    call_graph,
    program_dependence_graph,
)
from src.preprocess import normalize
from src.shared import cache_path

logger = logging.getLogger(__name__)

_NODE_STYLE = {
    "shape": "ellipse",
    "style": "filled",
    "fillcolor": "#eef3fb",
    "color": "#3b6ea5",
    "fontname": "Helvetica",
    "fontsize": "10",
}
_GRAPH_ATTR = {"rankdir": "TB", "labelloc": "b", "fontname": "Helvetica", "fontsize": "10"}
_EDGE_STYLE = {
    "control": {"color": "red", "style": "dashed"},
    "data": {"color": "blue", "style": "solid"},
    "both": {"color": "purple", "style": "solid"},
}
_CALL_EDGE_STYLE = {"color": "blue", "style": "solid"}


def _add_legend(dot: graphviz.Digraph, entries: list) -> None:
    """Legend box keyed by real sample arrows (color + line style), not words."""
    with dot.subgraph(name="cluster_legend") as lg:
        lg.attr(label="", color="gray80", style="rounded")
        lg.attr("node", fontname="Helvetica", fontsize="9")
        for i, (label, style) in enumerate(entries):
            tail, head = f"_lg{i}a", f"_lg{i}b"
            lg.node(tail, label="", shape="box", style="invis", width="0.45")
            lg.node(head, label=label, shape="plaintext")
            lg.edge(tail, head, arrowsize="0.7", **style)


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
    m = _metrics_from_call_graph(graph)
    caption = (
        '<<font point-size="13"><b>Call graph</b></font><br/><br/>'
        f'<font point-size="9">n_nodes={m["cg_n_nodes"]}&nbsp;&nbsp;&nbsp;n_edges={m["cg_n_edges"]}&nbsp;&nbsp;&nbsp;'
        f'max_out_degree={m["cg_max_out_degree"]}&nbsp;&nbsp;&nbsp;longest_chain={m["cg_longest_chain"]}&nbsp;&nbsp;&nbsp;'
        f'density={m["cg_density"]:.2f}&nbsp;&nbsp;&nbsp;n_cycles={m["cg_n_cycles"]}</font>>'
    )
    dot = graphviz.Digraph("call_graph", engine="dot")
    dot.attr(label=caption, **_GRAPH_ATTR)
    dot.attr("node", **_NODE_STYLE)
    for node in graph.nodes():
        dot.node(str(node), node.rsplit(".", 1)[-1])
    for u, v in graph.edges():
        dot.edge(str(u), str(v), **_CALL_EDGE_STYLE)
    _add_legend(dot, [("function call", _CALL_EDGE_STYLE)])
    return _render(dot, out_path, fmt)


def _pdg_label(line: int, lines: list) -> str:
    text = lines[line - 1].strip() if 0 < line <= len(lines) else ""
    if len(text) > 50:
        text = text[:47] + "..."
    return f"{line}: {text}"


def render_pdg(code: str, out_path: str = None, function: str = None, fmt: str = "pdf"):
    graph = program_dependence_graph(code, function)
    lines = code.splitlines()
    m = _metrics_from_pdg(graph)
    caption = (
        '<<font point-size="13"><b>Program dependence graph</b></font><br/><br/>'
        f'<font point-size="9">control_data_ratio={m["pdg_control_data_ratio"]:.2f}&nbsp;&nbsp;&nbsp;'
        f'longest_chain={m["pdg_longest_chain"]}&nbsp;&nbsp;&nbsp;'
        f'max_defuse_distance={m["pdg_max_defuse_distance"]}&nbsp;&nbsp;&nbsp;'
        f'unconditional_fraction={m["pdg_unconditional_fraction"]:.2f}&nbsp;&nbsp;&nbsp;'
        f'n_control_regions={m["pdg_n_control_regions"]}</font>>'
    )
    dot = graphviz.Digraph("pdg", engine="dot")
    dot.attr(label=caption, **_GRAPH_ATTR)
    dot.attr("node", **_NODE_STYLE)
    for node in graph.nodes():
        dot.node(str(node), _pdg_label(node, lines))
    for u, v, data in graph.edges(data=True):
        kind = data.get("kind")
        if kind == "both":
            dot.edge(str(u), str(v), **_EDGE_STYLE["control"])
            data_style = dict(_EDGE_STYLE["data"], label=f" {abs(u - v)}", fontsize="9", fontcolor="blue")
            dot.edge(str(u), str(v), **data_style)
        else:
            style = dict(_EDGE_STYLE.get(kind, {}))
            if kind == "data":
                style.update(label=f" {abs(u - v)}", fontsize="9", fontcolor="blue")
            dot.edge(str(u), str(v), **style)
    _add_legend(
        dot,
        [
            ("control dependence", _EDGE_STYLE["control"]),
            ("data dependence  (number = def-use distance)", _EDGE_STYLE["data"]),
        ],
    )
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
