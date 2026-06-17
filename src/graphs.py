"""Scalpel call graph and program dependence graph."""

import logging
import multiprocessing as mp
import os
import tempfile
import time
from queue import Empty

import networkx as nx
from scalpel.SSA.const import SSA
from scalpel.call_graph.pycg import CallGraphGenerator
from scalpel.cfg import CFGBuilder

logger = logging.getLogger(__name__)

_CALL_GRAPH_TIMEOUT = int(os.environ.get("GRAPH_CALL_TIMEOUT", "300"))


def _longest_chain(graph: nx.DiGraph) -> int:
    if graph.number_of_nodes() == 0:
        return 0
    return nx.dag_longest_path_length(nx.condensation(graph))


def _n_cycles(graph: nx.DiGraph) -> int:
    selfloops = nx.number_of_selfloops(graph)
    nontrivial = sum(1 for c in nx.strongly_connected_components(graph) if len(c) > 1)
    return selfloops + nontrivial


def _zero_call_graph_metrics() -> dict:
    return {
        "cg_n_nodes": 0,
        "cg_n_edges": 0,
        "cg_max_out_degree": 0,
        "cg_longest_chain": 0,
        "cg_density": 0.0,
        "cg_n_cycles": 0,
    }


def _metrics_from_call_graph(graph: nx.DiGraph) -> dict:
    n = graph.number_of_nodes()
    if n == 0:
        return _zero_call_graph_metrics()
    out_degrees = [d for _, d in graph.out_degree()]
    return {
        "cg_n_nodes": n,
        "cg_n_edges": graph.number_of_edges(),
        "cg_max_out_degree": max(out_degrees),
        "cg_longest_chain": _longest_chain(graph),
        "cg_density": nx.density(graph),
        "cg_n_cycles": _n_cycles(graph),
    }


def _internal_functions(generator: CallGraphGenerator) -> set:
    names = set()
    for mod_name, mod in generator.output_internal_mods().items():
        for qualified in mod.get("methods", {}):
            if qualified != mod_name:
                names.add(qualified)
    return names


def _build_call_graph(code: str) -> nx.DiGraph:
    with tempfile.TemporaryDirectory() as tmp:
        module = os.path.join(tmp, "module.py")
        with open(module, "w", encoding="utf-8") as fh:
            fh.write(code)
        generator = CallGraphGenerator([module], tmp)
        generator.analyze()
        functions = _internal_functions(generator)
        edges = [
            (caller, callee)
            for caller, callee in generator.output_edges()
            if caller in functions and callee in functions
        ]

    graph = nx.DiGraph()
    graph.add_nodes_from(functions)
    graph.add_edges_from(edges)
    return graph


def _call_graph_worker(code: str, out_queue) -> None:
    try:
        out_queue.put(("ok", _metrics_from_call_graph(_build_call_graph(code))))
    except Exception as e:
        out_queue.put(("err", f"{type(e).__name__}: {e}"))


def _call_graph_metrics(code: str) -> dict:
    n_lines = code.count("\n") + 1
    logger.info(f"call graph: analyzing {n_lines} lines with PyCG ...")
    start = time.perf_counter()

    ctx = mp.get_context("fork")
    out_queue = ctx.Queue()
    proc = ctx.Process(target=_call_graph_worker, args=(code, out_queue), daemon=True)
    proc.start()
    try:
        status, payload = out_queue.get(timeout=_CALL_GRAPH_TIMEOUT)
    except Empty:
        proc.terminate()
        proc.join()
        logger.warning(
            f"call graph TIMED OUT after {_CALL_GRAPH_TIMEOUT}s ({n_lines} lines); using zeros"
        )
        return _zero_call_graph_metrics()
    finally:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()

    if status == "err":
        logger.warning(f"call graph FAILED ({n_lines} lines): {payload}")
        return _zero_call_graph_metrics()

    elapsed = time.perf_counter() - start
    if elapsed > 1:
        logger.info(
            f"call graph: done in {elapsed:.0f}s "
            f"({payload['cg_n_nodes']} nodes, {payload['cg_n_edges']} edges)"
        )
    return payload


def _zero_data_metrics() -> dict:
    return {
        "pdg_longest_chain": 0,
        "pdg_max_defuse_distance": 0,
        "pdg_density": 0.0,
        "pdg_max_fan_in": 0,
        "pdg_max_fan_out": 0,
        "pdg_mean_degree": 0.0,
        "pdg_slice_mean": 0.0,
        "pdg_slice_max": 0.0,
        "pdg_n_cycles": 0,
    }


def _zero_control_metrics() -> dict:
    return {
        "pdg_control_data_ratio": 0.0,
        "pdg_control_span_max": 0,
        "pdg_control_span_mean": 0.0,
        "pdg_unconditional_fraction": 0.0,
        "pdg_n_control_regions": 0,
        "pdg_control_betweenness_max": 0.0,
    }


def _zero_pdg_metrics() -> dict:
    return {**_zero_data_metrics(), **_zero_control_metrics()}


def _edges_of_kind(graph: nx.DiGraph, kinds: set) -> list:
    return [(u, v) for u, v, d in graph.edges(data=True) if d.get("kind") in kinds]


def _data_metrics(data_edges: list) -> dict:
    graph = nx.DiGraph()
    graph.add_edges_from(data_edges)
    n = graph.number_of_nodes()
    if n == 0:
        return _zero_data_metrics()
    in_degrees = [d for _, d in graph.in_degree()]
    out_degrees = [d for _, d in graph.out_degree()]
    degrees = [d for _, d in graph.degree()]
    distances = [abs(u - v) for u, v in graph.edges()]
    slices = [len(nx.ancestors(graph, node)) / n for node in graph.nodes()]
    return {
        "pdg_longest_chain": _longest_chain(graph),
        "pdg_max_defuse_distance": max(distances) if distances else 0,
        "pdg_density": graph.number_of_edges() / n,
        "pdg_max_fan_in": max(in_degrees),
        "pdg_max_fan_out": max(out_degrees),
        "pdg_mean_degree": sum(degrees) / n,
        "pdg_slice_mean": sum(slices) / n,
        "pdg_slice_max": max(slices),
        "pdg_n_cycles": _n_cycles(graph),
    }


def _control_metrics(all_nodes: list, control_edges: list, n_data_edges: int) -> dict:
    graph = nx.DiGraph()
    graph.add_nodes_from(all_nodes)
    graph.add_edges_from(control_edges)
    n = graph.number_of_nodes()
    if n == 0:
        return _zero_control_metrics()

    spans = []
    for node in graph.nodes():
        if graph.out_degree(node) > 0:
            reach = nx.single_source_shortest_path_length(graph, node)
            spans.extend(d for d in reach.values() if d > 0)
    regions = {frozenset(graph.predecessors(node)) for node in graph.nodes()}
    betweenness = nx.betweenness_centrality(graph)
    unconditional = sum(1 for node in graph.nodes() if graph.in_degree(node) == 0)

    return {
        "pdg_control_data_ratio": (
            len(control_edges) / n_data_edges if n_data_edges else 0.0
        ),
        "pdg_control_span_max": max(spans) if spans else 0,
        "pdg_control_span_mean": sum(spans) / len(spans) if spans else 0.0,
        "pdg_unconditional_fraction": unconditional / n,
        "pdg_n_control_regions": len(regions),
        "pdg_control_betweenness_max": (
            max(betweenness.values()) if betweenness else 0.0
        ),
    }


def _metrics_from_pdg(graph: nx.DiGraph) -> dict:
    data_edges = _edges_of_kind(graph, {"data", "both"})
    control_edges = _edges_of_kind(graph, {"control", "both"})
    return {
        **_data_metrics(data_edges),
        **_control_metrics(list(graph.nodes()), control_edges, len(data_edges)),
    }


def _all_scope_cfgs(cfg) -> list:
    scopes = []
    stack = [cfg]
    while stack:
        scope = stack.pop()
        scopes.append(scope)
        stack.extend(scope.functioncfgs.values())
        stack.extend(getattr(scope, "class_cfgs", {}).values())
    return scopes


def _add_typed_edge(graph: nx.DiGraph, u: int, v: int, kind: str) -> None:
    if graph.has_edge(u, v):
        if graph[u][v].get("kind") != kind:
            graph[u][v]["kind"] = "both"
    else:
        graph.add_edge(u, v, kind=kind)


def _add_statement_nodes(scope_cfg, graph: nx.DiGraph) -> None:
    for block in scope_cfg.get_all_blocks():
        for statement in block.statements:
            line = getattr(statement, "lineno", None)
            if line is not None:
                graph.add_node(line)


def _add_defuse_edges(scope_cfg, graph: nx.DiGraph) -> None:
    ssa_results, const_dict = SSA().compute_SSA(scope_cfg)

    def_line = {}
    for key, value in const_dict.items():
        line = getattr(value, "lineno", None)
        if line is not None:
            def_line[key] = line

    for block in scope_cfg.get_all_blocks():
        statement_uses = ssa_results.get(block.id, [])
        for i, uses in enumerate(statement_uses):
            if i >= len(block.statements):
                continue
            use_line = getattr(block.statements[i], "lineno", None)
            if use_line is None:
                continue
            for name, versions in uses.items():
                for version in versions:
                    source = def_line.get((name, version))
                    if source is not None and source != use_line:
                        _add_typed_edge(graph, source, use_line, "data")


def _add_control_edges(scope_cfg, graph: nx.DiGraph) -> None:
    blocks = list(scope_cfg.get_all_blocks())
    if not blocks:
        return
    exit_node = "__exit__"
    cfg_graph = nx.DiGraph()
    for block in blocks:
        cfg_graph.add_node(block.id)
        for link in block.exits:
            cfg_graph.add_edge(block.id, link.target.id)
    for block in blocks:
        if not block.exits:
            cfg_graph.add_edge(block.id, exit_node)
    if exit_node not in cfg_graph:
        return

    ipdom = nx.immediate_dominators(cfg_graph.reverse(), exit_node)
    block_by_id = {block.id: block for block in blocks}

    def postdominates(b, a):
        node = a
        while True:
            if node == b:
                return True
            nxt = ipdom.get(node)
            if nxt is None or nxt == node:
                return node == b
            node = nxt

    for a, b in cfg_graph.edges():
        if a == exit_node or b == exit_node or postdominates(b, a):
            continue
        stop = ipdom.get(a)
        node = b
        while node is not None and node != stop and node != exit_node:
            _link_control(block_by_id.get(a), block_by_id.get(node), graph)
            nxt = ipdom.get(node)
            if nxt == node:
                break
            node = nxt


def _link_control(predicate_block, dependent_block, graph: nx.DiGraph) -> None:
    if (
        predicate_block is None
        or dependent_block is None
        or not predicate_block.statements
    ):
        return
    predicate_line = getattr(predicate_block.statements[-1], "lineno", None)
    if predicate_line is None:
        return
    for statement in dependent_block.statements:
        line = getattr(statement, "lineno", None)
        if line is not None and line != predicate_line:
            _add_typed_edge(graph, predicate_line, line, "control")


def _build_pdg(code: str, scope_names: set = None) -> nx.DiGraph:
    cfg = CFGBuilder().build_from_src("module", code)
    graph = nx.DiGraph()
    for scope_cfg in _all_scope_cfgs(cfg):
        if scope_names is not None and scope_cfg.name not in scope_names:
            continue
        _add_statement_nodes(scope_cfg, graph)
        for add_edges in (_add_defuse_edges, _add_control_edges):
            try:
                add_edges(scope_cfg, graph)
            except Exception as e:
                logger.debug(
                    f"{add_edges.__name__} failed for one scope: {type(e).__name__}: {e}"
                )
    return graph


def _pdg_metrics(code: str) -> dict:
    start = time.perf_counter()
    try:
        metrics = _metrics_from_pdg(_build_pdg(code))
    except Exception as e:
        logger.warning(
            f"program dependence graph FAILED after {time.perf_counter() - start:.0f}s: "
            f"{type(e).__name__}: {e}"
        )
        return _zero_pdg_metrics()
    elapsed = time.perf_counter() - start
    if elapsed > 1:
        logger.info(f"program dependence graph: done in {elapsed:.0f}s")
    return metrics


def compute_graph_metrics(code: str) -> dict:
    return {**_call_graph_metrics(code), **_pdg_metrics(code)}


def call_graph(code: str) -> nx.DiGraph:
    return _build_call_graph(code)


def program_dependence_graph(code: str, function: str = None) -> nx.DiGraph:
    return _build_pdg(code, {function} if function else None)
