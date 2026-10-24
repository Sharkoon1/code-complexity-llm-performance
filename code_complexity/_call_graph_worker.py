"""Subprocess entry point used by code_complexity.graphs for the PyCG call graph.

Usage: ``python -m code_complexity._call_graph_worker <python-file>`` prints the
call-graph metrics of the file as JSON. Kept outside of ``graphs`` so that running it
as ``__main__`` does not execute the ``graphs`` module twice.
"""

import json
import sys
from pathlib import Path

from code_complexity.graphs import _build_call_graph, _metrics_from_call_graph


def main(argv: list) -> None:
    code = Path(argv[1]).read_text(encoding="utf-8")
    print(json.dumps(_metrics_from_call_graph(_build_call_graph(code))))


if __name__ == "__main__":
    main(sys.argv)
