"""code-complexity: structural complexity metrics for Python source files.

Public API::

    from code_complexity import measure_file, measure_code

    metrics = measure_file("module.py")      # dict with loc, nesting_max, pdg_max_fan_in, cg_max_out_degree, ...

The core install computes the classical, AST and graph metrics; LM-CC needs the ``lmcc`` extra.
"""

from code_complexity.graphs import compute_graph_metrics
from code_complexity.metrics import measure_code, measure_file, metrics_for_code
from code_complexity.preprocess import normalize

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "compute_graph_metrics",
    "measure_code",
    "measure_file",
    "metrics_for_code",
    "normalize",
]
