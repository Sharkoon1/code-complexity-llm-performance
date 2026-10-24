"""Tests for the public measure_code / measure_file API and its agreement with the project-work results."""

from pathlib import Path

import pytest

import code_complexity
from code_complexity import measure_code, measure_file

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "swe_bench_verified_whole_file_result.parquet"
FILTERED = ROOT / "data" / "interim" / "swebench_verified_filtered.parquet"
CODE_CACHE = ROOT / "data" / "cache" / "code"

CODE = (
    "def f(x):\n"
    "    a = x + 1\n"
    "    if a > 0:\n"
    "        for i in range(a):\n"
    "            a += i\n"
    "    return a\n"
)

WHOLE_FILE_KEYS = ("loc", "nesting_max", "pdg_max_fan_in", "cg_max_out_degree")


def test_version():
    assert code_complexity.__version__ == "1.0.0"


def test_measure_code_whole_file_keys():
    metrics = measure_code(CODE)
    for key in WHOLE_FILE_KEYS + ("cc_sum", "cognitive_sum", "n_functions", "pdg_longest_chain"):
        assert key in metrics
    assert metrics["nesting_max"] == 2
    assert metrics["n_functions"] == 1
    assert "lm_cc_score" not in metrics  # LM-CC is off by default


def test_measure_file_equals_measure_code(tmp_path):
    path = tmp_path / "module.py"
    path.write_text(CODE, encoding="utf-8")
    assert measure_file(path) == measure_code(CODE)


def test_measure_code_rejects_unformattable_source():
    with pytest.raises(ValueError, match="black"):
        measure_code("def f(:\n    pass\n")


@pytest.mark.skipif(
    not (RESULTS.exists() and FILTERED.exists()),
    reason="project-work result files not present",
)
@pytest.mark.parametrize(
    "instance_id",
    [
        "django__django-11239",
        "django__django-15572",
        "django__django-16255",
        "pydata__xarray-4075",
    ],
)
def test_reproduces_project_work_results(instance_id):
    """measure_file on a cached pre-patch file must reproduce the whole-file parquet row."""
    pd = pytest.importorskip("pandas")
    expected = pd.read_parquet(RESULTS).set_index("instance_id").loc[instance_id]
    task = pd.read_parquet(FILTERED).set_index("instance_id").loc[instance_id]
    path = (
        CODE_CACHE
        / task["repo"].replace("/", "__")
        / task["base_commit"]
        / task["python_files"][0].replace("/", "__")
    )
    metrics = measure_file(path)
    for key in WHOLE_FILE_KEYS:
        assert metrics[key] == expected[key], key
