.PHONY: format test run-swe-bench run-swe-bench-live verify verify-model

format:
	uv run black src tests

test:
	uv run pytest

run-swe-bench:
	uv run run_swe_bench.py

run-swe-bench-live:
	uv run run_swe_bench_live.py

verify:
	uv run python -m scripts.verify_xcodeeval --mode model-free

verify-model:
	uv run python -m scripts.verify_xcodeeval --mode with-model
