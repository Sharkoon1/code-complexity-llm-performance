.PHONY: format test run verify verify-model

format:
	uv run black src tests

test:
	uv run pytest

run:
	uv run main.py

verify:
	uv run python -m src.verify_xcodeeval --mode model-free

verify-model:
	uv run python -m src.verify_xcodeeval --mode with-model
