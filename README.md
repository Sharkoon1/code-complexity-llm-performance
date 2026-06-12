# Code Complexity and LLM Performance

Bachelor's thesis investigating whether code complexity metrics predict LLM performance on real-world software engineering tasks. Compares classical metrics (cyclomatic, Halstead, LOC) and LM-CC against whether an agent resolves the task.

## Datasets

The same two agents and agent framework are run on both benchmarks:

- SWE-agent + Claude 3.7 Sonnet
- OpenHands + Qwen3-Coder-480B-A35B

SWE-bench Verified (500 tasks). Agent labels come from the [SWE-bench experiments repo](https://github.com/SWE-bench/experiments): `evaluation/verified/20250225_sweagent_claude-3-7-sonnet` and `evaluation/verified/20250805_openhands-Qwen3-Coder-480B-A35B-Instruct`. A cross-agent resolution rate over all submitted agents is also computed.

SWE-bench-Live lite (300 tasks). Agent labels come from the [SWE-bench-Live submission repo](https://github.com/SWE-bench-Live/submission): `submissions/lite/20250501-sweagent-claude37` and `submissions/lite/20250725-openhands-Qwen3-Coder-480B-A35B`.

Metrics are computed once per benchmark.

## Approach

For each task:
1. Extract the target Python file and the patched function from the task repository at its base commit
2. Compute complexity metrics (cyclomatic, Halstead, LOC, LM-CC) at both file and function level
3. Join each agent's resolved labels
4. Correlate metric values with task success and difficulty

[LM-CC](src/lm_cc.py) follows [Xie et al., 2026](https://arxiv.org/abs/2602.07882): entropy-based segmentation via a code language model, tree-sitter hierarchical decomposition, aggregation over depth and branching factor.

## Setup

```bash
git clone https://github.com/Sharkoon1/code-complexity-llm-performance.git
cd code-complexity-llm-performance

uv sync
```

Requires Python 3.12+

For LM-CC, a local language model (default `Qwen/Qwen2.5-Coder-0.5B`) is downloaded on the  first run. The Qwen default is for testing. For the pipeline [CodeLlama-7b](https://huggingface.co/meta-llama/CodeLlama-7b-hf).

## Usage

SWE-bench Verified:

```bash
uv run run_swe_bench.py
```

Loads SWE-bench Verified, the SWE-agent Claude 3.7 and OpenHands Qwen3-Coder-480B labels, and cross-agent resolution rates. Writes `results/swe_bench_verified_whole_file_result.parquet` and `results/swe_bench_verified_function_result.parquet`.

SWE-bench-Live:

```bash
uv run run_swe_bench_live.py
```

Loads SWE-bench-Live lite, the SWE-agent Claude 3.7 and OpenHands Qwen3-Coder-480B labels. Writes `results/swe_bench_live_whole_file_result.parquet` and `results/swe_bench_live_function_result.parquet`.

The LM-CC forward pass runs locally on the model named by the environment variable `LM_CC_MODEL` (default `Qwen/Qwen2.5-Coder-0.5B`). The device is selected automatically (CUDA, then MPS, then CPU) and can be overridden with `LM_CC_DEVICE`. `LM_CC_ENTROPY_CHUNK` (default `256`) splits the entropy reduction along the sequence dimension to bound GPU memory on long files.

## Notebooks

Analysis notebooks live in `notebooks/`, one per agent and level (whole file and function):

- SWE-bench Verified: `swe_bench_verified_sweagent_claude37_*`, `swe_bench_verified_openhands_qwen3coder_*`, `swe_bench_verified_all_agents`, and `swe_bench_verified_metrics_overview`
- SWE-bench-Live: `swe_bench_live_sweagent_claude37_*`, `swe_bench_live_openhands_qwen3coder_*`, and `swe_bench_live_metrics_overview`
- Cross-benchmark: `swe_bench_verified_vs_live` compares the LM-CC vs resolution relationship across both benchmarks for both agents

## References

- Xie, C., Shi, Y., Gu, X., & Shen, B. (2026). *Rethinking Code Complexity Through the Lens of Large Language Models*. arXiv:2602.07882
- Jimenez, C. E. et al. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024. [arXiv:2310.06770](https://arxiv.org/abs/2310.06770). https://github.com/SWE-bench/SWE-bench
- Zhang, L. et al. (2025). *SWE-bench Goes Live!* [arXiv:2505.23419](https://arxiv.org/abs/2505.23419). https://github.com/microsoft/SWE-bench-Live

## License

MIT
