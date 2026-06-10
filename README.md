# Code Complexity and LLM Performance

Bachelor's thesis investigating whether code complexity metrics predict LLM performance on real-world software engineering tasks. Compares classical metrics (cyclomatic, Halstead, LOC) and LM-CC on SWE-bench Verified.

## Approach

For each task in SWE-bench Verified:
1. Extract the target Python file and the patched function from the swe bench repository
2. Computes complexity metrics (cyclomatic, Halstead, LOC, LM-CC) at both **file** and **function** level
3. Label each task with whether Claude 3.5 Sonnet (Agentless) resolved it, and with the resolution rate across all SWE-bench agents (task difficulty)
4. Correlate metric values with task success / difficulty

LM-CC follows [Xie et al., 2026](https://arxiv.org/abs/2602.07882) - entropy-based segmentation via a code language model, tree-sitter hierarchical decomposition, aggregation over depth and branching factor. 

## Setup

```bash
git clone https://github.com/Sharkoon1/code-complexity-llm-performance.git
cd code-complexity-llm-performance

uv sync
```

Requires Python 3.12+

For LM-CC, a local language model (default: `Qwen/Qwen2.5-Coder-0.5B`) is downloaded on first run. **The Qwen default** for testing, for the real results I used [CodeLlama-7b](https://huggingface.co/meta-llama/CodeLlama-7b-hf)

## Usage

```bash
uv run main.py
```

This will:
1. Load SWE-bench Verified, the Agentless/Claude-3.5-Sonnet labels, and cross-agent resolution rates
2. Compute metrics for each task's target file and patched function
3. Save results to `results/swe_bench_whole_file_result.parquet` and `results/swe_bench_function_result.parquet`

The LM-CC forward pass runs locally on the model named by the environment variable `LM_CC_MODEL` (default `Qwen/Qwen2.5-Coder-0.5B`). The device is selected automatically (CUDA, then MPS, then CPU) and can be overridden with `LM_CC_DEVICE`.

`LM_CC_ENTROPY_CHUNK` (default `256`) splits the entropy reduction along the sequence dimension to bound GPU memory on long files.

## References

- Xie, C., Shi, Y., Gu, X., & Shen, B. (2026). *Rethinking Code Complexity Through the Lens of Large Language Models*. arXiv:2602.07882
- Jimenez, C. E. et al. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024.

## License

MIT
