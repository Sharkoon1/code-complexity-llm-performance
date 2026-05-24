# Code Complexity and LLM Performance

[![Runpod](https://api.runpod.io/badge/Sharkoon1/code-complexity-llm-performance)](https://console.runpod.io/hub/Sharkoon1/code-complexity-llm-performance)

Bachelor's thesis investigating whether code complexity metrics predict LLM performance on real-world software engineering tasks. Compares classical metrics (cyclomatic, Halstead, LOC) and LM-CC on SWE-bench Verified.

## Approach

For each task in SWE-bench Verified:
1. Extract the target Python file from the repository
2. Compute complexity metrics (cyclomatic, Halstead, LOC, LM-CC)
3. Evaluate Claude Sonnet on the task
4. Correlate metric values with task success

LM-CC follows [Xie et al., 2026](https://arxiv.org/abs/2602.07882) - entropy-based segmentation via a code language model, hierarchical decomposition by indentation, aggregation over depth and branching factor.

## Setup

```bash
git clone https://github.com/Sharkoon1/code-complexity-llm-performance.git
cd code-complexity-llm-performance

uv sync
```

Requires Python 3.12+. For LM-CC computation, a local language model (default: `Qwen/Qwen2.5-Coder-0.5B`) is downloaded on first run.

## Usage

```bash
uv run main.py
```

This will:
1. Load SWE-bench Verified
2. Compute metrics for each task's target file
3. Save results to `results/result.parquet`

**Local vs. remote LM-CC.** By default (`LOCAL_LM_CC` unset or `true`) the LM-CC forward pass runs locally. Set `LOCAL_LM_CC=false` to offload it — see below.

## Remote inference (optional)

LM-CC can offload the forward pass to any HTTP endpoint - useful for running a larger model (e.g. CodeLlama-7b) on a GPU host while the rest of the pipeline stays local. Set `LOCAL_LM_CC=false` plus the endpoint in a `.env` file (see [.env.example](.env.example)):

```
LOCAL_LM_CC=false
LM_CC_REMOTE_URL=https://your-endpoint/...
LM_CC_REMOTE_KEY=<bearer-token>
```

Then run `uv run --env-file .env main.py`. If `LOCAL_LM_CC` is unset or set to `true`, the remote env vars are ignored and the pipeline runs the local model.

**Endpoint contract**:

- `POST <url>` with header `Authorization: Bearer <key>`
- Request body: `{"input": {"code": "<source>"}}`
- Response body: `{"output": {"tokens": [...], "entropy": [...], "offsets": [[s, e], ...]}}`
  - `tokens`: `int[n]`
  - `entropy`: `float[n]`
  - `offsets`: `int[n][2]` — char spans into the input string

## References

- Xie, C., Shi, Y., Gu, X., & Shen, B. (2026). *Rethinking Code Complexity Through the Lens of Large Language Models*. arXiv:2602.07882
- Jimenez, C. E. et al. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024.

## License

MIT