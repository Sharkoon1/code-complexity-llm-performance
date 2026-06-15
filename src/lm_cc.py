"""LM-CC: model-perceived code complexity (Xie et al., arXiv:2602.07882)."""

import logging
import os
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.block_tree import CodeBlockProcessor, get_code_with_boundaries

logger = logging.getLogger(__name__)

_THRESHOLD = 0.67

_tokenizer = None
_model = None


def _get_device() -> str:
    override = os.environ.get("LM_CC_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load():
    global _tokenizer, _model
    if _model is not None:
        return

    model_name = os.environ.get("LM_CC_MODEL", "Qwen/Qwen2.5-Coder-0.5B")

    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        ).to(_get_device())
        _model.eval()
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise


@dataclass
class TokenFeatures:
    tokens: torch.Tensor  # (n-1,)
    entropy: torch.Tensor  # (n-1,)


@torch.no_grad()
def _compute_token_features(code: str) -> TokenFeatures:
    _load()
    inputs = _tokenizer(code, return_tensors="pt").to(_model.device)

    hidden = _model.model(**inputs).last_hidden_state  # (1, n, hidden_dim)

    chunk_size = int(os.environ.get("LM_CC_ENTROPY_CHUNK", "256"))
    n = hidden.shape[1]
    entropy = torch.empty((1, n), device=hidden.device, dtype=torch.float32)

    for i in range(0, n, chunk_size):
        sl = slice(i, i + chunk_size)
        chunk_logits = _model.lm_head(hidden[:, sl])
        log_probs = F.log_softmax(chunk_logits, dim=-1)
        entropy[:, sl] = -(log_probs.exp() * log_probs).sum(dim=-1)
        del chunk_logits, log_probs

    del hidden

    return TokenFeatures(
        tokens=inputs.input_ids[0, 1:].cpu(),
        entropy=entropy[0, :-1].cpu(),
    )


def _get_block_cnt(node) -> int:
    if not node:
        return 0
    total = 1
    for child in node.get("children", []):
        total += _get_block_cnt(child)
    return total


def _get_total_branch(node) -> int:
    return _get_block_cnt(node) - 1


def _get_depth_sum(node, depth=1) -> int:
    if not node:
        return 0
    total = depth
    if "children" in node:
        for child in node["children"]:
            total += _get_depth_sum(child, depth + 1)
    return total


def _get_max_depth(node, depth=1) -> int:
    if not node:
        return 0
    children = node.get("children")
    if not children:
        return depth
    max_d = depth
    for child in children:
        d = _get_max_depth(child, depth + 1)
        if d > max_d:
            max_d = d
    return max_d


def _get_avg_depth(node, depth=1) -> float:
    if not node:
        return 0.0
    total_depth = _get_depth_sum(node, depth)
    total_nodes = _get_block_cnt(node)
    if total_nodes == 0:
        return 0.0
    return float(total_depth) / total_nodes


def _get_max_width(node) -> int:
    if not node:
        return 0
    level_counts = defaultdict(int)

    def traverse(node, level):
        weight = node.get("weight", 1)
        level_counts[level] += weight
        for child in node.get("children", []):
            traverse(child, level + 1)

    traverse(node, 1)
    return max(level_counts.values())


def _get_avg_children(node) -> float:
    if not node:
        return 0.0
    internal_count = 0
    total_children = 0

    def traverse(n):
        nonlocal internal_count, total_children
        children = n.get("children", []) if isinstance(n, dict) else []
        if children:
            internal_count += 1
            total_children += len(children)
            for c in children:
                traverse(c)

    traverse(node)
    return float(total_children) / internal_count if internal_count > 0 else 0.0


def _get_lmcc(tree_node) -> float:
    ALPHA = 0.8
    total_branch = _get_total_branch(tree_node)
    depth_sum = _get_depth_sum(tree_node, depth=1)
    return depth_sum * (1 - ALPHA) + total_branch * ALPHA


def _units_at_depth(node, target, depth=1):
    found = [node] if depth == target else []
    for child in node.get("children", []):
        found += _units_at_depth(child, target, depth + 1)
    return found


def _deepest_unit_entropy(tree_node, entropies) -> float:
    deepest = _units_at_depth(tree_node, _get_max_depth(tree_node))
    values = [
        e
        for node in deepest
        for e in entropies[node["start_token"] : node["end_token"] + 1]
    ]
    return float(sum(values) / len(values)) if values else 0.0


def _mean_entropy(entropies) -> float:
    return float(sum(entropies) / len(entropies)) if entropies else 0.0


def _high_entropy_fraction(entropies) -> float:
    if not entropies:
        return 0.0
    return sum(1 for e in entropies if e >= _THRESHOLD) / len(entropies)


def _unit_mean_entropy(node, entropies) -> float:
    values = entropies[node["start_token"] : node["end_token"] + 1]
    return sum(values) / len(values) if values else 0.0


def _depth_weighted_entropy(tree_node, entropies) -> float:
    weighted = 0.0
    weight = 0
    stack = [(tree_node, 1)]
    while stack:
        node, depth = stack.pop()
        weighted += depth * _unit_mean_entropy(node, entropies)
        weight += depth
        for child in node.get("children", []):
            stack.append((child, depth + 1))
    return float(weighted / weight) if weight else 0.0


def _zero_metrics() -> dict:
    return {
        "lm_cc_score": 0.0,
        "lm_cc_total_comp": 0,
        "lm_cc_total_branch": 0,
        "lm_cc_max_comp": 0,
        "lm_cc_avg_comp": 0.0,
        "lm_cc_max_branch": 0,
        "lm_cc_avg_branch": 0.0,
        "deepest_unit_entropy": 0.0,
        "mean_entropy": 0.0,
        "high_entropy_fraction": 0.0,
        "depth_weighted_entropy": 0.0,
    }


def _metrics_from_tree(block_tree: dict) -> dict:
    return {
        "lm_cc_score": _get_lmcc(block_tree),
        "lm_cc_total_comp": _get_depth_sum(block_tree),
        "lm_cc_total_branch": _get_total_branch(block_tree),
        "lm_cc_max_comp": _get_max_depth(block_tree),
        "lm_cc_avg_comp": _get_avg_depth(block_tree),
        "lm_cc_max_branch": _get_max_width(block_tree),
        "lm_cc_avg_branch": _get_avg_children(block_tree),
    }


def compute_lm_cc(code: str) -> dict:
    features = _compute_token_features(code)
    if features.entropy.numel() == 0:
        return _zero_metrics()

    token_strings = _tokenizer.convert_ids_to_tokens(features.tokens.tolist())
    entropies = features.entropy.tolist()

    # Line reconstruction needs SentencePiece tokens (CodeLlama)
    if not any(("▁" in t) or (t == "<0x0A>") for t in token_strings[:64]):
        logger.warning("LM_CC expects a SentencePiece tokenizer (e.g. CodeLlama)")

    code_with_boundaries, _, start_end_tokens = get_code_with_boundaries(
        token_strings, entropies, threshold=_THRESHOLD
    )
    block_tree = CodeBlockProcessor().parse_code_blocks(
        code_with_boundaries, tokens=token_strings, start_end_tokens=start_end_tokens
    )
    metrics = _metrics_from_tree(block_tree)
    metrics["deepest_unit_entropy"] = _deepest_unit_entropy(block_tree, entropies)
    metrics["mean_entropy"] = _mean_entropy(entropies)
    metrics["high_entropy_fraction"] = _high_entropy_fraction(entropies)
    metrics["depth_weighted_entropy"] = _depth_weighted_entropy(block_tree, entropies)
    return metrics
