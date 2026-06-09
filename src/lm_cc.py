import io
import logging
import os
import tokenize
import torch
import ast
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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
            torch_dtype=torch.float16,
        ).to(_get_device())
        _model.eval()
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise


@dataclass
class SemanticUnit:
    char_start: int  #  from segment (horizontal info)
    char_end: int  #  from segment (horizontal info)
    depth: int  #  from hierarchy (vertical info)
    indent: int  #  leading-whitespace width of the unit's first line (root = -1)
    children: list[...]  #  from hierarchy (vertical info)


@dataclass
class TokenFeatures:
    tokens: torch.Tensor  # (n-1,) token IDs
    entropy: torch.Tensor  # (n-1,) per-token entropy
    offsets: torch.Tensor  # (n-1, 2) char spans in source


def compute_lm_cc(code: str) -> dict:
    # 1. normalize
    processed_code = _preprocess_code(code)
    # 2. entropy
    features = compute_token_features(processed_code)

    # Edge case: trivial code with no meaningful tokens
    if features.entropy.numel() == 0:
        empty_root = SemanticUnit(
            char_start=0, char_end=len(processed_code),
            depth=0, indent=-1, children=[],
        )
        return _aggregate(empty_root)

    # 3. semantic units
    boundaries = _detect_boundaries(processed_code, features.offsets, features.entropy)
    segments = _mask_to_segments(boundaries)
    # 4. semantic hierachy
    hierachy = _build_hierarchy(segments, features, processed_code)

    return _aggregate(hierachy)


def _docstring_token_starts(code: str) -> set[tuple[int, int]]:
    starts: set[tuple[int, int]] = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return starts
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            # Skip a docstring that is the sole statement of a function/class:
            # removing it would leave an empty (invalid) body. (An empty module
            # is fine, so module docstrings may always be removed.)
            if (body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                    and (isinstance(node, ast.Module) or len(body) > 1)):
                doc = body[0].value
                starts.add((doc.lineno, doc.col_offset))
    return starts


def _preprocess_code(code: str) -> str:
    docstring_starts = _docstring_token_starts(code)

    io_object = io.StringIO(code)
    kept_tokens = []
    last_appended_line = -1

    for token in tokenize.generate_tokens(io_object.readline):
        token_type = token.type
        token_start_line, _ = token.start

        if token_type in (tokenize.INDENT, tokenize.DEDENT):
            continue

        # filter comments
        if token_type in (tokenize.COMMENT, tokenize.TYPE_COMMENT):
            continue

        # filter docstrings 
        if token_type == tokenize.STRING and token.start in docstring_starts:
            continue

        # filter blank lines
        if token_type in (tokenize.NL, tokenize.NEWLINE):
            if token_start_line != last_appended_line:
                continue

        kept_tokens.append(token)
        last_appended_line = token_start_line

    text = tokenize.untokenize(kept_tokens)
    # untokenize leaves a "\" line-continuation placeholder wherever a token
    # was removed
    lines = [ln for ln in text.splitlines(keepends=True) if ln.strip() != "\\"]
    return "".join(lines)


def _aggregate(root: SemanticUnit, alpha: float = 0.8) -> dict:
    comp_levels = []      # d(v) for real units (depth >= 1)
    branch_factors = []   # b(v) for every unit incl. root

    def walk(node):
        branch_factors.append(len(node.children))
        if node.depth > 0:
            comp_levels.append(node.depth)
        for child in node.children:
            walk(child)

    walk(root)

    if not comp_levels:
        return {
            "lm_cc_score": 0.0,
            "lm_cc_max_comp": 0,
            "lm_cc_avg_comp": 0.0,
            "lm_cc_total_comp": 0,
            "lm_cc_max_branch": 0,
            "lm_cc_avg_branch": 0.0,
            "lm_cc_total_branch": 0,
        }

    total_comp = sum(comp_levels)
    total_branch = sum(branch_factors)
    score = alpha * total_branch + (1 - alpha) * total_comp

    return {
        "lm_cc_score": score,
        "lm_cc_max_comp": max(comp_levels),
        "lm_cc_avg_comp": total_comp / len(comp_levels),
        "lm_cc_total_comp": total_comp,
        "lm_cc_max_branch": max(branch_factors),
        "lm_cc_avg_branch": total_branch / len(branch_factors),
        "lm_cc_total_branch": total_branch,
    }


def _line_indent(code: str, char_pos: int) -> int:
    line_start = code.rfind("\n", 0, char_pos) + 1
    indent = 0
    for ch in code[line_start:]:
        if ch == " ":
            indent += 1
        elif ch == "\t":
            indent += 8
        else:
            break
    return indent


def _build_hierarchy(
    segments: list[tuple[int, int]],
    features: TokenFeatures,
    processed_code: str,
) -> SemanticUnit:
    root = SemanticUnit(
        char_start=0,
        char_end=len(processed_code),
        depth=0,
        indent=-1,
        children=[],
    )
    stack = [root]

    for tok_start, tok_end in segments:
        # skip empty segments
        if tok_end <= tok_start:
            continue

        # char position
        char_start = features.offsets[tok_start, 0].item()
        char_end = features.offsets[tok_end - 1, 1].item()

        # raw indentation of the line this unit starts on
        level = _line_indent(processed_code, char_start)

        # find parent on stack
        while stack[-1].indent >= level:
            stack.pop()
        parent = stack[-1]

        # create node and append
        node = SemanticUnit(
            char_start=char_start,
            char_end=char_end,
            depth=parent.depth + 1,
            indent=level,
            children=[],
        )
        parent.children.append(node)
        stack.append(node)

    return root


def _mask_to_segments(boundaries: torch.Tensor) -> list:
    segments = []
    segment_start = 0
    for i, boundary in enumerate(boundaries.tolist()):
        # i > segment_start avoids an empty segment when the first
        # token itself is a boundary (boundaries[0] is True).
        if boundary and i > segment_start:
            segments.append((segment_start, i))  # (start, end)
            segment_start = i
    segments.append(
        (segment_start, len(boundaries))
    )  # last segment has no end, doesnt access loop

    return segments


def _build_line_offsets(code: str) -> list[int]:
    offsets = [0]
    for i, char in enumerate(code):
        if char == "\n":
            offsets.append(i + 1)
    return offsets

def _find_syntactic_boundaries(code: str) -> set[int]:
    boundary_chars = set()
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        logger.warning(
            "Syntactic boundary detection skipped: processed code did not parse; "
            "falling back to entropy-only segmentation."
        )
        return boundary_chars

    line_offsets = _build_line_offsets(code)
    
    target_types = (
        ast.If, ast.For, ast.While, ast.AsyncFor,
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
        ast.Try, ast.With, ast.AsyncWith,
    )
    
    for node in ast.walk(tree):
        if isinstance(node, target_types):
            if hasattr(node, 'end_lineno') and node.end_lineno is not None:
                char_pos = line_offsets[node.end_lineno - 1] + node.end_col_offset
                boundary_chars.add(char_pos)
    
    return boundary_chars

def _syntactic_boundary_mask(
    code: str,
    offsets: torch.Tensor,
) -> torch.Tensor:
    boundary_chars = _find_syntactic_boundaries(code)
    n_tokens = offsets.shape[0]
    
    if not boundary_chars:
        return torch.zeros(n_tokens, dtype=torch.bool)
    
    boundary_tensor = torch.tensor(sorted(boundary_chars), dtype=offsets.dtype)
    token_starts = offsets[:, 0].unsqueeze(1)  # (n_tokens, 1)
    token_ends = offsets[:, 1].unsqueeze(1)    # (n_tokens, 1)
    boundaries = boundary_tensor.unsqueeze(0)  # (1, n_boundaries)
    
    #  (n_tokens, n_boundaries)
    in_range = (token_starts <= boundaries) & (boundaries < token_ends)

    return in_range.any(dim=1)

def _detect_boundaries(
    code:str, offsets: torch.Tensor, entropy: torch.Tensor, tau_quantile: float = 0.67
) -> torch.Tensor:
    if entropy.numel() == 0:
        return torch.zeros(0, dtype=torch.bool)
    
    tau = torch.quantile(entropy.float(), tau_quantile)
    entropy_boundary_mask = entropy > tau 
    syntatic_boundary_mask = _syntactic_boundary_mask(code, offsets)
    return entropy_boundary_mask | syntatic_boundary_mask


@torch.no_grad()
def compute_token_features(code: str) -> TokenFeatures:
    _load()
    inputs = _tokenizer(code, return_tensors="pt", return_offsets_mapping=True)
    offsets = inputs.pop("offset_mapping")
    inputs = inputs.to(_model.device)

    hidden = _model.model(**inputs).last_hidden_state   # (1, n, hidden_dim)

    chunk_size = int(os.environ.get("LM_CC_ENTROPY_CHUNK", "256"))
    n = hidden.shape[1]
    entropy = torch.empty((1, n), device=hidden.device, dtype=torch.float32)
    
    for i in range(0, n, chunk_size):
        sl = slice(i, i + chunk_size)
        chunk_logits = _model.lm_head(hidden[:, sl])           # only chunk logits
        log_probs = F.log_softmax(chunk_logits, dim=-1)
        entropy[:, sl] = -(log_probs.exp() * log_probs).sum(dim=-1)
        del chunk_logits, log_probs
    
    del hidden
    
    return TokenFeatures(
        tokens=inputs.input_ids[0, 1:].cpu(),
        entropy=entropy[0, :-1].cpu(),
        offsets=offsets[0, 1:],
    )
