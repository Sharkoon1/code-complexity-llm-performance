import io
import logging
import os
import time
import tokenize
import requests
import torch
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
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(_get_device())
    _model.eval()


@dataclass
class SemanticUnit:
    char_start: int  #  from segment (horizontal info)
    char_end: int  #  from segment (horizontal info)
    depth: int  #  from hierarchy (vertical info)
    indent: int  #  from source (raw indentation level)
    children: list[...]  #  from hierarchy (vertical info)


@dataclass
class TokenFeatures:
    tokens: torch.Tensor  # (n-1,) token IDs
    entropy: torch.Tensor  # (n-1,) per-token entropy
    offsets: torch.Tensor  # (n-1, 2) char spans in source


def compute_lm_cc(code: str) -> dict:
    # 1. normalize
    processed_code, indent_levels = _preprocess_code(code)
    # 2. entropy
    features = compute_token_features(processed_code)
    # 3. semantic units
    boundaries = _detect_boundaries(features.entropy)
    segments = _mask_to_segments(boundaries)
    # 4. semantic hierachy
    hierachy = _build_hierarchy(segments, features, processed_code, indent_levels)
    score = _aggregate(hierachy)

    return {"lm_cc": score}


def _preprocess_code(code: str) -> tuple[str, dict[int, int]]:
    io_object = io.StringIO(code)
    kept_tokens = []
    last_appended_line = -1
    indent_levels = {}
    current_level = 0

    for token in tokenize.generate_tokens(io_object.readline):
        token_type = token.type
        token_start_line, _ = token.start

        # track nesting via INDENT/DEDENT
        if token_type == tokenize.INDENT:
            current_level += 1
            continue
        if token_type == tokenize.DEDENT:
            current_level -= 1
            continue

        # record indent for this line
        if token_start_line not in indent_levels:
            indent_levels[token_start_line] = current_level

        # filter commments and doc strings
        if token_type in (tokenize.COMMENT, tokenize.TYPE_COMMENT):
            continue

        # filter blank lines
        if token_type in (tokenize.NL, tokenize.NEWLINE):
            if token_start_line != last_appended_line:
                continue

        kept_tokens.append(token)
        last_appended_line = token_start_line

    return tokenize.untokenize(kept_tokens), indent_levels


def _aggregate(root: SemanticUnit, alpha: float = 0.8) -> float:
    score = 0.0

    def walk(node):
        nonlocal score
        if node.depth > 0:  # skip root
            b = len(node.children)  # branching factor (number of tree children)
            d = node.depth  # compositional level (deepness of tree)
            score += alpha * b + (1 - alpha) * d
        for child in node.children:
            walk(child)

    walk(root)
    return score


def _build_hierarchy(
    segments: list[tuple[int, int]],
    features: TokenFeatures,
    processed_code: str,
    indent_levels: dict[int, int],
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
        # char position
        char_start = features.offsets[tok_start, 0].item()
        char_end = features.offsets[tok_end - 1, 1].item()

        # line number
        line_number = processed_code[:char_start].count("\n") + 1
        level = indent_levels.get(line_number, 0)

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
        if boundary:
            segments.append((segment_start, i))  # (start, end)
            segment_start = i
    segments.append(
        (segment_start, len(boundaries))
    )  # last segment has no end, doesnt access loop

    return segments


def _detect_boundaries(
    entropy: torch.Tensor, tau_quantile: float = 0.67
) -> torch.Tensor:
    tau = torch.quantile(entropy.float(), tau_quantile)
    is_boundary = entropy > tau
    return is_boundary


def compute_token_features(code: str) -> TokenFeatures:
    if os.environ.get("LOCAL_LM_CC", "true").lower() == "true":
        return _compute_token_features_local(code)
    return _compute_token_features_remote(code)


@torch.no_grad()
def _compute_token_features_local(code: str) -> TokenFeatures:
    _load()
    inputs = _tokenizer(code, return_tensors="pt", return_offsets_mapping=True)
    offsets = inputs.pop("offset_mapping")
    inputs = inputs.to(_model.device)

    logits = _model(**inputs).logits

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)

    return TokenFeatures(
        tokens=inputs.input_ids[0, 1:].cpu(),
        entropy=entropy[0, :-1].cpu(),
        offsets=offsets[0, 1:],
    )


def _compute_token_features_remote(code: str) -> TokenFeatures:
    endpoint_url = os.environ.get("LM_CC_REMOTE_URL")
    api_key = os.environ.get("LM_CC_REMOTE_KEY")
    if not endpoint_url or not api_key:
        raise RuntimeError("LM_CC_REMOTE_URL and LM_CC_REMOTE_KEY missing.")

    timeout_seconds = float(os.environ.get("LM_CC_REMOTE_TIMEOUT", "600"))

    try:
        response = requests.post(
            endpoint_url,
            json={"input": {"code": code}},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Remote inference failed.")
        raise RuntimeError(f"Remote inference request failed: {exc}.") from exc

    try:
        response_body = response.json()
    except ValueError as exc:
        logger.error("Remote returned unparsable json body: %s.", response.text[:500])
        raise RuntimeError("Remote returned unparsable json body.") from exc

    output = response_body.get("output")
    if not output or not all(
        field in output for field in ("tokens", "entropy", "offsets")
    ):
        logger.error("Response missing expected fields: %s.", response_body)
        raise RuntimeError("Response missing tokens/entropy/offsets.")

    return TokenFeatures(
        tokens=torch.tensor(output["tokens"], dtype=torch.int64),
        entropy=torch.tensor(output["entropy"], dtype=torch.float32),
        offsets=torch.tensor(output["offsets"], dtype=torch.int64),
    )
