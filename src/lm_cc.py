import io
import logging
import os
import time
import tokenize
import requests
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
    boundaries = _detect_boundaries(processed_code, features.offsets, features.entropy)
    segments = _mask_to_segments(boundaries)
    # 4. semantic hierachy
    hierachy = _build_hierarchy(segments, features, processed_code, indent_levels)

    return _aggregate(hierachy)


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

        # filter commments 
        if token_type in (tokenize.COMMENT, tokenize.TYPE_COMMENT):
            continue

        # filter blank lines
        if token_type in (tokenize.NL, tokenize.NEWLINE):
            if token_start_line != last_appended_line:
                continue

        kept_tokens.append(token)
        last_appended_line = token_start_line

    return tokenize.untokenize(kept_tokens), indent_levels


def _aggregate(root: SemanticUnit, alpha: float = 0.8) -> dict:
    depths = []
    branches = []

    def walk(node):
        if node.depth > 0:  # skip root
            depths.append(node.depth)
            branches.append(len(node.children))
        for child in node.children:
            walk(child)

    walk(root)

    if not depths:
        return {
            "lm_cc_score": 0.0,
            "lm_cc_max_comp": 0,
            "lm_cc_avg_comp": 0.0,
            "lm_cc_total_comp": 0,
            "lm_cc_max_branch": 0,
            "lm_cc_avg_branch": 0.0,
            "lm_cc_total_branch": 0,
        }

    total_comp = sum(depths)
    total_branch = sum(branches)
    score = alpha * total_branch + (1 - alpha) * total_comp

    return {
        "lm_cc_score": score,
        "lm_cc_max_comp": max(depths),
        "lm_cc_avg_comp": total_comp / len(depths),
        "lm_cc_total_comp": total_comp,
        "lm_cc_max_branch": max(branches),
        "lm_cc_avg_branch": total_branch / len(branches),
        "lm_cc_total_branch": total_branch,
    }


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
    in_range = (token_starts <= boundaries) & (boundaries <= token_ends)
    
    return in_range.any(dim=1)

def _detect_boundaries(
    code:str, offsets: torch.Tensor, entropy: torch.Tensor, tau_quantile: float = 0.67
) -> torch.Tensor:
    tau = torch.quantile(entropy.float(), tau_quantile)
    entropy_boundary_mask = entropy > tau 
    syntatic_boundary_mask = _syntactic_boundary_mask(code, offsets)
    return entropy_boundary_mask | syntatic_boundary_mask


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


def _compute_token_features_remote(code: str) -> TokenFeatures:
    endpoint_url = os.environ.get("LM_CC_REMOTE_URL")
    api_key = os.environ.get("LM_CC_REMOTE_KEY")
    if not endpoint_url or not api_key:
        raise RuntimeError("LM_CC_REMOTE_URL and LM_CC_REMOTE_KEY missing.")

    timeout_seconds = float(os.environ.get("LM_CC_REMOTE_TIMEOUT", "600"))
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.post(
            endpoint_url,
            json={"input": {"code": code}},
            headers=headers,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        response_body = response.json()
    except requests.RequestException as exc:
        logger.error("Remote inference failed.")
        raise RuntimeError(f"Remote inference request failed: {exc}.") from exc
    except ValueError as exc:
        logger.error("Remote returned unparsable json body: %s.", response.text[:500])
        raise RuntimeError("Remote returned unparsable json body.") from exc

    status = response_body.get("status")
    if status in ("IN_QUEUE", "IN_PROGRESS"):
        job_id = response_body.get("id")
        if not job_id:
            raise RuntimeError(f"Response has status {status} but no job id.")
        response_body = _poll_until_complete(
            endpoint_url, job_id, headers, timeout_seconds
        )
    elif status in ("FAILED", "CANCELLED", "TIMED_OUT"):
        raise RuntimeError(f"Remote job ended with status {status}: {response_body}.")

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


def _poll_until_complete(
    endpoint_url: str, job_id: str, headers: dict, timeout_seconds: float
) -> dict:
    base, _, _ = endpoint_url.rpartition("/")
    status_url = f"{base}/status/{job_id}"

    deadline = time.monotonic() + timeout_seconds
    while True:
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"Remote job {job_id} did not complete in {timeout_seconds}s."
            )
        try:
            response = requests.get(status_url, headers=headers, timeout=30)
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(f"Polling job {job_id} failed: {exc}.") from exc

        status = body.get("status")
        if status == "COMPLETED":
            return body
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError(
                f"Remote job {job_id} ended with status {status}: {body}."
            )
        time.sleep(1.0)
