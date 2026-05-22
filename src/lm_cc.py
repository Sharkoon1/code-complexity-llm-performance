import io
import tokenize
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass

# for mac
device = "mps" if torch.backends.mps.is_available() else "cpu"

model_name = "Qwen/Qwen2.5-Coder-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
).to(device)
model.eval()

@dataclass
class SemanticUnit:
    char_start: int       #  from segment (horizontal info)
    char_end: int         #  from segment (horizontal info)
    depth: int            #  from hierarchy (vertical info)
    indent:int            #  from source (raw indentation level)
    children: list[...]   #  from hierarchy (vertical info)

@dataclass
class TokenFeatures:
    tokens: torch.Tensor      # (n-1,) token IDs
    entropy: torch.Tensor     # (n-1,) per-token entropy
    offsets: torch.Tensor     # (n-1, 2) char spans in source

def compute_lm_cc(code: str) -> dict:
    # 1. normalize
    processed_code, indent_levels = _preprocess_code(code)
    # 2. entropy
    features = _compute_token_features(processed_code)
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
        if node.depth > 0:                           # skip root 
            b = len(node.children)                   # branching factor (number of tree children)
            d = node.depth                           # compositional level (deepness of tree)
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
            segments.append((segment_start, i)) # (start, end)
            segment_start = i
    segments.append((segment_start, len(boundaries))) # last segment has no end, doesnt access loop

    return segments


def _detect_boundaries(entropy: torch.Tensor, tau_quantile: float = 0.67) -> torch.Tensor:
    tau = torch.quantile(entropy.float(), tau_quantile)
    is_boundary = entropy > tau
    return is_boundary   

@torch.no_grad()
def _compute_token_features(code: str) -> TokenFeatures:
    inputs = tokenizer(code, return_tensors="pt", return_offsets_mapping=True)
    offsets = inputs.pop("offset_mapping")        
    inputs = inputs.to(model.device)
    
    logits = model(**inputs).logits

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)

    return TokenFeatures(
        tokens=inputs.input_ids[0, 1:].cpu(),
        entropy=entropy[0, :-1].cpu(),
        offsets=offsets[0, 1:],
    )