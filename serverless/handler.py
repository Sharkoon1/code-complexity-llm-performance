import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import runpod
from src.lm_cc import _compute_token_features_local


def handler(event):
    code = event["input"]["code"]
    features = _compute_token_features_local(code)
    return {
        "tokens": features.tokens.tolist(),
        "entropy": features.entropy.tolist(),
        "offsets": features.offsets.tolist(),
    }


runpod.serverless.start({"handler": handler})
