import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

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
