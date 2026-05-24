FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ARG REMOTE_MODEL=Qwen/Qwen2.5-Coder-7B
ARG HF_TOKEN=""

ENV HF_TOKEN=${HF_TOKEN}
ENV LM_CC_MODEL=${REMOTE_MODEL}
ENV LM_CC_DEVICE=cuda

WORKDIR /app

RUN pip install --no-cache-dir \
    "transformers==4.46.3" \
    "accelerate==1.1.1" \
    "huggingface_hub>=0.26,<0.28" \
    runpod requests

RUN python -c "import os; from huggingface_hub import snapshot_download; \
    token = os.environ.get('HF_TOKEN') or None; \
    snapshot_download(repo_id=os.environ['LM_CC_MODEL'], token=token)"

COPY src/ /app/src/
COPY serverless/handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
