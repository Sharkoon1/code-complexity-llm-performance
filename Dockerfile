FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
ARG REMOTE_MODEL=Qwen/Qwen2.5-Coder-7B
ENV LM_CC_MODEL=${REMOTE_MODEL}
ENV LM_CC_DEVICE=cuda
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV HF_HOME=/runpod-volume/.huggingface

WORKDIR /app

RUN pip install --no-cache-dir \
    "transformers==4.46.3" \
    "accelerate==1.1.1" \
    "huggingface_hub>=0.26,<0.28" \
    runpod requests

COPY src/ /app/src/
COPY serverless/handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]