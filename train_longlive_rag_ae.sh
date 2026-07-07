#!/bin/bash
# Train the LongLive-RAG retrieval autoencoder (single GPU, CPU-only safe).
# All hyperparameters live in configs/longlive_rag/ae_delta.yaml.
#
# Override GPU via CUDA_VISIBLE_DEVICES:
#   CUDA_VISIBLE_DEVICES=2 bash train_longlive_rag_ae.sh
# Override config:
#   CONFIG=configs/longlive_rag/ae_delta.yaml bash train_longlive_rag_ae.sh

set -uo pipefail

CONFIG="${CONFIG:-configs/longlive_rag/ae_delta.yaml}"

python -m methods.longlive_rag.ae.train --config "${CONFIG}"