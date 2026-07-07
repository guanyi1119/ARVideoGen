#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# LongLive-RAG Latent Corpus Generator — Multi-Node Torchrun Launcher
#
# Generates a flat dataset of latent_{global_pos:06d}.pt files from
# text prompts using the Reward-Forcing backbone (optionally with a
# LongLive LoRA adapter).
#
# Environment variables (all optional, with defaults):
#   NNODES          Number of nodes (default: 1)
#   NODE_RANK       Rank of this node (default: 0)
#   NPROC_PER_NODE  GPUs per node   (default: 8)
#   MASTER_ADDR     Master node address (default: 127.0.0.1)
#   MASTER_PORT     Master node port    (default: 29500)
#   CONFIG          Path to the YAML config (default: see below)
#
# Usage (single node, 8 GPUs, RF base):
#   bash generate_longlive_rag_latent.sh
#
# Usage (single node, 8 GPUs, LongLive LoRA):
#   CONFIG=configs/longlive_rag/generate_latent_longlive.yaml \
#       bash generate_longlive_rag_latent.sh
#
# Usage (multi-node, 2 nodes × 8 GPUs):
#   # On node 0:
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
#       bash generate_longlive_rag_latent.sh
#   # On node 1:
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 \
#       bash generate_longlive_rag_latent.sh
#
# Usage (single GPU debug):
#   NPROC_PER_NODE=1 bash generate_longlive_rag_latent.sh
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration (with defaults) ──────────────────────────────────
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
CONFIG=${CONFIG:-configs/longlive_rag/generate_latent_reward_forcing.yaml}

# ── Resolve script directory for torchrun ──────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/scripts/generate_longlive_rag_latent.py"

echo "━━━ LongLive-RAG Latent Generator ━━━"
echo "  Nodes:       ${NNODES}"
echo "  Node rank:   ${NODE_RANK}"
echo "  GPUs/node:   ${NPROC_PER_NODE}"
echo "  Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Config:      ${CONFIG}"
echo "  Script:      ${SCRIPT}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT}" \
    --config_path "${CONFIG}" \
    "$@"
