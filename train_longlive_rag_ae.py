"""Thin entry point for LongLive-RAG retrieval autoencoder training.

Usage:
    python train_longlive_rag_ae.py
    # or override config:
    python train_longlive_rag_ae.py --config configs/longlive_rag/ae_delta.yaml

All real logic lives in ``methods.longlive_rag.ae.train`` so that this file
stays a stable invocation target (mirrors the train_<method>.py pattern in
the repo). NPU support follows the same convention as every other
``train_<method>.py`` here: read DEVICE_TYPE from env, transfer to NPU when
running on Ascend hardware.
"""
import os
import sys

# NPU support (mirrors every train_<method>.py / inference_<method>.py in repo)
DEVICE_TYPE = os.environ.get('DEVICE_TYPE', 'cuda')
if DEVICE_TYPE == "npu":
    from torch_npu.contrib import transfer_to_npu


def main():
    # Make the repo root importable when this file is run from anywhere.
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from methods.longlive_rag.ae.train import main as ae_main
    ae_main()


if __name__ == "__main__":
    main()