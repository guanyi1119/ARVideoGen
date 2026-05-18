from core.data.lmdb_utils import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os
from methods.anyflow.far.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MultiODERegressionLMDBDataset(Dataset):
    """LMDB dataset spanning multiple shards (sub-directories).

    Used by Causal-Forcing and Self-Forcing for large-scale data.
    """
    def __init__(self, data_path: str, folder_name_pattern: str, max_pair: int = int(1e8)):
        self.envs = []
        self.latents_shape = []
        self.index = []
        valid_fnames = []

        for fname in sorted(os.listdir(data_path)):
            if not (folder_name_pattern in fname):
                continue
            path = os.path.join(data_path, fname)
            try:
                env = lmdb.open(path,
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)
                latents_shape = get_array_shape_from_lmdb(env, 'latents')
                valid_fnames.append(fname)
                self.envs.append(env)
                self.latents_shape.append(latents_shape)
            except:
                continue
        print(f"Loaded {len(valid_fnames)} datasets: {valid_fnames}")
        for shard_id, latents_shape in enumerate(self.latents_shape):
            for local_i in range(latents_shape[0]):
                self.index.append((shard_id, local_i))

        # self.latents_shape = [None] * len(self.envs)
        # for shard_id, env in enumerate(self.envs):
        #     self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
        #     for local_i in range(self.latents_shape[shard_id][0]):
        #         self.index.append((shard_id, local_i))

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width).
                  It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        return {
            "prompts": prompts,
            "latents": torch.tensor(latents, dtype=torch.float32)[-1]
        }
