from core.data.lmdb_utils import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os


class TextDataset(Dataset):
    """Text prompt dataset for Causal-Forcing / Self-Forcing / LongLive.

    Reads a text file with one prompt per line. Optionally loads an extended
    prompt file (used for teacher-forcing training in Causal-Forcing).

    [DIFF-CausVid] CausVid has a simpler TextDataset that only reads lines
    without extended_prompt_path support and returns plain strings instead
    of dicts. See CausVid's own data.py if needed.
    """
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class ODERegressionLMDBDataset(Dataset):
    """ODE regression dataset stored in LMDB format.

    Used by Causal-Forcing, Self-Forcing, CausVid, and LongLive.
    """
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width).
              It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


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
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


# [DIFF-CausVid] CausVid-specific: in-memory ODE dataset using torch.load
# (alternative to LMDB for small datasets)
class ODERegressionDataset(Dataset):
    """In-memory ODE regression dataset. Loads all data via torch.load.

    Used by CausVid for small-scale training where LMDB is not needed.
    """
    def __init__(self, data_path, max_pair=int(1e8)):
        self.data_dict = torch.load(data_path, weights_only=False)
        self.max_pair = max_pair

    def __len__(self):
        return min(len(self.data_dict['prompts']), self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width).
              It is ordered from pure noise to clean image.
        """
        return {
            "prompts": self.data_dict['prompts'][idx],
            "ode_latent": self.data_dict['latents'][idx].squeeze(0),
        }


# [DIFF-Causal-Forcing] Causal-Forcing-specific: clean latent supervision dataset
class LatentLMDBDataset(Dataset):
    """LMDB dataset that returns only the final (clean) latent for supervision.

    Used by Causal-Forcing for clean latent loss training.
    """
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - clean_latent: Tensor of shape (num_frames, num_channels, height, width).
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "clean_latent": torch.tensor(latents, dtype=torch.float32)[-1]
        }


# [DIFF-Causal-Forcing] Causal-Forcing-specific: multi-shard LMDB dataset
class ShardingLMDBDataset(Dataset):
    """LMDB dataset spanning multiple shards (sub-directories).

    Used by Causal-Forcing and Self-Forcing for large-scale data.
    """
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

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
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


# [DIFF-Causal-Forcing] Causal-Forcing-specific: I2V training dataset
class TextImagePairDataset(Dataset):
    """Image-text pair dataset for image-to-video training.

    Used by Causal-Forcing for I2V mode.
    """
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        aspect_ratio = metadata_path.stem.split('_')[-1]

        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }


# [DIFF-LongLive] LongLive-specific: two-prompt dataset for prompt-switch training
class TwoTextDataset(Dataset):
    """Dataset that returns two text prompts per sample for prompt-switch training.

    Used by LongLive for streaming training with prompt switches.

    Args:
        prompt_path (str): Path to text file with first-segment prompts.
        switch_prompt_path (str): Path to text file with second-segment prompts.
    """
    def __init__(self, prompt_path: str, switch_prompt_path: str):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        with open(switch_prompt_path, encoding="utf-8") as f:
            self.switch_prompt_list = [line.rstrip() for line in f]

        assert len(self.switch_prompt_list) == len(self.prompt_list), (
            "The two prompt files must contain the same number of lines so that "
            "each first-segment prompt is paired with exactly one second-segment prompt."
        )

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        return {
            "prompts": self.prompt_list[idx],
            "switch_prompts": self.switch_prompt_list[idx],
            "idx": idx,
        }


# [DIFF-LongLive] LongLive-specific: multi-segment prompt dataset from JSONL
class MultiTextDataset(Dataset):
    """Dataset for multi-segment prompts stored in a JSONL file.

    Each line is a JSON object, e.g. {"prompts": ["a cat", "a dog", "a bird"]}
    Used by LongLive for multi-segment streaming training.

    Args:
        prompt_path: Path to the JSONL file.
        field: Name of the list-of-strings field (default "prompts").
        cache_dir: cache_dir passed to HF Datasets (optional).
    """
    def __init__(self, prompt_path: str, field: str = "prompts", cache_dir: str | None = None):
        import datasets as hf_datasets

        self.ds = hf_datasets.load_dataset(
            "json",
            data_files=prompt_path,
            split="train",
            cache_dir=cache_dir,
            streaming=False,
        )

        assert len(self.ds) > 0, "JSONL is empty"
        assert field in self.ds.column_names, f"Missing field '{field}'"

        seg_len = len(self.ds[0][field])
        for i, ex in enumerate(self.ds):
            val = ex[field]
            assert isinstance(val, list), f"Line {i} field '{field}' is not a list"
            assert len(val) == seg_len, f"Line {i} list length mismatch"

        self.field = field

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        return {
            "idx": idx,
            "prompts_list": self.ds[idx][self.field],
        }


def cycle(dl):
    while True:
        for data in dl:
            yield data
