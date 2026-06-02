"""
Create LMDB from processed latent files (.npz or .pt).

Features:
1. Read all .npz or .pt files from directory (supports moxing/obs://)
2. Write data to LMDB (latents/prompts for npz, ODE pairs for pt)
3. Compatible with MultiODERegressionLMDBDataset
4. Optional sharding into multiple LMDB files

Usage:
    # Local files, npz format
    python scripts/create_lmdb_custom_data.py \
        --input_dir processed_data \
        --lmdb_path output.lmdb

    # Local files, pt format (ODE pairs)
    python scripts/create_lmdb_custom_data.py \
        --input_dir ode_pairs \
        --lmdb_path output.lmdb \
        --input_format pt

    # With moxing for remote paths
    python scripts/create_lmdb_custom_data.py \
        --input_dir obs://bucket/processed_data \
        --lmdb_path obs://bucket/output.lmdb \
        --use_moxing

    # Shard into multiple LMDB files
    python scripts/create_lmdb_custom_data.py \
        --input_dir processed_data \
        --lmdb_path output.lmdb \
        --num_shards 4
"""
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

import argparse
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
# NPU support
DEVICE_TYPE = os.environ.get('DEVICE_TYPE', 'cuda')
if DEVICE_TYPE == "npu":
    from torch_npu.contrib import transfer_to_npu

from core.data.lmdb_utils import store_arrays_to_lmdb


# Optional moxing import
mox = None
def try_import_moxing():
    global mox
    try:
        import moxing as mox
        return True
    except ImportError:
        return False


def list_files(input_dir, use_moxing, file_ext):
    """List all files with given extension in directory, sorted by name."""
    is_remote = use_moxing and (input_dir.startswith('obs://') or input_dir.startswith('s3://'))

    if is_remote and mox is not None:
        # List remote files
        all_files = mox.file.list_directory(input_dir, recursive=False)
        files = [f for f in all_files if f.endswith(file_ext)]
        files.sort()
        return [os.path.join(input_dir, f) for f in files]
    else:
        # List local files
        path = Path(input_dir)
        if not path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        files = sorted(list(path.glob(f"*{file_ext}")))
        return [str(f) for f in files]


def load_from_npz(file_path, use_moxing):
    """Load latent and prompt from .npz file."""
    is_remote = use_moxing and (file_path.startswith('obs://') or file_path.startswith('s3://'))

    if is_remote and mox is not None:
        # Read remote file
        with mox.file.File(file_path, 'rb') as f:
            bio = BytesIO(f.read())
        data = np.load(bio, allow_pickle=True)
    else:
        # Read local file
        data = np.load(file_path, allow_pickle=True)

    latent = data['latent']
    prompt = str(data['prompt'])
    return [{'latent': latent, 'prompt': prompt}]


def load_from_pt(file_path, use_moxing):
    """Load data dict from .pt file (ODE pairs format)."""
    is_remote = use_moxing and (file_path.startswith('obs://') or file_path.startswith('s3://'))

    if is_remote and mox is not None:
        # Read remote file
        with mox.file.File(file_path, 'rb') as f:
            bio = BytesIO(f.read())
        data_dict = torch.load(bio)
    else:
        # Read local file
        data_dict = torch.load(file_path)

    # Process data dict and return list of samples
    # Format: {prompt: stored_data_tensor}
    samples = []
    for prompt, stored_data in data_dict.items():
        sample = {
            'prompt': str(prompt),
            'latents': stored_data
        }
        samples.append(sample)
    return samples


def build_data_dict_from_sample(sample):
    """Build data dict from sample for LMDB storage."""
    data_dict = {}
    for key, val in sample.items():
        if key == 'prompt':
            data_dict['prompts'] = np.array([val], dtype=object)
        elif key == 'latent':
            data_dict['latents'] = np.expand_dims(val, axis=0)
        elif key == 'latents':
            data_dict['latents'] = np.expand_dims(val, axis=0)
        else:
            data_dict[key] = np.expand_dims(val, axis=0)
    return data_dict


def main():
    parser = argparse.ArgumentParser(description="Create LMDB from processed latent files")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input files")
    parser.add_argument("--lmdb_path", type=str, required=True, help="Path to output LMDB")
    parser.add_argument("--use_moxing", action="store_true", help="Enable moxing for remote file access")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of LMDB shards to split data into (default: 1, no sharding)")
    parser.add_argument("--input_format", type=str, default="npz", choices=["npz", "pt"], help="Input file format: npz or pt (default: npz)")
    args = parser.parse_args()

    # Import moxing if needed
    if args.use_moxing:
        if not try_import_moxing():
            print("Warning: moxing not available, falling back to local file access only")
            args.use_moxing = False

    print(f"Reading {args.input_format} files from: {args.input_dir}")

    # List all input files
    file_ext = f".{args.input_format}"
    input_files = list_files(args.input_dir, args.use_moxing, file_ext)

    if not input_files:
        print(f"No {file_ext} files found in {args.input_dir}")
        return

    print(f"Found {len(input_files)} input files")

    num_shards = max(1, args.num_shards)
    if num_shards > 1:
        print(f"Sharding into {num_shards} LMDB files")

    # Check if LMDB path is remote
    is_lmdb_remote = args.use_moxing and (args.lmdb_path.startswith('obs://') or args.lmdb_path.startswith('s3://'))

    # Build per-shard state
    import lmdb
    map_size = 1024 ** 4
    shards = []

    for shard_idx in range(num_shards):
        if num_shards == 1:
            shard_lmdb_path = args.lmdb_path
        else:
            shard_lmdb_path = f"{args.lmdb_path}_shard_{shard_idx}"

        if is_lmdb_remote:
            temp_dir = tempfile.mkdtemp(prefix=f"create_lmdb_shard{shard_idx}_")
            local_lmdb_path = os.path.join(temp_dir, "lmdb_temp")
        else:
            local_lmdb_path = shard_lmdb_path
            temp_dir = None

        os.makedirs(os.path.dirname(local_lmdb_path) or '.', exist_ok=True)
        env = lmdb.open(local_lmdb_path, map_size=map_size)

        shards.append({
            'lmdb_path': shard_lmdb_path,
            'local_lmdb_path': local_lmdb_path,
            'env': env,
            'counter': 0,
            'first_sample': None,
            'temp_dir': temp_dir,
        })

    # Select load function based on input format
    if args.input_format == "npz":
        load_func = load_from_npz
    else:
        load_func = load_from_pt

    # Process each file, round-robin into shards
    total_samples = 0
    for file_idx, file_path in enumerate(tqdm(input_files, desc="Writing LMDB")):
        shard_idx = file_idx % num_shards
        shard = shards[shard_idx]
        try:
            samples = load_func(file_path, args.use_moxing)

            for sample in samples:
                # Keep first sample for shape info
                if shard['counter'] == 0:
                    shard['first_sample'] = sample

                # Build data dict and store
                data_dict = build_data_dict_from_sample(sample)
                store_arrays_to_lmdb(shard['env'], data_dict, start_index=shard['counter'])
                shard['counter'] += 1
                total_samples += 1

        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            continue

    # Finalize each shard
    total_written = 0
    for shard_idx, shard in enumerate(shards):
        counter = shard['counter']
        total_written += counter

        if counter == 0:
            shard['env'].close()
            if shard['temp_dir'] is not None:
                shutil.rmtree(shard['temp_dir'])
            if num_shards > 1:
                print(f"Shard {shard_idx}: no files processed, skipped")
            continue

        # Write shape information
        if num_shards > 1:
            print(f"Writing shape info for shard {shard_idx}...")
        else:
            print("Writing shape info...")
        with shard['env'].begin(write=True) as txn:
            # Use first sample to get all keys
            first_sample = shard['first_sample']

            # Build sample dict with all keys
            sample_dict = build_data_dict_from_sample(first_sample)

            # Write shape for each key
            for key, val in sample_dict.items():
                array_shape = np.array(val.shape)
                array_shape[0] = counter  # total count in this shard
                shape_key = f"{key}_shape".encode()
                shape_str = " ".join(map(str, array_shape))
                txn.put(shape_key, shape_str.encode())
                print(f"  {key}: shape {array_shape}")

        shard['env'].close()

        # If LMDB path is remote, copy local temp LMDB to remote
        if is_lmdb_remote:
            remote_shard_path = shard['lmdb_path']
            print(f"Copying local LMDB to remote: {remote_shard_path}")
            try:
                mox.file.make_dirs(os.path.dirname(remote_shard_path))
            except Exception:
                pass
            local_lmdb_dir = Path(shard['local_lmdb_path'])
            for f in local_lmdb_dir.iterdir():
                if f.is_file():
                    remote_path = os.path.join(remote_shard_path, f.name)
                    mox.file.copy(str(f), remote_path)
            if shard['temp_dir'] is not None:
                shutil.rmtree(shard['temp_dir'])

    # Summary
    if num_shards > 1:
        per_shard = [s['counter'] for s in shards]
        print(f"Done! Wrote {total_written} samples across {num_shards} shards: {per_shard}")
    else:
        print(f"Done! Successfully wrote {total_written} samples to LMDB: {args.lmdb_path}")


if __name__ == "__main__":
    main()
