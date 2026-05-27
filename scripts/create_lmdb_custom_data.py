"""
Create LMDB from processed latent files (.npz).

Features:
1. Read all .npz files from directory (supports moxing/obs://)
2. Write latents and prompts to LMDB
3. Compatible with MultiODERegressionLMDBDataset

Usage:
    # Local files
    python scripts/create_lmdb_custom_data.py \
        --input_dir processed_data \
        --lmdb_path output.lmdb

    # With moxing for remote paths
    python scripts/create_lmdb_custom_data.py \
        --input_dir obs://bucket/processed_data \
        --lmdb_path obs://bucket/output.lmdb \
        --use_moxing
"""
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

import argparse
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


def list_npz_files(input_dir, use_moxing):
    """List all .npz files in directory, sorted by name."""
    is_remote = use_moxing and (input_dir.startswith('obs://') or input_dir.startswith('s3://'))

    if is_remote and mox is not None:
        # List remote files
        all_files = mox.file.list_directory(input_dir, recursive=False)
        npz_files = [f for f in all_files if f.endswith('.npz')]
        npz_files.sort()
        return [os.path.join(input_dir, f) for f in npz_files]
    else:
        # List local files
        path = Path(input_dir)
        if not path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        npz_files = sorted(list(path.glob("*.npz")))
        return [str(f) for f in npz_files]


def load_latent_and_prompt(file_path, use_moxing):
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
    return latent, prompt


def main():
    parser = argparse.ArgumentParser(description="Create LMDB from processed latent files")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with .npz latent files")
    parser.add_argument("--lmdb_path", type=str, required=True, help="Path to output LMDB")
    parser.add_argument("--use_moxing", action="store_true", help="Enable moxing for remote file access")
    args = parser.parse_args()

    # Import moxing if needed
    if args.use_moxing:
        if not try_import_moxing():
            print("Warning: moxing not available, falling back to local file access only")
            args.use_moxing = False

    print(f"Reading latent files from: {args.input_dir}")

    # List all npz files
    npz_files = list_npz_files(args.input_dir, args.use_moxing)

    if not npz_files:
        print(f"No .npz files found in {args.input_dir}")
        return

    print(f"Found {len(npz_files)} latent files")

    # Check if LMDB path is remote
    is_lmdb_remote = args.use_moxing and (args.lmdb_path.startswith('obs://') or args.lmdb_path.startswith('s3://'))

    # Determine actual LMDB path (local temp if remote)
    if is_lmdb_remote:
        # Create local temp LMDB first
        temp_dir = tempfile.mkdtemp(prefix="create_lmdb_")
        local_lmdb_path = os.path.join(temp_dir, "lmdb_temp")
        print(f"Creating local temp LMDB first: {local_lmdb_path}")
    else:
        local_lmdb_path = args.lmdb_path

    # Create LMDB
    os.makedirs(os.path.dirname(local_lmdb_path) or '.', exist_ok=True)

    # Set map size: 1TB as safety
    import lmdb
    map_size = 1024 ** 4
    env = lmdb.open(local_lmdb_path, map_size=map_size)

    # Keep first sample for shape info
    first_latent = None
    first_prompt = None
    counter = 0

    # Process each file
    for file_path in tqdm(npz_files, desc="Writing LMDB"):
        try:
            latent, prompt = load_latent_and_prompt(file_path, args.use_moxing)

            # Keep first sample
            if counter == 0:
                first_latent = latent
                first_prompt = prompt

            # Write single entry
            data_dict = {
                'latents': np.expand_dims(latent, axis=0),  # (1, T, C, H, W)
                'prompts': np.array([prompt], dtype=object)
            }
            store_arrays_to_lmdb(env, data_dict, start_index=counter)
            counter += 1

        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            continue

    if counter == 0:
        print("No files processed successfully!")
        env.close()
        return

    # Write shape information
    print("Writing shape info...")
    with env.begin(write=True) as txn:
        # Use first entry to determine shape, set count to counter
        latents_sample = np.expand_dims(first_latent, axis=0)
        prompts_sample = np.array([first_prompt], dtype=object)

        for key, val in [('latents', latents_sample), ('prompts', prompts_sample)]:
            array_shape = np.array(val.shape)
            array_shape[0] = counter  # total count
            shape_key = f"{key}_shape".encode()
            shape_str = " ".join(map(str, array_shape))
            txn.put(shape_key, shape_str.encode())
            print(f"Wrote {key}: shape {array_shape}")

    env.close()

    # If LMDB path is remote, copy local temp LMDB to remote
    if is_lmdb_remote:
        print(f"Copying local LMDB to remote: {args.lmdb_path}")
        # Make sure remote directory exists
        try:
            mox.file.make_dirs(os.path.dirname(args.lmdb_path))
        except Exception:
            pass
        # Copy all files in LMDB directory
        local_lmdb_dir = Path(local_lmdb_path)
        for f in local_lmdb_dir.iterdir():
            if f.is_file():
                remote_path = os.path.join(args.lmdb_path, f.name)
                mox.file.copy(str(f), remote_path)
        # Clean up local temp files
        import shutil
        shutil.rmtree(temp_dir)

    print(f"Done! Successfully wrote {counter} videos to LMDB: {args.lmdb_path}")


if __name__ == "__main__":
    main()
