import numpy as np
import os
import random
import torch


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def merge_dict_list(dict_list):
    if len(dict_list) == 1:
        return dict_list[0]

    merged_dict = {}
    for k, v in dict_list[0].items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                merged_dict[k] = torch.stack([d[k] for d in dict_list], dim=0)
            else:
                merged_dict[k] = torch.cat([d[k] for d in dict_list], dim=0)
        else:
            merged_dict[k] = v
    return merged_dict


def cycle(dl):
    while True:
        for data in dl:
            yield data


def init_logging_folder(args):
    """Initialize TensorBoard logging and create output directory."""
    from datetime import datetime
    from .tensorboard_utils import TensorBoardLogger

    date = str(datetime.now()).replace(" ", "-").replace(":", "-")
    output_path = os.path.join(
        args.output_path,
        f"{date}_seed{args.seed}"
    )
    os.makedirs(output_path, exist_ok=False)

    os.makedirs(args.output_path, exist_ok=True)

    tensorboard_dir = os.path.join(output_path, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = TensorBoardLogger(
        log_dir=tensorboard_dir,
        config=args,
        name=getattr(args, 'wandb_name', None)
    )

    return output_path, writer


def prepare_for_saving(tensor, fps=16, caption=None):
    """Convert range [-1, 1] to [0, 1] and format for logging.

    Returns:
        3D tensor [C, H, W] for images (grid), or
        5D tensor [N, T, C, H, W] for videos.
    """
    from torchvision.utils import make_grid

    tensor = (tensor * 0.5 + 0.5).clamp(0, 1).detach()

    if tensor.ndim == 4:
        # [B, C, H, W] → grid image [C, H', W']
        return make_grid(tensor, 4, padding=0, normalize=False)
    elif tensor.ndim == 5:
        # [B, T, C, H, W] → keep as-is for add_video
        return tensor
    else:
        raise ValueError("Unsupported tensor shape for saving. Expected 4D (image) or 5D (video) tensor.")
