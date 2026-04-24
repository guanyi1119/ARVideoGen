import numpy as np
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


# [DIFF-CausVid] CausVid-specific: wandb-based logging and saving utilities
def init_logging_folder(args):
    """Initialize wandb logging and create output directory.

    Used by CausVid for experiment tracking.
    """
    from datetime import datetime
    from omegaconf import OmegaConf
    import wandb

    date = str(datetime.now()).replace(" ", "-").replace(":", "-")
    output_path = os.path.join(
        args.output_path,
        f"{date}_seed{args.seed}"
    )
    os.makedirs(output_path, exist_ok=False)

    os.makedirs(args.output_path, exist_ok=True)
    wandb.login(host=args.wandb_host, key=args.wandb_key)
    run = wandb.init(config=OmegaConf.to_container(args, resolve=True), dir=args.output_path, **
                     {"mode": "online", "entity": args.wandb_entity, "project": args.wandb_project})
    wandb.run.log_code(".")
    wandb.run.name = args.wandb_name
    print(f"run dir: {run.dir}")
    wandb_folder = run.dir
    os.makedirs(wandb_folder, exist_ok=True)

    return output_path, wandb_folder


# [DIFF-CausVid] CausVid-specific: prepare tensors for wandb logging
def prepare_for_saving(tensor, fps=16, caption=None):
    """Convert range [-1, 1] to [0, 1] and format for wandb logging.

    Used by CausVid for visualization.
    """
    from torchvision.utils import make_grid
    import wandb

    tensor = (tensor * 0.5 + 0.5).clamp(0, 1).detach()

    if tensor.ndim == 4:
        tensor = make_grid(tensor, 4, padding=0, normalize=False)
        return wandb.Image((tensor * 255).cpu().numpy().astype(np.uint8), caption=caption)
    elif tensor.ndim == 5:
        return wandb.Video((tensor * 255).cpu().numpy().astype(np.uint8), fps=fps, format="webm", caption=caption)
    else:
        raise ValueError("Unsupported tensor shape for saving. Expected 4D (image) or 5D (video) tensor.")
