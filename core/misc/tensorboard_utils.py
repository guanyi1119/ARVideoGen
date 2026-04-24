import os
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf


class TensorBoardLogger:
    """TensorBoard-based logger, replacing wandb for local experiment tracking."""

    def __init__(self, log_dir, config=None, name=None):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir
        if config is not None:
            config_path = os.path.join(log_dir, "config.yaml")
            OmegaConf.save(config, config_path)

    def log(self, metrics: dict, step: int):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, global_step=step)
            elif hasattr(value, 'numpy'):
                if value.numel() == 1:
                    self.writer.add_scalar(key, value.item(), global_step=step)
                else:
                    self.writer.add_scalar(key, value.mean().item(), global_step=step)

    def log_image(self, tag, tensor, step):
        """Log image tensor [C, H, W] or [N, C, H, W]."""
        self.writer.add_image(tag, tensor, global_step=step)

    def log_video(self, tag, video_tensor, step, fps=16):
        """Log video tensor [N, T, C, H, W] in [0, 1] range."""
        self.writer.add_video(tag, video_tensor, global_step=step, fps=fps)

    def close(self):
        self.writer.close()
