import gc
import logging
from datetime import datetime
from core.data.dataset import ODERegressionLMDBDataset, cycle
from methods.causal_forcing import ODERegression
from collections import defaultdict
from core.misc import (
    set_seed,
    TensorBoardLogger
)
import torch.distributed as dist
from omegaconf import OmegaConf
import torch

import time
import os

from core.distributed import barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.disable_logging = getattr(config, 'disable_logging', False) or getattr(config, 'disable_wandb', False)

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        self.output_path = config.logdir

        if self.is_main_process and not self.disable_logging:
            tensorboard_dir = os.path.join(self.output_path, "tensorboard")
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = TensorBoardLogger(
                log_dir=tensorboard_dir,
                config=config,
                name=getattr(config, 'config_name', None)
            )

        # Step 2: Initialize the model and optimizer

        assert config.trainer == "ode", "Only ODE loss is supported for ODE training"
        self.model = ODERegression(config, device=self.device)

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        dataset = ODERegressionLMDBDataset(
            config.data_path, max_pair=getattr(config, "max_pair", int(1e8)))
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8)
        self.dataloader = cycle(dataloader)

        self.step = 0

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
                fixed = {}
                for k, v in state_dict.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "generator_ema" in state_dict:
                gen_sd = state_dict["generator_ema"]
                fixed = {}
                for k, v in gen_sd.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        ##############################################################################################################

        self.max_grad_norm = 10.0
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        state_dict = {
            "generator": generator_state_dict
        }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def train_one_step(self, loss_scale=1.0):
        
        self.model.eval()  # prevent any randomness (e.g. dropout)

        VISUALIZE = self.step % self.config.log_iters == 0 and not self.config.no_visualize

        # Step 1: Get the next batch of text prompts
        batch = next(self.dataloader)
        text_prompts = batch["prompts"]
        ode_latent = batch["ode_latent"].to(
            device=self.device, dtype=self.dtype)
        batch_size = len(text_prompts)
        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

        # Step 3: Train the generator
        generator_loss, log_dict = self.model.generator_loss(
            ode_latent=ode_latent,
            conditional_dict=conditional_dict
        )

        unnormalized_loss = log_dict["unnormalized_loss"]
        timestep = log_dict["timestep"]

        if self.world_size > 1:
            gathered_unnormalized_loss = torch.zeros(
                [self.world_size, *unnormalized_loss.shape],
                dtype=unnormalized_loss.dtype, device=self.device)
            gathered_timestep = torch.zeros(
                [self.world_size, *timestep.shape],
                dtype=timestep.dtype, device=self.device)

            dist.all_gather_into_tensor(
                gathered_unnormalized_loss, unnormalized_loss)
            dist.all_gather_into_tensor(gathered_timestep, timestep)
        else:
            gathered_unnormalized_loss = unnormalized_loss
            gathered_timestep = timestep

        loss_breakdown = defaultdict(list)
        stats = {}

        for index, t in enumerate(timestep):
            loss_breakdown[str(int(t.item()) // 250 * 250)].append(
                unnormalized_loss[index].item())

        for key_t in loss_breakdown.keys():
            stats["loss_at_time_" + key_t] = sum(loss_breakdown[key_t]) / \
                len(loss_breakdown[key_t])

        self.generator_optimizer.zero_grad()
        (generator_loss * loss_scale).backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm)
        self.generator_optimizer.step()

        # Step 4: Logging
        if self.is_main_process and not self.disable_logging:
            wandb_loss_dict = {
                "generator_loss": generator_loss.item(),
                "generator_grad_norm": generator_grad_norm.item(),
                **stats
            }
            self.writer.log(wandb_loss_dict, step=self.step)
            if VISUALIZE:
                noisy = self.model.vae.decode_to_pixel(log_dict["input"]).squeeze(1).add_(1.0).div_(2.0).clamp_(0.0, 1.0).mul_(255).cpu().to(torch.uint8).numpy()
                pred = self.model.vae.decode_to_pixel(log_dict["output"]).squeeze(1).add_(1.0).div_(2.0).clamp_(0.0, 1.0).mul_(255).cpu().to(torch.uint8).numpy()
                self.writer.log_video("noisy", noisy, self.step, fps=16)
                self.writer.log_video("pred", pred, self.step, fps=16)

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()

        if (self.step + 1) % 1 == 0:
            end_time = time.time()
            end_step = self.step + 1

            # 计算吞吐量
            step_diff = end_step - self.start_step
            time_diff = end_time - self.start_time
            seconds_per_iter = time_diff / step_diff
            throughput = batch_size / seconds_per_iter

            # 打印训练日志，吞吐加在最后
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"{timestamp}: [step {self.step}]" \
                f"generator_loss: {wandb_loss_dict['generator_loss']:.4f}" \
                f"DI_throughput: {throughput:.2f} samples/s/npu"
            )
            self.start_time = time.time()
            self.start_step = end_step

    def train(self):
        self.start_step = 0
        self.start_time = time.time()
        while True:
            self.train_one_step()
            if (not self.config.no_save) and self.step % self.config.log_iters == 0 and self.step > 0:
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_logging:
                        self.writer.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            self.step += 1
