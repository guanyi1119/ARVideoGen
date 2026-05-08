from datetime import datetime
from methods.causvid.data import ODERegressionDataset, ODERegressionLMDBDataset
from core.data.dataset import MultiODERegressionLMDBDataset
from methods.causvid.ode_regression import ODERegression
from transformers.models.t5.modeling_t5 import T5Block
from collections import defaultdict
from methods.causvid.util import (
    launch_distributed_job,
    set_seed, init_logging_folder,
    fsdp_wrap, cycle,
    fsdp_state_dict,
    barrier
)
import torch.distributed as dist
from omegaconf import OmegaConf
import argparse
import torch
import time
import os


class Trainer:
    def __init__(self, config):
        self.config = config

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process:
            self.output_path, self.writer = init_logging_folder(config)

        # Step 2: Initialize the model and optimizer

        if config.distillation_loss == "ode":
            self.distillation_model = ODERegression(config, device=self.device)
        else:
            raise ValueError("Invalid distillation loss type")

        self.distillation_model.generator = fsdp_wrap(
            self.distillation_model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            transformer_module=(T5Block,
                                ) if config.generator_fsdp_wrap_strategy == "transformer" else None
        )
        self.distillation_model.text_encoder = fsdp_wrap(
            self.distillation_model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            transformer_module=(T5Block,
                                ) if config.text_encoder_fsdp_wrap_strategy == "transformer" else None
        )

        if not config.no_visualize:
            self.distillation_model.vae = self.distillation_model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.distillation_model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2)
        )

        # Step 3: Initialize the dataloader
        # dataset = ODERegressionDataset(config.data_path)
        data_name_pattern = getattr(config, "data_name_pattern", "")
        if data_name_pattern:
            dataset = MultiODERegressionLMDBDataset(
                config.data_path, data_name_pattern, max_pair=getattr(config, "max_pair", int(1e8))
            )
        else:
            dataset = ODERegressionLMDBDataset(
                config.data_path, max_pair=getattr(config, "max_pair", int(1e8)))
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8)
        self.dataloader = cycle(dataloader)

        self.step = 0
        self.max_grad_norm = 10.0
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.distillation_model.generator)
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

    def train_one_step(self):
        self.distillation_model.eval()  # prevent any randomness (e.g. dropout)

        VISUALIZE = self.step % self.config.log_iters == 0 and not self.config.no_visualize

        # Step 1: Get the next batch of text prompts
        batch = next(self.dataloader)
        text_prompts = batch["prompts"]
        ode_latent = batch["ode_latent"].to(
            device=self.device, dtype=self.dtype)
        batch_size = len(text_prompts)

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.distillation_model.text_encoder(
                text_prompts=text_prompts)

        # Step 3: Train the generator
        generator_loss, log_dict = self.distillation_model.generator_loss(
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
        generator_loss.backward()
        generator_grad_norm = self.distillation_model.generator.clip_grad_norm_(
            self.max_grad_norm)
        self.generator_optimizer.step()

        # Step 4: Logging
        log_dict_write = {
            "generator_loss": generator_loss.item(),
            "generator_grad_norm": generator_grad_norm.item(),
            **stats
        }
        if self.is_main_process:
            self.writer.log(log_dict_write, step=self.step)
            if VISUALIZE:
                noisy = self.distillation_model.vae.decode_to_pixel(log_dict["input"]).squeeze(1).add_(1.0).div_(2.0).clamp_(0.0, 1.0).mul_(255).cpu().to(torch.uint8).numpy()
                pred = self.distillation_model.vae.decode_to_pixel(log_dict["output"]).squeeze(1).add_(1.0).div_(2.0).clamp_(0.0, 1.0).mul_(255).cpu().to(torch.uint8).numpy()
                self.writer.log_video("noisy", noisy, self.step, fps=16)
                self.writer.log_video("pred", pred, self.step, fps=16)

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
                f"{timestamp}: [step {self.step}] " \
                f"generator_loss: {log_dict_write['generator_loss']:.4f} " \
                f"DI_throughput: {throughput:.2f} samples/s/npu"
            )
            self.start_time = time.time()
            self.start_step = end_step

    def train(self):
        self.start_step = 0
        self.start_time = time.time()
        while True:
            self.train_one_step()
            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    self.writer.log({"per iteration time": current_time -
                              self.previous_time}, step=self.step)
                    self.previous_time = current_time

            self.step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_save", action="store_true")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    config.no_save = args.no_save

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
