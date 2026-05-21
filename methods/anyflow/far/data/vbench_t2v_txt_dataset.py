# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from typing import Any, Optional

from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.utils.data import Dataset

from far.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VBenchT2VTXTDataset(Dataset, ConfigMixin):
    
    config_name = 'vbench_t2v_txt_dataset_config.json'
    
    @register_to_config
    def __init__(
        self,
        txt_path: str = 'prompts/self_forcing_all_dimension_extended.txt',
        num_samples_per_prompt: Optional[int] = 5,
    ):
        super().__init__()

        with open(txt_path, 'r') as txt_file:
            prompts = [line.strip() for line in txt_file.readlines()]

        self.samples = []

        for prompt in prompts:
            for idx in range(num_samples_per_prompt):
                prompt_item_ = {
                    "aug_prompt_en": prompt,
                    "video_path": f"{prompt[:100]}-{idx}.mp4"
                }
                self.samples.append(prompt_item_)

        # append sample seed to each samples for metric reimplementation
        for idx, _ in enumerate(self.samples):
            self.samples[idx]['seed'] = idx

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]
