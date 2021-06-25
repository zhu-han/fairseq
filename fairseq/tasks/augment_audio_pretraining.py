# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
import torch

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any
from omegaconf import MISSING, II, OmegaConf

from fairseq.data import (
    AddTargetDataset,
    BinarizedAudioDataset,
    Dictionary,
    AugmentFileAudioDataset,
    encoders,
)
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig

from . import register_task
from .. import utils
from ..logging import metrics

from .audio_pretraining import AudioPretrainingConfig, AudioPretrainingTask, LabelEncoder

logger = logging.getLogger(__name__)


@dataclass
class AugAudioPretrainingConfig(AudioPretrainingConfig):
    aug_sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )
    rir_dict: str = field(
        default="None",
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )  
    noise_dict: str = field(
        default="None",
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )

@register_task("augment_audio_pretraining", dataclass=AugAudioPretrainingConfig)
class AugAudioPretrainingTask(AudioPretrainingTask):
    """ """

    cfg: AugAudioPretrainingConfig

    def __init__(
        self,
        cfg: AugAudioPretrainingConfig,
    ):
        super().__init__(cfg)


    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        # upgrade old task
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == "ctc"

        if getattr(task_cfg, "binarized_dataset", False):
            self.datasets[split] = BinarizedAudioDataset(
                data_path,
                split=split,
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=task_cfg.labels is not None or task_cfg.enable_padding,
                normalize=task_cfg.normalize,
                num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                compute_mask_indices=(self.cfg.precompute_mask_indices or self.cfg.tpu),
                **self._get_mask_precompute_kwargs(task_cfg),
            )
        else:
            manifest_path = os.path.join(data_path, "{}.tsv".format(split))

            self.datasets[split] = AugmentFileAudioDataset(
                manifest_path=manifest_path,
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=task_cfg.labels is not None or task_cfg.enable_padding,
                normalize=task_cfg.normalize,
                num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                compute_mask_indices=(self.cfg.precompute_mask_indices or self.cfg.tpu),
                augment_config={
                    "seed": 8888, 
                    "rir_dict":self.cfg.rir_dict, 
                    "noise_dice":self.cfg.noise_dict, 
                    "sample_rate": self.cfg.aug_sample_rate
                },
                **self._get_mask_precompute_kwargs(task_cfg),
            )

        if self.cfg.tpu and task_cfg["mask_channel_prob"] == 0.0:
            logger.info(
                "Pretraining on TPUs may suffer convergence "
                "issues when training with `mask_channel_prob` value of "
                "0. You may want to set this to a low value close to 0."
            )

        if task_cfg.labels:
            label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
            skipped_indices = getattr(self.datasets[split], "skipped_indices", set())
            with open(label_path, "r") as f:
                labels = [line for i, line in enumerate(f) if i not in skipped_indices]

            assert len(labels) == len(self.datasets[split]), (
                f"labels length ({len(labels)}) and dataset length "
                f"({len(self.datasets[split])}) do not match"
            )

            process_label = LabelEncoder(self.target_dictionary)

            self.datasets[split] = AddTargetDataset(
                self.datasets[split],
                labels,
                pad=self.target_dictionary.pad(),
                eos=self.target_dictionary.eos(),
                batch_targets=True,
                process_label=process_label,
                add_to_input=task_cfg.get("autoregressive", False),
            )

 