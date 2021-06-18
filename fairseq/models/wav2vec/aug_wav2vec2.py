# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
    MultiheadAttention,
    SamePad,
    TransposeLast,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import buffered_arange, index_put, is_xla_tensor
from .wav2vec2 import Wav2Vec2Config, Wav2Vec2Model

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])


@dataclass
class AugWav2Vec2Config(Wav2Vec2Config):
    pass

@register_model("aug_wav2vec2", dataclass=AugWav2Vec2Config)
class AugWav2Vec2Model(Wav2Vec2Model):
    def __init__(self, cfg: AugWav2Vec2Config):
        super().__init__(cfg)
        
        
    def forward(
        self,
        source,
        padding_mask=None,
        mask=True,
        features_only=False,
        layer=None,
        mask_indices=None,
        mask_channel_indices=None,
        padding_count=None,
    ):

        if len(source)== 1 or isinstance(source, torch.Tensor):
            return super().forward(
                source[0], 
                padding_mask=padding_mask,
                mask=mask,
                features_only=features_only,
                layer=layer,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
                padding_count=padding_count,
            )

        features_list = []
        features_pen_list = []
        unmasked_features_list = []
        for i in range(len(source)):
            if self.feature_grad_mult > 0:
                features = self.feature_extractor(source[i])
                if self.feature_grad_mult != 1.0:
                    features = GradMultiply.apply(features, self.feature_grad_mult)
            else:
                with torch.no_grad():
                    features = self.feature_extractor(source[i])
 
            features_pen = features.float().pow(2).mean()
            features = features.transpose(1, 2)
            features = self.layer_norm(features)
            unmasked_features = features.clone()
            
            features_list.append(features)   
            features_pen_list.append(features_pen)
            unmasked_features_list.append(unmasked_features)

        features_pen = torch.mean(torch.stack(features_pen_list))

        if padding_mask is not None and padding_mask.any():
            features_0 = features_list[0]
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features_0.shape[:2], dtype=features_0.dtype, device=features_0.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        channel_num = len(features_list)
        num_vars_list = [0] * channel_num
        code_ppl_list = [0] * channel_num
        prob_ppl_list = [0] * channel_num
        curr_temp_list = [0] * channel_num
        x_list = [0] * channel_num
        y_list = [0] * channel_num
        neg_list = [0] * channel_num


        for i in range(channel_num):
            if self.post_extract_proj is not None:
                features_list[i] = self.post_extract_proj(features_list[i])
            features_list[i] = self.dropout_input(features_list[i])
            unmasked_features_list[i] = self.dropout_features(unmasked_features_list[i])


            if self.input_quantizer:
                q = self.input_quantizer(features, produce_targets=False)
                features_list[i] = q["x"]
                num_vars_list[i] = q["num_vars"]
                code_ppl_list[i] = q["code_perplexity"]
                prob_ppl_list[i] = q["prob_perplexity"]
                curr_temp_list[i] = q["temp"]
                features_list[i] = self.project_inp(features_list[i])

            if mask:
                x, mask_indices = self.apply_mask(
                    features_list[i],
                    padding_mask,
                    mask_indices=mask_indices,
                    mask_channel_indices=mask_channel_indices,
                )
                if not is_xla_tensor(x) and mask_indices is not None:
                    # tpu-comment: reducing the size in a dynamic way causes
                    # too many recompilations on xla.
                    y = unmasked_features_list[i][mask_indices].view(
                        unmasked_features_list[i].size(0), -1, unmasked_features_list[i].size(-1)
                    )
                else:
                    y = unmasked_features_list[i]
            else:
                x = features_list[i]
                y = unmasked_features_list[i]
                mask_indices = None

            x, layer_results = self.encoder(x, padding_mask=padding_mask, layer=layer)

            if features_only:
                return {
                    "x": x,
                    "padding_mask": padding_mask,
                    "features": unmasked_features,
                    "layer_results": layer_results,
                }

            if self.quantizer:
                q = self.quantizer(y, produce_targets=False)
                y = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]

                y = self.project_q(y)

                if self.negatives_from_everywhere:
                    neg_cands = self.quantizer(unmasked_features, produce_targets=False)[
                        "x"
                    ]
                    negs, _ = self.sample_negatives(
                        neg_cands,
                        y.size(1),
                        padding_count=padding_count,
                    )
                    negs = self.project_q(negs)

                else:
                    negs, _ = self.sample_negatives(
                        y,
                        y.size(1),
                        padding_count=padding_count,
                    )

                if self.codebook_negatives > 0:
                    cb_negs = self.quantizer.sample_from_codebook(
                        y.size(0) * y.size(1), self.codebook_negatives
                    )
                    cb_negs = cb_negs.view(
                        self.codebook_negatives, y.size(0), y.size(1), -1
                    )  # order doesnt matter
                    cb_negs = self.project_q(cb_negs)
                    negs = torch.cat([negs, cb_negs], dim=0)
            else:
                y = self.project_q(y)

                if self.negatives_from_everywhere:
                    negs, _ = self.sample_negatives(
                        unmasked_features,
                        y.size(1),
                        padding_count=padding_count,
                    )
                    negs = self.project_q(negs)
                else:
                    negs, _ = self.sample_negatives(
                        y,
                        y.size(1),
                        padding_count=padding_count,
                    )

            if not is_xla_tensor(x):
                # tpu-comment: reducing the size in a dynamic way causes
                # too many recompilations on xla.
                x = x[mask_indices].view(x.size(0), -1, x.size(-1))

            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)

            x = self.final_proj(x)

            x_list[i], y_list[i], neg_list[i] = x, y, negs

        x = self.compute_preds_list(x_list, y_list, neg_list)

        result = {
            "x": x,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result

    def compute_preds_list(self, x_list, y_list, negatives_list):
        import random
        x = random.choice(x_list)
        y = random.choice(y_list)
        neg = torch.cat(negatives_list)
        neg_idex = list(range(neg.size(0)))
        neg_idex = random.sample(neg_idex, self.cfg.num_negatives) 
        neg = neg[neg_idex]
        return self.compute_preds(x, y, neg)
