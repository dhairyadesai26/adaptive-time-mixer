"""
Ablation variants for the Adaptive FMM extension.

These models allow controlled experiments to understand
which components of the adaptive mixing contribute to performance:

1. NoContextAdaptiveFMM:   Adaptive weights without input history context
2. NoHorizonAdaptiveFMM:   Adaptive weights without horizon conditioning
3. FixedWeightFMM:         Learned but fixed (not input-dependent) weights
"""
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from baseline.models.TimeMixer import Model as BaselineTimeMixer
from extension.layers.adaptive_layers import AdaptiveScaleMixer


class NoContextAdaptiveFMM(BaselineTimeMixer):
    """
    Ablation: Adaptive FMM WITHOUT input history context.

    The weights are predicted from horizon embedding only,
    not from the input data. This tests whether the adaptive
    benefit comes from horizon-awareness alone.

    weight = softmax(weight_net(horizon_embedding))
    """

    def __init__(self, configs):
        super(NoContextAdaptiveFMM, self).__init__(configs)
        num_scales = configs.down_sampling_layers + 1

        # Weight prediction from horizon only
        self.horizon_weight_net = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, num_scales),
        )
        self.horizon_embed = nn.Sequential(
            nn.Linear(1, configs.d_model // 2),
            nn.GELU(),
            nn.Linear(configs.d_model // 2, configs.d_model),
        )
        self.num_scales = num_scales
        self._last_weights = None

        # Init to uniform
        nn.init.zeros_(self.horizon_weight_net[-1].weight)
        nn.init.zeros_(self.horizon_weight_net[-1].bias)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_future_temporal_feature:
            if self.channel_independence == 1:
                B, T, N = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)

        x_enc, x_mark_enc = self._Model__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)
                enc_out_list.append(enc_out)

        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        # Horizon-only weighting (NO context from input)
        h = torch.tensor([[float(self.pred_len)]],
                         device=dec_out_list[0].device)
        horizon_emb = self.horizon_embed(h)  # [1, d_model]
        logits = self.horizon_weight_net(horizon_emb)  # [1, num_scales]
        weights = F.softmax(logits, dim=-1) * self.num_scales  # [1, num_scales]

        stacked = torch.stack(dec_out_list, dim=-1)
        weights_expanded = weights.unsqueeze(1).unsqueeze(2)
        dec_out = (stacked * weights_expanded).sum(dim=-1)

        self._last_weights = weights.detach()
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out


class NoHorizonAdaptiveFMM(BaselineTimeMixer):
    """
    Ablation: Adaptive FMM WITHOUT horizon conditioning.

    Weights depend on input history context only.
    This tests whether horizon-awareness adds value.

    weight = softmax(weight_net(context_from_input))
    """

    def __init__(self, configs):
        super(NoHorizonAdaptiveFMM, self).__init__(configs)
        num_scales = configs.down_sampling_layers + 1

        # Adaptive mixer without horizon
        self.adaptive_mixer = AdaptiveScaleMixer(
            d_model=configs.d_model,
            num_scales=num_scales,
            use_horizon=False,  # Key difference
        )
        self._last_weights = None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_future_temporal_feature:
            if self.channel_independence == 1:
                B, T, N = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)

        x_enc, x_mark_enc = self._Model__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)
                enc_out_list.append(enc_out)

        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        # Context-only weighting (NO horizon)
        dec_out, weights = self.adaptive_mixer(
            enc_out_list, dec_out_list, pred_len=None
        )
        self._last_weights = weights.detach()
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out


class FixedWeightFMM(BaselineTimeMixer):
    """
    Ablation: Learned but FIXED weights (not input-dependent).

    The scale weights are nn.Parameters optimized during training
    but do NOT change based on input. This tests whether the
    benefit comes from learning scale importance vs. adapting it.

    weight = softmax(learnable_params)
    """

    def __init__(self, configs):
        super(FixedWeightFMM, self).__init__(configs)
        num_scales = configs.down_sampling_layers + 1

        # Fixed learnable weights (not input-dependent)
        self.scale_logits = nn.Parameter(torch.zeros(num_scales))
        self._last_weights = None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_future_temporal_feature:
            if self.channel_independence == 1:
                B, T, N = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)

        x_enc, x_mark_enc = self._Model__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)
                enc_out_list.append(enc_out)

        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        # Fixed (non-input-dependent) weighting (scaled by num_scales)
        weights = F.softmax(self.scale_logits, dim=-1) * self.scale_logits.size(0)  # [num_scales]
        stacked = torch.stack(dec_out_list, dim=-1)
        weights_expanded = weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        dec_out = (stacked * weights_expanded).sum(dim=-1)

        self._last_weights = weights.detach().unsqueeze(0)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out
