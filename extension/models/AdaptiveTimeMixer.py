"""
Adaptive TimeMixer: TimeMixer + Adaptive Multiscale FMM.

This is our research contribution. It inherits the entire TimeMixer
architecture (PDM, normalization, embedding, etc.) and ONLY modifies
the Future Multipredictor Mixing (FMM) stage.

Original FMM:
    forecast = Σ forecast_i   (uniform summation)

Our Adaptive FMM:
    w_1, ..., w_k = softmax(weight_net(context))
    forecast = Σ w_i * forecast_i   (input-dependent weighting)

Key design decision: we initialize the adaptive weights to produce
uniform weights, so the model starts with behavior identical to
the baseline. The adaptive weights are then learned during training.
"""
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from baseline.models.TimeMixer import Model as BaselineTimeMixer
from extension.layers.adaptive_layers import AdaptiveScaleMixer


class AdaptiveTimeMixerModel(BaselineTimeMixer):
    """
    TimeMixer with Adaptive Multiscale Forecast Mixing.

    Inherits everything from baseline TimeMixer.
    Overrides ONLY the future_multi_mixing() method to use
    AdaptiveScaleMixer instead of uniform summation.

    Additional attributes:
        adaptive_mixer: AdaptiveScaleMixer module
        _last_weights: cached scale weights from last forward pass (for visualization)
    """

    def __init__(self, configs):
        super(AdaptiveTimeMixerModel, self).__init__(configs)

        # Number of scales = down_sampling_layers + 1
        num_scales = configs.down_sampling_layers + 1

        # Whether to use horizon conditioning
        use_horizon = getattr(configs, 'adaptive_use_horizon', True)
        temperature = getattr(configs, 'adaptive_temperature', 1.0)

        # Create the adaptive scale mixer
        self.adaptive_mixer = AdaptiveScaleMixer(
            d_model=configs.d_model,
            num_scales=num_scales,
            use_horizon=use_horizon,
            max_horizon=getattr(configs, 'max_horizon', 720),
            temperature=temperature,
        )

        # Cache for visualization
        self._last_weights = None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Override forecast to use adaptive mixing instead of uniform sum.
        The PDM encoding path is IDENTICAL to baseline.
        Only the final combination step changes.

        Handles channel_independence=1 where batch dim becomes B*N inside encoder.
        """
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

        # Embedding
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

        # Past Decomposable Mixing (identical to baseline)
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing — OUR MODIFICATION
        # Step 1: Generate per-scale forecasts (same as baseline)
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        # Step 2: Adaptive combination instead of uniform sum
        # dec_out_list items are [B, pred_len, c_out] after baseline's reshape
        # enc_out_list items are [B*N, T_i, d_model] when channel_independence=1
        # We compute weights from the encoder outputs (B*N batches), then
        # average across channels to get [B, num_scales] weights
        weights = self._compute_adaptive_weights(enc_out_list, B, N if self.channel_independence == 1 else 1)

        # Weighted combination
        stacked = torch.stack(dec_out_list, dim=-1)  # [B, pred_len, c_out, num_scales]
        weights_expanded = weights.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, num_scales]
        dec_out = (stacked * weights_expanded).sum(dim=-1)  # [B, pred_len, c_out]

        # Cache weights for visualization
        self._last_weights = weights.detach()

        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def _compute_adaptive_weights(self, enc_out_list, B, N):
        """
        Compute adaptive scale weights, handling channel_independence reshape.

        When channel_independence=1, enc_out_list has [B*N, T_i, d_model] tensors.
        We pool across channels to get per-batch weights.
        """
        import torch.nn.functional as F

        # Get context from each scale via global average pooling
        scale_contexts = []
        for enc_out in enc_out_list:
            pooled = enc_out.mean(dim=1)  # [B*N, d_model] or [B, d_model]
            if N > 1:
                # Average across channels: reshape [B*N, d_model] → [B, N, d_model] → mean → [B, d_model]
                pooled = pooled.reshape(B, N, -1).mean(dim=1)
            scale_contexts.append(pooled)

        # Stack and process through adaptive mixer's weight network
        # Simple aggregation: concatenate scale contexts
        d_model = scale_contexts[0].size(-1)
        num_scales = len(scale_contexts)

        # Project each scale context
        projected = []
        for i, ctx in enumerate(scale_contexts):
            projected.append(self.adaptive_mixer.context_aggregator.scale_projections[i](ctx))

        combined = torch.cat(projected, dim=-1)  # [B, d_model * num_scales]
        context = self.adaptive_mixer.context_aggregator.combine(combined)  # [B, d_model]

        # Optionally add horizon conditioning
        if self.adaptive_mixer.use_horizon and self.pred_len is not None:
            horizon_emb = self.adaptive_mixer.horizon_encoder(self.pred_len)
            horizon_emb = horizon_emb.expand(B, -1)
            conditioning = torch.cat([context, horizon_emb], dim=-1)
        else:
            conditioning = context

        # Predict weights (scaled by num_scales so uniform init corresponds to baseline sum)
        logits = self.adaptive_mixer.weight_net(conditioning)
        weights = F.softmax(logits / self.adaptive_mixer.temperature, dim=-1) * num_scales  # [B, num_scales]

        return weights

    def get_last_weights(self):
        """
        Returns the scale weights from the most recent forward pass.
        Useful for visualization and interpretability analysis.

        Returns:
            weights: [B, num_scales] tensor or None if no forward pass yet
        """
        return self._last_weights

    def get_adaptive_params_count(self):
        """
        Returns the number of additional parameters introduced by
        the adaptive mixer (for reporting overhead).
        """
        return sum(p.numel() for p in self.adaptive_mixer.parameters())

    def get_baseline_params_count(self):
        """
        Returns the number of parameters in the baseline TimeMixer parts.
        """
        total = sum(p.numel() for p in self.parameters())
        adaptive = self.get_adaptive_params_count()
        return total - adaptive
