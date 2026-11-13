"""
Adaptive Layers for Multiscale Forecast Mixing.

These layers implement the core contribution of our extension:
input-dependent, learnable scale weighting for TimeMixer's FMM stage.

Architecture:
  scale_1 ──► predictor_1 ──┐
  scale_2 ──► predictor_2 ──┤
  scale_3 ──► predictor_3 ──┼──► AdaptiveScaleMixer ──► final forecast
  scale_4 ──► predictor_4 ──┘
                                    ▲
                                    │
                          input-dependent weights (softmax)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleContextAggregator(nn.Module):
    """
    Aggregates multiscale encoder outputs into a single context vector
    that conditions the adaptive weight prediction.

    Takes a list of encoder outputs at different scales and produces
    a fixed-size context representation.
    """

    def __init__(self, d_model, num_scales):
        super(ScaleContextAggregator, self).__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        # Project each scale to a common space, then combine
        self.scale_projections = nn.ModuleList([
            nn.Linear(d_model, d_model)
            for _ in range(num_scales)
        ])
        self.combine = nn.Sequential(
            nn.Linear(d_model * num_scales, d_model),
            nn.GELU(),
        )

    def forward(self, enc_out_list):
        """
        Args:
            enc_out_list: list of [B, T_i, d_model] tensors at different scales

        Returns:
            context: [B, d_model] aggregated context vector
        """
        projected = []
        for i, enc_out in enumerate(enc_out_list):
            # Global average pooling over temporal dimension
            pooled = enc_out.mean(dim=1)  # [B, d_model]
            projected.append(self.scale_projections[i](pooled))

        # Concatenate and combine
        combined = torch.cat(projected, dim=-1)  # [B, d_model * num_scales]
        context = self.combine(combined)  # [B, d_model]
        return context


class HorizonEncoder(nn.Module):
    """
    Encodes the forecast horizon as a conditioning signal.
    This allows the model to learn that different horizons
    should weight scales differently.

    For example:
        short horizon → fine-scale features may matter more
        long horizon  → coarse trend features may matter more
    """

    def __init__(self, d_model, max_horizon=720):
        super(HorizonEncoder, self).__init__()
        self.d_model = d_model
        self.horizon_embedding = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, pred_len):
        """
        Args:
            pred_len: int or tensor, the forecast horizon length

        Returns:
            horizon_emb: [1, d_model] or [B, d_model] horizon embedding
        """
        if isinstance(pred_len, int):
            h = torch.tensor([[float(pred_len)]], device=next(self.parameters()).device)
        else:
            h = pred_len.float().unsqueeze(-1)
        return self.horizon_embedding(h)


class AdaptiveScaleMixer(nn.Module):
    """
    Core contribution: Adaptive Multiscale Forecast Mixing.

    Instead of uniformly summing predictions from different scales
    (original FMM: forecast = Σ forecast_i),
    we learn input-dependent weights:

        w_1, w_2, ..., w_k = softmax(weight_net(context))
        forecast = Σ w_i * forecast_i

    The weights depend on:
        1. Input history (via ScaleContextAggregator)
        2. Forecast horizon (via HorizonEncoder) — optional
        3. Scale representations

    This allows the model to adaptively emphasize different scales
    based on the input pattern and forecasting task.
    """

    def __init__(self, d_model, num_scales, use_horizon=True, max_horizon=720, temperature=1.0):
        super(AdaptiveScaleMixer, self).__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        self.use_horizon = use_horizon
        self.temperature = temperature

        # Context aggregator
        self.context_aggregator = ScaleContextAggregator(d_model, num_scales)

        # Optional horizon encoder
        if use_horizon:
            self.horizon_encoder = HorizonEncoder(d_model, max_horizon)
            input_dim = d_model * 2  # context + horizon
        else:
            self.horizon_encoder = None
            input_dim = d_model

        # Weight prediction network
        self.weight_net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_scales),
        )

        # Initialize weights to produce roughly uniform distribution initially
        # This ensures the model starts close to the baseline behavior
        nn.init.zeros_(self.weight_net[-1].weight)
        nn.init.zeros_(self.weight_net[-1].bias)

    def forward(self, enc_out_list, forecasts, pred_len=None):
        """
        Args:
            enc_out_list: list of [B, T_i, d_model] encoder outputs per scale
            forecasts: list of [B, pred_len, c_out] forecast tensors per scale
            pred_len: int, forecast horizon (used if use_horizon=True)

        Returns:
            output: [B, pred_len, c_out] adaptively weighted forecast
            weights: [B, num_scales] the scale weights (for visualization)
        """
        # Get input-dependent context
        context = self.context_aggregator(enc_out_list)  # [B, d_model]

        # Optionally add horizon conditioning
        if self.use_horizon and pred_len is not None:
            horizon_emb = self.horizon_encoder(pred_len)  # [1, d_model]
            horizon_emb = horizon_emb.expand(context.size(0), -1)  # [B, d_model]
            conditioning = torch.cat([context, horizon_emb], dim=-1)  # [B, 2*d_model]
        else:
            conditioning = context  # [B, d_model]

        # Predict scale weights (scaled by num_scales so uniform init corresponds to baseline sum)
        logits = self.weight_net(conditioning)  # [B, num_scales]
        weights = F.softmax(logits / self.temperature, dim=-1) * self.num_scales  # [B, num_scales]

        # Weighted combination of forecasts
        # forecasts[i]: [B, pred_len, c_out]
        # weights[:, i]: [B]
        stacked = torch.stack(forecasts, dim=-1)  # [B, pred_len, c_out, num_scales]
        weights_expanded = weights.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, num_scales]
        output = (stacked * weights_expanded).sum(dim=-1)  # [B, pred_len, c_out]

        return output, weights

    def get_weight_stats(self, enc_out_list, pred_len=None):
        """
        Utility method to get weight statistics for visualization/analysis.
        Returns the raw weights without requiring forecasts.
        """
        context = self.context_aggregator(enc_out_list)

        if self.use_horizon and pred_len is not None:
            horizon_emb = self.horizon_encoder(pred_len)
            horizon_emb = horizon_emb.expand(context.size(0), -1)
            conditioning = torch.cat([context, horizon_emb], dim=-1)
        else:
            conditioning = context

        logits = self.weight_net(conditioning)
        weights = F.softmax(logits / self.temperature, dim=-1) * self.num_scales

        return {
            'weights': weights,  # [B, num_scales]
            'logits': logits,    # [B, num_scales] (pre-softmax)
            'context': context,  # [B, d_model]
        }
