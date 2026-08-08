"""Conditional VAE inpainting model.

Two mechanisms make this behave like inpainting rather than forecasting-with-a-VAE-
label: (1) it consumes the full 73-slot padded tensor with padding_mask/target_mask
channels and horizon values zeroed, exactly like a masked image region, and (2) it
splits into a recognition network (sees the true, unmasked horizon -- training only)
and a context-conditioned prior network (sees only the masked input -- training and
inference). Only the prior + decoder run at inference time.

The decoder predicts a per-component (mean, logvar) pair, not just a point value -- see
decode()'s docstring. This is what lets losses.py train against a real NLL instead of
plain MSE, so the loss itself can reward calibrated sample diversity instead of relying
entirely on the KL term to inject it indirectly (see cvae_direction_collapse.md's
"generative pivot" discussion for why plain MSE was never going to be enough: the
optimizer can rationally let z collapse and get "diversity" for free by decoding noise
around a context-only prediction, since MSE alone never penalizes under-dispersion).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data_pipeline import MAX_LOG_RETURN, N_CHANNELS


class ConvEncoder(nn.Module):
    """1D-conv stack over the time axis -> pooled fixed-size representation. Doesn't
    need TOTAL_LEN (73) to divide evenly by anything, unlike the patch-based model."""

    def __init__(self, in_channels: int = N_CHANNELS, hidden: int = 32, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T=73, C=9) -> (B, out_dim)."""
        x = x.transpose(1, 2)  # (B, C, T) for Conv1d
        h = self.net(x)
        h = self.pool(h).squeeze(-1)
        return self.proj(h)


class CVAEInpainting(nn.Module):
    def __init__(
        self,
        hidden: int = 32,
        ctx_dim: int = 64,
        z_dim: int = 16,
        decoder_hidden: int = 128,
        ctx_dropout: float = 0.0,
        decoder_ctx_dim: int | None = None,
        in_channels: int = N_CHANNELS,
        price_logvar_range: tuple[float, float] = (-14.0, -6.0),
        vol_logvar_range: tuple[float, float] = (-6.0, 3.0),
    ):
        super().__init__()
        self.z_dim = z_dim
        # Separate bounds: price components live at raw log-return scale (std ~1e-3-1e-2)
        # while log_volume_norm is already z-scored to unit variance -- one shared range
        # would be miscalibrated for one of the two groups. Bounded via a sigmoid squash
        # in decode() (smooth everywhere), not torch.clamp (zero gradient outside range),
        # same rationale as MAX_LOG_RETURN bounding the mean below.
        #
        # price_logvar_range=(-14,-6) -> predicted std in [0.0009, 0.050]: floor comfortably
        # tighter than real body_ret's own std (~0.003, letting the model be confident when
        # appropriate), ceiling well under MAX_LOG_RETURN=0.15 so sampled noise (see
        # sample()) only rarely approaches the hard architectural cap at ~3 sigma, rather
        # than routinely blowing through it the way an upper bound near MAX_LOG_RETURN
        # itself would (a logvar of -2, an earlier draft of this range, implies std=0.37 --
        # more than DOUBLE the entire cap, clearly wrong).
        # vol_logvar_range=(-6,3) -> std in [0.050, 4.48]: log_volume_norm has unit variance
        # by construction, so this spans confidently-tight to a few multiples of baseline.
        self.price_logvar_range = price_logvar_range
        self.vol_logvar_range = vol_logvar_range
        self.context_encoder = ConvEncoder(in_channels=in_channels, hidden=hidden, out_dim=ctx_dim)
        self.recognition_encoder = ConvEncoder(in_channels=in_channels, hidden=hidden, out_dim=ctx_dim)

        self.prior_head = nn.Linear(ctx_dim, 2 * z_dim)
        self.recognition_head = nn.Linear(ctx_dim, 2 * z_dim)

        # Bottlenecks the decoder's OWN view of context to a much smaller dimension than
        # what the prior sees (prior_head above still takes the full ctx_dim) -- a
        # permanent, deterministic version of what ctx_dropout only does stochastically
        # during training. Default None (identity, full ctx_dim) so old checkpoints saved
        # without this param in their config (see evaluate.py, which reconstructs the
        # model from a checkpoint's saved config dict) still load correctly.
        self.decoder_ctx_dim = decoder_ctx_dim if decoder_ctx_dim is not None else ctx_dim
        self.decoder_ctx_proj = (
            nn.Linear(ctx_dim, decoder_ctx_dim) if decoder_ctx_dim is not None else nn.Identity()
        )

        # Dropped only inside decode() (see below), never before prior_head -- the prior
        # still sees the full, clean context; only the decoder's direct context bypass is
        # weakened. Default 0.0 (identity) so old checkpoints saved without this param in
        # their config (see evaluate.py, which reconstructs the model from a checkpoint's
        # saved config dict) still load correctly.
        self.ctx_dropout = nn.Dropout(ctx_dropout)

        # 30, not 15: one (mean, logvar) pair per component -- see decode()'s docstring.
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + self.decoder_ctx_dim, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, 30),
        )

    def encode_prior(self, masked_tensor: torch.Tensor):
        ctx_repr = self.context_encoder(masked_tensor)
        mu_p, logvar_p = self.prior_head(ctx_repr).chunk(2, dim=-1)
        return mu_p, logvar_p, ctx_repr

    def encode_recognition(self, full_tensor: torch.Tensor):
        rec_repr = self.recognition_encoder(full_tensor)
        mu_q, logvar_q = self.recognition_head(rec_repr).chunk(2, dim=-1)
        return mu_q, logvar_q

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _bounded_logvar(raw: torch.Tensor, logvar_range: tuple[float, float]) -> torch.Tensor:
        """Sigmoid squash into [lo, hi] -- smooth everywhere (unlike torch.clamp, which has
        zero gradient outside the range), same rationale as MAX_LOG_RETURN bounding the
        decoder's mean output below."""
        lo, hi = logvar_range
        return lo + (hi - lo) * torch.sigmoid(raw)

    def decode(self, z: torch.Tensor, ctx_repr: torch.Tensor):
        """decoder_ctx_proj + ctx_dropout are applied here, not before -- both weaken the
        decoder's direct, always-available context signal (a classic posterior-collapse
        driver: if the decoder can already reconstruct well from context alone, there's no
        loss pressure to ever route information through z) without touching the prior's
        own view of context (prior_head, in encode_prior, sees the un-bottlenecked
        ctx_repr). decoder_ctx_proj is a permanent, deterministic bottleneck (active at
        both train and inference time); ctx_dropout on top of it is a no-op whenever the
        model is in eval() mode, so inference (sample(), always preceded by .eval() --
        see evaluate.py) only feels the projection, not the dropout.

        Returns (price_mean, price_logvar, volume_mean, vol_logvar), each mean/logvar pair
        the same shape -- one (mean, logvar) per component, not just a point prediction.
        This is what lets losses.py's cvae_loss(reconstruction="nll") train against a real
        NLL: the loss can now reward getting the *spread* right, not just the mean, instead
        of leaving all anti-collapse pressure to the KL term. The mean's tanh/sigmoid
        squashing (bounded to +-MAX_LOG_RETURN / [0, MAX_LOG_RETURN]) is unchanged from
        before this was added; logvar gets its own, separate bounded squash per group (see
        _bounded_logvar) since price and volume live on very different natural scales."""
        ctx_repr = self.decoder_ctx_proj(ctx_repr)
        ctx_repr = self.ctx_dropout(ctx_repr)
        raw = self.decoder(torch.cat([z, ctx_repr], dim=-1))
        raw_mean, raw_logvar = raw.chunk(2, dim=-1)  # same idiom as prior_head/recognition_head

        price_raw = raw_mean[:, :12].reshape(-1, 3, 4)
        # Bounded to +-MAX_LOG_RETURN (open_ret/body_ret) or [0, MAX_LOG_RETURN] (wicks)
        # so an undertrained/unstable head can't blow up into unrealistic price moves.
        open_ret = MAX_LOG_RETURN * torch.tanh(price_raw[..., 0])
        body_ret = MAX_LOG_RETURN * torch.tanh(price_raw[..., 1])
        upper_wick = MAX_LOG_RETURN * torch.sigmoid(price_raw[..., 2])
        lower_wick = MAX_LOG_RETURN * torch.sigmoid(price_raw[..., 3])
        price = torch.stack([open_ret, body_ret, upper_wick, lower_wick], dim=-1)
        volume = raw_mean[:, 12:15]

        price_logvar = self._bounded_logvar(raw_logvar[:, :12].reshape(-1, 3, 4), self.price_logvar_range)
        vol_logvar = self._bounded_logvar(raw_logvar[:, 12:15], self.vol_logvar_range)
        return price, price_logvar, volume, vol_logvar

    def forward(self, masked_tensor: torch.Tensor, full_tensor: torch.Tensor):
        """Training forward. z ~ q(z | full window) via the recognition network;
        decoder conditions on the *masked* context encoding, never the recognition
        encoding directly -- only z carries information from the true horizon."""
        mu_p, logvar_p, ctx_repr = self.encode_prior(masked_tensor)
        mu_q, logvar_q = self.encode_recognition(full_tensor)
        z = self.reparameterize(mu_q, logvar_q)
        price, price_logvar, volume, vol_logvar = self.decode(z, ctx_repr)
        return price, price_logvar, volume, vol_logvar, mu_p, logvar_p, mu_q, logvar_q

    @torch.no_grad()
    def sample(self, masked_tensor: torch.Tensor, k: int = 5, sample_noise: bool = True):
        """Inference: only the prior network + decoder run. Returns price (K,B,3,4),
        volume (K,B,3).

        sample_noise: when True (default), each draw adds calibrated aleatoric noise on
        top of its z-conditioned mean (price_mean + eps*exp(0.5*price_logvar), same for
        volume) -- diversity across the k draws now comes from both the latent z AND the
        decoder's own learned per-example uncertainty, not z alone. Set False to reproduce
        the old z-only-diversity behavior (e.g. to ablate how much of the diversity comes
        from which source) -- decode()'s mean output is unaffected either way, only whether
        its logvar actually gets used to perturb the returned sample.

        Noise is added post-squash (in real log-return units, not pre-tanh/sigmoid), so a
        large enough draw could in principle push open_ret/body_ret outside
        +-MAX_LOG_RETURN or a wick negative -- re-clamped back into each component's valid
        range after adding noise, so the "architecturally impossible to violate" guarantee
        MAX_LOG_RETURN was originally built for (see decode()) still holds for every
        returned sample, not just the mean."""
        mu_p, logvar_p, ctx_repr = self.encode_prior(masked_tensor)
        prices, volumes = [], []
        for _ in range(k):
            z = self.reparameterize(mu_p, logvar_p)
            price, price_logvar, volume, vol_logvar = self.decode(z, ctx_repr)
            if sample_noise:
                price = price + torch.randn_like(price) * torch.exp(0.5 * price_logvar)
                volume = volume + torch.randn_like(volume) * torch.exp(0.5 * vol_logvar)
                price = torch.stack([
                    price[..., 0].clamp(-MAX_LOG_RETURN, MAX_LOG_RETURN),  # open_ret
                    price[..., 1].clamp(-MAX_LOG_RETURN, MAX_LOG_RETURN),  # body_ret
                    price[..., 2].clamp(0.0, MAX_LOG_RETURN),              # upper_wick
                    price[..., 3].clamp(0.0, MAX_LOG_RETURN),              # lower_wick
                ], dim=-1)
            prices.append(price)
            volumes.append(volume)
        return torch.stack(prices), torch.stack(volumes)
