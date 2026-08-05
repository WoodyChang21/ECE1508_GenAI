"""A Mamba backbone adapted to multivariate time-series forecasting."""

from __future__ import annotations

import torch
from torch import nn
from transformers import MambaConfig, MambaModel


class MambaForecaster(nn.Module):
    """Predict the next target value from a window of continuous features.

    Hugging Face's bare Mamba model normally receives token embeddings. Here a
    learned linear layer embeds each multivariate market observation instead.
    The prediction head reads the final causal hidden state.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        state_size: int = 16,
        num_layers: int = 3,
        expand: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError("n_features must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.n_features = n_features
        self.input_projection = nn.Linear(n_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.backbone = MambaModel(
            MambaConfig(
                # The token embedding table is unused because forward receives
                # inputs_embeds, so keep it deliberately tiny.
                vocab_size=1,
                hidden_size=d_model,
                state_size=state_size,
                num_hidden_layers=num_layers,
                expand=expand,
                conv_kernel=conv_kernel,
                use_cache=False,
            )
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return one prediction per window.

        Args:
            values: Float tensor shaped ``(batch, sequence, features)``.
        """
        if values.ndim != 3:
            raise ValueError("values must have shape (batch, sequence, features)")
        if values.shape[-1] != self.n_features:
            raise ValueError(
                f"expected {self.n_features} features, got {values.shape[-1]}"
            )

        embeddings = self.input_norm(self.input_projection(values))
        hidden = self.backbone(
            inputs_embeds=embeddings,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        return self.head(self.dropout(hidden[:, -1])).squeeze(-1)
