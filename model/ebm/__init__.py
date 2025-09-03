"""
Energy-Based Model (EBM) package for DPPO integration.
"""

import torch.nn as nn
from .ebm import EBM as OriginalEBM


class EBMWrapper(nn.Module):
    """Wrapper for Original EBM to provide a simple Hydra-compatible interface."""

    def __init__(self, obs_dim, action_dim, hidden_dim=256, num_layers=3, **kwargs):
        super().__init__()

        class Config:
            def __init__(self):
                self.ebm = type("obj", (object,), {
                    "embed_dim": hidden_dim,
                    "state_dim": obs_dim,
                    "action_dim": action_dim,
                    "nhead": kwargs.get("nhead", 8),
                    "depth": num_layers,
                    "dropout": kwargs.get("dropout", 0.1),
                    "use_cls_token": kwargs.get("use_cls_token", False),
                    "num_views": kwargs.get("num_views", None),
                })()

        self.config = Config()
        self.model = OriginalEBM(self.config)

    def forward(self, k_idx, t_idx,views=None, poses=None, actions=None, **kwargs):
        return self.model(k_idx=k_idx, t_idx=t_idx, views=views, poses=poses, actions=actions)

    def __getattr__(self, name):
        if name in ["model", "config", "__dict__"]:
            return super().__getattr__(name)
        return getattr(self.model, name)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)


# -----------------------------------------------------------------------------
# Export API
# -----------------------------------------------------------------------------
EBM = EBMWrapper
__all__ = ["EBM", "EBMWrapper", "OriginalEBM"]
