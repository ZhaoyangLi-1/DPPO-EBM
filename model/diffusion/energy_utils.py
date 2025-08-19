"""
Energy-based model utilities for DPPO integration.

This module provides utilities for integrating energy-based models (EBMs) 
with diffusion policy optimization, including the EnergyScalerPerK class
for per-step energy normalization.
"""

import torch
import torch.nn as nn
import math
import logging
from typing import Optional, List, Dict, Any

log = logging.getLogger(__name__)


@torch.no_grad()
def _logmeanexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute log-mean-exp operation for numerical stability."""
    m = x.max(dim=dim, keepdim=True).values
    return (m + torch.log(torch.clamp(torch.mean(torch.exp(x - m), dim=dim, keepdim=True), min=1e-12))).squeeze(dim)


class EnergyScalerPerK:
    """
    Per-step energy normalization scaler for diffusion denoising steps.
    
    This class maintains running statistics (mean/median and standard deviation/MAD)
    for energy values at each denoising step k, allowing for proper normalization
    of energy-based rewards across different denoising steps.
    
    Args:
        K: Number of denoising steps
        momentum: Momentum for updating running statistics (default: 0.99)
        eps: Small constant for numerical stability (default: 1e-6)
        use_mad: Whether to use Median Absolute Deviation instead of standard deviation (default: True)
    """
    
    def __init__(self, K: int, momentum: float = 0.99, eps: float = 1e-6, use_mad: bool = True):
        self.K = K
        self.m = torch.zeros(K + 1)  # Running mean/median for each step k
        self.s = torch.ones(K + 1)   # Running std/MAD for each step k
        self.count = torch.zeros(K + 1)  # Count of updates for each step k
        self.momentum = momentum
        self.eps = eps
        self.use_mad = use_mad

    @torch.no_grad()
    def update(self, k_vec: torch.Tensor, E_vec: torch.Tensor):
        """
        Update running statistics with new energy values.
        
        Args:
            k_vec: Denoising step indices [B] or [B, 1]
            E_vec: Energy values corresponding to each step [B] or [B, 1]
        """
        k = k_vec.reshape(-1).cpu()
        e = E_vec.reshape(-1).detach().cpu()
        
        for kk in k.unique():
            mask = (k == kk)
            if mask.sum() == 0:
                continue
                
            vals = e[mask]
            # Use median for robust statistics if use_mad=True, otherwise mean
            mu = vals.median() if self.use_mad else vals.mean()
            # Use MAD for robust scale if use_mad=True, otherwise std
            sd = (vals - mu).abs().median() * 1.4826 if self.use_mad else vals.std(unbiased=False)
            
            kk = int(kk.item())
            # Update running statistics with momentum
            self.m[kk] = self.momentum * self.m[kk] + (1 - self.momentum) * mu
            self.s[kk] = self.momentum * self.s[kk] + (1 - self.momentum) * max(sd, 1e-3)
            self.count[kk] += mask.sum()

    @torch.no_grad()
    def normalize(self, k_vec: torch.Tensor, E_vec: torch.Tensor) -> torch.Tensor:
        """
        Normalize energy values using running statistics.
        
        Args:
            k_vec: Denoising step indices [B] or [B, 1]
            E_vec: Energy values to normalize [B] or [B, 1]
            
        Returns:
            Normalized energy values [B] or [B, 1]
        """
        m = self.m.to(E_vec.device)[k_vec]
        s = self.s.to(E_vec.device)[k_vec]
        return (E_vec - m) / (s + self.eps)

    def reset(self):
        """Reset all running statistics."""
        self.m.zero_()
        self.s.fill_(1.0)
        self.count.zero_()

    def get_stats(self, k: int) -> Dict[str, float]:
        """Get current statistics for a specific denoising step k."""
        if k < 0 or k > self.K:
            raise ValueError(f"k must be between 0 and {self.K}")
        return {
            "mean": self.m[k].item(),
            "std": self.s[k].item(),
            "count": self.count[k].item()
        }


class KFreePotential:
    """
    K-marginalized free energy potential for reward shaping.
    
    This class implements potential-based reward shaping using energy-based models
    and diffusion policies. It computes potentials φ(s) using k-marginalized free energy:
    
    Φ_k(s) = (1/β) * log E_{a~q_k}[exp(-β * Ẽ(s,a,k))]
    φ(s) = α * Σ_k w_k Φ_k(s)
    
    where q_k is the diffusion policy at denoising step k.
    
    Args:
        ebm_model: Energy-based model for computing E(s,a,k)
        pi0: Frozen diffusion policy for sampling actions
        K: Number of denoising steps
        stats: Normalization statistics for actions
        k_use: List of denoising steps to use for potential computation
        beta_k: Inverse temperature parameter (default: 1.0)
        alpha: Scaling factor for the potential (default: 1.0)
        M: Number of samples for Monte Carlo estimation (default: 1)
        eta: Weight decay parameter for step weighting (default: 0.3)
        use_mu_only: Whether to use only mean actions (faster but less accurate)
        device: Device to use for computations
    """
    
    def __init__(self, ebm_model: nn.Module, pi0: nn.Module, K: int, stats: Dict[str, Any],
                 k_use: List[int], beta_k: float = 1.0, alpha: float = 1.0, M: int = 1,
                 eta: float = 0.3, use_mu_only: bool = True, device: Optional[torch.device] = None):
        
        self.ebm = ebm_model.eval()
        self.pi0 = pi0.eval()
        
        # Freeze the prior policy parameters
        for p in self.pi0.parameters():
            p.requires_grad = False
            
        self.K = K
        self.k_use = sorted(set([int(min(max(1, k), K)) for k in k_use]))
        self.beta = beta_k
        self.alpha = alpha
        self.M = max(1, int(M))
        self.use_mu_only = use_mu_only
        self.device = device or next(ebm_model.parameters()).device
        
        # Weight early denoising steps higher (exploration-friendly)
        w = torch.tensor([math.exp(eta * (K - k)) for k in range(K + 1)], dtype=torch.float32)
        self.w = (w / w.sum()).to(self.device)
        
        # Energy scaler for normalization
        self.scaler = EnergyScalerPerK(K=K, momentum=0.99, use_mad=True)
        self.stats = stats

    @torch.no_grad()
    def _phi_k_from_actions(self, poses: torch.Tensor, views, t_idx: torch.Tensor, 
                           k: int, A_k: torch.Tensor) -> torch.Tensor:
        """Compute Φ_k(s) for a specific denoising step k."""
        B = poses.size(0)
        k_vec = torch.full((B,), int(k), dtype=torch.long, device=self.device)
        
        # Normalize actions using statistics
        act_min_t = torch.tensor(self.stats["act_min"], dtype=torch.float32, device=poses.device)
        act_max_t = torch.tensor(self.stats["act_max"], dtype=torch.float32, device=poses.device)
        if act_min_t.ndim == 2:
            act_min_t = act_min_t.min(dim=0).values
        if act_max_t.ndim == 2:
            act_max_t = act_max_t.max(dim=0).values
        act_min_t = act_min_t.view(1, 1, -1)
        act_max_t = act_max_t.view(1, 1, -1)
        
        # Normalize actions to [-1, 1] range
        a_norm = torch.clamp((A_k - act_min_t) / (act_max_t - act_min_t + 1e-6) * 2 - 1, -1.0, 1.0)
        
        # Compute energy using EBM
        E_out = self.ebm(k_idx=k_vec, views=None, poses=poses, actions=a_norm)
        E = E_out[0] if isinstance(E_out, (tuple, list)) else E_out
        
        # Average over additional dimensions if present
        if E.dim() >= 2 and E.size(0) == B:
            E = E.mean(dim=tuple(range(1, E.dim())))
            
        # Update scaler and normalize energy
        self.scaler.update(k_vec, E)
        En = self.scaler.normalize(k_vec, E)
        x = -self.beta * En
        
        return x if self.use_mu_only else (1.0 / self.beta) * x

    @torch.no_grad()
    def phi(self, obs_state: torch.Tensor, t_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the potential φ(s) for given observations.
        
        Args:
            obs_state: Observation states [B, obs_dim] or [B, 1, obs_dim]
            t_idx: Optional timestep indices [B]
            
        Returns:
            Potential values [B]
        """
        if obs_state.dim() == 3 and obs_state.size(1) == 1:
            poses = obs_state[:, 0, :]
        else:
            poses = obs_state
            
        B = poses.size(0)
        device = poses.device
        Phi_list = []
        
        if self.use_mu_only or self.M == 1:
            # Use deterministic sampling (zero noise)
            z0 = torch.zeros((B, self.pi0.eps_model.out_shape[0], self.pi0.eps_model.out_shape[1]), device=device)
            noise_zero = self._make_zero_noise_list(B, *self.pi0.eps_model.out_shape, self.K, device)
            
            _, chain = self.pi0.decode_chunk(
                poses, z0=z0,
                cond={"state": poses, "gamma_schedule": "const", "norm_stats": self.stats},
                noise_list=noise_zero
            )
            
            for k in self.k_use:
                A_k = chain[:, (self.K - k)]  # Index j=K-k corresponds to a^k
                Phi_k = self._phi_k_from_actions(
                    poses, None, 
                    t_idx if t_idx is not None else torch.zeros(B, dtype=torch.long, device=device), 
                    k, A_k
                )
                Phi_list.append(Phi_k if self.use_mu_only else (1.0 / self.beta) * Phi_k)
        else:
            # Use Monte Carlo sampling
            Xk_accum = {k: [] for k in self.k_use}
            for m in range(self.M):
                z0 = torch.randn((B, self.pi0.eps_model.out_shape[0], self.pi0.eps_model.out_shape[1]), device=device)
                noise_list = self._make_randn_noise_list(B, *self.pi0.eps_model.out_shape, self.K, device)
                
                _, chain = self.pi0.decode_chunk(
                    poses, z0=z0,
                    cond={"state": poses, "gamma_schedule": "const", "norm_stats": self.stats},
                    noise_list=noise_list
                )
                
                for k in self.k_use:
                    A_k = chain[:, (self.K - k)]
                    x_or_phi = self._phi_k_from_actions(
                        poses, None,
                        t_idx if t_idx is not None else torch.zeros(B, dtype=torch.long, device=device),
                        k, A_k
                    )
                    Xk_accum[k].append(x_or_phi)
                    
            for k in self.k_use:
                x_stack = torch.stack(Xk_accum[k], dim=1)  # [B, M]
                Phi_k = (1.0 / self.beta) * _logmeanexp(x_stack, dim=1)
                Phi_list.append(Phi_k)
                
        # Weighted combination of potentials
        w = self.w[self.k_use]
        w = w / w.sum()
        P = torch.stack(Phi_list, dim=1)  # [B, len(k_use)]
        phi = self.alpha * (P * w.view(1, -1)).sum(dim=1)
        
        return phi  # [B]

    @torch.no_grad()
    def shape_reward(self, s_t: torch.Tensor, s_tp1: torch.Tensor, gamma: float,
                    t_t: Optional[torch.Tensor] = None, t_tp1: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute shaped reward using potential-based reward shaping.
        
        Args:
            s_t: Current states [B, obs_dim]
            s_tp1: Next states [B, obs_dim]
            gamma: Discount factor
            t_t: Current timesteps [B] (optional)
            t_tp1: Next timesteps [B] (optional)
            
        Returns:
            Shaped reward values [B]
        """
        phi_t = self.phi(s_t, t_t)
        phi_tp1 = self.phi(s_tp1, t_tp1)
        return gamma * phi_tp1 - phi_t

    def _make_zero_noise_list(self, B: int, H: int, A: int, K: int, device):
        """Create list of zero noise tensors for deterministic sampling."""
        zero = torch.zeros((B, H, A), device=device)
        return [zero.clone() for _ in range(K)]

    def _make_randn_noise_list(self, B: int, H: int, A: int, K: int, device):
        """Create list of random noise tensors for stochastic sampling."""
        return [torch.randn((B, H, A), device=device) for _ in range(K)]


def build_k_use_indices(mode: str, K: int) -> List[int]:
    """
    Build list of denoising steps to use based on mode.
    
    Args:
        mode: Mode string ('all', 'last_half', 'tail:N')
        K: Total number of denoising steps
        
    Returns:
        List of denoising step indices to use
    """
    if mode == "all":
        return list(range(1, K + 1))
    if mode == "last_half":
        return list(range(max(1, K // 2), K + 1))
    if mode.startswith("tail:"):
        n = int(mode.split(":")[1])
        n = max(1, n)
        return list(range(max(1, K - n + 1), K + 1))
    # Default: last 6 steps
    return list(range(max(1, K - 6 + 1), K + 1))
