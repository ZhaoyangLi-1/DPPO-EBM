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
        # Ensure both tensors have the same shape
        k = k_vec.reshape(-1).cpu()
        e = E_vec.reshape(-1).detach().cpu()
        
        # Check if shapes match
        if k.size(0) != e.size(0):
            log.warning(f"Shape mismatch in EnergyScalerPerK.update: k_vec.shape={k_vec.shape}, E_vec.shape={E_vec.shape}")
            # Use the minimum length to avoid index errors
            min_len = min(k.size(0), e.size(0))
            k = k[:min_len]
            e = e[:min_len]
        
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
        if k < 0 or k >= self.K:
            raise ValueError(f"k must be between 0 and {self.K-1}")
        return {
            "mean": self.m[k].item(),
            "std": self.s[k].item(),
            "count": self.count[k].item()
        }


class KFreePotential:
    """
    Path free energy potential φ(s) for PBRS, matching the theory:
      φ(s) = (α/β) * log[ 1/(M|K|) * Σ_{m=1..M} Σ_{k∈K} exp(-β E_θ(s, a_k^(m), k)) ]

    - Actions a_k^(m) are sampled from a frozen diffusion prior π₀ at denoising step k
    - K includes denoising indices in 0..K-1 (0 is the final fully denoised step, K-1 is fully noisy)
    - Aggregation is a pure log-mean-exp over all sampled terms (no additional weights)
    """

    def __init__(
        self,
        ebm_model: nn.Module,
        pi0: nn.Module,
        K: int,
        stats: Dict[str, Any],
        k_use: List[int],
        beta_k: float = 1.0,
        alpha: float = 1.0,
        M: int = 1,
        use_mu_only: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.ebm = ebm_model.eval()
        self.pi0 = pi0.eval()

        # Freeze the prior policy parameters
        for p in self.pi0.parameters():
            p.requires_grad = False

        self.K = int(K)
        # Allow k=0..K-1 (0 is fully denoised, K-1 is fully noisy)
        self.k_use = sorted(set(int(min(max(0, k), self.K - 1)) for k in k_use))
        
        # Log k_use range validation
        log.info(f"KFreePotential k_use validation: K={self.K}, k_use={self.k_use}, valid range: 0 to {self.K-1}")
        if any(k < 0 or k >= self.K for k in self.k_use):
            log.warning(f"Invalid k values in k_use: {[k for k in self.k_use if k < 0 or k >= self.K]}")
        
        self.beta = float(beta_k)
        self.alpha = float(alpha)
        self.M = max(1, int(M))
        self.use_mu_only = bool(use_mu_only)
        self.device = device or next(ebm_model.parameters()).device

        # Normalization stats for actions
        self.stats = stats

    @torch.no_grad()
    def _neg_beta_energy_from_actions(
        self,
        poses: torch.Tensor,
        k: int,
        A_k: torch.Tensor,
        t: int = 0,
    ) -> torch.Tensor:
        """Return x = -β E_θ(s, a_k, k) with action normalization to [-1, 1]."""
        # Log k range validation (only once)
        if not hasattr(self, '_logged_k_range'):
            log.info(f"EBM k range validation: k={k}, K={self.K}, valid range: 0 to {self.K-1}")
            if k < 0 or k >= self.K:
                log.warning(f"Invalid k value: {k}, should be in range [0, {self.K-1}]")
            self._logged_k_range = True
        
        B = poses.size(0)
        k_vec = torch.full((B,), int(k), dtype=torch.long, device=poses.device)
        t_vec = torch.full((B,), int(t), dtype=torch.long, device=poses.device)

        # Ensure actions have shape [B, H, A]
        if A_k.dim() == 2:  # [B, A]
            A_k = A_k.unsqueeze(1)
        elif A_k.dim() != 3:
            log.warning(f"Unexpected A_k shape: {A_k.shape}")
            return torch.zeros(B, device=poses.device)

        # Normalize actions to [-1, 1]
        try:
            act_min_t = torch.tensor(self.stats["act_min"], dtype=torch.float32, device=poses.device)
            act_max_t = torch.tensor(self.stats["act_max"], dtype=torch.float32, device=poses.device)
            if act_min_t.ndim == 2:
                act_min_t = act_min_t.min(dim=0).values
            if act_max_t.ndim == 2:
                act_max_t = act_max_t.max(dim=0).values
            act_min_t = act_min_t.view(1, 1, -1)
            act_max_t = act_max_t.view(1, 1, -1)
            a_norm = torch.clamp((A_k - act_min_t) / (act_max_t - act_min_t + 1e-6) * 2 - 1, -1.0, 1.0)
        except Exception as e:
            log.warning(f"Action normalization failed: {e}")
            a_norm = A_k

        # Compute energy - adapt to new EBM interface
        try:
            # Check if EBM has forward_batch method for batch processing
            if hasattr(self.ebm, 'forward_batch'):
                # For forward_batch, we need [B, 4, T, A] format
                # If we have single action [B, 1, A], we need to pad to 4 actions
                H = a_norm.size(1)
                if H < 4:
                    # Pad with zeros or repeat the action to get 4 actions
                    a_norm_padded = a_norm.repeat(1, 4, 1)[:, :4, :]  # [B, 4, A]
                else:
                    a_norm_padded = a_norm[:, :4, :]  # Take first 4 if more than 4
                
                # Add temporal dimension for EBM: [B, 4, T=1, A]
                a_norm_batch = a_norm_padded.unsqueeze(2)  # [B, 4, 1, A]
                
                # Call forward_batch
                E_out = self.ebm.forward_batch(k_idx=k_vec, t_idx=t_vec, views=None, poses=poses, actions=a_norm_batch)
                E = E_out[0] if isinstance(E_out, (tuple, list)) else E_out
                
                # E should be [B, 4], we take mean across 4 actions
                if E.dim() == 2 and E.size(1) == 4:
                    E = E.mean(dim=1)  # [B]
                elif E.dim() >= 2 and E.size(0) == B:
                    E = E.mean(dim=tuple(range(1, E.dim())))
            else:
                # Fallback to regular forward for single action
                # a_norm should be [B, T=1, A] for regular forward
                E_out = self.ebm(k_idx=k_vec, t_idx=t_vec, views=None, poses=poses, actions=a_norm)
                E = E_out[0] if isinstance(E_out, (tuple, list)) else E_out
                if E.dim() >= 2 and E.size(0) == B:
                    E = E.mean(dim=tuple(range(1, E.dim()))))
        except Exception as e:
            log.warning(f"EBM forward failed: {e}")
            return torch.zeros(B, device=poses.device)

        return -self.beta * E

    @torch.no_grad()
    def phi(self, obs_state: torch.Tensor, t: int = 0) -> torch.Tensor:
        """Compute φ(s) via log-mean-exp over sampled path energies exactly as in the theory."""
        # Accept [B, obs] or [B, 1, obs] or [B, T_cond, obs] -> use last cond step
        if obs_state.dim() == 3:
            poses = obs_state[:, -1, :]
        else:
            poses = obs_state

        B = poses.size(0)
        device = poses.device
        x_terms: List[torch.Tensor] = []  # each [B]

        # Sample chains from frozen prior
        if self.use_mu_only or self.M == 1:
            try:
                samples = self.pi0.forward(cond={"state": poses}, deterministic=True, return_chain=True)
                chain = samples.chains  # [B, K+1, H, A], index -1 is final action
                T = chain.size(1)
                for k in self.k_use:
                    if 0 <= k < self.K and T > 0:
                        idx = max(0, min(T - 1, T - 1 - k))
                        A_k = chain[:, idx]
                        x = self._neg_beta_energy_from_actions(poses, k, A_k, t)  # [B]
                        x_terms.append(x)
            except Exception as e:
                log.warning(f"Deterministic chain sampling failed: {e}")
                return torch.zeros(B, device=device)
        else:
            try:
                for m in range(self.M):
                    samples = self.pi0.forward(cond={"state": poses}, deterministic=False, return_chain=True)
                    chain = samples.chains  # [B, K+1, H, A]
                    T = chain.size(1)
                    for k in self.k_use:
                        if 0 <= k < self.K and T > 0:
                            idx = max(0, min(T - 1, T - 1 - k))
                            A_k = chain[:, idx]
                            x = self._neg_beta_energy_from_actions(poses, k, A_k, t)
                            x_terms.append(x)
            except Exception as e:
                log.warning(f"Stochastic chain sampling failed: {e}")
                return torch.zeros(B, device=device)

        if len(x_terms) == 0:
            return torch.zeros(B, device=device)

        # Stack to [B, M*|K|] and apply log-mean-exp
        X = torch.stack(x_terms, dim=1)
        log_mean_exp = _logmeanexp(X, dim=1)  # [B]
        return (self.alpha / self.beta) * log_mean_exp

    @torch.no_grad()
    def shape_reward(self, s_t: torch.Tensor, s_tp1: torch.Tensor, gamma: float, t: int = 0) -> torch.Tensor:
        """r_shape = γ φ(s') - φ(s)."""
        phi_t = self.phi(s_t, t)
        phi_tp1 = self.phi(s_tp1, t+1)
        return gamma * phi_tp1 - phi_t

    def _make_zero_noise_list(self, B: int, H: int, A: int, K: int, device):
        zero = torch.zeros((B, H, A), device=device)
        return [zero.clone() for _ in range(K)]

    def _make_randn_noise_list(self, B: int, H: int, A: int, K: int, device):
        return [torch.randn((B, H, A), device=device) for _ in range(K)]


def build_k_use_indices(mode: str, K: int) -> List[int]:
    """
    Build list of denoising steps to use based on mode, with 0-based indexing (0..K-1).
    
    k=0: completely denoised action (no noise)
    k=K-1: fully noisy action (maximum noise)
    
    - "all": use all steps [0, 1, ..., K-1]
    - "last_half": use the last half of denoising (low-noise end): [0, ..., floor((K-1)/2)]
    - "tail:N": use the last N denoising steps near k=0: [0, 1, ..., min(N-1, K)]
    """
    K = int(K)
    result = None
    
    if mode == "all":
        result = list(range(0, K))  # 0 to K-1
    elif mode == "last_half":
        result = list(range(0, K // 2))  # 0 to (K//2)-1
    elif mode.startswith("tail:"):
        try:
            n = int(mode.split(":")[1])
        except Exception:
            n = 6
        n = max(1, n)
        result = list(range(0, min(n, K)))  # 0 to min(n-1, K-1)
    else:
        # Default: tail:6
        result = list(range(0, min(6, K)))  # 0 to min(5, K-1)
    
    # Log k range validation (only once per mode)
    if not hasattr(build_k_use_indices, '_logged_modes'):
        build_k_use_indices._logged_modes = set()
    
    if mode not in build_k_use_indices._logged_modes:
        log.info(f"build_k_use_indices: mode='{mode}', K={K}, result={result}, valid range: 0 to {K-1}")
        build_k_use_indices._logged_modes.add(mode)
    
    return result
