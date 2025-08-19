# DPPO + Energy-Based Model Integration

This document explains how to integrate Energy-Based Models (EBMs) with the DPPO (Diffusion Policy Policy Optimization) framework, including the `EnergyScalerPerK` class for per-step energy normalization and potential-based reward shaping.

## Overview

The EBM integration adds two main capabilities to DPPO:

1. **EnergyScalerPerK**: Per-step energy normalization for diffusion denoising steps
2. **Potential-Based Reward Shaping (PBRS)**: Using EBMs to compute potential functions for reward shaping

## Key Components

### 1. EnergyScalerPerK (`model/diffusion/energy_utils.py`)

The `EnergyScalerPerK` class maintains running statistics for energy values at each denoising step k, allowing for proper normalization across different denoising steps.

**Key Features:**
- Per-step running statistics (mean/median and standard deviation/MAD)
- Momentum-based updates for stability
- Support for both mean/std and median/MAD statistics
- Automatic device handling

**Usage:**
```python
from model.diffusion.energy_utils import EnergyScalerPerK

# Initialize scaler
scaler = EnergyScalerPerK(K=20, momentum=0.99, use_mad=True)

# Update with new energy values
k_vec = torch.tensor([1, 2, 3, 4, 5])  # Denoising step indices
energy_vec = torch.randn(5)  # Energy values
scaler.update(k_vec, energy_vec)

# Normalize energy values
normalized_energy = scaler.normalize(k_vec, energy_vec)
```

### 2. KFreePotential (`model/diffusion/energy_utils.py`)

The `KFreePotential` class implements potential-based reward shaping using k-marginalized free energy:

```
Φ_k(s) = (1/β) * log E_{a~q_k}[exp(-β * Ẽ(s,a,k))]
φ(s) = α * Σ_k w_k Φ_k(s)
```

**Key Features:**
- K-marginalized free energy computation
- Support for both deterministic and Monte Carlo sampling
- Configurable temperature and weighting parameters
- Integration with frozen diffusion policies

### 3. PPODiffusionEBM (`model/diffusion/diffusion_ppo_ebm.py`)

Enhanced version of `PPODiffusion` that integrates EBM functionality:

**New Parameters:**
- `use_energy_scaling`: Enable EnergyScalerPerK
- `use_pbrs`: Enable potential-based reward shaping
- `pbrs_lambda`: Weight for PBRS reward
- `pbrs_beta`: Inverse temperature parameter
- `pbrs_alpha`: Scaling factor for potential
- `pbrs_M`: Number of Monte Carlo samples
- `pbrs_eta`: Weight decay parameter for step weighting
- `pbrs_use_mu_only`: Use only mean actions (faster)
- `pbrs_k_use_mode`: Which denoising steps to use for PBRS

### 4. TrainPPODiffusionEBMAgent (`agent/finetune/train_ppo_diffusion_ebm_agent.py`)

Enhanced training agent that integrates EBM functionality into the training loop.

## Installation and Setup

### 1. File Structure

The integration adds the following files to the DPPO codebase:

```
dppo/
├── model/
│   ├── diffusion/
│   │   ├── energy_utils.py          # EnergyScalerPerK and KFreePotential
│   │   └── diffusion_ppo_ebm.py     # Enhanced PPODiffusion with EBM
│   └── ebm/
│       └── example_ebm.py           # Example EBM models
├── agent/
│   └── finetune/
│       └── train_ppo_diffusion_ebm_agent.py  # Enhanced training agent
└── cfg/
    └── gym/
        └── finetune/
            └── hopper-v2/
                └── ft_ppo_diffusion_ebm_mlp.yaml  # Example configuration
```

### 2. Configuration

To use the EBM integration, modify your configuration file to include EBM-specific parameters:

```yaml
model:
  _target_: model.diffusion.diffusion_ppo_ebm.PPODiffusionEBM
  
  # Energy scaling parameters
  use_energy_scaling: True
  energy_scaling_momentum: 0.99
  energy_scaling_eps: 1e-6
  energy_scaling_use_mad: True
  
  # Potential-based reward shaping parameters
  use_pbrs: True
  pbrs_lambda: 1.0
  pbrs_beta: 1.0
  pbrs_alpha: 1.0
  pbrs_M: 1
  pbrs_eta: 0.3
  pbrs_use_mu_only: True
  pbrs_k_use_mode: "tail:6"
  
  # EBM model configuration
  ebm:
    _target_: model.ebm.example_ebm.ExampleEBM
    obs_dim: ${obs_dim}
    action_dim: ${action_dim}
    hidden_dim: 256
    num_layers: 3
  
  # EBM checkpoint path (optional)
  ebm_ckpt_path: ${oc.env:DPPO_LOG_DIR}/ebm_checkpoints/example_ebm.pt
```

### 3. Training Agent

Update your training script to use the enhanced agent:

```yaml
_target_: agent.finetune.train_ppo_diffusion_ebm_agent.TrainPPODiffusionEBMAgent
```

## Usage Examples

### 1. Basic EBM Integration

```bash
# Run training with EBM integration
python script/run.py --config-name=ft_ppo_diffusion_ebm_mlp \
    --config-dir=cfg/gym/finetune/hopper-v2
```

### 2. Energy Scaling Only

To use only energy scaling without PBRS:

```yaml
model:
  use_energy_scaling: True
  use_pbrs: False
  # ... other parameters
```

### 3. PBRS Only

To use only potential-based reward shaping without energy scaling:

```yaml
model:
  use_energy_scaling: False
  use_pbrs: True
  # ... PBRS parameters
```

### 4. Custom EBM Model

Replace the example EBM with your own model:

```python
# In your EBM model file
class MyEBM(nn.Module):
    def __init__(self, obs_dim, action_dim, **kwargs):
        super().__init__()
        # Your EBM implementation
        
    def forward(self, poses, actions, k_idx=None, views=None):
        # Your forward pass
        return energy_values
```

Then update the configuration:

```yaml
model:
  ebm:
    _target_: path.to.your.ebm.MyEBM
    obs_dim: ${obs_dim}
    action_dim: ${action_dim}
    # ... your EBM parameters
```

## Key Parameters

### Energy Scaling Parameters

- `use_energy_scaling`: Enable/disable energy scaling
- `energy_scaling_momentum`: Momentum for running statistics (default: 0.99)
- `energy_scaling_eps`: Small constant for numerical stability (default: 1e-6)
- `energy_scaling_use_mad`: Use Median Absolute Deviation instead of std (default: True)

### PBRS Parameters

- `use_pbrs`: Enable/disable potential-based reward shaping
- `pbrs_lambda`: Weight for PBRS reward (default: 1.0)
- `pbrs_beta`: Inverse temperature parameter (default: 1.0)
- `pbrs_alpha`: Scaling factor for potential (default: 1.0)
- `pbrs_M`: Number of Monte Carlo samples (default: 1)
- `pbrs_eta`: Weight decay parameter for step weighting (default: 0.3)
- `pbrs_use_mu_only`: Use only mean actions (default: True)
- `pbrs_k_use_mode`: Which denoising steps to use ("all", "last_half", "tail:N")

## Monitoring and Logging

The enhanced training agent provides additional logging for EBM components:

### Energy Scaling Statistics

```python
# Log energy scaling statistics for key denoising steps
wandb.log({
    "energy_scaling/step_1/mean": stats["mean"],
    "energy_scaling/step_1/std": stats["std"],
    "energy_scaling/step_1/count": stats["count"],
})
```

### PBRS Rewards

```python
# Log PBRS reward statistics
wandb.log({
    "train/pbrs_reward": pbrs_reward,
    "train/mean_pbrs_reward": pbrs_reward_trajs.mean(),
})
```

## Advanced Usage

### 1. Custom Energy Functions

You can implement custom energy functions by extending the EBM models:

```python
class CustomEBM(nn.Module):
    def forward(self, poses, actions, k_idx=None, views=None):
        # Custom energy computation
        energy = your_custom_energy_function(poses, actions, k_idx)
        return energy
```

### 2. Multi-Step Energy Computation

For more complex scenarios, you can compute energy across multiple denoising steps:

```python
# In your training loop
energy_values = []
for k in range(1, K + 1):
    energy_k = ebm_model(obs, actions_k, k_idx=torch.full((B,), k))
    energy_values.append(energy_k)
energy_values = torch.stack(energy_values, dim=1)  # [B, K]
```

### 3. Adaptive Energy Scaling

You can implement adaptive energy scaling by modifying the `EnergyScalerPerK` class:

```python
class AdaptiveEnergyScaler(EnergyScalerPerK):
    def update(self, k_vec, energy_vec):
        # Custom update logic
        super().update(k_vec, energy_vec)
        # Additional adaptive logic
```

## Troubleshooting

### Common Issues

1. **EBM Model Not Found**: Ensure your EBM model is properly imported and configured
2. **Memory Issues**: Reduce batch size or use `pbrs_use_mu_only=True` for faster computation
3. **Numerical Instability**: Adjust `energy_scaling_eps` or use `energy_scaling_use_mad=True`

### Debugging

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger('model.diffusion.energy_utils').setLevel(logging.DEBUG)
```

## Performance Considerations

1. **Energy Scaling**: Minimal overhead, mainly for normalization
2. **PBRS**: Can be computationally expensive, especially with Monte Carlo sampling
3. **Memory Usage**: Increases with batch size and number of denoising steps
4. **Training Speed**: Use `pbrs_use_mu_only=True` for faster training

## Future Extensions

The EBM integration is designed to be extensible. Potential future extensions include:

1. **Adaptive Energy Scaling**: Dynamic adjustment of scaling parameters
2. **Multi-Modal EBMs**: Support for different types of energy functions
3. **Hierarchical Energy Models**: Multi-level energy computation
4. **Online EBM Training**: Joint training of EBM and policy

## References

- [DPPO Paper](https://arxiv.org/abs/2409.00588)
- [Energy-Based Models](https://arxiv.org/abs/2002.05616)
- [Potential-Based Reward Shaping](https://arxiv.org/abs/1901.08652)
