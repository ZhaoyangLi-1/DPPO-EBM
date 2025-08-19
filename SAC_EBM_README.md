# SAC + EBM Integration

This document explains how to use the Soft Actor-Critic (SAC) with Energy-Based Model (EBM) integration for diffusion policy training.

## Overview

The SAC + EBM integration combines the sample efficiency of SAC with the guidance capabilities of energy-based models through potential-based reward shaping.

## Key Features

- **SAC Algorithm**: Off-policy actor-critic with entropy regularization
- **EBM Integration**: Energy-based model for potential-based reward shaping
- **Diffusion Policies**: Support for diffusion-based action generation
- **Automatic Entropy Tuning**: Adaptive temperature parameter
- **Experience Replay**: Efficient sample utilization

## Files Structure

```
DPPO-EBM/
├── model/diffusion/
│   ├── diffusion_sac.py              # Base SAC implementation
│   └── diffusion_sac_ebm.py          # SAC + EBM integration
├── agent/finetune/
│   └── train_sac_diffusion_ebm_agent.py  # SAC + EBM training agent
├── cfg/gym/finetune/hopper-v2/
│   └── ft_sac_diffusion_ebm_mlp.yaml # Configuration file
└── script/
    └── run_sac_ebm_example.py        # Example usage script
```

## Quick Start

### 1. Basic Training

```bash
# Train SAC + EBM on Hopper environment
python script/run_sac_ebm_example.py --env hopper-v2

# Train without EBM (baseline SAC)
python script/run_sac_ebm_example.py --env hopper-v2 --no-ebm
```

### 2. Custom EBM Checkpoint

```bash
# Use custom EBM checkpoint
python script/run_sac_ebm_example.py \
    --env hopper-v2 \
    --ebm-ckpt /path/to/your/ebm_checkpoint.pt
```

### 3. Adjust EBM Parameters

```bash
# Customize EBM parameters
python script/run_sac_ebm_example.py \
    --env hopper-v2 \
    --lambda 0.3 \
    --beta 1.5 \
    --alpha 0.05
```

## Configuration Parameters

### SAC Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.005 | Target network update rate |
| `alpha` | 0.2 | Entropy regularization coefficient |
| `automatic_entropy_tuning` | True | Enable automatic alpha tuning |
| `target_entropy` | -3 | Target entropy for auto-tuning |

### EBM Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_ebm_reward_shaping` | True | Enable EBM reward shaping |
| `pbrs_lambda` | 0.5 | EBM reward weight |
| `pbrs_beta` | 1.0 | Inverse temperature parameter |
| `pbrs_alpha` | 0.1 | Potential scaling factor |
| `pbrs_M` | 4 | Number of Monte Carlo samples |
| `pbrs_use_mu_only` | True | Use only mean actions |
| `pbrs_k_use_mode` | "tail:6" | Denoising steps to use |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `buffer_size` | 100000 | Replay buffer size |
| `batch_size` | 256 | Training batch size |
| `actor_lr` | 1e-4 | Actor learning rate |
| `critic_lr` | 1e-3 | Critic learning rate |

## Advanced Usage

### 1. Custom Configuration

Create your own configuration file:

```yaml
# my_sac_ebm_config.yaml
defaults:
  - ft_sac_diffusion_ebm_mlp
  - _self_

model:
  pbrs_lambda: 0.8
  pbrs_beta: 2.0
  pbrs_alpha: 0.2

train:
  buffer_size: 200000
  batch_size: 512
  actor_lr: 5e-5
```

### 2. Direct Command Line Usage

```bash
python script/run.py \
    --config-name=ft_sac_diffusion_ebm_mlp \
    --config-dir=cfg/gym/finetune/hopper-v2 \
    model.pbrs_lambda=0.7 \
    model.pbrs_beta=1.2 \
    train.batch_size=512
```

### 3. Environment-Specific Configurations

Different environments may require different settings:

```bash
# Walker2d (more complex locomotion)
python script/run_sac_ebm_example.py \
    --env walker2d-v2 \
    --lambda 0.3 \
    --beta 1.5

# HalfCheetah (faster dynamics)
python script/run_sac_ebm_example.py \
    --env halfcheetah-v2 \
    --lambda 0.6 \
    --beta 0.8
```

## Monitoring and Logging

The training automatically logs various metrics:

### SAC Metrics
- `loss/q1`, `loss/q2`, `loss/v`: Critic losses
- `loss/actor`: Actor loss
- `loss/alpha`: Alpha loss (if auto-tuning enabled)
- `alpha`: Current alpha value

### EBM Metrics
- `avg PBR reward - train`: Average EBM reward during training
- `avg PBR reward - eval`: Average EBM reward during evaluation

### Performance Metrics
- `avg episode reward - train`: Training episode rewards
- `avg episode reward - eval`: Evaluation episode rewards
- `success rate - eval`: Success rate during evaluation

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Reduce batch size and buffer size
   train.batch_size=128 train.buffer_size=50000
   ```

2. **Training Instability**
   ```bash
   # Reduce EBM weight and increase temperature
   model.pbrs_lambda=0.2 model.pbrs_beta=2.0
   ```

3. **Slow Convergence**
   ```bash
   # Increase learning rates and batch size
   train.actor_lr=2e-4 train.critic_lr=2e-3 train.batch_size=512
   ```

### Performance Tips

1. **EBM Weight Tuning**: Start with `lambda=0.3-0.5` and adjust based on performance
2. **Temperature Parameter**: Higher `beta` values make EBM more conservative
3. **Batch Size**: Larger batches generally improve stability
4. **Buffer Size**: Larger buffers improve sample efficiency

## Comparison with Other Algorithms

| Algorithm | Sample Efficiency | Training Stability | EBM Compatibility | Implementation Complexity |
|-----------|------------------|-------------------|-------------------|---------------------------|
| **SAC + EBM** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| PPO + EBM | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Cal-QL + EBM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| AWR + EBM | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

## References

- [SAC Paper](https://arxiv.org/abs/1801.01290)
- [EBM Integration](https://arxiv.org/abs/2002.05616)
- [Potential-Based Reward Shaping](https://arxiv.org/abs/1901.08652)
- [DPPO Framework](https://arxiv.org/abs/2409.00588)

## License

This implementation follows the same license as the main DPPO-EBM project.
