# FDPP: Fine-tune Diffusion Policy with Human Preference

Implementation of FDPP (Fine-tune Diffusion Policy with Human Preference) based on the paper: "FDPP: Fine-tune Diffusion Policy with Human Preference" by Chen et al.

## Overview

FDPP is a method for fine-tuning pre-trained diffusion policies to align with human preferences while maintaining task performance. The key components are:

1. **Preference-based Reward Learning**: Learn reward functions from human preference labels using the Bradley-Terry model
2. **KL Regularization**: Prevent over-fitting to preferences while preserving original task capabilities
3. **PPO-based Fine-tuning**: Use reinforcement learning to optimize the policy with learned rewards

## Architecture

```
DPPO-EBM/
├── model/
│   └── rl/
│       └── preference_reward_model.py    # Preference reward models
├── agent/
│   └── finetune/
│       └── train_fdpp_diffusion_agent.py # Main FDPP agent
├── cfg/
│   ├── fdpp_pusht_diffusion.yaml        # Push-T task config
│   └── fdpp_stack_diffusion.yaml        # Stack task config
├── script/
│   └── run_fdpp.py                      # Python runner script
└── run_fdpp.sh                          # Shell script runner
```

## Key Features

### 1. Preference-based Reward Learning
- **Bradley-Terry Model**: Models preference probabilities between state/trajectory pairs
- **Flexible Architecture**: Supports both state-level and trajectory-level rewards
- **Online Learning**: Updates preference model during policy fine-tuning

### 2. KL Regularization
- **Reference Policy**: Maintains frozen copy of pre-trained policy
- **Upper Bound Approximation**: Computes KL divergence at each denoising step
- **Adaptive Weight**: Tunable α parameter balances preference alignment vs. task performance

### 3. Integration with DPPO
- **Extends TrainPPODiffusionAgent**: Builds on existing DPPO infrastructure
- **Compatible with Diffusion Models**: Works with various diffusion architectures
- **Parallel Environment Support**: Efficient data collection with vectorized environments

## Usage

### Basic Usage

#### Using Shell Script
```bash
# Fine-tune on Gym tasks
./run_fdpp_gym_robomimic.sh gym hopper-v2 0.01 42 0
./run_fdpp_gym_robomimic.sh gym walker2d-v2 0.01 42 0
./run_fdpp_gym_robomimic.sh gym halfcheetah-v2 0.01 42 0

# Fine-tune on Robomimic tasks  
./run_fdpp_gym_robomimic.sh robomimic lift 0.02 42 0
./run_fdpp_gym_robomimic.sh robomimic can 0.02 42 0
```

#### Using Python Script
```bash
# Gym tasks
python script/run_fdpp_gym_robomimic.py --task_type gym --task_name hopper-v2 --kl_weight 0.01

# Robomimic tasks
python script/run_fdpp_gym_robomimic.py --task_type robomimic --task_name lift --kl_weight 0.02

# With custom parameters
python script/run_fdpp_gym_robomimic.py \
    --task_type robomimic \
    --task_name lift \
    --kl_weight 0.02 \
    --preference_lr 3e-4 \
    --n_itr 200 \
    --n_envs 20

# Disable preference learning (only KL regularization)
python script/run_fdpp_gym_robomimic.py --task_type gym --task_name hopper-v2 --no_preference --kl_weight 0.1
```

### Configuration Parameters

#### FDPP-specific Parameters
- `use_preference_reward`: Enable preference-based reward learning (default: True)
- `kl_weight`: KL regularization weight α (default: 0.01)
- `preference_buffer_size`: Size of preference dataset (default: 10000)
- `preference_batch_size`: Batch size for preference learning (default: 256)
- `preference_lr`: Learning rate for preference model (default: 1e-4)
- `preference_update_freq`: How often to update preference model (default: 10)
- `n_preference_epochs`: Epochs per preference update (default: 3)
- `use_trajectory_reward`: Use trajectory-level vs state-level rewards (default: False)

#### Network Architecture
- `preference_hidden_dims`: Hidden layer dimensions for preference model
- `preference_activation`: Activation function (relu, tanh, leaky_relu)
- `preference_output_activation`: Output activation (tanh, sigmoid, none)

## Implementation Details

### Preference Collection

Currently, the implementation includes simulated preference labeling for demonstration. In practice, you should implement actual human feedback collection:

```python
def _query_human_preference(self, obs_0, obs_1):
    """
    Implement your human feedback interface here.
    Should return:
    - 0: obs_0 preferred
    - 1: obs_1 preferred
    - -1: equal preference
    """
    # Your implementation
    pass
```

### Custom Preference Criteria

Modify the `_simulate_preference` method for task-specific preferences:

```python
def _simulate_preference(self, obs_0, obs_1):
    # Example: Prefer states closer to goal
    dist_0 = compute_distance_to_goal(obs_0)
    dist_1 = compute_distance_to_goal(obs_1)
    
    if abs(dist_0 - dist_1) < threshold:
        return -1  # Equal
    return 0 if dist_0 < dist_1 else 1
```

### KL Weight Selection

Guidelines for choosing KL weight (α):
- **Small α (0.001-0.01)**: Strong preference alignment, may forget original task
- **Medium α (0.01-0.1)**: Balanced trade-off
- **Large α (0.1-1.0)**: Preserves original policy, weak preference alignment

## Experiments

### Gym Locomotion Tasks
Test preference learning on continuous control:
```bash
# Hopper locomotion with preference for specific gait patterns
python script/run_fdpp_gym_robomimic.py --task_type gym --task_name hopper-v2 --kl_weight 0.01 --n_itr 100

# Walker2D with preference for energy efficiency 
python script/run_fdpp_gym_robomimic.py --task_type gym --task_name walker2d-v2 --kl_weight 0.01 --n_itr 100

# HalfCheetah with preference for stable running
python script/run_fdpp_gym_robomimic.py --task_type gym --task_name halfcheetah-v2 --kl_weight 0.01 --n_itr 100
```

### Robomimic Manipulation Tasks
Test on manipulation tasks with different preferences:
```bash
# Lift task with preference for smooth movements
python script/run_fdpp_gym_robomimic.py --task_type robomimic --task_name lift --kl_weight 0.02 --n_itr 200

# Can task with preference for gentle manipulation
python script/run_fdpp_gym_robomimic.py --task_type robomimic --task_name can --kl_weight 0.02 --n_itr 200

# Square assembly with preference for precision
python script/run_fdpp_gym_robomimic.py --task_type robomimic --task_name square --kl_weight 0.02 --n_itr 200
```

## Monitoring

Training progress is logged to WandB with metrics:
- `reward/average`: Task reward
- `reward/preference_dataset_size`: Number of preference labels
- `loss/policy`: PPO policy loss
- `loss/kl_regularization`: KL divergence loss
- `metrics/kl_divergence`: Average KL divergence

## Troubleshooting

### Common Issues

1. **High KL Divergence**: Reduce learning rate or increase KL weight
2. **Poor Task Performance**: Decrease KL weight or collect more preferences
3. **Unstable Training**: Reduce batch size or use smaller learning rates
4. **Memory Issues**: Reduce `n_envs` or `preference_buffer_size`

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Citation

If you use FDPP in your research, please cite:
```bibtex
@article{chen2025fdpp,
  title={FDPP: Fine-tune Diffusion Policy with Human Preference},
  author={Chen, Yuxin and others},
  journal={arXiv preprint arXiv:2501.08259},
  year={2025}
}
```

## Future Extensions

- [ ] GUI for human preference collection
- [ ] Active learning for efficient preference queries
- [ ] Multi-task preference learning
- [ ] Adaptive KL weight scheduling
- [ ] Integration with vision-language models for automatic preference generation