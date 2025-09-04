# Simple PPO Evaluation Guide

## 📋 Overview

This guide shows how to evaluate trained Simple PPO models (both environment-only and EBM-guided versions) using the new evaluation configurations.

## 🚀 Quick Start

### 1. Environment-Only PPO Evaluation
```bash
python script/run.py \
    --config-name=eval_simple_ppo_env \
    --config-dir=cfg/gym/eval/hopper-v2 \
    model_path="/path/to/your/trained/env/model/state_199.pt"
```

### 2. EBM-Guided PPO Evaluation
```bash
python script/run.py \
    --config-name=eval_simple_ppo_ebm \
    --config-dir=cfg/gym/eval/hopper-v2 \
    model_path="/path/to/your/trained/ebm/model/state_199.pt"
```

## 📁 File Structure

```
cfg/gym/eval/hopper-v2/
├── eval_simple_ppo_env.yaml    # Environment-only evaluation
└── eval_simple_ppo_ebm.yaml    # EBM-guided evaluation

agent/finetune/eval/
└── eval_simple_ppo_agent.py    # Evaluation agent implementation
```

## ⚙️ Configuration Details

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_path` | Path to trained model checkpoint | **REQUIRED** |
| `n_steps` | Total evaluation steps | 5000 |
| `env.n_envs` | Number of parallel environments | 40 |
| `use_ebm_reward` | Whether model was trained with EBM | env: False, ebm: True |

### Model Path Examples

**Environment-only trained model:**
```yaml
model_path: ${oc.env:DPPO_LOG_DIR}/gym-finetune/hopper-medium-v2_simple_mlp_ppo_env_ta4/2024-09-04_12-30-45_42/checkpoint/state_199.pt
```

**EBM-guided trained model:**
```yaml
model_path: ${oc.env:DPPO_LOG_DIR}/gym-finetune/hopper-medium-v2_simple_mlp_ppo_ebm_ta4/2024-09-04_15-20-30_42/checkpoint/state_199.pt
```

## 🔧 Customization

### Modify Evaluation Duration
```bash
python script/run.py \
    --config-name=eval_simple_ppo_env \
    --config-dir=cfg/gym/eval/hopper-v2 \
    n_steps=10000 \
    model_path="/path/to/model.pt"
```

### Change Environment Count
```bash
python script/run.py \
    --config-name=eval_simple_ppo_env \
    --config-dir=cfg/gym/eval/hopper-v2 \
    env.n_envs=80 \
    model_path="/path/to/model.pt"
```

### Enable Rendering (for visualization)
```bash
python script/run.py \
    --config-name=eval_simple_ppo_env \
    --config-dir=cfg/gym/eval/hopper-v2 \
    render_num=4 \
    model_path="/path/to/model.pt"
```

## 📊 Output Metrics

The evaluation will provide:

- **Mean Episode Reward**: Average reward per episode
- **Standard Deviation**: Reward variance
- **Min/Max Rewards**: Performance range
- **Episode Length**: Average episode duration
- **Success Rate**: If applicable (based on threshold)
- **Total Episodes**: Number of completed episodes

### Sample Output
```
==============================================================
EVALUATION RESULTS
==============================================================
Total episodes: 156
Total steps: 5000
Mean episode reward: 2847.34 ± 423.21
Episode reward range: [1234.56, 3456.78]
Median episode reward: 2901.45
Mean episode length: 845.2 ± 123.4
Success rate: 0.834
==============================================================
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Path Not Found**
   ```bash
   # Check if your model path exists
   ls -la /path/to/your/model/checkpoint/
   ```

2. **Configuration Mismatch**
   - Ensure the model config in eval yaml matches training config
   - Check `mlp_dims`, `use_ebm_reward`, etc.

3. **EBM Checkpoint Missing**
   ```bash
   # Verify EBM checkpoint exists
   ls -la /scr/zhaoyang/outputs/gym_one_action_t_idx/Hopper-v2-no-gp/ebm_best_val_hopper-v2-medium.pt
   ```

### Debug Mode
```bash
python script/run.py \
    --config-name=eval_simple_ppo_env \
    --config-dir=cfg/gym/eval/hopper-v2 \
    model_path="/path/to/model.pt" \
    hydra/job_logging=colorlog \
    hydra.verbose=true
```

## 🎯 Best Practices

1. **Always use deterministic evaluation** (already set in agent)
2. **Run multiple seeds** for statistical significance
3. **Use sufficient steps** (5000+ recommended)
4. **Match training and evaluation configs** exactly
5. **Save results** (automatically saved to `eval_results.pkl`)

## 📈 Comparing Results

To compare ENV vs EBM performance:

1. Run both evaluations with same settings
2. Compare mean episode rewards and success rates
3. Look at variance to assess stability
4. Consider episode length differences

The evaluation will objectively measure environment performance regardless of training rewards used!