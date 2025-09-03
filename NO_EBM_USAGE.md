# DPPO-EBM 取消奖励塑形使用说明

本文档说明如何在 DPPO-EBM 项目中完全取消 EBM (Energy-Based Model) 奖励塑形功能，使用标准的 DPPO 进行训练。

## 修改概览

为了取消奖励塑形功能，我们进行了以下修改：

### 1. 配置文件修改

**原有 EBM 配置（已禁用）：**
- `use_energy_scaling: False` - 禁用能量缩放
- `use_ebm_reward_shaping: False` - 禁用潜在奖励塑形 (PBRS)
- `use_ebm_reward: False` - 禁用 EBM 标量奖励替换
- EBM 模型配置已注释掉
- EBM 检查点路径已注释掉

**修改的配置文件：**
- `/linting-slow-vol/DPPO-EBM/cfg/gym/finetune/hopper-v2/ft_ppo_diffusion_ebm_mlp.yaml`
- `/linting-slow-vol/DPPO-EBM/cfg/gym/finetune/walker2d-v2/ft_ppo_diffusion_ebm_mlp.yaml`
- `/linting-slow-vol/DPPO-EBM/cfg/gym/finetune/halfcheetah-v2/ft_ppo_diffusion_ebm_mlp.yaml`

### 2. 新增无 EBM 配置文件

为了更清晰地使用标准 DPPO，我们创建了新的配置文件：

**新配置文件：**
- `/linting-slow-vol/DPPO-EBM/cfg/gym/finetune/hopper-v2/ft_ppo_diffusion_mlp_no_ebm.yaml`
- `/linting-slow-vol/DPPO-EBM/cfg/gym/finetune/walker2d-v2/ft_ppo_diffusion_mlp_no_ebm.yaml`
- `/linting-slow-vol/DPPO-EBM/cfg/gym/finetune/halfcheetah-v2/ft_ppo_diffusion_mlp_no_ebm.yaml`

这些配置文件使用标准的 `PPODiffusion` 而不是 `PPODiffusionEBM`。

## 使用方法

### 方法一：使用修改后的 EBM 配置文件（EBM 功能已禁用）

```bash
# Hopper 环境
python script/run.py --config-name=ft_ppo_diffusion_ebm_mlp \
    --config-dir=cfg/gym/finetune/hopper-v2

# Walker2d 环境  
python script/run.py --config-name=ft_ppo_diffusion_ebm_mlp \
    --config-dir=cfg/gym/finetune/walker2d-v2

# HalfCheetah 环境
python script/run.py --config-name=ft_ppo_diffusion_ebm_mlp \
    --config-dir=cfg/gym/finetune/halfcheetah-v2
```

### 方法二：使用新的无 EBM 配置文件（推荐）

```bash
# Hopper 环境
python script/run.py --config-name=ft_ppo_diffusion_mlp_no_ebm \
    --config-dir=cfg/gym/finetune/hopper-v2

# Walker2d 环境  
python script/run.py --config-name=ft_ppo_diffusion_mlp_no_ebm \
    --config-dir=cfg/gym/finetune/walker2d-v2

# HalfCheetah 环境
python script/run.py --config-name=ft_ppo_diffusion_mlp_no_ebm \
    --config-dir=cfg/gym/finetune/halfcheetah-v2
```

## 配置验证

运行测试脚本来验证 EBM 功能已正确禁用：

```bash
cd /linting-slow-vol/DPPO-EBM
python test_no_ebm_config.py
```

预期输出：
```
Testing EBM-disabled configuration...
==================================================
✅ PASS: use_energy_scaling = False
✅ PASS: use_ebm_reward_shaping = False
✅ PASS: use_ebm_reward = False
✅ PASS: EBM configuration is disabled
✅ PASS: EBM checkpoint path is disabled

🎉 All tests passed! EBM functionality is properly disabled.
```

## 主要差异

### 禁用 EBM 后的系统行为：

1. **奖励系统**：使用原始环境奖励，不进行任何 EBM 基于的奖励塑形或替换
2. **训练智能体**：使用标准 `TrainPPODiffusionAgent` 或 `TrainPPODiffusionEBMAgent`（EBM 功能已禁用）
3. **模型**：使用标准 `PPODiffusion` 或 `PPODiffusionEBM`（EBM 功能已禁用）
4. **计算开销**：大幅减少，因为不需要计算能量函数和势函数

### 训练参数保持不变：

- PPO 超参数（学习率、批次大小、更新频率等）
- 扩散模型参数（去噪步数、噪声调度等）  
- 环境设置（并行环境数、最大步数等）

## 日志和监控

训练时，WandB 日志将只包含标准的 PPO 指标：

- `loss`: 总损失
- `pg loss`: 策略梯度损失
- `value loss`: 价值函数损失
- `avg episode reward - train/eval`: 平均回合奖励（原始环境奖励）
- `success rate - eval`: 成功率（如果适用）

不会记录以下 EBM 相关指标：
- `train/pbrs_reward`: PBRS 奖励
- `train/mean_pbrs_reward`: 平均 PBRS 奖励
- `energy_scaling/*`: 能量缩放统计信息

## 文件清单

### 修改的文件：
1. `cfg/gym/finetune/hopper-v2/ft_ppo_diffusion_ebm_mlp.yaml`
2. `cfg/gym/finetune/walker2d-v2/ft_ppo_diffusion_ebm_mlp.yaml` 
3. `cfg/gym/finetune/halfcheetah-v2/ft_ppo_diffusion_ebm_mlp.yaml`

### 新增的文件：
1. `cfg/gym/finetune/hopper-v2/ft_ppo_diffusion_mlp_no_ebm.yaml`
2. `cfg/gym/finetune/walker2d-v2/ft_ppo_diffusion_mlp_no_ebm.yaml`
3. `cfg/gym/finetune/halfcheetah-v2/ft_ppo_diffusion_mlp_no_ebm.yaml`
4. `test_no_ebm_config.py`
5. `NO_EBM_USAGE.md`

## 注意事项

1. **预训练模型**：仍然使用相同的预训练扩散策略检查点
2. **归一化统计**：仍然需要相同的观测和动作归一化文件
3. **环境依赖**：所有 MuJoCo/Gym 环境依赖保持不变
4. **性能基线**：这些配置提供了标准 DPPO 的性能基线，可用于与 EBM 增强版本进行比较

## 故障排除

如果遇到问题：

1. **检查配置**：确认所有 EBM 相关参数都设置为 `False`
2. **检查路径**：确认预训练模型和归一化文件路径正确
3. **检查依赖**：确认所有必需的包都已安装
4. **查看日志**：检查训练日志中是否有 EBM 相关的错误或警告

通过这些修改，DPPO-EBM 项目现在可以在完全禁用 EBM 功能的情况下运行，提供标准 DPPO 的基线性能。