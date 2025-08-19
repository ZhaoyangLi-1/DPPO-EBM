# Walker2d 和 HalfCheetah 环境配置文件

本目录包含了walker2d和halfcheetah环境的PPO+EBM、SAC、SAC+EBM配置文件。

## 📁 配置文件结构

### Walker2d 环境
```
cfg/gym/finetune/walker2d-v2/
├── ft_ppo_diffusion_ebm_mlp.yaml    # PPO + EBM
├── ft_sac_diffusion_mlp.yaml        # 纯SAC
└── ft_sac_diffusion_ebm_mlp.yaml    # SAC + EBM
```

### HalfCheetah 环境
```
cfg/gym/finetune/halfcheetah-v2/
├── ft_ppo_diffusion_ebm_mlp.yaml    # PPO + EBM
├── ft_sac_diffusion_mlp.yaml        # 纯SAC
└── ft_sac_diffusion_ebm_mlp.yaml    # SAC + EBM
```

## 🚀 运行实验

### 方法1: 使用自动化脚本

```bash
# 运行所有实验
python script/run_walker_halfcheetah_experiments.py

# 只运行特定环境
python script/run_walker_halfcheetah_experiments.py --env walker2d
python script/run_walker_halfcheetah_experiments.py --env halfcheetah

# 只运行特定算法
python script/run_walker_halfcheetah_experiments.py --algorithm sac_ebm
python script/run_walker_halfcheetah_experiments.py --algorithm ppo_ebm

# 指定GPU
python script/run_walker_halfcheetah_experiments.py --gpu 1

# 只查看将要运行的命令（不执行）
python script/run_walker_halfcheetah_experiments.py --dry-run
```

### 方法2: 手动运行单个实验

```bash
# Walker2d - PPO + EBM
python main.py --config-path cfg/gym/finetune/walker2d-v2 --config-name ft_ppo_diffusion_ebm_mlp.yaml

# Walker2d - SAC
python main.py --config-path cfg/gym/finetune/walker2d-v2 --config-name ft_sac_diffusion_mlp.yaml

# Walker2d - SAC + EBM
python main.py --config-path cfg/gym/finetune/walker2d-v2 --config-name ft_sac_diffusion_ebm_mlp.yaml

# HalfCheetah - PPO + EBM
python main.py --config-path cfg/gym/finetune/halfcheetah-v2 --config-name ft_ppo_diffusion_ebm_mlp.yaml

# HalfCheetah - SAC
python main.py --config-path cfg/gym/finetune/halfcheetah-v2 --config-name ft_sac_diffusion_mlp.yaml

# HalfCheetah - SAC + EBM
python main.py --config-path cfg/gym/finetune/halfcheetah-v2 --config-name ft_sac_diffusion_ebm_mlp.yaml
```

## ⚙️ 配置参数说明

### 通用参数
- `env_name`: 环境名称 (walker2d-medium-v2 / halfcheetah-medium-v2)
- `obs_dim`: 观察维度 (17)
- `action_dim`: 动作维度 (6)
- `denoising_steps`: 预训练去噪步数 (20)
- `ft_denoising_steps`: 微调去噪步数 (10)
- `horizon_steps`: 动作序列长度 (4)

### PPO 特定参数
- `actor_lr`: Actor学习率 (1e-4)
- `critic_lr`: Critic学习率 (1e-3)
- `batch_size`: 批次大小 (50000)
- `update_epochs`: 更新轮数 (5)
- `gae_lambda`: GAE参数 (0.95)

### SAC 特定参数
- `actor_lr`: Actor学习率 (3e-4)
- `critic_lr`: Critic学习率 (3e-4)
- `value_lr`: Value网络学习率 (3e-4)
- `tau`: 目标网络更新率 (0.005)
- `alpha`: 熵正则化系数 (0.2)
- `auto_alpha`: 自动调整alpha (True)
- `buffer_size`: 经验回放缓冲区大小 (1000000)
- `batch_size`: 批次大小 (256)

### EBM 特定参数
- `use_ebm_reward_shaping`: 启用EBM奖励塑造 (True)
- `pbrs_lambda`: EBM奖励权重 (0.5)
- `pbrs_beta`: 逆温度参数 (1.0)
- `pbrs_alpha`: 势能缩放因子 (0.1)
- `pbrs_M`: 蒙特卡洛采样数 (4)
- `pbrs_use_mu_only`: 只使用确定性采样 (True)
- `pbrs_k_use_mode`: 去噪步骤选择模式 ("tail:6")

## 📊 实验对比

| 算法 | 类型 | 样本效率 | 训练稳定性 | EBM集成 |
|------|------|----------|------------|---------|
| PPO | On-policy | 中等 | 高 | ❌ |
| PPO + EBM | On-policy | 高 | 高 | ✅ |
| SAC | Off-policy | 高 | 高 | ❌ |
| SAC + EBM | Off-policy | 很高 | 很高 | ✅ |

## 🔧 超参数调优建议

### PPO + EBM
- 调整 `pbrs_lambda` (0.1-1.0) 控制EBM影响强度
- 调整 `pbrs_beta` (0.5-2.0) 控制温度
- 调整 `pbrs_alpha` (0.05-0.5) 控制势能缩放

### SAC + EBM
- 调整 `alpha` (0.1-0.3) 控制探索
- 调整 `tau` (0.001-0.01) 控制目标网络更新
- 调整 `pbrs_lambda` (0.3-0.7) 平衡环境奖励和EBM奖励

## 📈 监控指标

### 训练指标
- `actor_loss`: Actor网络损失
- `critic_loss`: Critic网络损失
- `value_loss`: Value网络损失 (SAC)
- `entropy`: 策略熵 (SAC)
- `alpha`: 熵系数 (SAC)

### EBM指标
- `ebm_reward`: EBM塑造奖励
- `potential_value`: 势能函数值
- `energy_values`: 能量值

### 性能指标
- `episode_reward`: 回合奖励
- `episode_length`: 回合长度
- `success_rate`: 成功率

## 🐛 常见问题

### 1. 内存不足
- 减少 `batch_size`
- 减少 `buffer_size` (SAC)
- 减少 `n_envs`

### 2. 训练不稳定
- 调整学习率
- 增加 `update_epochs` (PPO)
- 调整 `tau` (SAC)

### 3. EBM效果不明显
- 增加 `pbrs_lambda`
- 调整 `pbrs_beta`
- 检查EBM模型配置

## 📝 日志和结果

实验结果将保存在：
```
${DPPO_LOG_DIR}/gym-finetune/${experiment_name}/${timestamp}/
```

包含：
- 模型检查点
- 训练日志
- 评估结果
- WandB记录
