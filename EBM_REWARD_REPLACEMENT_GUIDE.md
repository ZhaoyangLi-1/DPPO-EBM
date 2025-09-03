# DPPO-EBM 完全替代环境奖励使用指南

本指南说明如何使用简单的 MLP EBM 模型完全替代环境奖励进行 DPPO 训练。

## 系统概览

EBM 奖励替代系统包含以下核心组件：

1. **Simple MLP EBM**: 基于多层感知机的能量模型
2. **Energy Scaling**: 每个去噪步骤的能量归一化
3. **Reward Replacement**: 完全用 EBM 计算的效用替代环境奖励
4. **Training Pipeline**: 端到端的训练和评估流程

## 系统架构

### EBM 奖励替代原理

```
环境奖励 R_env(s,a) → EBM 效用 U_EBM(s,a,k)

其中:
U_EBM(s,a,k) = -(E_θ(s,a,k) - β_BC(s,k)) / τ_k

- E_θ(s,a,k): EBM 预测的能量值  
- β_BC(s,k): 基线策略的能量期望
- τ_k: 第 k 步的能量缩放因子
```

### 支持的奖励模式

- **k0 模式**: 仅使用最终去噪步骤 (k=0) 的效用
- **dense 模式**: 对所有去噪步骤的效用进行加权求和

## 文件结构

```
DPPO-EBM/
├── model/
│   └── ebm/
│       └── simple_mlp_ebm.py          # 简单 MLP EBM 实现
├── cfg/gym/finetune/
│   ├── hopper-v2/
│   │   └── ft_ppo_diffusion_mlp_ebm_reward_only.yaml
│   ├── walker2d-v2/
│   │   └── ft_ppo_diffusion_mlp_ebm_reward_only.yaml
│   └── halfcheetah-v2/
│       └── ft_ppo_diffusion_mlp_ebm_reward_only.yaml
├── train_simple_mlp_ebm.py            # EBM 训练脚本
├── test_ebm_reward_replacement.py     # 测试脚本
└── checkpoints/                       # EBM 模型检查点
```

## 使用步骤

### 第一步：训练 EBM 模型

```bash
# 为每个环境训练 EBM
python train_simple_mlp_ebm.py --env_name hopper --epochs 50
python train_simple_mlp_ebm.py --env_name walker2d --epochs 50  
python train_simple_mlp_ebm.py --env_name halfcheetah --epochs 50

# 使用自定义数据训练
python train_simple_mlp_ebm.py \
    --env_name hopper \
    --data_path /path/to/trajectory_data.npz \
    --epochs 100 \
    --batch_size 512 \
    --lr 1e-3
```

**训练数据格式**:
- `states`: [N, obs_dim] 状态观测
- `actions`: [N, horizon_steps, action_dim] 动作序列  
- `rewards`: [N] 奖励值

### 第二步：验证 EBM 模型

```bash
# 运行测试验证所有组件
python test_ebm_reward_replacement.py
```

预期输出：
```
🎉 All tests passed! EBM reward replacement is ready to use.
```

### 第三步：运行 DPPO 训练

```bash
# 使用 EBM 完全替代环境奖励
python script/run.py --config-name=ft_ppo_diffusion_mlp_ebm_reward_only \
    --config-dir=cfg/gym/finetune/hopper-v2

python script/run.py --config-name=ft_ppo_diffusion_mlp_ebm_reward_only \
    --config-dir=cfg/gym/finetune/walker2d-v2

python script/run.py --config-name=ft_ppo_diffusion_mlp_ebm_reward_only \
    --config-dir=cfg/gym/finetune/halfcheetah-v2
```

## 配置详解

### EBM 模型配置

```yaml
model:
  ebm:
    _target_: model.ebm.simple_mlp_ebm.EBMWrapper
    obs_dim: 11                    # 观测维度
    action_dim: 3                  # 动作维度  
    horizon_steps: 4               # 动作序列长度
    hidden_dims: [512, 256, 128]   # 隐藏层维度
    activation: "gelu"             # 激活函数
    use_layer_norm: true           # 层归一化
    dropout: 0.1                   # Dropout 率
    use_time_embedding: true       # 时间步嵌入
    max_timesteps: 1000           # 最大时间步
    max_denoising_steps: 20       # 最大去噪步骤
```

### 奖励替代配置

```yaml
model:
  # 禁用 PBRS，只使用奖励替代
  use_ebm_reward_shaping: false
  
  # 启用能量缩放
  use_energy_scaling: true
  energy_scaling_momentum: 0.99
  energy_scaling_use_mad: true
  
  # EBM 奖励替代设置
  use_ebm_reward: true
  ebm_reward_mode: k0              # k0 或 dense
  ebm_reward_clip_u_max: 5.0       # 效用裁剪阈值
  ebm_reward_lambda: 2.0           # 效用缩放因子
  ebm_reward_baseline_M: 8         # 基线样本数
  ebm_reward_baseline_use_mu_only: false  # 基线采样策略
```

## EBM 模型特性

### Simple MLP EBM 架构

- **输入**: 状态 + 动作序列 + 时间嵌入 + 去噪步骤嵌入
- **网络**: 多层感知机 + 层归一化 + Dropout
- **输出**: 标量能量值
- **温度参数**: 可学习的能量缩放因子

### 训练特点

- **损失函数**: MSE 回归或排序损失
- **优化器**: Adam + 学习率衰减
- **正则化**: 权重衰减 + Dropout + 梯度裁剪
- **数据**: 支持真实轨迹数据和合成数据

## 监控与调试

### 训练指标

- **Loss**: EBM 预测损失
- **Correlation**: 能量与奖励的相关性
- **Temperature**: 能量缩放温度
- **Energy Range**: 能量值的范围

### WandB 日志

```yaml
wandb:
  project: gym-${env_name}-finetune-ebm
  run: ${name}_EBM_Reward_Only
  tags: [DPPO+EBM-Reward-Only]
```

训练过程中会记录：
- 训练和验证损失
- 能量与奖励的相关性
- 温度参数变化
- 原始环境奖励 vs EBM 奖励

### 调试技巧

1. **检查能量相关性**: EBM 能量应与负奖励呈正相关
2. **监控温度参数**: 温度过高或过低可能导致不稳定
3. **验证基线计算**: 确保基线策略能量计算正确
4. **观察奖励分布**: EBM 奖励应在合理范围内

## 性能调优

### EBM 模型调优

- **网络容量**: 增加隐藏层维度提高表达能力
- **激活函数**: GELU 通常比 ReLU 表现更好
- **Dropout**: 0.1-0.2 的 dropout 有助于泛化
- **批次大小**: 较大批次有助于稳定训练

### 奖励替代调优

- **效用裁剪**: 防止极端效用值影响训练稳定性
- **缩放因子**: 调节 EBM 奖励的整体幅度
- **基线样本数**: 更多样本提高基线估计精度
- **能量缩放**: MAD 通常比标准差更稳健

## 实验对比

### 基线对比

可以与以下配置进行对比：

1. **标准 DPPO**: `ft_ppo_diffusion_mlp_no_ebm.yaml`
2. **PBRS 增强**: `ft_ppo_diffusion_ebm_mlp.yaml` (启用 PBRS)
3. **EBM 奖励替代**: `ft_ppo_diffusion_mlp_ebm_reward_only.yaml`

### 评估指标

- **Episode Reward**: 真实环境奖励 (用于公平比较)
- **EBM Reward**: EBM 计算的效用值
- **Success Rate**: 任务成功率 (如果适用)
- **Training Stability**: 训练过程的稳定性

## 故障排除

### 常见问题

1. **EBM 训练不收敛**
   - 检查数据质量和标签正确性
   - 降低学习率或增加正则化
   - 使用更简单的网络架构

2. **奖励替代不稳定**
   - 调整效用裁剪阈值
   - 增加基线样本数
   - 使用 MAD 而非标准差进行缩放

3. **训练性能下降**
   - 检查 EBM 与环境奖励的相关性
   - 验证归一化统计是否正确
   - 确认检查点文件路径正确

### 调试命令

```bash
# 检查 EBM 模型
python -c "
import torch
from model.ebm.simple_mlp_ebm import EBMWrapper
model = EBMWrapper(obs_dim=11, action_dim=3, horizon_steps=4)
print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')"

# 验证配置
python test_ebm_reward_replacement.py

# 检查检查点
python -c "
import torch
ckpt = torch.load('checkpoints/simple_mlp_ebm_hopper.pt')
print(f'Val loss: {ckpt[\"val_loss\"]:.4f}')
print(f'Epoch: {ckpt[\"epoch\"]}')"
```

## 高级用法

### 自定义 EBM 架构

可以通过修改 `SimpleMLPEBM` 类来实现自定义架构：

```python
class CustomEBM(SimpleMLPEBM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 添加自定义层
        self.custom_layer = nn.Linear(...)
    
    def forward(self, k_idx, t_idx, views, poses, actions):
        # 自定义前向传播
        pass
```

### 多环境 EBM

可以训练一个 EBM 处理多个环境：

```python
# 联合训练多环境数据
python train_simple_mlp_ebm.py \
    --env_name multi \
    --data_path combined_data.npz \
    --obs_dim 17 \
    --action_dim 6
```

### 在线 EBM 更新

可以在 DPPO 训练过程中持续更新 EBM：

```yaml
model:
  ebm_online_update: true
  ebm_update_freq: 100
  ebm_update_lr: 1e-4
```

这个系统提供了一个完整的 EBM 奖励替代解决方案，能够完全替代环境奖励进行强化学习训练。