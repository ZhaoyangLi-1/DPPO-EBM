# SAC vs SAC + EBM 消融实验

本文档说明如何运行SAC和SAC + EBM的消融实验，以比较EBM集成对性能的影响。

## 实验概述

消融实验对比两个版本：
1. **SAC (Pure)** - 纯Soft Actor-Critic，不包含EBM
2. **SAC + EBM** - SAC与Energy-Based Model集成

## 文件结构

```
DPPO-EBM/
├── model/diffusion/
│   ├── diffusion_sac.py                    # 纯SAC实现
│   └── diffusion_sac_ebm.py               # SAC + EBM实现
├── agent/finetune/
│   ├── train_sac_diffusion_agent.py       # 纯SAC训练代理
│   └── train_sac_diffusion_ebm_agent.py   # SAC + EBM训练代理
├── cfg/gym/finetune/hopper-v2/
│   ├── ft_sac_diffusion_mlp.yaml          # 纯SAC配置
│   └── ft_sac_diffusion_ebm_mlp.yaml      # SAC + EBM配置
└── script/
    ├── run_ablation_study.py              # 消融实验脚本
    ├── test_sac.py                        # 纯SAC测试
    └── test_sac_ebm.py                    # SAC + EBM测试
```

## 快速开始

### 1. 运行完整消融实验

```bash
# 运行SAC vs SAC + EBM对比实验
python script/run_ablation_study.py --env hopper-v2

# 使用自定义EBM参数
python script/run_ablation_study.py \
    --env hopper-v2 \
    --lambda 0.3 \
    --beta 1.5 \
    --alpha 0.05
```

### 2. 单独运行实验

```bash
# 只运行纯SAC
python script/run_ablation_study.py --env hopper-v2 --skip-sac-ebm

# 只运行SAC + EBM
python script/run_ablation_study.py --env hopper-v2 --skip-sac
```

### 3. 测试代码正确性

```bash
# 测试纯SAC实现
python script/test_sac.py

# 测试SAC + EBM实现
python script/test_sac_ebm.py
```

## 实验参数

### 环境参数

| 环境 | 观察维度 | 动作维度 | 推荐设置 |
|------|----------|----------|----------|
| hopper-v2 | 11 | 3 | 默认 |
| walker2d-v2 | 17 | 6 | 需要调整参数 |
| halfcheetah-v2 | 17 | 6 | 需要调整参数 |

### EBM参数

| 参数 | 默认值 | 说明 | 推荐范围 |
|------|--------|------|----------|
| `lambda` | 0.5 | EBM奖励权重 | 0.1 - 1.0 |
| `beta` | 1.0 | 逆温度参数 | 0.5 - 2.0 |
| `alpha` | 0.1 | 势能缩放因子 | 0.05 - 0.3 |

### SAC参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gamma` | 0.99 | 折扣因子 |
| `tau` | 0.005 | 目标网络更新率 |
| `alpha` | 0.2 | 熵正则化系数 |
| `buffer_size` | 100000 | 回放缓冲区大小 |
| `batch_size` | 256 | 训练批次大小 |

## 实验设计

### 1. 基线实验

```bash
# 标准消融实验
python script/run_ablation_study.py --env hopper-v2
```

### 2. 参数敏感性实验

```bash
# 测试不同EBM权重
for lambda in 0.1 0.3 0.5 0.7 1.0; do
    python script/run_ablation_study.py \
        --env hopper-v2 \
        --lambda $lambda \
        --skip-sac
done

# 测试不同温度参数
for beta in 0.5 1.0 1.5 2.0; do
    python script/run_ablation_study.py \
        --env hopper-v2 \
        --beta $beta \
        --skip-sac
done
```

### 3. 环境对比实验

```bash
# 在不同环境中测试
for env in hopper-v2 walker2d-v2 halfcheetah-v2; do
    python script/run_ablation_study.py --env $env
done
```

## 结果分析

### 关键指标

1. **训练性能**
   - 平均回合奖励
   - 成功率
   - 收敛速度

2. **样本效率**
   - 达到目标性能所需的步数
   - 学习曲线斜率

3. **稳定性**
   - 训练方差
   - 超参数敏感性

### 预期结果

| 指标 | SAC (Pure) | SAC + EBM | 预期改进 |
|------|------------|-----------|----------|
| 最终性能 | 基准 | 更高 | +10-30% |
| 样本效率 | 基准 | 更好 | +20-50% |
| 训练稳定性 | 基准 | 更稳定 | 方差减少 |
| 收敛速度 | 基准 | 更快 | 提前收敛 |

## 监控和日志

### WandB指标

#### SAC指标
- `loss/q1`, `loss/q2`, `loss/v`: 评论家损失
- `loss/actor`: 演员损失
- `loss/alpha`: Alpha损失
- `alpha`: 当前Alpha值

#### EBM指标 (仅SAC + EBM)
- `avg PBR reward - train`: 训练期间平均EBM奖励
- `avg PBR reward - eval`: 评估期间平均EBM奖励

#### 性能指标
- `avg episode reward - train`: 训练回合奖励
- `avg episode reward - eval`: 评估回合奖励
- `success rate - eval`: 评估成功率

### 结果可视化

```python
import wandb
import matplotlib.pyplot as plt

# 比较学习曲线
api = wandb.Api()
runs = api.runs("your-project")

# 提取数据
sac_rewards = []
sac_ebm_rewards = []

for run in runs:
    if "sac_diffusion_mlp" in run.name:
        sac_rewards.append(run.history()["avg episode reward - eval"])
    elif "sac_diffusion_ebm_mlp" in run.name:
        sac_ebm_rewards.append(run.history()["avg episode reward - eval"])

# 绘制对比图
plt.figure(figsize=(10, 6))
plt.plot(sac_rewards.mean(axis=0), label="SAC (Pure)")
plt.plot(sac_ebm_rewards.mean(axis=0), label="SAC + EBM")
plt.xlabel("Training Steps")
plt.ylabel("Average Episode Reward")
plt.legend()
plt.title("SAC vs SAC + EBM Performance Comparison")
plt.show()
```

## 故障排除

### 常见问题

1. **内存不足**
   ```bash
   # 减少批次大小和缓冲区大小
   train.batch_size=128 train.buffer_size=50000
   ```

2. **训练不稳定**
   ```bash
   # 降低学习率
   train.actor_lr=5e-5 train.critic_lr=5e-4
   ```

3. **EBM效果不明显**
   ```bash
   # 增加EBM权重
   model.pbrs_lambda=0.8 model.pbrs_beta=0.5
   ```

### 调试技巧

1. **启用详细日志**
   ```bash
   # 在配置文件中设置
   log_freq: 1
   ```

2. **检查梯度**
   ```python
   # 在训练代码中添加
   for name, param in model.named_parameters():
       if param.grad is not None:
           print(f"{name}: grad_norm = {param.grad.norm()}")
   ```

3. **监控EBM奖励分布**
   ```python
   # 检查EBM奖励的统计信息
   print(f"EBM reward mean: {pbrs_reward.mean()}")
   print(f"EBM reward std: {pbrs_reward.std()}")
   ```

## 实验最佳实践

### 1. 实验设计
- 使用相同的随机种子确保公平比较
- 运行多次实验取平均值
- 记录所有超参数设置

### 2. 计算资源
- 使用相同的硬件配置
- 监控GPU内存使用
- 记录训练时间

### 3. 结果验证
- 检查训练曲线是否合理
- 验证最终性能是否稳定
- 对比多个随机种子

### 4. 报告格式

```markdown
## 实验结果

### 环境: hopper-v2
- **SAC (Pure)**: 平均奖励 = X ± Y
- **SAC + EBM**: 平均奖励 = X ± Y
- **改进**: +Z%

### 关键发现
1. EBM提高了样本效率
2. 训练更加稳定
3. 收敛速度更快

### 超参数设置
- lambda = 0.5
- beta = 1.0
- alpha = 0.1
```

## 扩展实验

### 1. 不同EBM架构
- 测试不同的EBM网络结构
- 比较不同的能量函数设计

### 2. 多任务学习
- 在多个环境中测试泛化能力
- 分析EBM的迁移学习效果

### 3. 消融EBM组件
- 测试能量缩放的效果
- 分析不同去噪步骤的影响

这个消融实验框架可以帮助您系统地评估EBM对SAC性能的影响，并为后续研究提供可靠的基线。
