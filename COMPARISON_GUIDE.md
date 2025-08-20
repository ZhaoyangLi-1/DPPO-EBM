# PPO, PPO+EBM, SAC, SAC+EBM 比较指南

本指南详细说明如何运行和比较四种强化学习方法：PPO、PPO+EBM、SAC、SAC+EBM。

## 概述

### 四种方法对比

| 方法 | 描述 | 配置文件 | 特点 |
|------|------|----------|------|
| **PPO** | Proximal Policy Optimization (基线) | `ft_ppo_diffusion_mlp.yaml` | 在线策略算法，稳定但样本效率较低 |
| **PPO+EBM** | PPO + Energy-Based Model Reward Shaping | `ft_ppo_diffusion_ebm_mlp.yaml` | 使用EBM进行奖励塑形，提高样本效率 |
| **SAC** | Soft Actor-Critic (基线) | `ft_sac_diffusion_mlp.yaml` | 离线策略算法，高样本效率 |
| **SAC+EBM** | SAC + Energy-Based Model Reward Shaping | `ft_sac_diffusion_ebm_mlp.yaml` | SAC结合EBM奖励塑形，进一步提升性能 |

### 关键差异

1. **算法类型**：
   - PPO: 在线策略 (on-policy)
   - SAC: 离线策略 (off-policy)

2. **EBM集成**：
   - 基线方法：无EBM
   - EBM方法：使用能量模型进行奖励塑形

3. **样本效率**：
   - SAC > PPO (通常)
   - EBM方法 > 基线方法 (预期)

## 快速开始

### 1. 运行完整比较实验

```bash
# 使用快速比较脚本
python script/quick_comparison.py --env walker2d-v2 --seeds 42 123 456

# 或使用详细比较脚本
python script/run_comparison_experiments.py --env walker2d-v2 --seeds 42 123 456
```

### 2. 单独运行每种方法

```bash
# PPO 基线
python script/run.py --config-path cfg/gym/finetune/walker2d-v2/ft_ppo_diffusion_mlp.yaml seed=42

# PPO + EBM
python script/run.py --config-path cfg/gym/finetune/walker2d-v2/ft_ppo_diffusion_ebm_mlp.yaml seed=42

# SAC 基线
python script/run.py --config-path cfg/gym/finetune/walker2d-v2/ft_sac_diffusion_mlp.yaml seed=42

# SAC + EBM
python script/run.py --config-path cfg/gym/finetune/walker2d-v2/ft_sac_diffusion_ebm_mlp.yaml seed=42
```

### 3. 评估训练好的模型

```bash
# 评估所有训练好的模型
python script/evaluate_trained_models.py \
    --model-dirs outputs/ppo_walker2d-v2_42 outputs/ppo_ebm_walker2d-v2_42 \
    --model-dirs outputs/sac_walker2d-v2_42 outputs/sac_ebm_walker2d-v2_42 \
    --env walker2d-v2 --n-episodes 100
```

## 实验设计

### 推荐实验设置

1. **环境选择**：
   - `walker2d-v2`: 中等复杂度，适合初步比较
   - `hopper-v2`: 简单环境，快速验证
   - `halfcheetah-v2`: 复杂环境，全面测试

2. **随机种子**：
   - 至少使用3个不同的种子：`[42, 123, 456]`
   - 确保结果的统计显著性

3. **训练步数**：
   - 快速测试：500,000步
   - 完整实验：1,000,000步
   - 深度比较：2,000,000步

### 实验配置

```yaml
# 在配置文件中设置
max_steps: 1000000          # 训练步数
eval_freq: 10000           # 评估频率
save_freq: 50000           # 保存频率
log_freq: 1000             # 日志频率
```

## 评估指标

### 主要指标

1. **学习曲线**：
   - 平均奖励 vs 训练步数
   - 成功率 vs 训练步数
   - 收敛速度

2. **最终性能**：
   - 平均奖励（最后10%的训练步数）
   - 成功率
   - 标准差

3. **样本效率**：
   - 达到特定性能水平所需的步数
   - 学习曲线的斜率

4. **稳定性**：
   - 不同种子间的性能方差
   - 训练过程中的稳定性

### 统计显著性测试

使用以下方法进行统计比较：

1. **t检验**：比较两种方法的平均性能
2. **效应量**：Cohen's d，量化性能差异的大小
3. **置信区间**：95%置信区间

## 结果分析

### 预期结果

1. **算法比较**：
   - SAC > PPO (样本效率)
   - SAC+EBM > SAC (性能提升)
   - PPO+EBM > PPO (性能提升)

2. **EBM效果**：
   - 更快的收敛速度
   - 更高的最终性能
   - 更好的样本效率

### 结果解释

1. **如果EBM方法表现更好**：
   - EBM奖励塑形有效
   - 能量模型提供了有用的指导

2. **如果基线方法表现更好**：
   - EBM参数可能需要调整
   - 能量模型质量需要检查

3. **如果差异不显著**：
   - 可能需要更多训练步数
   - 环境可能不适合EBM方法

## 可视化结果

### 生成的图表

1. **学习曲线比较**：
   ```
   learning_curves.png
   ```

2. **最终性能比较**：
   ```
   final_performance.png
   ```

3. **成功率比较**：
   ```
   success_rates.png
   ```

4. **统计显著性测试**：
   ```
   statistical_analysis.png
   ```

### 使用WandB进行实时监控

```bash
# 设置WandB
export DPPO_WANDB_ENTITY=your-entity
export DPPO_WANDB_PROJECT=ppo-sac-ebm-comparison

# 运行实验
python script/quick_comparison.py --env walker2d-v2
```

## 故障排除

### 常见问题

1. **训练失败**：
   - 检查GPU内存
   - 验证配置文件路径
   - 检查依赖项安装

2. **性能差异不明显**：
   - 增加训练步数
   - 调整EBM参数
   - 检查能量模型质量

3. **内存不足**：
   - 减少batch size
   - 使用更小的网络
   - 减少并行环境数量

### 调试技巧

1. **启用详细日志**：
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **检查中间结果**：
   - 监控训练过程中的奖励
   - 检查EBM能量值
   - 验证奖励塑形效果

3. **逐步调试**：
   - 先运行基线方法
   - 再添加EBM功能
   - 逐步调整参数

## 高级分析

### 消融研究

1. **EBM参数消融**：
   - 不同的β值（温度参数）
   - 不同的λ值（奖励权重）
   - 不同的k_use模式

2. **算法组件消融**：
   - 仅使用能量缩放
   - 仅使用奖励塑形
   - 不同的潜在函数

### 超参数优化

使用Optuna或类似工具进行超参数优化：

```python
import optuna

def objective(trial):
    # 定义超参数搜索空间
    beta = trial.suggest_float('beta', 0.1, 10.0)
    lambda_val = trial.suggest_float('lambda', 0.1, 2.0)
    
    # 运行实验
    result = run_experiment(beta, lambda_val)
    return result['mean_reward']

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

## 报告生成

### 自动生成报告

运行比较脚本后，会自动生成：

1. **JSON报告**：`comparison_report.json`
2. **Markdown报告**：`comparison_report.md`
3. **可视化图表**：各种PNG文件

### 手动创建报告

```bash
# 生成自定义报告
python -c "
import json
with open('comparison_report.json', 'r') as f:
    data = json.load(f)
print('Best method:', max(data['results_summary'].items(), key=lambda x: x[1]['mean_reward'])[0])
"
```

## 最佳实践

### 实验设计

1. **控制变量**：
   - 保持相同的环境设置
   - 使用相同的随机种子
   - 保持相同的训练步数

2. **多次运行**：
   - 至少3个不同的随机种子
   - 考虑计算资源限制

3. **公平比较**：
   - 相同的计算预算
   - 相同的超参数调优程度

### 结果记录

1. **详细记录**：
   - 实验配置
   - 运行环境
   - 硬件规格

2. **版本控制**：
   - 代码版本
   - 依赖项版本
   - 配置文件版本

3. **可重现性**：
   - 保存随机种子
   - 记录所有参数
   - 保存模型检查点

## 结论

通过系统性的比较实验，您可以：

1. **量化EBM的效果**：了解EBM集成带来的性能提升
2. **选择最佳方法**：根据具体需求选择最适合的算法
3. **指导未来工作**：识别需要改进的方面

记住，强化学习实验的结果可能因环境、超参数和随机性而有所不同。因此，进行多次运行和统计分析非常重要。
