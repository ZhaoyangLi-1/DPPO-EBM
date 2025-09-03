# PPO 奖励下降问题诊断和修复指南

## 问题症状
- 训练初期奖励正常，但随着训练进行奖励持续下降
- 策略变得过于确定性，失去探索能力
- 价值函数估计不准确

## 常见原因

### 1. 学习率过高
**症状**: 快速收敛但很快崩溃
**解决方案**: 
- Actor LR: `3e-4` → `1e-4`
- Critic LR: `1e-3` → `3e-4`

### 2. PPO裁剪过于激进
**症状**: 策略更新幅度太大
**解决方案**: 
- `clip_ploss_coef: 0.2` → `0.1` 或 `0.05`

### 3. 缺乏探索
**症状**: 熵快速下降为0
**解决方案**: 
- 增加 `ent_coef: 0.01` → `0.02-0.05`
- 使用固定标准差: `fixed_std: 0.1`

### 4. 批次大小和更新次数
**症状**: 过拟合单个批次
**解决方案**: 
- 减少 `batch_size: 10000` → `2000`
- 减少 `update_epochs: 10` → `5`

### 5. 奖励缩放问题
**症状**: 奖励数值不稳定
**解决方案**: 
- 关闭运行时奖励缩放: `reward_scale_running: False`
- 设置 `reward_scale_const: 1.0`

### 6. 梯度爆炸
**症状**: 损失值突然跳跃
**解决方案**: 
- 添加梯度裁剪: `max_grad_norm: 0.5`

## 修复步骤

### 步骤 1: 使用保守配置
```bash
# 使用改进的配置文件
./run_simple_mlp_ppo.sh hopper-v2 42 env
```

### 步骤 2: 实时监控
```bash
# 实时监控训练
python monitor_ppo_training.py --continuous --plot
```

### 步骤 3: 关键指标检查
- **策略损失**: 应该小且稳定 (0.01-0.1)
- **价值损失**: 应该逐渐下降
- **熵**: 应该缓慢下降，不应该为0
- **解释方差**: 应该 > 0.5
- **KL散度**: 应该 < target_kl

### 步骤 4: 紧急修复
如果奖励仍然下降，立即应用这些设置：

```yaml
train:
  actor_lr: 1e-4          # 更低的学习率
  critic_lr: 3e-4
  batch_size: 1000        # 更小的批次
  update_epochs: 3        # 更少的更新
  ent_coef: 0.05          # 更高的熵
  max_grad_norm: 0.3      # 更严格的梯度裁剪
  target_kl: 0.01         # 更保守的KL目标

model:
  clip_ploss_coef: 0.05   # 更保守的裁剪
  actor:
    fixed_std: 0.1        # 固定标准差确保探索
```

## 高级诊断

### 检查环境问题
```python
# 验证环境奖励分布
import gym
env = gym.make('Hopper-v2')
rewards = []
for _ in range(100):
    env.reset()
    ep_reward = 0
    for _ in range(1000):
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        ep_reward += reward
        if done:
            break
    rewards.append(ep_reward)
print(f"Random policy reward range: {min(rewards):.1f} - {max(rewards):.1f}")
```

### 检查策略分布
```python
# 在训练过程中监控动作标准差
# 应该逐渐下降但不应该接近0
```

## 应急方案

如果所有方法都失败：

1. **重新初始化**: 从随机权重重新开始
2. **预训练**: 先用行为克隆预训练几个epoch
3. **更简单的网络**: 使用 `[128, 128]` 而不是 `[256, 256, 256]`
4. **不同的优化器**: 尝试 Adam 的不同变种

## 成功指标

训练成功的标志：
- 奖励稳定增长或保持稳定
- 策略损失 < 0.1 且稳定
- 熵缓慢下降但保持 > 0.1
- 解释方差 > 0.5
- KL散度在目标范围内

## 预防措施

1. 始终从保守的超参数开始
2. 实时监控关键指标
3. 设置早停机制
4. 定期保存检查点
5. 使用多个随机种子验证