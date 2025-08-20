#!/bin/bash

# PPO, PPO+EBM, SAC, SAC+EBM 比较实验示例脚本
# 这个脚本展示了如何运行完整的比较实验

set -e  # 遇到错误时退出

echo "=========================================="
echo "PPO, PPO+EBM, SAC, SAC+EBM 比较实验"
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export DPPO_LOG_DIR="./outputs"
export DPPO_DATA_DIR="/path/to/your/data"  # 请修改为实际路径

# 创建输出目录
mkdir -p outputs
mkdir -p comparison_results

# 定义实验参数
ENV_NAME="walker2d-v2"
SEEDS=(42 123 456)
MAX_STEPS=1000000

echo "环境: $ENV_NAME"
echo "随机种子: ${SEEDS[@]}"
echo "最大训练步数: $MAX_STEPS"
echo ""

# 方法配置
METHODS=("ppo" "ppo_ebm" "sac" "sac_ebm")
METHOD_NAMES=("PPO" "PPO+EBM" "SAC" "SAC+EBM")

# 记录开始时间
START_TIME=$(date +%s)

# 运行训练实验
echo "开始训练实验..."
echo ""

for i in "${!METHODS[@]}"; do
    METHOD=${METHODS[$i]}
    METHOD_NAME=${METHOD_NAMES[$i]}
    
    echo "=========================================="
    echo "训练 $METHOD_NAME"
    echo "=========================================="
    
    for SEED in "${SEEDS[@]}"; do
        echo "种子: $SEED"
        
        # 构建配置文件路径
        CONFIG_PATH="cfg/gym/finetune/$ENV_NAME/ft_${METHOD}_diffusion_mlp.yaml"
        
        # 检查配置文件是否存在
        if [ ! -f "$CONFIG_PATH" ]; then
            echo "错误: 配置文件不存在: $CONFIG_PATH"
            exit 1
        fi
        
        # 运行训练
        python script/run.py \
            --config-path "$CONFIG_PATH" \
            "seed=$SEED" \
            "env_name=$ENV_NAME" \
            "logdir=outputs/${METHOD}_${ENV_NAME}_${SEED}" \
            "device=cuda:0" \
            "max_steps=$MAX_STEPS"
        
        echo "✅ $METHOD_NAME (种子 $SEED) 训练完成"
        echo ""
        
        # 添加延迟，避免资源冲突
        sleep 10
    done
done

# 计算训练时间
END_TIME=$(date +%s)
TRAINING_TIME=$((END_TIME - START_TIME))
echo "训练完成！总耗时: $((TRAINING_TIME / 60)) 分钟 $((TRAINING_TIME % 60)) 秒"
echo ""

# 运行评估
echo "=========================================="
echo "开始模型评估"
echo "=========================================="

# 构建模型目录列表
MODEL_DIRS=""
for METHOD in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        MODEL_DIRS="$MODEL_DIRS outputs/${METHOD}_${ENV_NAME}_${SEED}"
    done
done

# 运行评估
python script/evaluate_trained_models.py \
    --model-dirs $MODEL_DIRS \
    --env "$ENV_NAME" \
    --n-episodes 100 \
    --output-dir "comparison_results"

echo "✅ 评估完成"
echo ""

# 生成比较报告
echo "=========================================="
echo "生成比较报告"
echo "=========================================="

# 使用快速比较脚本生成报告
python script/quick_comparison.py \
    --env "$ENV_NAME" \
    --seeds "${SEEDS[@]}" \
    --skip-training

echo "✅ 比较报告生成完成"
echo ""

# 显示结果摘要
echo "=========================================="
echo "实验完成摘要"
echo "=========================================="

echo "📁 输出目录:"
echo "  - 训练结果: ./outputs/"
echo "  - 比较结果: ./comparison_results/"
echo "  - 日志文件: ./comparison_experiments.log"

echo ""
echo "📊 生成的文件:"
echo "  - 学习曲线: comparison_results/learning_curves.png"
echo "  - 性能比较: comparison_results/final_performance.png"
echo "  - 成功率比较: comparison_results/success_rates.png"
echo "  - 统计分析: comparison_results/statistical_analysis.png"
echo "  - 评估报告: comparison_results/evaluation_report.md"

echo ""
echo "🎯 下一步建议:"
echo "  1. 查看生成的图表和报告"
echo "  2. 使用WandB查看详细的学习曲线"
echo "  3. 分析统计显著性结果"
echo "  4. 根据结果调整EBM参数"

echo ""
echo "=========================================="
echo "实验完成！"
echo "=========================================="

# 可选：显示最佳方法
if [ -f "comparison_results/evaluation_report.json" ]; then
    echo ""
    echo "🏆 最佳方法:"
    python -c "
import json
try:
    with open('comparison_results/evaluation_report.json', 'r') as f:
        data = json.load(f)
    if 'results_summary' in data and data['results_summary']:
        best_method = max(data['results_summary'].items(), key=lambda x: x[1]['mean_reward'])[0]
        print(f'  {best_method}')
    else:
        print('  数据不完整，请检查评估结果')
except Exception as e:
    print(f'  无法读取结果: {e}')
"
fi
