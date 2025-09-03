#!/bin/bash
# Quick start script for EBM reward replacement

set -e

echo "🚀 DPPO-EBM 奖励替代快速入门"
echo "================================"

# Check if we're in the right directory
if [ ! -f "train_simple_mlp_ebm.py" ]; then
    echo "❌ 错误: 请在 DPPO-EBM 项目根目录运行此脚本"
    exit 1
fi

# Default settings
ENV_NAME="hopper"
EPOCHS=20
DEVICE="cuda"
NO_WANDB=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV_NAME="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --cpu)
            DEVICE="cpu"
            shift
            ;;
        --no-wandb)
            NO_WANDB="--no_wandb"
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --env ENV       环境名称 (hopper|walker2d|halfcheetah, 默认: hopper)"
            echo "  --epochs N      训练轮数 (默认: 20)"
            echo "  --cpu           使用 CPU 而非 GPU"
            echo "  --no-wandb      禁用 WandB 日志"
            echo "  --help          显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --env walker2d --epochs 50"
            echo "  $0 --cpu --no-wandb"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo "设置: 环境=$ENV_NAME, 轮数=$EPOCHS, 设备=$DEVICE"
echo ""

# Step 1: Run tests
echo "📋 步骤 1: 运行系统测试"
echo "------------------------"
if python test_ebm_reward_replacement.py; then
    echo "✅ 系统测试通过"
else
    echo "❌ 系统测试失败，请检查安装"
    exit 1
fi
echo ""

# Step 2: Train EBM
echo "🧠 步骤 2: 训练 EBM 模型"
echo "------------------------"
echo "正在为 $ENV_NAME 环境训练 EBM..."

python train_simple_mlp_ebm.py \
    --env_name $ENV_NAME \
    --epochs $EPOCHS \
    --device $DEVICE \
    --batch_size 256 \
    --lr 1e-3 \
    $NO_WANDB

echo "✅ EBM 训练完成"
echo ""

# Step 3: Verify checkpoint
echo "🔍 步骤 3: 验证 EBM 检查点"
echo "-------------------------"
CHECKPOINT_PATH="checkpoints/simple_mlp_ebm_${ENV_NAME}.pt"

if [ -f "$CHECKPOINT_PATH" ]; then
    echo "✅ 找到 EBM 检查点: $CHECKPOINT_PATH"
    
    # Check checkpoint info
    python -c "
import torch
try:
    ckpt = torch.load('$CHECKPOINT_PATH', map_location='cpu')
    print(f'  验证损失: {ckpt[\"val_loss\"]:.4f}')
    print(f'  训练损失: {ckpt[\"train_loss\"]:.4f}')
    print(f'  训练轮数: {ckpt[\"epoch\"]}')
    print('✅ 检查点信息正常')
except Exception as e:
    print(f'❌ 检查点验证失败: {e}')
    exit(1
"
else
    echo "❌ 未找到 EBM 检查点: $CHECKPOINT_PATH"
    exit 1
fi
echo ""

# Step 4: Show next steps
echo "🎯 步骤 4: 下一步操作"
echo "--------------------"
echo "EBM 模型训练完成！现在可以运行 DPPO 训练："
echo ""

# Convert env name for config path
case $ENV_NAME in
    hopper)
        CONFIG_ENV="hopper-v2"
        ;;
    walker2d) 
        CONFIG_ENV="walker2d-v2"
        ;;
    halfcheetah)
        CONFIG_ENV="halfcheetah-v2"
        ;;
    *)
        CONFIG_ENV="$ENV_NAME-v2"
        ;;
esac

echo "python script/run.py \\"
echo "    --config-name=ft_ppo_diffusion_mlp_ebm_reward_only \\"
echo "    --config-dir=cfg/gym/finetune/$CONFIG_ENV"
echo ""

echo "或者运行其他环境的 EBM 训练:"
for env in hopper walker2d halfcheetah; do
    if [ "$env" != "$ENV_NAME" ]; then
        echo "  $0 --env $env --epochs $EPOCHS"
    fi
done
echo ""

echo "📚 更多信息请查看:"
echo "  - EBM_REWARD_REPLACEMENT_GUIDE.md (详细使用指南)"
echo "  - test_ebm_reward_replacement.py (测试脚本)"
echo ""

echo "🎉 快速入门完成！"