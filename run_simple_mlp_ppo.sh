#!/bin/bash

# Simple MLP PPO Baseline Runner
# This script runs both environment reward and EBM reward versions of Simple MLP PPO
# 
# Usage:
#   ./run_simple_mlp_ppo.sh <env_name> <seed> [reward_type]
#
# Examples:
#   ./run_simple_mlp_ppo.sh hopper-v2 42 env     # Run with environment rewards
#   ./run_simple_mlp_ppo.sh hopper-v2 42 ebm     # Run with EBM rewards
#   ./run_simple_mlp_ppo.sh hopper-v2 42 both    # Run both configurations

set -e

# Default values
ENV_NAME=${1:-hopper-v2}
SEED=${2:-42}
REWARD_TYPE=${3:-env}

# Environment mapping
case $ENV_NAME in
    "hopper-v2")
        ENV_FULL="hopper-medium-v2"
        ;;
    "walker2d-v2")
        ENV_FULL="walker2d-medium-v2"
        ;;
    "halfcheetah-v2")
        ENV_FULL="halfcheetah-medium-v2"
        ;;
    *)
        echo "Error: Unsupported environment $ENV_NAME"
        echo "Supported environments: hopper-v2, walker2d-v2, halfcheetah-v2"
        exit 1
        ;;
esac

echo "============================================"
echo "Simple MLP PPO Baseline Training"
echo "Environment: $ENV_FULL"
echo "Seed: $SEED"
echo "Reward Type: $REWARD_TYPE"
echo "============================================"

# Function to run training
run_training() {
    local reward_type=$1
    local config_name="ft_simple_mlp_ppo_${reward_type}"
    local config_dir="cfg/gym/finetune/${ENV_NAME}"
    
    echo ""
    echo "Starting training with $reward_type rewards..."
    echo "Config: $config_name"
    echo "Config Dir: $config_dir"
    
    python script/run.py \
        --config-name=$config_name \
        --config-dir=$config_dir \
        seed=$SEED \
        env_name=$ENV_FULL
}

# Run based on reward type selection
case $REWARD_TYPE in
    "env")
        echo "Running with environment rewards only..."
        run_training "env"
        ;;
    "ebm")
        echo "Running with EBM rewards only..."
        run_training "ebm"
        ;;
    "both")
        echo "Running both environment and EBM reward configurations..."
        run_training "env"
        echo ""
        echo "Environment reward training completed. Starting EBM reward training..."
        sleep 2
        run_training "ebm"
        ;;
    *)
        echo "Error: Invalid reward type '$REWARD_TYPE'"
        echo "Valid options: env, ebm, both"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Training completed successfully!"
echo "============================================"