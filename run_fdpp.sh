#!/bin/bash

# FDPP: Fine-tune Diffusion Policy with Human Preference
# Example script for running FDPP experiments

# Set environment variables
export DPPO_LOG_DIR=${DPPO_LOG_DIR:-"./log"}
export DPPO_DATA_DIR=${DPPO_DATA_DIR:-"./data"}
export DPPO_WANDB_ENTITY=${DPPO_WANDB_ENTITY:-"your_wandb_entity"}

# Function to print colored output
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# Parse command line arguments
TASK=${1:-"pusht"}
KL_WEIGHT=${2:-"0.01"}
SEED=${3:-"42"}
GPU=${4:-"0"}

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU

print_info "Starting FDPP experiment"
print_info "Task: $TASK"
print_info "KL Weight: $KL_WEIGHT"
print_info "Seed: $SEED"
print_info "GPU: $GPU"

# Select configuration based on task
if [ "$TASK" == "pusht" ]; then
    CONFIG="cfg/fdpp_pusht_diffusion.yaml"
    print_info "Using Push-T configuration"
elif [ "$TASK" == "stack-dist" ]; then
    CONFIG="cfg/fdpp_stack_diffusion.yaml"
    EXTRA_ARGS="preference_type=dist"
    print_info "Using Stack task with distance preference"
elif [ "$TASK" == "stack-align" ]; then
    CONFIG="cfg/fdpp_stack_diffusion.yaml"
    EXTRA_ARGS="preference_type=align"
    print_info "Using Stack task with alignment preference"
else
    print_error "Unknown task: $TASK"
    echo "Usage: $0 [pusht|stack-dist|stack-align] [kl_weight] [seed] [gpu]"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    print_error "Configuration file not found: $CONFIG"
    exit 1
fi

# Create log directory
mkdir -p $DPPO_LOG_DIR/fdpp-finetune

# Run FDPP training
print_info "Starting FDPP training..."

python -m hydra.main \
    config_path=../cfg \
    config_name=$(basename $CONFIG .yaml) \
    kl_weight=$KL_WEIGHT \
    seed=$SEED \
    $EXTRA_ARGS \
    hydra.job.chdir=True

if [ $? -eq 0 ]; then
    print_success "FDPP training completed successfully!"
else
    print_error "FDPP training failed!"
    exit 1
fi