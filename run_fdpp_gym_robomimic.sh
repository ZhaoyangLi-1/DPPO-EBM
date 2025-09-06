#!/bin/bash

# FDPP for DPPO-EBM: Fine-tune Diffusion Policy with Human Preference
# Script for running FDPP on gym and robomimic tasks

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
TASK_TYPE=${1:-"gym"}  # gym or robomimic
TASK_NAME=${2:-"hopper-v2"}  # e.g., hopper-v2, walker2d-v2, lift, can
KL_WEIGHT=${3:-"0.01"}
SEED=${4:-"42"}
GPU=${5:-"0"}

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU

print_info "Starting FDPP experiment"
print_info "Task Type: $TASK_TYPE"
print_info "Task Name: $TASK_NAME"
print_info "KL Weight: $KL_WEIGHT"
print_info "Seed: $SEED"
print_info "GPU: $GPU"

# Select configuration based on task type
if [ "$TASK_TYPE" == "gym" ]; then
    # Gym tasks
    if [ "$TASK_NAME" == "hopper-v2" ] || [ "$TASK_NAME" == "hopper-medium-v2" ]; then
        CONFIG_PATH="cfg/gym/finetune/hopper-v2"
        ENV_NAME="hopper-medium-v2"
    elif [ "$TASK_NAME" == "walker2d-v2" ] || [ "$TASK_NAME" == "walker2d-medium-v2" ]; then
        CONFIG_PATH="cfg/gym/finetune/walker2d-v2"
        ENV_NAME="walker2d-medium-v2"
    elif [ "$TASK_NAME" == "halfcheetah-v2" ] || [ "$TASK_NAME" == "halfcheetah-medium-v2" ]; then
        CONFIG_PATH="cfg/gym/finetune/halfcheetah-v2"
        ENV_NAME="halfcheetah-medium-v2"
    else
        print_error "Unknown gym task: $TASK_NAME"
        echo "Available gym tasks: hopper-v2, walker2d-v2, halfcheetah-v2"
        exit 1
    fi
    CONFIG="${CONFIG_PATH}/ft_fdpp_diffusion_mlp.yaml"
    
elif [ "$TASK_TYPE" == "robomimic" ]; then
    # Robomimic tasks
    if [ "$TASK_NAME" == "lift" ]; then
        CONFIG_PATH="cfg/robomimic/finetune/lift"
    elif [ "$TASK_NAME" == "can" ]; then
        CONFIG_PATH="cfg/robomimic/finetune/can"
    elif [ "$TASK_NAME" == "square" ]; then
        CONFIG_PATH="cfg/robomimic/finetune/square"
    elif [ "$TASK_NAME" == "transport" ]; then
        CONFIG_PATH="cfg/robomimic/finetune/transport"
    else
        print_error "Unknown robomimic task: $TASK_NAME"
        echo "Available robomimic tasks: lift, can, square, transport"
        exit 1
    fi
    CONFIG="${CONFIG_PATH}/ft_fdpp_diffusion_mlp.yaml"
    ENV_NAME=$TASK_NAME
    
else
    print_error "Unknown task type: $TASK_TYPE"
    echo "Usage: $0 [gym|robomimic] [task_name] [kl_weight] [seed] [gpu]"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    print_error "Configuration file not found: $CONFIG"
    print_info "Creating configuration from template..."
    
    # Create the config if it doesn't exist
    if [ "$TASK_TYPE" == "gym" ]; then
        # Copy from hopper template
        mkdir -p $(dirname "$CONFIG")
        cp cfg/gym/finetune/hopper-v2/ft_fdpp_diffusion_mlp.yaml "$CONFIG"
        # Update env_name in the config
        sed -i "s/env_name: hopper-medium-v2/env_name: ${ENV_NAME}/g" "$CONFIG"
    elif [ "$TASK_TYPE" == "robomimic" ]; then
        # Copy from lift template
        mkdir -p $(dirname "$CONFIG")
        cp cfg/robomimic/finetune/lift/ft_fdpp_diffusion_mlp.yaml "$CONFIG"
        # Update env_name in the config
        sed -i "s/env_name: lift/env_name: ${ENV_NAME}/g" "$CONFIG"
    fi
fi

# Create log directory
mkdir -p $DPPO_LOG_DIR/${TASK_TYPE}-finetune-fdpp

# Run FDPP training
print_info "Starting FDPP training for $TASK_TYPE/$ENV_NAME..."
print_info "Config: $CONFIG"

python -m hydra.main \
    config_path="$(pwd)/$(dirname $CONFIG)" \
    config_name="$(basename $CONFIG .yaml)" \
    kl_weight=$KL_WEIGHT \
    seed=$SEED \
    env_name=$ENV_NAME \
    hydra.job.chdir=True

if [ $? -eq 0 ]; then
    print_success "FDPP training completed successfully!"
else
    print_error "FDPP training failed!"
    exit 1
fi