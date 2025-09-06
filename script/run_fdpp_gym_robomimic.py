"""
FDPP for DPPO-EBM: Fine-tune Diffusion Policy with Human Preference
Main script for running FDPP experiments on gym and robomimic tasks.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import torch
import numpy as np
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agent.finetune.train_fdpp_diffusion_agent import TrainFDPPDiffusionAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config_path(task_type: str, task_name: str) -> str:
    """Get configuration path for given task."""
    if task_type == "gym":
        # Map gym task names to their config paths
        gym_task_map = {
            "hopper-v2": "cfg/gym/finetune/hopper-v2/ft_fdpp_diffusion_mlp.yaml",
            "hopper-medium-v2": "cfg/gym/finetune/hopper-v2/ft_fdpp_diffusion_mlp.yaml",
            "walker2d-v2": "cfg/gym/finetune/walker2d-v2/ft_fdpp_diffusion_mlp.yaml", 
            "walker2d-medium-v2": "cfg/gym/finetune/walker2d-v2/ft_fdpp_diffusion_mlp.yaml",
            "halfcheetah-v2": "cfg/gym/finetune/halfcheetah-v2/ft_fdpp_diffusion_mlp.yaml",
            "halfcheetah-medium-v2": "cfg/gym/finetune/halfcheetah-v2/ft_fdpp_diffusion_mlp.yaml",
        }
        return gym_task_map.get(task_name, f"cfg/gym/finetune/{task_name}/ft_fdpp_diffusion_mlp.yaml")
    
    elif task_type == "robomimic":
        # Map robomimic task names
        robomimic_task_map = {
            "lift": "cfg/robomimic/finetune/lift/ft_fdpp_diffusion_mlp.yaml",
            "can": "cfg/robomimic/finetune/can/ft_fdpp_diffusion_mlp.yaml",
            "square": "cfg/robomimic/finetune/square/ft_fdpp_diffusion_mlp.yaml",
            "transport": "cfg/robomimic/finetune/transport/ft_fdpp_diffusion_mlp.yaml",
        }
        return robomimic_task_map.get(task_name, f"cfg/robomimic/finetune/{task_name}/ft_fdpp_diffusion_mlp.yaml")
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def validate_config(cfg: DictConfig) -> bool:
    """Validate the configuration parameters."""
    # Check required parameters
    required_params = [
        'env_name', 'obs_dim', 'action_dim', 'device',
        'kl_weight', 'use_preference_reward'
    ]
    
    for param in required_params:
        if param not in cfg:
            logger.error(f"Missing required parameter: {param}")
            return False
    
    # Validate KL weight
    if cfg.kl_weight < 0:
        logger.error(f"Invalid KL weight: {cfg.kl_weight}. Must be >= 0")
        return False
    
    # Validate device
    if not torch.cuda.is_available() and 'cuda' in cfg.device:
        logger.warning("CUDA not available, switching to CPU")
        cfg.device = 'cpu'
    
    return True


def run_fdpp_experiment(cfg: DictConfig):
    """Run FDPP experiment with given configuration."""
    logger.info("="*60)
    logger.info("FDPP for DPPO-EBM")
    logger.info("Fine-tune Diffusion Policy with Human Preference")
    logger.info("="*60)
    
    # Validate configuration
    if not validate_config(cfg):
        raise ValueError("Invalid configuration")
    
    # Set random seed
    set_seed(cfg.get('seed', 42))
    
    # Log experiment details
    logger.info(f"Task: {cfg.env_name}")
    logger.info(f"Observation Dim: {cfg.obs_dim}")
    logger.info(f"Action Dim: {cfg.action_dim}")
    logger.info(f"KL Weight (α): {cfg.kl_weight}")
    logger.info(f"Preference Learning: {cfg.use_preference_reward}")
    logger.info(f"Device: {cfg.device}")
    logger.info(f"Seed: {cfg.get('seed', 42)}")
    
    # Initialize FDPP agent
    logger.info("Initializing FDPP agent...")
    agent = TrainFDPPDiffusionAgent(cfg)
    
    # Load pre-trained policy if specified
    if 'base_policy_path' in cfg and Path(cfg.base_policy_path).exists():
        logger.info(f"Loading pre-trained policy from: {cfg.base_policy_path}")
        # Note: Loading logic would need to be implemented in the agent
    else:
        logger.warning("No pre-trained policy path specified or file not found")
    
    # Run training
    logger.info("Starting FDPP training...")
    try:
        agent.run()
        logger.info("FDPP training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        agent.save(agent.itr)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


def main():
    """Main entry point for FDPP experiments."""
    parser = argparse.ArgumentParser(
        description="FDPP for DPPO-EBM: Fine-tune Diffusion Policy with Human Preference"
    )
    
    # Task selection
    parser.add_argument(
        '--task_type', 
        type=str, 
        default='gym',
        choices=['gym', 'robomimic'],
        help='Type of task (gym or robomimic)'
    )
    
    parser.add_argument(
        '--task_name', 
        type=str, 
        default='hopper-v2',
        help='Name of the specific task'
    )
    
    # FDPP specific parameters
    parser.add_argument(
        '--kl_weight', 
        type=float, 
        default=0.01,
        help='KL regularization weight (alpha in paper)'
    )
    
    parser.add_argument(
        '--no_preference',
        action='store_true',
        help='Disable preference reward learning'
    )
    
    parser.add_argument(
        '--preference_buffer_size',
        type=int,
        default=10000,
        help='Size of preference dataset buffer'
    )
    
    parser.add_argument(
        '--preference_lr',
        type=float,
        default=1e-4,
        help='Learning rate for preference model'
    )
    
    # Training parameters
    parser.add_argument(
        '--n_itr',
        type=int,
        default=100,
        help='Number of training iterations'
    )
    
    parser.add_argument(
        '--n_envs',
        type=int,
        default=50,
        help='Number of parallel environments'
    )
    
    # Other parameters
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use (cuda:0, cpu, etc.)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom config file'
    )
    
    parser.add_argument(
        '--wandb_entity',
        type=str,
        default=None,
        help='WandB entity name'
    )
    
    parser.add_argument(
        '--wandb_project',
        type=str,
        default=None,
        help='WandB project name'
    )
    
    args = parser.parse_args()
    
    # Get configuration file
    if args.config:
        config_path = args.config
    else:
        config_path = get_config_path(args.task_type, args.task_name)
    
    # Load configuration
    if not Path(config_path).exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Available gym tasks: hopper-v2, walker2d-v2, halfcheetah-v2")
        logger.info("Available robomimic tasks: lift, can, square, transport")
        sys.exit(1)
    
    cfg = OmegaConf.load(config_path)
    
    # Override configuration with command line arguments
    cfg.kl_weight = args.kl_weight
    cfg.use_preference_reward = not args.no_preference
    cfg.preference_buffer_size = args.preference_buffer_size
    cfg.preference_lr = args.preference_lr
    cfg.train.n_train_itr = args.n_itr
    cfg.env.n_envs = args.n_envs
    cfg.seed = args.seed
    cfg.device = args.device
    
    # Handle environment name mapping for gym tasks
    if args.task_type == "gym":
        env_name_map = {
            "hopper-v2": "hopper-medium-v2",
            "walker2d-v2": "walker2d-medium-v2", 
            "halfcheetah-v2": "halfcheetah-medium-v2",
        }
        cfg.env_name = env_name_map.get(args.task_name, args.task_name)
    else:
        cfg.env_name = args.task_name
    
    # Override WandB settings
    if args.wandb_entity:
        cfg.wandb.entity = args.wandb_entity
    if args.wandb_project:
        cfg.wandb.project = args.wandb_project
    elif args.task_type == "gym":
        cfg.wandb.project = f"gym-{cfg.env_name}-finetune-fdpp"
    else:
        cfg.wandb.project = f"robomimic-{cfg.env_name}-finetune-fdpp"
    
    # Set environment variables
    os.environ.setdefault('DPPO_LOG_DIR', './log')
    os.environ.setdefault('DPPO_DATA_DIR', './data')
    if args.wandb_entity:
        os.environ['DPPO_WANDB_ENTITY'] = args.wandb_entity
    
    # Run experiment
    run_fdpp_experiment(cfg)


if __name__ == "__main__":
    main()