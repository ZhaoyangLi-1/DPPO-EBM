#!/usr/bin/env python3
"""
Quick comparison script for PPO, PPO+EBM, SAC, SAC+EBM.

This script provides a streamlined way to run training and evaluation
for all four methods with minimal configuration.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
import logging

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def run_single_experiment(env_name, method, seed, gpu_id=0, max_steps=1000000):
    """
    Run a single training experiment.
    
    Args:
        env_name: Environment name
        method: Method name ('ppo', 'ppo_ebm', 'sac', 'sac_ebm')
        seed: Random seed
        gpu_id: GPU ID to use
        max_steps: Maximum training steps
    """
    logger = logging.getLogger(__name__)
    
    # Define method configurations
    method_configs = {
        'ppo': 'cfg/gym/finetune/{env}/ft_ppo_diffusion_mlp.yaml',
        'ppo_ebm': 'cfg/gym/finetune/{env}/ft_ppo_diffusion_ebm_mlp.yaml',
        'sac': 'cfg/gym/finetune/{env}/ft_sac_diffusion_mlp.yaml',
        'sac_ebm': 'cfg/gym/finetune/{env}/ft_sac_diffusion_ebm_mlp.yaml'
    }
    
    config_path = method_configs[method].format(env=env_name)
    config_path = project_root / config_path
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False
    
    # Set environment variables
    env_vars = {
        'CUDA_VISIBLE_DEVICES': str(gpu_id),
        'DPPO_LOG_DIR': str(project_root / 'outputs'),
    }
    
    # Construct command
    cmd = [
        'python', 'script/run.py',
        '--config-path', str(config_path),
        f'seed={seed}',
        f'env_name={env_name}',
        f'logdir=outputs/{method}_{env_name}_{seed}',
        'device=cuda:0',
        f'max_steps={max_steps}'
    ]
    
    logger.info(f"Running {method} on {env_name} with seed {seed}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            env={**os.environ, **env_vars},
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            logger.info(f"✅ Successfully completed {method} on {env_name} with seed {seed}")
            return True
        else:
            logger.error(f"❌ Failed to run {method} on {env_name} with seed {seed}")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Timeout for {method} on {env_name} with seed {seed}")
        return False
    except Exception as e:
        logger.error(f"💥 Exception for {method} on {env_name} with seed {seed}: {e}")
        return False

def run_all_experiments(env_name, seeds, gpu_id=0, max_steps=1000000):
    """
    Run all experiments for the four methods.
    
    Args:
        env_name: Environment name
        seeds: List of random seeds
        gpu_id: GPU ID to use
        max_steps: Maximum training steps
    """
    logger = logging.getLogger(__name__)
    
    methods = ['ppo', 'ppo_ebm', 'sac', 'sac_ebm']
    method_names = ['PPO', 'PPO+EBM', 'SAC', 'SAC+EBM']
    
    results = {}
    
    logger.info("=" * 60)
    logger.info("Starting comprehensive comparison experiments")
    logger.info(f"Environment: {env_name}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Methods: {method_names}")
    logger.info("=" * 60)
    
    for method in methods:
        logger.info(f"\n🚀 Running experiments for {method}")
        results[method] = []
        
        for seed in seeds:
            success = run_single_experiment(env_name, method, seed, gpu_id, max_steps)
            results[method].append(success)
            
            if success:
                logger.info(f"  ✅ Seed {seed}: SUCCESS")
            else:
                logger.error(f"  ❌ Seed {seed}: FAILED")
            
            # Add delay between experiments
            time.sleep(5)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    
    for method, method_name in zip(methods, method_names):
        successes = sum(results[method])
        total = len(results[method])
        success_rate = successes / total if total > 0 else 0
        
        logger.info(f"{method_name:12}: {successes}/{total} successful ({success_rate:.1%})")
    
    return results

def generate_comparison_plots():
    """
    Generate comparison plots from training results.
    This is a simplified version that uses wandb or tensorboard logs.
    """
    logger = logging.getLogger(__name__)
    
    logger.info("\n📊 Generating comparison plots...")
    
    # This would typically involve:
    # 1. Reading wandb logs or tensorboard logs
    # 2. Extracting learning curves
    # 3. Creating comparison plots
    # 4. Saving to output directory
    
    logger.info("📈 Plots would be generated here (implement based on your logging setup)")

def main():
    """Main function for quick comparison."""
    parser = argparse.ArgumentParser(description='Quick comparison of PPO, PPO+EBM, SAC, SAC+EBM')
    parser.add_argument('--env', type=str, default='walker2d-v2',
                       help='Environment name (default: walker2d-v2)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123],
                       help='Random seeds to use (default: [42, 123])')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use (default: 0)')
    parser.add_argument('--max-steps', type=int, default=1000000,
                       help='Maximum training steps (default: 1000000)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only generate plots')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['ppo', 'ppo_ebm', 'sac', 'sac_ebm'],
                       help='Methods to run (default: all)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    if not args.skip_training:
        # Run training experiments
        results = run_all_experiments(args.env, args.seeds, args.gpu, args.max_steps)
        
        # Print final summary
        logger.info("\n" + "=" * 60)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 60)
        
        total_experiments = len(args.seeds) * len(args.methods)
        successful_experiments = sum(sum(results[method]) for method in results)
        
        logger.info(f"Total experiments: {total_experiments}")
        logger.info(f"Successful experiments: {successful_experiments}")
        logger.info(f"Success rate: {successful_experiments/total_experiments:.1%}")
        
        if successful_experiments > 0:
            logger.info("\n📁 Results saved to:")
            logger.info("  - outputs/ (training logs and checkpoints)")
            logger.info("  - wandb/ (if using wandb logging)")
            
            # Generate comparison plots
            generate_comparison_plots()
            
            logger.info("\n🎯 Next steps:")
            logger.info("  1. Check the training logs in outputs/")
            logger.info("  2. Use wandb dashboard to compare learning curves")
            logger.info("  3. Run evaluation script on trained models")
            logger.info("  4. Generate final comparison report")
        else:
            logger.error("❌ No experiments completed successfully!")
            logger.error("Check the error messages above for troubleshooting.")
    else:
        logger.info("⏭️ Skipping training (--skip-training flag used)")
        generate_comparison_plots()

if __name__ == '__main__':
    main()
