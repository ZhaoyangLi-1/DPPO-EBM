#!/usr/bin/env python3
"""
Comprehensive comparison script for PPO, PPO+EBM, SAC, SAC+EBM experiments.

This script runs training experiments for all four methods and provides
comprehensive comparison metrics and visualizations.
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('comparison_experiments.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_training_experiment(env_name, method, config_path, seed, logdir, gpu_id=0):
    """
    Run a single training experiment.
    
    Args:
        env_name: Environment name (e.g., 'walker2d-v2')
        method: Method name ('ppo', 'ppo_ebm', 'sac', 'sac_ebm')
        config_path: Path to configuration file
        seed: Random seed
        logdir: Logging directory
        gpu_id: GPU ID to use
    """
    logger = logging.getLogger(__name__)
    
    # Set environment variables
    env_vars = {
        'CUDA_VISIBLE_DEVICES': str(gpu_id),
        'DPPO_LOG_DIR': str(logdir),
        'DPPO_DATA_DIR': os.environ.get('DPPO_DATA_DIR', '/path/to/data'),
        'DPPO_WANDB_ENTITY': os.environ.get('DPPO_WANDB_ENTITY', 'your-entity'),
    }
    
    # Construct command
    cmd = [
        'python', 'script/run.py',
        '--config-path', str(config_path),
        'seed=' + str(seed),
        'env_name=' + env_name,
        'logdir=' + str(logdir / f"{method}_{env_name}_{seed}"),
        'device=cuda:0'
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
            timeout=7200  # 2 hours timeout
        )
        
        if result.returncode == 0:
            logger.info(f"Successfully completed {method} on {env_name} with seed {seed}")
            return True, result.stdout
        else:
            logger.error(f"Failed to run {method} on {env_name} with seed {seed}")
            logger.error(f"Error: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout for {method} on {env_name} with seed {seed}")
        return False, "Timeout"
    except Exception as e:
        logger.error(f"Exception for {method} on {env_name} with seed {seed}: {e}")
        return False, str(e)

def parse_training_logs(logdir, method, env_name, seed):
    """
    Parse training logs to extract metrics.
    
    Args:
        logdir: Logging directory
        method: Method name
        env_name: Environment name
        seed: Random seed
        
    Returns:
        Dictionary containing parsed metrics
    """
    log_file = logdir / f"{method}_{env_name}_{seed}" / "training.log"
    
    if not log_file.exists():
        return None
    
    metrics = {
        'method': method,
        'env_name': env_name,
        'seed': seed,
        'episode_rewards': [],
        'eval_returns': [],
        'success_rates': [],
        'training_steps': [],
        'wall_time': []
    }
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Parse different types of log lines
                if 'episode_reward' in line:
                    # Extract episode reward
                    pass
                elif 'eval/return_mean' in line:
                    # Extract evaluation return
                    pass
                elif 'eval/success_rate' in line:
                    # Extract success rate
                    pass
    except Exception as e:
        logging.warning(f"Failed to parse log file {log_file}: {e}")
    
    return metrics

def create_comparison_plots(results, output_dir):
    """
    Create comparison plots for all methods.
    
    Args:
        results: Dictionary containing results for all methods
        output_dir: Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Learning Curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    methods = ['ppo', 'ppo_ebm', 'sac', 'sac_ebm']
    method_names = ['PPO', 'PPO+EBM', 'SAC', 'SAC+EBM']
    
    for i, (method, method_name) in enumerate(zip(methods, method_names)):
        if method in results:
            data = results[method]
            # Plot learning curves
            # ... (implement plotting logic)
            axes[i].set_title(f'{method_name} Learning Curve')
            axes[i].set_xlabel('Training Steps')
            axes[i].set_ylabel('Episode Reward')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Final Performance Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    final_performances = []
    method_labels = []
    
    for method, method_name in zip(methods, method_names):
        if method in results:
            # Calculate final performance metrics
            # ... (implement calculation logic)
            final_performances.append(0)  # Placeholder
            method_labels.append(method_name)
    
    # Create bar plot
    bars = ax.bar(method_labels, final_performances)
    ax.set_title('Final Performance Comparison')
    ax.set_ylabel('Average Episode Reward')
    
    # Add value labels on bars
    for bar, value in zip(bars, final_performances):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Success Rate Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    success_rates = []
    for method, method_name in zip(methods, method_names):
        if method in results:
            # Calculate success rates
            # ... (implement calculation logic)
            success_rates.append(0)  # Placeholder
        else:
            success_rates.append(0)
    
    bars = ax.bar(method_labels, success_rates)
    ax.set_title('Success Rate Comparison')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(0, 1)
    
    for bar, value in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rates.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_comparison_report(results, output_dir):
    """
    Generate a comprehensive comparison report.
    
    Args:
        results: Dictionary containing results for all methods
        output_dir: Output directory for report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    report = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'methods_tested': list(results.keys()),
            'total_experiments': sum(len(data) for data in results.values())
        },
        'results_summary': {},
        'statistical_analysis': {},
        'recommendations': []
    }
    
    # Calculate summary statistics for each method
    for method, data in results.items():
        if data:
            # Calculate mean, std, min, max for key metrics
            # ... (implement calculation logic)
            report['results_summary'][method] = {
                'mean_reward': 0,  # Placeholder
                'std_reward': 0,   # Placeholder
                'mean_success_rate': 0,  # Placeholder
                'training_time': 0  # Placeholder
            }
    
    # Statistical analysis
    # ... (implement statistical tests)
    
    # Generate recommendations
    # ... (implement recommendation logic)
    
    # Save report
    with open(output_dir / 'comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate markdown report
    markdown_report = generate_markdown_report(report)
    with open(output_dir / 'comparison_report.md', 'w') as f:
        f.write(markdown_report)
    
    return report

def generate_markdown_report(report):
    """Generate a markdown version of the comparison report."""
    md = f"""# Experiment Comparison Report

Generated on: {report['experiment_info']['timestamp']}

## Overview

This report compares the performance of four methods:
- PPO (baseline)
- PPO + EBM Reward Shaping
- SAC (baseline)
- SAC + EBM Reward Shaping

## Results Summary

"""
    
    for method, stats in report['results_summary'].items():
        md += f"""
### {method.upper()}

- **Mean Episode Reward**: {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}
- **Mean Success Rate**: {stats['mean_success_rate']:.3f}
- **Average Training Time**: {stats['training_time']:.1f} minutes

"""
    
    md += """
## Statistical Analysis

[Statistical analysis results would go here]

## Recommendations

[Recommendations would go here]

## Plots

The following plots are generated:
- `learning_curves.png`: Learning curves for all methods
- `final_performance.png`: Final performance comparison
- `success_rates.png`: Success rate comparison

"""
    
    return md

def main():
    """Main function to run comparison experiments."""
    parser = argparse.ArgumentParser(description='Run comparison experiments for PPO, PPO+EBM, SAC, SAC+EBM')
    parser.add_argument('--env', type=str, default='walker2d-v2', 
                       help='Environment name (default: walker2d-v2)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                       help='Random seeds to use (default: [42, 123, 456])')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use (default: 0)')
    parser.add_argument('--logdir', type=str, default='./comparison_results',
                       help='Logging directory (default: ./comparison_results)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only generate plots/reports')
    parser.add_argument('--methods', type=str, nargs='+', 
                       default=['ppo', 'ppo_ebm', 'sac', 'sac_ebm'],
                       help='Methods to run (default: all)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting comparison experiments")
    
    # Create output directory
    logdir = Path(args.logdir)
    logdir.mkdir(exist_ok=True)
    
    # Define method configurations
    method_configs = {
        'ppo': {
            'config': f'cfg/gym/finetune/{args.env}/ft_ppo_diffusion_mlp.yaml',
            'description': 'PPO baseline'
        },
        'ppo_ebm': {
            'config': f'cfg/gym/finetune/{args.env}/ft_ppo_diffusion_ebm_mlp.yaml',
            'description': 'PPO + EBM reward shaping'
        },
        'sac': {
            'config': f'cfg/gym/finetune/{args.env}/ft_sac_diffusion_mlp.yaml',
            'description': 'SAC baseline'
        },
        'sac_ebm': {
            'config': f'cfg/gym/finetune/{args.env}/ft_sac_diffusion_ebm_mlp.yaml',
            'description': 'SAC + EBM reward shaping'
        }
    }
    
    # Filter methods based on user input
    methods_to_run = [m for m in args.methods if m in method_configs]
    
    results = {}
    
    if not args.skip_training:
        # Run training experiments
        for method in methods_to_run:
            logger.info(f"Running experiments for {method}")
            results[method] = []
            
            for seed in args.seeds:
                config_path = project_root / method_configs[method]['config']
                
                if not config_path.exists():
                    logger.warning(f"Config file not found: {config_path}")
                    continue
                
                success, output = run_training_experiment(
                    args.env, method, config_path, seed, logdir, args.gpu
                )
                
                if success:
                    # Parse results
                    metrics = parse_training_logs(logdir, method, args.env, seed)
                    if metrics:
                        results[method].append(metrics)
                
                # Add delay between experiments
                time.sleep(10)
    
    # Generate comparison plots and report
    logger.info("Generating comparison plots and report")
    create_comparison_plots(results, logdir)
    report = generate_comparison_report(results, logdir)
    
    logger.info(f"Comparison experiments completed. Results saved to {logdir}")
    logger.info("Generated files:")
    logger.info(f"  - {logdir}/learning_curves.png")
    logger.info(f"  - {logdir}/final_performance.png")
    logger.info(f"  - {logdir}/success_rates.png")
    logger.info(f"  - {logdir}/comparison_report.json")
    logger.info(f"  - {logdir}/comparison_report.md")

if __name__ == '__main__':
    main()
