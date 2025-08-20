#!/usr/bin/env python3
"""
Evaluation script for trained PPO, PPO+EBM, SAC, SAC+EBM models.

This script evaluates trained models and provides comprehensive comparison metrics.
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import logging
from datetime import datetime

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_trained_model(model_path, device='cuda'):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    logger = logging.getLogger(__name__)
    
    if not Path(model_path).exists():
        logger.error(f"Model checkpoint not found: {model_path}")
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # Load model based on checkpoint structure
        # This is a placeholder - implement based on your model structure
        logger.info(f"Successfully loaded model from {model_path}")
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None

def evaluate_model(model, env_name, n_episodes=100, render=False):
    """
    Evaluate a trained model on the environment.
    
    Args:
        model: Trained model
        env_name: Environment name
        n_episodes: Number of episodes to evaluate
        render: Whether to render episodes
        
    Returns:
        Dictionary containing evaluation metrics
    """
    logger = logging.getLogger(__name__)
    
    # This is a placeholder - implement actual evaluation logic
    # based on your environment and model structure
    
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    
    logger.info(f"Evaluating model on {env_name} for {n_episodes} episodes")
    
    # Placeholder evaluation loop
    for episode in range(n_episodes):
        # Implement actual evaluation logic here
        episode_reward = np.random.normal(100, 20)  # Placeholder
        episode_length = np.random.randint(100, 1000)  # Placeholder
        success = np.random.random() > 0.3  # Placeholder
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        success_rates.append(success)
        
        if episode % 10 == 0:
            logger.info(f"Completed episode {episode}/{n_episodes}")
    
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rates': success_rates,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': np.mean(success_rates),
        'n_episodes': n_episodes
    }
    
    logger.info(f"Evaluation completed. Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    logger.info(f"Success rate: {metrics['success_rate']:.3f}")
    
    return metrics

def compare_methods(evaluation_results, output_dir):
    """
    Compare evaluation results across different methods.
    
    Args:
        evaluation_results: Dictionary containing results for each method
        output_dir: Output directory for plots and reports
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    methods = ['ppo', 'ppo_ebm', 'sac', 'sac_ebm']
    method_names = ['PPO', 'PPO+EBM', 'SAC', 'SAC+EBM']
    
    # Extract metrics for comparison
    mean_rewards = []
    std_rewards = []
    success_rates = []
    
    for method in methods:
        if method in evaluation_results:
            results = evaluation_results[method]
            mean_rewards.append(results['mean_reward'])
            std_rewards.append(results['std_reward'])
            success_rates.append(results['success_rate'])
        else:
            mean_rewards.append(0)
            std_rewards.append(0)
            success_rates.append(0)
    
    # 1. Performance Comparison Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Mean reward comparison
    bars1 = ax1.bar(method_names, mean_rewards, yerr=std_rewards, capsize=5)
    ax1.set_title('Mean Episode Reward Comparison')
    ax1.set_ylabel('Mean Episode Reward')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value, std in zip(bars1, mean_rewards, std_rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                f'{value:.1f}±{std:.1f}', ha='center', va='bottom')
    
    # Success rate comparison
    bars2 = ax2.bar(method_names, success_rates)
    ax2.set_title('Success Rate Comparison')
    ax2.set_ylabel('Success Rate')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars2, success_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Learning Curve Comparison (if available)
    if any('learning_curves' in results for results in evaluation_results.values()):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for method, method_name in zip(methods, method_names):
            if method in evaluation_results and 'learning_curves' in evaluation_results[method]:
                learning_data = evaluation_results[method]['learning_curves']
                # Plot learning curves
                # ... (implement plotting logic)
                pass
        
        ax.set_title('Learning Curve Comparison')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Episode Reward')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'learning_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Statistical Analysis
    perform_statistical_analysis(evaluation_results, output_dir)
    
    # 4. Generate comparison report
    generate_evaluation_report(evaluation_results, output_dir)

def perform_statistical_analysis(evaluation_results, output_dir):
    """
    Perform statistical analysis to compare methods.
    
    Args:
        evaluation_results: Dictionary containing results for each method
        output_dir: Output directory for analysis results
    """
    from scipy import stats
    
    methods = ['ppo', 'ppo_ebm', 'sac', 'sac_ebm']
    method_names = ['PPO', 'PPO+EBM', 'SAC', 'SAC+EBM']
    
    # Extract reward data for statistical tests
    reward_data = {}
    for method in methods:
        if method in evaluation_results:
            reward_data[method] = evaluation_results[method]['episode_rewards']
    
    # Perform pairwise t-tests
    statistical_results = {}
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j and method1 in reward_data and method2 in reward_data:
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(reward_data[method1], reward_data[method2])
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(((len(reward_data[method1]) - 1) * np.var(reward_data[method1], ddof=1) +
                                    (len(reward_data[method2]) - 1) * np.var(reward_data[method2], ddof=1)) /
                                   (len(reward_data[method1]) + len(reward_data[method2]) - 2))
                cohens_d = (np.mean(reward_data[method1]) - np.mean(reward_data[method2])) / pooled_std
                
                statistical_results[f"{method1}_vs_{method2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < 0.05
                }
    
    # Save statistical results
    with open(output_dir / 'statistical_analysis.json', 'w') as f:
        json.dump(statistical_results, f, indent=2)
    
    # Create statistical summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    comparisons = []
    p_values = []
    
    for comparison, results in statistical_results.items():
        comparisons.append(comparison.replace('_', ' vs ').upper())
        p_values.append(results['p_value'])
    
    # Create p-value plot
    bars = ax.bar(comparisons, p_values)
    ax.axhline(y=0.05, color='red', linestyle='--', label='Significance threshold (p=0.05)')
    ax.set_title('Statistical Significance Tests')
    ax.set_ylabel('p-value')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    
    # Add significance indicators
    for bar, p_val in zip(bars, p_values):
        if p_val < 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   '***', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'statistical_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_evaluation_report(evaluation_results, output_dir):
    """
    Generate a comprehensive evaluation report.
    
    Args:
        evaluation_results: Dictionary containing results for each method
        output_dir: Output directory for report
    """
    methods = ['ppo', 'ppo_ebm', 'sac', 'sac_ebm']
    method_names = ['PPO', 'PPO+EBM', 'SAC', 'SAC+EBM']
    
    report = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'methods_evaluated': list(evaluation_results.keys()),
            'total_episodes': sum(results['n_episodes'] for results in evaluation_results.values())
        },
        'results_summary': {},
        'recommendations': []
    }
    
    # Calculate summary statistics
    for method, method_name in zip(methods, method_names):
        if method in evaluation_results:
            results = evaluation_results[method]
            report['results_summary'][method_name] = {
                'mean_reward': results['mean_reward'],
                'std_reward': results['std_reward'],
                'success_rate': results['success_rate'],
                'mean_length': results['mean_length'],
                'n_episodes': results['n_episodes']
            }
    
    # Generate recommendations
    if len(evaluation_results) >= 2:
        best_method = max(report['results_summary'].items(), 
                         key=lambda x: x[1]['mean_reward'])[0]
        best_success = max(report['results_summary'].items(), 
                          key=lambda x: x[1]['success_rate'])[0]
        
        report['recommendations'] = [
            f"Best overall performance: {best_method}",
            f"Best success rate: {best_success}",
            "Consider EBM integration if it shows consistent improvements",
            "Evaluate sample efficiency for practical deployment"
        ]
    
    # Save report
    with open(output_dir / 'evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate markdown report
    markdown_report = generate_markdown_evaluation_report(report)
    with open(output_dir / 'evaluation_report.md', 'w') as f:
        f.write(markdown_report)

def generate_markdown_evaluation_report(report):
    """Generate a markdown version of the evaluation report."""
    md = f"""# Model Evaluation Report

Generated on: {report['evaluation_info']['timestamp']}

## Overview

This report presents the evaluation results for trained models:
- PPO (baseline)
- PPO + EBM Reward Shaping
- SAC (baseline)
- SAC + EBM Reward Shaping

## Results Summary

| Method | Mean Reward | Std Reward | Success Rate | Mean Length |
|--------|-------------|------------|--------------|-------------|
"""
    
    for method_name, stats in report['results_summary'].items():
        md += f"| {method_name} | {stats['mean_reward']:.2f} | {stats['std_reward']:.2f} | {stats['success_rate']:.3f} | {stats['mean_length']:.1f} |\n"
    
    md += f"""
## Recommendations

"""
    
    for rec in report['recommendations']:
        md += f"- {rec}\n"
    
    md += """
## Generated Files

- `performance_comparison.png`: Performance comparison plots
- `learning_curves_comparison.png`: Learning curve comparison (if available)
- `statistical_analysis.png`: Statistical significance tests
- `evaluation_report.json`: Detailed results in JSON format
- `statistical_analysis.json`: Statistical test results

"""
    
    return md

def main():
    """Main function to run model evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate trained PPO, PPO+EBM, SAC, SAC+EBM models')
    parser.add_argument('--model-dirs', type=str, nargs='+', required=True,
                       help='Directories containing trained models')
    parser.add_argument('--env', type=str, default='walker2d-v2',
                       help='Environment name (default: walker2d-v2)')
    parser.add_argument('--n-episodes', type=int, default=100,
                       help='Number of episodes to evaluate (default: 100)')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory for results (default: ./evaluation_results)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation (default: cuda)')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes during evaluation')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting model evaluation")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Define method names
    method_names = ['ppo', 'ppo_ebm', 'sac', 'sac_ebm']
    
    evaluation_results = {}
    
    # Evaluate each model
    for model_dir, method in zip(args.model_dirs, method_names):
        logger.info(f"Evaluating {method} model from {model_dir}")
        
        # Find the best checkpoint (latest or best performing)
        model_dir = Path(model_dir)
        checkpoint_dir = model_dir / 'checkpoint'
        
        if checkpoint_dir.exists():
            # Find the latest checkpoint
            checkpoints = list(checkpoint_dir.glob('state_*.pt'))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[1]))
                logger.info(f"Using checkpoint: {latest_checkpoint}")
                
                # Load model
                model = load_trained_model(latest_checkpoint, args.device)
                
                if model is not None:
                    # Evaluate model
                    metrics = evaluate_model(model, args.env, args.n_episodes, args.render)
                    evaluation_results[method] = metrics
                else:
                    logger.error(f"Failed to load model for {method}")
            else:
                logger.error(f"No checkpoints found in {checkpoint_dir}")
        else:
            logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Compare methods
    if evaluation_results:
        logger.info("Generating comparison plots and reports")
        compare_methods(evaluation_results, output_dir)
        
        logger.info(f"Evaluation completed. Results saved to {output_dir}")
        logger.info("Generated files:")
        logger.info(f"  - {output_dir}/performance_comparison.png")
        logger.info(f"  - {output_dir}/learning_curves_comparison.png")
        logger.info(f"  - {output_dir}/statistical_analysis.png")
        logger.info(f"  - {output_dir}/evaluation_report.json")
        logger.info(f"  - {output_dir}/evaluation_report.md")
    else:
        logger.error("No models were successfully evaluated")

if __name__ == '__main__':
    main()
