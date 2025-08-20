#!/usr/bin/env python3
"""
Learning Speed and Stability Analysis for RL Methods

This script analyzes how fast and stable different RL methods learn,
comparing PPO, PPO+EBM, SAC, SAC+EBM.
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
import logging
from datetime import datetime

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

def load_training_data(log_dir, method, env_name, seed):
    """
    Load training data from logs.
    
    Args:
        log_dir: Logging directory
        method: Method name
        env_name: Environment name
        seed: Random seed
        
    Returns:
        Dictionary containing training metrics
    """
    logger = logging.getLogger(__name__)
    
    # Try different log file formats
    log_paths = [
        log_dir / f"{method}_{env_name}_{seed}" / "training.log",
        log_dir / f"{method}_{env_name}_{seed}" / "wandb" / "latest-run" / "files" / "wandb-events.jsonl",
        log_dir / f"{method}_{env_name}_{seed}" / "tensorboard" / "events.out.tfevents.*"
    ]
    
    training_data = {
        'steps': [],
        'rewards': [],
        'eval_rewards': [],
        'success_rates': [],
        'losses': [],
        'timestamps': []
    }
    
    # Try to load from different sources
    for log_path in log_paths:
        if log_path.exists():
            try:
                if log_path.suffix == '.log':
                    # Parse text log file
                    training_data = parse_text_log(log_path)
                elif log_path.suffix == '.jsonl':
                    # Parse wandb log
                    training_data = parse_wandb_log(log_path)
                elif 'tfevents' in str(log_path):
                    # Parse tensorboard log
                    training_data = parse_tensorboard_log(log_path)
                
                if training_data['steps']:
                    logger.info(f"Loaded training data from {log_path}")
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to parse {log_path}: {e}")
                continue
    
    return training_data

def parse_text_log(log_path):
    """Parse text-based training log."""
    training_data = {
        'steps': [],
        'rewards': [],
        'eval_rewards': [],
        'success_rates': [],
        'losses': [],
        'timestamps': []
    }
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                # Parse different log formats
                if 'episode_reward' in line and 'step' in line:
                    # Extract step and reward
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'step':
                            step = int(parts[i+1])
                            training_data['steps'].append(step)
                        elif part == 'episode_reward':
                            reward = float(parts[i+1])
                            training_data['rewards'].append(reward)
                            
                elif 'eval/return_mean' in line:
                    # Extract evaluation reward
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'eval/return_mean':
                            eval_reward = float(parts[i+1])
                            training_data['eval_rewards'].append(eval_reward)
                            
                elif 'eval/success_rate' in line:
                    # Extract success rate
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'eval/success_rate':
                            success_rate = float(parts[i+1])
                            training_data['success_rates'].append(success_rate)
    except Exception as e:
        logging.warning(f"Error parsing text log: {e}")
    
    return training_data

def parse_wandb_log(log_path):
    """Parse wandb log file."""
    training_data = {
        'steps': [],
        'rewards': [],
        'eval_rewards': [],
        'success_rates': [],
        'losses': [],
        'timestamps': []
    }
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'step' in data:
                    training_data['steps'].append(data['step'])
                if 'episode_reward' in data:
                    training_data['rewards'].append(data['episode_reward'])
                if 'eval/return_mean' in data:
                    training_data['eval_rewards'].append(data['eval/return_mean'])
                if 'eval/success_rate' in data:
                    training_data['success_rates'].append(data['eval/success_rate'])
    except Exception as e:
        logging.warning(f"Error parsing wandb log: {e}")
    
    return training_data

def parse_tensorboard_log(log_path):
    """Parse tensorboard log file."""
    # This is a placeholder - implement tensorboard parsing if needed
    return {
        'steps': [],
        'rewards': [],
        'eval_rewards': [],
        'success_rates': [],
        'losses': [],
        'timestamps': []
    }

def calculate_learning_speed_metrics(training_data, target_performance=0.8):
    """
    Calculate learning speed metrics.
    
    Args:
        training_data: Training data dictionary
        target_performance: Target performance threshold (0-1)
        
    Returns:
        Dictionary containing learning speed metrics
    """
    if not training_data['steps'] or not training_data['rewards']:
        return None
    
    steps = np.array(training_data['steps'])
    rewards = np.array(training_data['rewards'])
    
    # Normalize rewards to 0-1 scale (assuming max reward is known)
    max_reward = np.max(rewards) if len(rewards) > 0 else 1.0
    normalized_rewards = rewards / max_reward
    
    metrics = {}
    
    # 1. Time to target performance
    target_reached = normalized_rewards >= target_performance
    if np.any(target_reached):
        first_target_step = steps[np.where(target_reached)[0][0]]
        metrics['steps_to_target'] = first_target_step
        metrics['target_performance'] = target_performance
    else:
        metrics['steps_to_target'] = None
        metrics['target_performance'] = target_performance
    
    # 2. Learning rate (slope of reward curve)
    if len(rewards) > 10:
        # Use linear regression on smoothed curve
        smoothed_rewards = savgol_filter(rewards, min(11, len(rewards)//2), 3)
        slope, intercept, r_value, p_value, std_err = stats.linregress(steps, smoothed_rewards)
        metrics['learning_rate'] = slope
        metrics['learning_rate_r2'] = r_value**2
    else:
        metrics['learning_rate'] = None
        metrics['learning_rate_r2'] = None
    
    # 3. Initial learning speed (first 20% of training)
    if len(rewards) > 5:
        initial_idx = max(1, len(rewards) // 5)
        initial_slope, _, _, _, _ = stats.linregress(steps[:initial_idx], rewards[:initial_idx])
        metrics['initial_learning_rate'] = initial_slope
    else:
        metrics['initial_learning_rate'] = None
    
    # 4. Convergence speed (time to reach 90% of final performance)
    if len(rewards) > 10:
        final_performance = np.mean(rewards[-len(rewards)//10:])  # Last 10%
        convergence_threshold = 0.9 * final_performance
        convergence_reached = rewards >= convergence_threshold
        if np.any(convergence_reached):
            convergence_step = steps[np.where(convergence_reached)[0][0]]
            metrics['steps_to_convergence'] = convergence_step
            metrics['convergence_threshold'] = convergence_threshold
        else:
            metrics['steps_to_convergence'] = None
    else:
        metrics['steps_to_convergence'] = None
    
    return metrics

def calculate_stability_metrics(training_data):
    """
    Calculate stability metrics.
    
    Args:
        training_data: Training data dictionary
        
    Returns:
        Dictionary containing stability metrics
    """
    if not training_data['rewards']:
        return None
    
    rewards = np.array(training_data['rewards'])
    
    metrics = {}
    
    # 1. Reward variance
    metrics['reward_variance'] = np.var(rewards)
    metrics['reward_std'] = np.std(rewards)
    
    # 2. Coefficient of variation (CV = std/mean)
    if np.mean(rewards) != 0:
        metrics['coefficient_of_variation'] = np.std(rewards) / np.abs(np.mean(rewards))
    else:
        metrics['coefficient_of_variation'] = None
    
    # 3. Reward stability (using rolling standard deviation)
    if len(rewards) > 20:
        window_size = min(20, len(rewards) // 4)
        rolling_std = pd.Series(rewards).rolling(window=window_size).std()
        metrics['mean_rolling_std'] = np.mean(rolling_std.dropna())
        metrics['max_rolling_std'] = np.max(rolling_std.dropna())
    else:
        metrics['mean_rolling_std'] = None
        metrics['max_rolling_std'] = None
    
    # 4. Performance degradation (check if performance decreases significantly)
    if len(rewards) > 50:
        # Split into quarters and compare
        quarter_size = len(rewards) // 4
        q1_mean = np.mean(rewards[:quarter_size])
        q4_mean = np.mean(rewards[-quarter_size:])
        degradation = (q1_mean - q4_mean) / q1_mean if q1_mean != 0 else 0
        metrics['performance_degradation'] = degradation
    else:
        metrics['performance_degradation'] = None
    
    # 5. Monotonicity (how often performance increases)
    if len(rewards) > 1:
        increases = np.sum(np.diff(rewards) > 0)
        total_changes = len(rewards) - 1
        metrics['monotonicity_ratio'] = increases / total_changes
    else:
        metrics['monotonicity_ratio'] = None
    
    # 6. Catastrophic forgetting (large performance drops)
    if len(rewards) > 10:
        # Detect large drops (>50% of current performance)
        drops = np.diff(rewards)
        large_drops = np.sum(drops < -0.5 * np.abs(rewards[:-1]))
        metrics['catastrophic_drops'] = large_drops
        metrics['catastrophic_drop_ratio'] = large_drops / (len(rewards) - 1)
    else:
        metrics['catastrophic_drops'] = None
        metrics['catastrophic_drop_ratio'] = None
    
    return metrics

def analyze_across_seeds(log_dir, method, env_name, seeds):
    """
    Analyze learning speed and stability across multiple seeds.
    
    Args:
        log_dir: Logging directory
        method: Method name
        env_name: Environment name
        seeds: List of seeds
        
    Returns:
        Dictionary containing aggregated metrics
    """
    logger = logging.getLogger(__name__)
    
    all_speed_metrics = []
    all_stability_metrics = []
    all_training_data = []
    
    for seed in seeds:
        training_data = load_training_data(log_dir, method, env_name, seed)
        if training_data['steps']:
            all_training_data.append(training_data)
            
            speed_metrics = calculate_learning_speed_metrics(training_data)
            stability_metrics = calculate_stability_metrics(training_data)
            
            if speed_metrics:
                speed_metrics['seed'] = seed
                all_speed_metrics.append(speed_metrics)
            
            if stability_metrics:
                stability_metrics['seed'] = seed
                all_stability_metrics.append(stability_metrics)
    
    # Aggregate metrics across seeds
    aggregated_metrics = {
        'method': method,
        'env_name': env_name,
        'n_seeds': len(all_speed_metrics),
        'speed_metrics': {},
        'stability_metrics': {},
        'training_data': all_training_data
    }
    
    if all_speed_metrics:
        # Aggregate speed metrics
        for key in all_speed_metrics[0].keys():
            if key != 'seed':
                values = [m[key] for m in all_speed_metrics if m[key] is not None]
                if values:
                    aggregated_metrics['speed_metrics'][f'{key}_mean'] = np.mean(values)
                    aggregated_metrics['speed_metrics'][f'{key}_std'] = np.std(values)
                    aggregated_metrics['speed_metrics'][f'{key}_min'] = np.min(values)
                    aggregated_metrics['speed_metrics'][f'{key}_max'] = np.max(values)
    
    if all_stability_metrics:
        # Aggregate stability metrics
        for key in all_stability_metrics[0].keys():
            if key != 'seed':
                values = [m[key] for m in all_stability_metrics if m[key] is not None]
                if values:
                    aggregated_metrics['stability_metrics'][f'{key}_mean'] = np.mean(values)
                    aggregated_metrics['stability_metrics'][f'{key}_std'] = np.std(values)
                    aggregated_metrics['stability_metrics'][f'{key}_min'] = np.min(values)
                    aggregated_metrics['stability_metrics'][f'{key}_max'] = np.max(values)
    
    logger.info(f"Analyzed {method} across {len(all_speed_metrics)} seeds")
    return aggregated_metrics

def create_learning_speed_plots(results, output_dir):
    """
    Create learning speed comparison plots.
    
    Args:
        results: Dictionary containing results for all methods
        output_dir: Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    methods = ['ppo', 'ppo_ebm', 'sac', 'sac_ebm']
    method_names = ['PPO', 'PPO+EBM', 'SAC', 'SAC+EBM']
    
    # 1. Learning curves comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (method, method_name) in enumerate(zip(methods, method_names)):
        if method in results and results[method]['training_data']:
            ax = axes[i]
            
            # Plot all seeds
            for seed_data in results[method]['training_data']:
                if seed_data['steps'] and seed_data['rewards']:
                    # Smooth the curve
                    if len(seed_data['rewards']) > 10:
                        smoothed_rewards = savgol_filter(seed_data['rewards'], 
                                                       min(11, len(seed_data['rewards'])//2), 3)
                    else:
                        smoothed_rewards = seed_data['rewards']
                    
                    ax.plot(seed_data['steps'], smoothed_rewards, alpha=0.3, linewidth=1)
            
            # Plot mean curve
            if results[method]['training_data']:
                all_steps = []
                all_rewards = []
                for seed_data in results[method]['training_data']:
                    if seed_data['steps'] and seed_data['rewards']:
                        all_steps.extend(seed_data['steps'])
                        all_rewards.extend(seed_data['rewards'])
                
                if all_steps:
                    # Create binned average
                    df = pd.DataFrame({'steps': all_steps, 'rewards': all_rewards})
                    df['step_bin'] = pd.cut(df['steps'], bins=50)
                    binned = df.groupby('step_bin').agg({'rewards': 'mean', 'steps': 'mean'})
                    
                    ax.plot(binned['steps'], binned['rewards'], linewidth=3, 
                           label=f'{method_name} (mean)', color='red')
            
            ax.set_title(f'{method_name} Learning Curve')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Episode Reward')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Learning speed metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Steps to target performance
    ax = axes[0]
    target_steps = []
    method_labels = []
    
    for method, method_name in zip(methods, method_names):
        if method in results and 'steps_to_target' in results[method]['speed_metrics']:
            mean_steps = results[method]['speed_metrics']['steps_to_target_mean']
            if mean_steps is not None:
                target_steps.append(mean_steps)
                method_labels.append(method_name)
    
    if target_steps:
        bars = ax.bar(method_labels, target_steps)
        ax.set_title('Steps to Target Performance')
        ax.set_ylabel('Training Steps')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, target_steps):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(target_steps)*0.01,
                   f'{value:.0f}', ha='center', va='bottom')
    
    # Learning rate comparison
    ax = axes[1]
    learning_rates = []
    method_labels = []
    
    for method, method_name in zip(methods, method_names):
        if method in results and 'learning_rate' in results[method]['speed_metrics']:
            mean_rate = results[method]['speed_metrics']['learning_rate_mean']
            if mean_rate is not None:
                learning_rates.append(mean_rate)
                method_labels.append(method_name)
    
    if learning_rates:
        bars = ax.bar(method_labels, learning_rates)
        ax.set_title('Learning Rate (Slope)')
        ax.set_ylabel('Reward per Step')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, learning_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(learning_rates)*0.01,
                   f'{value:.4f}', ha='center', va='bottom')
    
    # Stability metrics
    ax = axes[2]
    stability_scores = []
    method_labels = []
    
    for method, method_name in zip(methods, method_names):
        if method in results and 'coefficient_of_variation' in results[method]['stability_metrics']:
            cv = results[method]['stability_metrics']['coefficient_of_variation_mean']
            if cv is not None:
                # Lower CV is better (more stable)
                stability_scores.append(1.0 / (1.0 + cv))  # Convert to stability score
                method_labels.append(method_name)
    
    if stability_scores:
        bars = ax.bar(method_labels, stability_scores)
        ax.set_title('Stability Score (1/(1+CV))')
        ax.set_ylabel('Stability Score')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, stability_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stability_scores)*0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    # Monotonicity ratio
    ax = axes[3]
    monotonicity_ratios = []
    method_labels = []
    
    for method, method_name in zip(methods, method_names):
        if method in results and 'monotonicity_ratio' in results[method]['stability_metrics']:
            ratio = results[method]['stability_metrics']['monotonicity_ratio_mean']
            if ratio is not None:
                monotonicity_ratios.append(ratio)
                method_labels.append(method_name)
    
    if monotonicity_ratios:
        bars = ax.bar(method_labels, monotonicity_ratios)
        ax.set_title('Monotonicity Ratio')
        ax.set_ylabel('Ratio of Performance Increases')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, monotonicity_ratios):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_speed_stability_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_learning_analysis_report(results, output_dir):
    """
    Generate comprehensive learning speed and stability report.
    
    Args:
        results: Dictionary containing results for all methods
        output_dir: Output directory for report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    methods = ['ppo', 'ppo_ebm', 'sac', 'sac_ebm']
    method_names = ['PPO', 'PPO+EBM', 'SAC', 'SAC+EBM']
    
    report = {
        'analysis_info': {
            'timestamp': datetime.now().isoformat(),
            'methods_analyzed': list(results.keys()),
            'total_seeds': sum(results[m]['n_seeds'] for m in results if m in results)
        },
        'learning_speed_ranking': {},
        'stability_ranking': {},
        'recommendations': []
    }
    
    # Rank methods by learning speed
    speed_scores = {}
    for method, method_name in zip(methods, method_names):
        if method in results:
            score = 0
            if 'steps_to_target_mean' in results[method]['speed_metrics']:
                steps = results[method]['speed_metrics']['steps_to_target_mean']
                if steps is not None:
                    score += 1.0 / (1.0 + steps / 1000000)  # Normalize
            
            if 'learning_rate_mean' in results[method]['speed_metrics']:
                rate = results[method]['speed_metrics']['learning_rate_mean']
                if rate is not None:
                    score += rate * 1000000  # Scale up
            
            speed_scores[method_name] = score
    
    # Rank methods by stability
    stability_scores = {}
    for method, method_name in zip(methods, method_names):
        if method in results:
            score = 0
            if 'coefficient_of_variation_mean' in results[method]['stability_metrics']:
                cv = results[method]['stability_metrics']['coefficient_of_variation_mean']
                if cv is not None:
                    score += 1.0 / (1.0 + cv)  # Lower CV is better
            
            if 'monotonicity_ratio_mean' in results[method]['stability_metrics']:
                ratio = results[method]['stability_metrics']['monotonicity_ratio_mean']
                if ratio is not None:
                    score += ratio
            
            stability_scores[method_name] = score
    
    # Sort rankings
    speed_ranking = sorted(speed_scores.items(), key=lambda x: x[1], reverse=True)
    stability_ranking = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
    
    report['learning_speed_ranking'] = dict(speed_ranking)
    report['stability_ranking'] = dict(stability_ranking)
    
    # Generate recommendations
    fastest_method = speed_ranking[0][0] if speed_ranking else "Unknown"
    most_stable_method = stability_ranking[0][0] if stability_ranking else "Unknown"
    
    report['recommendations'] = [
        f"Fastest learning: {fastest_method}",
        f"Most stable: {most_stable_method}",
        "Consider EBM integration if it improves both speed and stability",
        "For production use, prioritize stability over raw speed",
        "Monitor catastrophic forgetting in long training runs"
    ]
    
    # Save report
    with open(output_dir / 'learning_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate markdown report
    markdown_report = generate_markdown_learning_report(report, results)
    with open(output_dir / 'learning_analysis_report.md', 'w') as f:
        f.write(markdown_report)
    
    return report

def generate_markdown_learning_report(report, results):
    """Generate a markdown version of the learning analysis report."""
    md = f"""# Learning Speed and Stability Analysis Report

Generated on: {report['analysis_info']['timestamp']}

## Overview

This report analyzes the learning speed and stability of four RL methods:
- PPO (baseline)
- PPO + EBM Reward Shaping
- SAC (baseline)
- SAC + EBM Reward Shaping

## Learning Speed Ranking

| Rank | Method | Speed Score |
|------|--------|-------------|
"""
    
    for i, (method, score) in enumerate(report['learning_speed_ranking'].items(), 1):
        md += f"| {i} | {method} | {score:.4f} |\n"
    
    md += f"""
## Stability Ranking

| Rank | Method | Stability Score |
|------|--------|----------------|
"""
    
    for i, (method, score) in enumerate(report['stability_ranking'].items(), 1):
        md += f"| {i} | {method} | {score:.4f} |\n"
    
    md += f"""
## Detailed Metrics

### Learning Speed Metrics

| Method | Steps to Target | Learning Rate | Convergence Steps |
|--------|-----------------|---------------|-------------------|
"""
    
    methods = ['ppo', 'ppo_ebm', 'sac', 'sac_ebm']
    method_names = ['PPO', 'PPO+EBM', 'SAC', 'SAC+EBM']
    
    for method, method_name in zip(methods, method_names):
        if method in results:
            speed_metrics = results[method]['speed_metrics']
            steps_to_target = speed_metrics.get('steps_to_target_mean', 'N/A')
            learning_rate = speed_metrics.get('learning_rate_mean', 'N/A')
            convergence_steps = speed_metrics.get('steps_to_convergence_mean', 'N/A')
            
            md += f"| {method_name} | {steps_to_target:.0f if steps_to_target else 'N/A'} | {learning_rate:.6f if learning_rate else 'N/A'} | {convergence_steps:.0f if convergence_steps else 'N/A'} |\n"
    
    md += f"""
### Stability Metrics

| Method | CV | Monotonicity | Catastrophic Drops |
|--------|----|--------------|-------------------|
"""
    
    for method, method_name in zip(methods, method_names):
        if method in results:
            stability_metrics = results[method]['stability_metrics']
            cv = stability_metrics.get('coefficient_of_variation_mean', 'N/A')
            monotonicity = stability_metrics.get('monotonicity_ratio_mean', 'N/A')
            catastrophic_drops = stability_metrics.get('catastrophic_drop_ratio_mean', 'N/A')
            
            md += f"| {method_name} | {cv:.4f if cv else 'N/A'} | {monotonicity:.3f if monotonicity else 'N/A'} | {catastrophic_drops:.3f if catastrophic_drops else 'N/A'} |\n"
    
    md += f"""
## Recommendations

"""
    
    for rec in report['recommendations']:
        md += f"- {rec}\n"
    
    md += """
## Generated Files

- `learning_curves_comparison.png`: Learning curves for all methods
- `learning_speed_stability_metrics.png`: Comparison of key metrics
- `learning_analysis_report.json`: Detailed results in JSON format

## Key Insights

1. **Learning Speed**: Measures how quickly methods reach target performance
2. **Stability**: Measures consistency and reliability of learning
3. **EBM Impact**: Assesses whether EBM integration improves both metrics
4. **Algorithm Comparison**: Compares on-policy (PPO) vs off-policy (SAC) approaches

"""
    
    return md

def main():
    """Main function for learning speed and stability analysis."""
    parser = argparse.ArgumentParser(description='Analyze learning speed and stability of RL methods')
    parser.add_argument('--log-dir', type=str, default='./outputs',
                       help='Directory containing training logs (default: ./outputs)')
    parser.add_argument('--env', type=str, default='walker2d-v2',
                       help='Environment name (default: walker2d-v2)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                       help='Random seeds to analyze (default: [42, 123, 456])')
    parser.add_argument('--output-dir', type=str, default='./learning_analysis',
                       help='Output directory for analysis results (default: ./learning_analysis)')
    parser.add_argument('--target-performance', type=float, default=0.8,
                       help='Target performance threshold for speed analysis (default: 0.8)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting learning speed and stability analysis")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Define methods to analyze
    methods = ['ppo', 'ppo_ebm', 'sac', 'sac_ebm']
    method_names = ['PPO', 'PPO+EBM', 'SAC', 'SAC+EBM']
    
    results = {}
    
    # Analyze each method
    for method in methods:
        logger.info(f"Analyzing {method}")
        result = analyze_across_seeds(args.log_dir, method, args.env, args.seeds)
        if result['n_seeds'] > 0:
            results[method] = result
            logger.info(f"  - Analyzed {result['n_seeds']} seeds")
        else:
            logger.warning(f"  - No valid data found for {method}")
    
    if results:
        # Create plots
        logger.info("Creating learning speed and stability plots")
        create_learning_speed_plots(results, output_dir)
        
        # Generate report
        logger.info("Generating analysis report")
        report = generate_learning_analysis_report(results, output_dir)
        
        logger.info(f"Analysis completed. Results saved to {output_dir}")
        logger.info("Generated files:")
        logger.info(f"  - {output_dir}/learning_curves_comparison.png")
        logger.info(f"  - {output_dir}/learning_speed_stability_metrics.png")
        logger.info(f"  - {output_dir}/learning_analysis_report.json")
        logger.info(f"  - {output_dir}/learning_analysis_report.md")
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("LEARNING SPEED AND STABILITY SUMMARY")
        logger.info("=" * 60)
        
        if report['learning_speed_ranking']:
            fastest = list(report['learning_speed_ranking'].keys())[0]
            logger.info(f"Fastest learning: {fastest}")
        
        if report['stability_ranking']:
            most_stable = list(report['stability_ranking'].keys())[0]
            logger.info(f"Most stable: {most_stable}")
        
    else:
        logger.error("No valid results found. Check log directory and seed values.")

if __name__ == '__main__':
    main()
