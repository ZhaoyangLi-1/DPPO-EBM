"""
Monitor PPO training and provide real-time diagnostics.

This script helps monitor PPO training and provides alerts when issues are detected.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
import time
import pickle

def load_training_results(log_dir):
    """Load training results from pickle files."""
    results_files = list(Path(log_dir).rglob("result.pkl"))
    
    if not results_files:
        return None
    
    # Load the most recent results file
    latest_file = max(results_files, key=lambda x: x.stat().st_mtime)
    
    try:
        with open(latest_file, 'rb') as f:
            results = pickle.load(f)
        return results, latest_file
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def analyze_training_metrics(results):
    """Analyze training metrics and detect issues."""
    if not results:
        return {"status": "No data"}
    
    # Extract metrics
    iterations = []
    train_rewards = []
    eval_rewards = []
    
    for result in results:
        iterations.append(result.get('itr', 0))
        if 'train_episode_reward' in result:
            train_rewards.append(result['train_episode_reward'])
        if 'eval_episode_reward' in result:
            eval_rewards.append(result['eval_episode_reward'])
    
    analysis = {
        "status": "OK",
        "iterations": len(iterations),
        "issues": [],
        "recommendations": []
    }
    
    # Check for decreasing rewards
    if len(train_rewards) >= 5:
        recent_trend = np.polyfit(range(len(train_rewards[-5:])), train_rewards[-5:], 1)[0]
        if recent_trend < -50:  # Significant negative trend
            analysis["issues"].append(f"Training rewards decreasing (slope: {recent_trend:.1f})")
            analysis["recommendations"].append("Consider lowering learning rates or increasing entropy coefficient")
    
    # Check for reward collapse
    if len(train_rewards) >= 3:
        if all(r < 100 for r in train_rewards[-3:]) and len(train_rewards) > 10:
            analysis["issues"].append("Potential reward collapse detected")
            analysis["recommendations"].append("Restart with more conservative hyperparameters")
    
    # Check for stagnation
    if len(train_rewards) >= 10:
        recent_var = np.var(train_rewards[-10:])
        if recent_var < 100:  # Very low variance
            analysis["issues"].append("Training appears to have stagnated")
            analysis["recommendations"].append("Increase exploration (entropy coefficient or learning rates)")
    
    analysis["train_rewards"] = train_rewards
    analysis["eval_rewards"] = eval_rewards
    
    return analysis

def plot_training_curves(results, save_path=None):
    """Plot training curves."""
    if not results or not results.get("train_rewards"):
        print("No training data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("PPO Training Monitoring", fontsize=16)
    
    # Training rewards
    train_rewards = results["train_rewards"]
    axes[0, 0].plot(train_rewards, 'b-', alpha=0.7, label='Training Reward')
    axes[0, 0].set_title("Training Reward")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add trend line
    if len(train_rewards) >= 3:
        z = np.polyfit(range(len(train_rewards)), train_rewards, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(range(len(train_rewards)), p(range(len(train_rewards))), 
                       'r--', alpha=0.8, label=f'Trend (slope: {z[0]:.1f})')
        axes[0, 0].legend()
    
    # Evaluation rewards
    eval_rewards = results.get("eval_rewards", [])
    if eval_rewards:
        axes[0, 1].plot(eval_rewards, 'g-', alpha=0.7, label='Eval Reward')
        axes[0, 1].set_title("Evaluation Reward")
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("Reward")
        axes[0, 1].grid(True, alpha=0.3)
    
    # Moving average of training rewards
    if len(train_rewards) >= 5:
        window = min(10, len(train_rewards) // 3)
        moving_avg = pd.Series(train_rewards).rolling(window=window, min_periods=1).mean()
        axes[1, 0].plot(train_rewards, 'b-', alpha=0.3, label='Raw')
        axes[1, 0].plot(moving_avg, 'b-', linewidth=2, label=f'MA({window})')
        axes[1, 0].set_title("Smoothed Training Reward")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("Reward")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Recent performance
    if len(train_rewards) >= 10:
        recent_rewards = train_rewards[-10:]
        axes[1, 1].bar(range(len(recent_rewards)), recent_rewards, alpha=0.7)
        axes[1, 1].set_title("Last 10 Training Rewards")
        axes[1, 1].set_xlabel("Recent Iterations")
        axes[1, 1].set_ylabel("Reward")
        axes[1, 1].axhline(y=np.mean(recent_rewards), color='r', linestyle='--', 
                          label=f'Mean: {np.mean(recent_rewards):.1f}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig

def print_status_report(analysis, results_file):
    """Print a comprehensive status report."""
    print("=" * 60)
    print("PPO TRAINING STATUS REPORT")
    print("=" * 60)
    print(f"Results file: {results_file}")
    print(f"Iterations processed: {analysis['iterations']}")
    print(f"Status: {analysis['status']}")
    print()
    
    if analysis['issues']:
        print("⚠️  ISSUES DETECTED:")
        for i, issue in enumerate(analysis['issues'], 1):
            print(f"  {i}. {issue}")
        print()
        
        print("💡 RECOMMENDATIONS:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
        print()
    else:
        print("✅ No major issues detected!")
        print()
    
    # Performance summary
    if analysis.get('train_rewards'):
        rewards = analysis['train_rewards']
        print("📊 TRAINING PERFORMANCE:")
        print(f"  Latest reward: {rewards[-1]:.1f}")
        print(f"  Best reward: {max(rewards):.1f}")
        print(f"  Average reward: {np.mean(rewards):.1f}")
        print(f"  Reward std: {np.std(rewards):.1f}")
        
        if len(rewards) >= 5:
            recent_avg = np.mean(rewards[-5:])
            early_avg = np.mean(rewards[:5]) if len(rewards) > 5 else np.mean(rewards)
            improvement = recent_avg - early_avg
            print(f"  Improvement (recent vs early): {improvement:+.1f}")

def main():
    parser = argparse.ArgumentParser(description="Monitor PPO training")
    parser.add_argument("--log-dir", "-l", 
                       default="/linting-slow-vol/DPPO-EBM/log/gym-finetune",
                       help="Training log directory")
    parser.add_argument("--env", "-e", default="hopper-medium-v2",
                       help="Environment name to monitor")
    parser.add_argument("--continuous", "-c", action="store_true",
                       help="Continuous monitoring mode")
    parser.add_argument("--interval", "-i", type=int, default=30,
                       help="Monitoring interval in seconds")
    parser.add_argument("--plot", "-p", action="store_true",
                       help="Generate and save plots")
    
    args = parser.parse_args()
    
    # Look for the most recent Simple MLP PPO run
    log_pattern = f"{args.env}_simple_mlp_ppo_env*"
    log_dirs = list(Path(args.log_dir).glob(log_pattern))
    
    if not log_dirs:
        print(f"No training logs found for pattern: {log_pattern}")
        print(f"Searched in: {args.log_dir}")
        return
    
    # Get the most recent log directory
    latest_log_dir = max(log_dirs, key=lambda x: x.stat().st_mtime)
    print(f"Monitoring: {latest_log_dir}")
    
    if args.continuous:
        print(f"Continuous monitoring every {args.interval} seconds. Press Ctrl+C to stop.")
        print()
        
        try:
            while True:
                result = load_training_results(latest_log_dir)
                if result:
                    results, results_file = result
                    analysis = analyze_training_metrics(results)
                    
                    # Clear screen for continuous monitoring
                    os.system('clear' if os.name == 'posix' else 'cls')
                    print_status_report(analysis, results_file)
                    
                    if args.plot:
                        plot_path = latest_log_dir / "training_monitor.png"
                        plot_training_curves(analysis, plot_path)
                else:
                    print("No training data found yet...")
                
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        # One-time monitoring
        result = load_training_results(latest_log_dir)
        if result:
            results, results_file = result
            analysis = analyze_training_metrics(results)
            print_status_report(analysis, results_file)
            
            if args.plot:
                plot_path = latest_log_dir / "training_monitor.png"
                fig = plot_training_curves(analysis, plot_path)
                if fig and not args.continuous:
                    plt.show()
        else:
            print("No training data found.")

if __name__ == "__main__":
    main()