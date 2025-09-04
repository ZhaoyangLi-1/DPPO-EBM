#!/usr/bin/env python3
"""
Script to view evaluation results from Simple PPO evaluation.
"""

import pickle
import numpy as np
import argparse
import os
import glob

def load_and_display_results(pkl_path):
    """Load and display results from a pickle file."""
    print(f"\n📊 Loading results from: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    
    stats = results['stats']
    episode_rewards = results['episode_rewards']
    episode_lengths = results['episode_lengths']
    config = results['config']
    model_path = results['model_path']
    
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Model Path: {model_path}")
    print(f"Config Name: {config.get('name', 'N/A')}")
    print(f"Use EBM Reward: {config.get('use_ebm_reward', 'N/A')}")
    print(f"Seed: {config.get('seed', 'N/A')}")
    print("-" * 80)
    
    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Total steps: {stats['total_steps']}")
    print(f"Mean episode reward: {stats['mean_episode_reward']:.3f} ± {stats['std_episode_reward']:.3f}")
    print(f"Episode reward range: [{stats['min_episode_reward']:.3f}, {stats['max_episode_reward']:.3f}]")
    print(f"Median episode reward: {stats['median_episode_reward']:.3f}")
    print(f"Mean episode length: {stats['mean_episode_length']:.1f} ± {stats['std_episode_length']:.1f}")
    
    if 'success_rate' in stats:
        print(f"Success rate: {stats['success_rate']:.3f}")
    
    # Additional statistics
    print("\n📈 Additional Statistics:")
    print(f"25th percentile reward: {np.percentile(episode_rewards, 25):.3f}")
    print(f"75th percentile reward: {np.percentile(episode_rewards, 75):.3f}")
    print(f"95th percentile reward: {np.percentile(episode_rewards, 95):.3f}")
    print(f"Coefficient of Variation: {stats['std_episode_reward'] / stats['mean_episode_reward'] * 100:.2f}%")
    
    print("=" * 80)
    return results

def compare_results(pkl_paths):
    """Compare results from multiple pickle files."""
    all_results = []
    for pkl_path in pkl_paths:
        results = load_and_display_results(pkl_path)
        all_results.append(results)
    
    if len(all_results) > 1:
        print("\n🔍 COMPARISON SUMMARY")
        print("=" * 80)
        for i, results in enumerate(all_results):
            stats = results['stats']
            config = results['config']
            model_type = "EBM" if config.get('use_ebm_reward', False) else "ENV"
            seed = config.get('seed', 'N/A')
            print(f"Model {i+1} ({model_type}, seed={seed}): {stats['mean_episode_reward']:.3f} ± {stats['std_episode_reward']:.3f}")
        print("=" * 80)

def find_eval_results(base_dir="/scr/zhaoyang/projects/DPPO-EBM/log/gym-eval"):
    """Find all evaluation result files."""
    pattern = os.path.join(base_dir, "**/eval_results.pkl")
    pkl_files = glob.glob(pattern, recursive=True)
    
    if pkl_files:
        print(f"\n🔍 Found {len(pkl_files)} evaluation result files:")
        for i, pkl_file in enumerate(pkl_files, 1):
            # Extract info from path
            path_parts = pkl_file.split('/')
            if len(path_parts) >= 3:
                experiment_name = path_parts[-3]  # hopper-medium-v2_eval_simple_ppo_env_ta4
                timestamp = path_parts[-2]        # 2025-09-04_01-47-03_42
                print(f"  {i}. {experiment_name} ({timestamp})")
        return pkl_files
    else:
        print("❌ No evaluation result files found!")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View Simple PPO evaluation results")
    parser.add_argument("--pkl", type=str, help="Path to specific pickle file")
    parser.add_argument("--compare", nargs="+", help="Compare multiple pickle files")
    parser.add_argument("--find", action="store_true", help="Find all evaluation results")
    
    args = parser.parse_args()
    
    if args.find:
        pkl_files = find_eval_results()
        if pkl_files:
            print("\nUse --pkl <path> to view specific results or --compare <path1> <path2> ... to compare")
    elif args.compare:
        compare_results(args.compare)
    elif args.pkl:
        load_and_display_results(args.pkl)
    else:
        print("Usage examples:")
        print("  python view_eval_results.py --find")
        print("  python view_eval_results.py --pkl /path/to/eval_results.pkl")
        print("  python view_eval_results.py --compare result1.pkl result2.pkl")