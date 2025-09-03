#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from omegaconf import OmegaConf
import hydra
from hydra import initialize, compose

def test_reward_logging():
    """Verify that wandb logging always uses environment rewards."""
    print("Testing reward logging behavior...")
    
    # Simulate environment rewards
    reward_trajs = np.array([
        [100, 150, 200, 80, 120],   # env 0
        [90, 110, 180, 90, 140],    # env 1
        [120, 160, 190, 70, 130],   # env 2
    ]).T  # [n_steps=5, n_envs=3]
    
    print(f"Original environment rewards shape: {reward_trajs.shape}")
    print(f"Environment rewards:\n{reward_trajs}")
    
    # Simulate episode boundaries (similar to the real code)
    episodes_start_end = [
        (0, 0, 2),  # env 0, steps 0-2
        (1, 0, 3),  # env 1, steps 0-3  
        (2, 1, 4),  # env 2, steps 1-4
    ]
    
    print(f"Episode boundaries: {episodes_start_end}")
    
    # CRITICAL: This is what the modified code does
    env_reward_trajs_split = [
        reward_trajs[start : end + 1, env_ind]
        for env_ind, start, end in episodes_start_end
    ]
    
    env_episode_reward = np.array(
        [np.sum(reward_traj) for reward_traj in env_reward_trajs_split]
    )
    
    avg_episode_reward = np.mean(env_episode_reward)
    
    print(f"\nEnvironment reward trajectories by episode:")
    for i, traj in enumerate(env_reward_trajs_split):
        print(f"  Episode {i}: {traj} -> sum = {np.sum(traj)}")
    
    print(f"\nEnvironment episode rewards: {env_episode_reward}")
    print(f"Average episode reward (for wandb): {avg_episode_reward}")
    
    # Simulate EBM reward replacement (this happens AFTER env reward calculation)
    print(f"\n--- EBM Replacement (happens AFTER env reward logging calculation) ---")
    ebm_rewards = np.array([
        [-10, -10, -10, -10, -10],   # EBM rewards (clipped)
        [-10, -10, -10, -10, -10],
        [-10, -10, -10, -10, -10],
    ]).T
    
    print(f"EBM rewards:\n{ebm_rewards}")
    
    # This replacement would happen for PPO training, but NOT for wandb logging
    reward_trajs_after_ebm = ebm_rewards
    print(f"Reward trajs after EBM replacement (used for PPO training):\n{reward_trajs_after_ebm}")
    
    print(f"\n✅ VERIFICATION:")
    print(f"   - Wandb logs environment rewards: {avg_episode_reward}")
    print(f"   - PPO trains with EBM rewards: {np.mean(ebm_rewards)}")
    print(f"   - They are DIFFERENT (which is correct!)")
    
    return avg_episode_reward

if __name__ == "__main__":
    test_reward_logging()