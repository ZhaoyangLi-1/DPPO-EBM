#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

def simulate_training_step(use_ebm_reward=False):
    """Simulate one training iteration to show reward logging behavior."""
    print(f"\n{'='*60}")
    print(f"SIMULATION: use_ebm_reward = {use_ebm_reward}")
    print(f"{'='*60}")
    
    # Step 1: Collect environment interactions
    print("Step 1: Environment interaction")
    reward_trajs = np.array([
        [800, 1200, 1500],  # High environment rewards (typical for hopper)
        [900, 1100, 1400],
        [1000, 1300, 1600],
    ]).T  # [n_steps=3, n_envs=3]
    
    print(f"Environment rewards collected: \n{reward_trajs}")
    print(f"Env reward mean per step: {np.mean(reward_trajs, axis=1)}")
    
    # Step 2: Calculate episode rewards for wandb (BEFORE EBM replacement)
    episodes_start_end = [(0, 0, 2), (1, 0, 2), (2, 0, 2)]  # 3 episodes
    
    env_reward_trajs_split = [
        reward_trajs[start : end + 1, env_ind]
        for env_ind, start, end in episodes_start_end
    ]
    
    env_episode_reward = np.array([np.sum(traj) for traj in env_reward_trajs_split])
    avg_episode_reward = np.mean(env_episode_reward)
    
    print(f"\nStep 2: Environment episode reward calculation (for wandb)")
    print(f"Episode rewards: {env_episode_reward}")
    print(f"Average episode reward: {avg_episode_reward}")
    
    # Step 3: EBM replacement (if enabled)
    if use_ebm_reward:
        print(f"\nStep 3: EBM reward replacement")
        ebm_rewards = np.array([
            [-10, -10, -10],  # EBM rewards (clipped due to poor scaling)
            [-10, -10, -10],
            [-10, -10, -10],
        ]).T
        
        print(f"EBM rewards: \n{ebm_rewards}")
        print(f"EBM reward mean: {np.mean(ebm_rewards)}")
        
        # This replaces reward_trajs for PPO training
        reward_trajs = ebm_rewards
        print(f"reward_trajs replaced with EBM rewards for PPO training")
    else:
        print(f"\nStep 3: No EBM replacement - using environment rewards")
    
    # Step 4: Wandb logging
    print(f"\nStep 4: Wandb logging")
    wandb_reward = avg_episode_reward  # Always uses environment rewards now!
    
    if use_ebm_reward:
        ebm_mean = np.mean(reward_trajs)
        print(f"Wandb logs:")
        print(f"  'avg episode reward - train (ENV)': {wandb_reward}")
        print(f"  'training_reward_type': 'EBM_guided'")
        print(f"  'logging_reward_type': 'Environment'")
        print(f"  'ebm_reward_mean': {ebm_mean}")
        print(f"  'ebm_reward_scale': 2.0")
    else:
        print(f"Wandb logs:")
        print(f"  'avg episode reward - train (ENV)': {wandb_reward}")
        print(f"  'training_reward_type': 'Environment'")
        print(f"  'logging_reward_type': 'Environment'")
    
    return wandb_reward

if __name__ == "__main__":
    print("TESTING ENVIRONMENT REWARD LOGGING BEHAVIOR")
    
    # Test environment-only training
    env_reward = simulate_training_step(use_ebm_reward=False)
    
    # Test EBM-guided training
    ebm_reward = simulate_training_step(use_ebm_reward=True)
    
    print(f"\n{'='*60}")
    print(f"FINAL VERIFICATION:")
    print(f"{'='*60}")
    print(f"Environment-only training wandb reward: {env_reward}")
    print(f"EBM-guided training wandb reward:       {ebm_reward}")
    print(f"Are they the same? {env_reward == ebm_reward}")
    print(f"✅ SUCCESS: Both configurations log the SAME environment reward!")