
"""
Real-time EBM reward debugger for DPPO training.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

class EBMRewardDebugger:
    def __init__(self):
        self.env_rewards = []
        self.ebm_rewards = []
        self.ebm_energies = []
        self.reward_ratios = []
    
    def log_rewards(self, env_reward, ebm_reward, ebm_energy):
        """Log rewards for comparison."""
        self.env_rewards.append(env_reward)
        self.ebm_rewards.append(ebm_reward) 
        self.ebm_energies.append(ebm_energy)
        
        if env_reward != 0:
            ratio = abs(ebm_reward / env_reward)
            self.reward_ratios.append(ratio)
    
    def check_reward_sanity(self):
        """Check if rewards are reasonable."""
        if len(self.env_rewards) < 10:
            return {"status": "insufficient_data"}
        
        env_mean = np.mean(self.env_rewards[-50:])
        ebm_mean = np.mean(self.ebm_rewards[-50:])
        
        issues = []
        
        # Check scale mismatch
        if abs(ebm_mean) < abs(env_mean) * 0.01:
            issues.append("EBM rewards too small vs env rewards")
        
        # Check if EBM rewards are reasonable
        if abs(ebm_mean) < 1:
            issues.append("EBM rewards may be too small to provide learning signal")
        
        # Check for constant rewards (no learning signal)
        if np.std(self.ebm_rewards[-20:]) < 0.1:
            issues.append("EBM rewards too constant, no learning signal")
            
        return {
            "status": "ok" if not issues else "issues_found",
            "issues": issues,
            "env_mean": env_mean,
            "ebm_mean": ebm_mean,
            "reward_ratio": abs(ebm_mean / env_mean) if env_mean != 0 else 0
        }
    
    def plot_rewards(self, save_path=None):
        """Plot reward comparison."""
        if len(self.env_rewards) < 5:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Environment rewards
        axes[0,0].plot(self.env_rewards[-100:], 'b-', alpha=0.7)
        axes[0,0].set_title("Environment Rewards")
        axes[0,0].grid(True, alpha=0.3)
        
        # EBM rewards  
        axes[0,1].plot(self.ebm_rewards[-100:], 'r-', alpha=0.7)
        axes[0,1].set_title("EBM Rewards")
        axes[0,1].grid(True, alpha=0.3)
        
        # Reward ratio
        if self.reward_ratios:
            axes[1,0].plot(self.reward_ratios[-100:], 'g-', alpha=0.7)
            axes[1,0].set_title("EBM/Env Reward Ratio")
            axes[1,0].grid(True, alpha=0.3)
        
        # Both on same plot (normalized)
        if len(self.env_rewards) >= 10:
            env_norm = np.array(self.env_rewards[-100:]) / np.std(self.env_rewards[-100:])
            ebm_norm = np.array(self.ebm_rewards[-100:]) / np.std(self.ebm_rewards[-100:])
            
            axes[1,1].plot(env_norm, 'b-', alpha=0.7, label='Env (normalized)')
            axes[1,1].plot(ebm_norm, 'r-', alpha=0.7, label='EBM (normalized)')
            axes[1,1].set_title("Normalized Rewards")
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig

# Usage in training loop:
# debugger = EBMRewardDebugger()
# 
# # In training step:
# debugger.log_rewards(env_reward, ebm_reward, ebm_energy)
# 
# # Check periodically:
# status = debugger.check_reward_sanity()
# if status["status"] == "issues_found":
#     print("EBM Reward Issues:", status["issues"])
