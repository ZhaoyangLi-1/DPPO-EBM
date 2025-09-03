
"""
EBM Reward Validation Script - Run this during training to check EBM reward quality.
"""

import numpy as np
import matplotlib.pyplot as plt

class EBMRewardValidator:
    def __init__(self):
        self.env_rewards = []
        self.ebm_rewards = []
        self.iterations = []
        
    def add_data(self, iteration, env_reward, ebm_reward):
        """Add reward data from training."""
        self.iterations.append(iteration)
        self.env_rewards.append(env_reward)
        self.ebm_rewards.append(ebm_reward)
    
    def validate_current_state(self):
        """Check if EBM rewards are behaving correctly."""
        if len(self.ebm_rewards) < 10:
            return {"status": "insufficient_data"}
        
        recent_ebm = self.ebm_rewards[-10:]
        recent_env = self.env_rewards[-10:] if self.env_rewards else None
        
        issues = []
        
        # Check 1: EBM rewards not all zeros
        if all(abs(r) < 1e-6 for r in recent_ebm):
            issues.append("EBM rewards are all zero - model may not be working")
        
        # Check 2: EBM rewards not constant  
        if np.std(recent_ebm) < 1.0:
            issues.append("EBM rewards too constant - no learning signal")
        
        # Check 3: EBM rewards in reasonable range for hopper
        ebm_range = max(recent_ebm) - min(recent_ebm)
        if ebm_range < 100:
            issues.append("EBM reward range too small - may need higher lambda")
        elif ebm_range > 5000:
            issues.append("EBM reward range too large - may need lower lambda")
        
        # Check 4: No extreme outliers
        ebm_mean = np.mean(recent_ebm)
        ebm_std = np.std(recent_ebm)
        outliers = [r for r in recent_ebm if abs(r - ebm_mean) > 3 * ebm_std]
        if len(outliers) > 2:
            issues.append("Too many EBM reward outliers - check clipping")
        
        return {
            "status": "ok" if not issues else "issues_found",
            "issues": issues,
            "stats": {
                "ebm_mean": np.mean(recent_ebm),
                "ebm_std": np.std(recent_ebm), 
                "ebm_range": ebm_range,
                "num_outliers": len(outliers)
            }
        }
    
    def plot_rewards(self, save_path="ebm_validation.png"):
        """Plot reward comparison."""
        if len(self.ebm_rewards) < 5:
            return
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.iterations, self.ebm_rewards, 'r-', alpha=0.7, label='EBM Rewards')
        if self.env_rewards:
            plt.plot(self.iterations, self.env_rewards, 'b-', alpha=0.7, label='Env Rewards')
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        plt.title('Raw Rewards')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        if len(self.ebm_rewards) >= 20:
            # Moving average
            window = 10
            ebm_smooth = []
            env_smooth = []
            for i in range(window, len(self.ebm_rewards)):
                ebm_smooth.append(np.mean(self.ebm_rewards[i-window:i]))
                if self.env_rewards:
                    env_smooth.append(np.mean(self.env_rewards[i-window:i]))
            
            plt.plot(self.iterations[window:], ebm_smooth, 'r-', linewidth=2, label='EBM (smoothed)')
            if env_smooth:
                plt.plot(self.iterations[window:], env_smooth, 'b-', linewidth=2, label='Env (smoothed)')
            
            plt.xlabel('Iteration')
            plt.ylabel('Reward')
            plt.title('Smoothed Rewards')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Validation plot saved: {save_path}")

# Usage example:
# validator = EBMRewardValidator()
# 
# # In your training loop:
# validator.add_data(iteration, env_reward, ebm_reward)
# 
# # Check periodically:
# if iteration % 10 == 0:
#     result = validator.validate_current_state()
#     if result["status"] == "issues_found":
#         print("EBM VALIDATION ISSUES:", result["issues"])
#         print("Consider stopping training and adjusting parameters")
#     
#     validator.plot_rewards(f"ebm_validation_itr_{iteration}.png")
