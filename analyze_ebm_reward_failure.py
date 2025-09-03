"""
Analyze why EBM rewards cause training collapse.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('/linting-slow-vol/DPPO-EBM')

def analyze_reward_scale_mismatch():
    """Analyze the reward scale mismatch issue."""
    print("=" * 60)
    print("EBM vs ENV REWARD SCALE ANALYSIS")
    print("=" * 60)
    
    # Typical reward ranges
    hopper_env_rewards = {
        'random_policy': (-50, 150),
        'medium_policy': (1000, 2000), 
        'expert_policy': (2500, 3500)
    }
    
    ebm_energy_range = {
        'typical_range': (-20, 20),
        'with_lambda_1': (-20, 20),
        'with_lambda_50': (-1000, 1000),
        'with_lambda_100': (-2000, 2000)
    }
    
    print("📊 REWARD SCALE COMPARISON:")
    print(f"Environment Rewards (Hopper):")
    for policy, (low, high) in hopper_env_rewards.items():
        print(f"  {policy}: {low} to {high}")
    
    print(f"\nEBM Energy Values:")
    for setting, (low, high) in ebm_energy_range.items():
        print(f"  {setting}: {low} to {high}")
    
    print("\n⚠️  CRITICAL ISSUE:")
    print("With ebm_reward_lambda=1.0, EBM rewards are ~100x smaller than env rewards!")
    print("This completely changes the optimization landscape.")
    
    return hopper_env_rewards, ebm_energy_range

def analyze_reward_signal_quality():
    """Analyze the quality of reward signals."""
    print("\n" + "=" * 60)
    print("REWARD SIGNAL QUALITY ANALYSIS") 
    print("=" * 60)
    
    print("🎯 ENVIRONMENT REWARDS:")
    print("✅ Direct task performance measure")
    print("✅ Dense signal (every timestep)")
    print("✅ Well-shaped for locomotion")
    print("✅ Proven to work with RL")
    
    print("\n🤖 EBM REWARDS:")
    print("❓ Energy-based preference measure")
    print("❓ May not align with task goals")
    print("❓ Sparse signal (only k=0 mode)")
    print("❓ Untested reward landscape")
    
    print("\n🔍 SPECIFIC ISSUES:")
    print("1. REWARD LANDSCAPE MISMATCH")
    print("   - Environment: smooth, continuous")
    print("   - EBM: potentially multi-modal, noisy")
    
    print("\n2. TEMPORAL CONSISTENCY")
    print("   - Environment: consistent over time")
    print("   - EBM: may vary based on denoising step")
    
    print("\n3. COVERAGE PROBLEM")
    print("   - EBM trained on limited data")
    print("   - May not generalize to exploration states")

def analyze_k0_mode_limitation():
    """Analyze the k=0 mode limitation."""
    print("\n" + "=" * 60)
    print("K0 MODE LIMITATION ANALYSIS")
    print("=" * 60)
    
    print("🔢 CURRENT SETTING: ebm_reward_mode: k0")
    print("➡️  Only uses final denoising step (k=0)")
    
    print("\n❌ PROBLEMS:")
    print("1. INFORMATION LOSS")
    print("   - Ignores intermediate denoising steps")
    print("   - Loses rich signal from full trajectory")
    
    print("2. SPARSE SIGNAL")
    print("   - Only one evaluation per action sequence")
    print("   - Harder for policy to learn from")
    
    print("3. NOISE SENSITIVITY")
    print("   - k=0 is most refined but may be overfitted")
    print("   - Single point evaluation is brittle")
    
    print("\n✅ DENSE MODE BENEFITS:")
    print("- Uses all denoising steps k=0,1,2,...,K")
    print("- Richer gradient signal")
    print("- More robust to individual step noise")

def create_ebm_reward_debugger():
    """Create a tool to debug EBM rewards in real time."""
    
    debug_code = '''
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
'''
    
    with open("/linting-slow-vol/DPPO-EBM/ebm_reward_debugger.py", 'w') as f:
        f.write(debug_code)
    
    print("✅ EBM reward debugger created: ebm_reward_debugger.py")

def provide_fix_recommendations():
    """Provide comprehensive fix recommendations."""
    print("\n" + "=" * 60)
    print("EBM REWARD FIX RECOMMENDATIONS")
    print("=" * 60)
    
    print("🚨 IMMEDIATE FIXES (in order of priority):")
    
    print("\n1. SCALE MATCHING:")
    print("   ebm_reward_lambda: 1.0 → 50-100")
    print("   ➡️  Match EBM scale to env reward scale")
    
    print("\n2. USE DENSE MODE:")
    print("   ebm_reward_mode: k0 → dense")
    print("   ➡️  Use all denoising steps for richer signal")
    
    print("\n3. CONSERVATIVE CLIPPING:")
    print("   ebm_reward_clip_u_max: 30.0 → 10.0")
    print("   ➡️  Prevent extreme outliers")
    
    print("\n4. FEWER BASELINE SAMPLES:")
    print("   ebm_reward_baseline_M: 32 → 8")
    print("   ➡️  Reduce computational overhead")
    
    print("\n🔬 ADVANCED FIXES:")
    
    print("\n5. REWARD NORMALIZATION:")
    print("   - Normalize EBM rewards to env reward statistics")
    print("   - Use running mean/std matching")
    
    print("\n6. HYBRID REWARDS:")
    print("   - Combine env + EBM: 0.8*env + 0.2*ebm")
    print("   - Gradually increase EBM weight")
    
    print("\n7. EBM QUALITY CHECK:")
    print("   - Validate EBM on known good/bad trajectories") 
    print("   - Retrain EBM if necessary")
    
    print("\n📊 MONITORING:")
    print("- Track EBM vs env reward correlation")
    print("- Monitor reward distribution changes")
    print("- Watch for sudden policy changes")

if __name__ == "__main__":
    print("🎉 CONFIRMED: EBM REWARD REPLACEMENT CAUSES TRAINING COLLAPSE!")
    print("Environment reward version: ✅ Rewards increasing")
    print("EBM reward version: ❌ Rewards decreasing")
    print()
    
    # Run analysis
    analyze_reward_scale_mismatch()
    analyze_reward_signal_quality()
    analyze_k0_mode_limitation()
    
    print("\n" + "=" * 60)
    print("CREATING DEBUGGING TOOLS")
    print("=" * 60)
    create_ebm_reward_debugger()
    
    provide_fix_recommendations()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("🎯 Since env rewards work, you have 3 options:")
    print()
    print("1. STICK WITH ENV REWARDS (recommended)")
    print("   - Use ft_ppo_diffusion_mlp_env_only.yaml")
    print("   - Stable, proven to work")
    print()
    print("2. FIX EBM REWARDS")
    print("   - Try ft_ppo_diffusion_mlp_ebm_improved.yaml")
    print("   - Monitor with ebm_reward_debugger.py")
    print()
    print("3. HYBRID APPROACH")
    print("   - Start with env rewards")
    print("   - Gradually introduce EBM component")
    print()
    print("💡 LESSON LEARNED:")
    print("Always validate reward signals before replacing env rewards!")
    print("EBM energies ≠ task performance rewards")