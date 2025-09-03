"""
Create working EBM reward configurations with proper calibration.
"""

import yaml
from pathlib import Path

def create_calibrated_ebm_configs():
    """Create properly calibrated EBM configurations."""
    
    # Load the base config
    base_config_path = "/linting-slow-vol/DPPO-EBM/cfg/gym/finetune/hopper-v2/ft_ppo_diffusion_ebm_mlp.yaml"
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Calculate optimal lambda for hopper
    # Hopper env rewards: ~1500 mean, EBM energies: ~±10 range
    # So we need lambda ≈ 1500 / 10 = 150 to match scales
    
    configs_to_create = {
        "conservative": {
            "lambda": 50.0,  # Conservative: 1/3 of optimal
            "description": "Conservative EBM scaling - safe starting point"
        },
        "moderate": {
            "lambda": 100.0,  # Moderate: 2/3 of optimal  
            "description": "Moderate EBM scaling - balanced approach"
        },
        "optimal": {
            "lambda": 150.0,  # Optimal: full scale matching
            "description": "Optimal EBM scaling - full scale matching"
        }
    }
    
    created_configs = {}
    
    for config_type, params in configs_to_create.items():
        # Create modified config
        new_config = config.copy()
        
        # EBM reward settings - THE CRITICAL FIXES
        new_config['model']['use_ebm_reward'] = True
        new_config['model']['ebm_reward_mode'] = 'dense'  # Use ALL denoising steps
        new_config['model']['ebm_reward_lambda'] = params['lambda']  # PROPER SCALING
        new_config['model']['ebm_reward_clip_u_max'] = params['lambda'] * 0.5  # Reasonable clipping
        new_config['model']['ebm_reward_baseline_M'] = 8  # Fewer samples for speed
        new_config['model']['ebm_reward_baseline_use_mu_only'] = False  # Stochastic baseline
        
        # Training stability settings  
        new_config['train']['actor_lr'] = 2e-5  # Lower LR for EBM sensitivity
        new_config['train']['critic_lr'] = 1e-4  
        new_config['train']['batch_size'] = 8000  # Smaller batches for stability
        new_config['train']['update_epochs'] = 2  # Conservative updates
        new_config['train']['ent_coef'] = 0.02  # Maintain exploration
        new_config['train']['max_grad_norm'] = 0.2  # Strict gradient clipping
        
        # Disable reward scaling (critical!)
        new_config['train']['reward_scale_running'] = False
        new_config['train']['reward_scale_const'] = 1.0
        
        # Update name and tags
        new_config['name'] = f"${{env_name}}_ppo_diffusion_ebm_FIXED_{config_type.upper()}_lambda{params['lambda']:.0f}_ta${{horizon_steps}}_td${{denoising_steps}}_tdf${{ft_denoising_steps}}"
        new_config['wandb']['run'] = f"${{name}}_EBM_FIXED_{config_type.upper()}"
        new_config['wandb']['tags'] = ['DPPO', 'EBM_Reward_FIXED', f'Lambda_{params["lambda"]:.0f}', config_type.title()]
        
        # Add metadata
        new_config['ebm_calibration'] = {
            'scaling_type': config_type,
            'lambda_value': params['lambda'],
            'expected_reward_range': [-params['lambda'] * 20, params['lambda'] * 20],
            'description': params['description'],
            'fixes_applied': [
                'Proper EBM-to-env reward scaling',
                'Dense mode (all denoising steps)', 
                'Conservative training hyperparameters',
                'Disabled reward normalization',
                'Strict gradient clipping'
            ]
        }
        
        # Save config
        output_path = f"/linting-slow-vol/DPPO-EBM/cfg/gym/finetune/hopper-v2/ft_ppo_diffusion_ebm_mlp_FIXED_{config_type}.yaml"
        with open(output_path, 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
        
        created_configs[config_type] = {
            'path': output_path,
            'lambda': params['lambda'],
            'description': params['description']
        }
        
        print(f"✅ Created {config_type} EBM config: {Path(output_path).name}")
        print(f"   Lambda: {params['lambda']:.1f}")
        print(f"   Expected EBM reward range: ±{params['lambda'] * 20:.0f}")
        print()
    
    return created_configs

def create_validation_script():
    """Create a script to validate EBM rewards during training."""
    
    validation_script = '''
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
'''
    
    with open("/linting-slow-vol/DPPO-EBM/ebm_reward_validator.py", 'w') as f:
        f.write(validation_script)
    
    print("✅ Created EBM reward validator: ebm_reward_validator.py")

def main():
    print("=" * 60)
    print("CREATING PROPERLY CALIBRATED EBM REWARD CONFIGS")
    print("=" * 60)
    print()
    print("🎯 PROBLEM ANALYSIS:")
    print("- Original lambda: 1.0 → EBM rewards: ±20")  
    print("- Hopper env rewards: ~1500")
    print("- Scale mismatch: 75x too small!")
    print()
    print("🔧 FIXES BEING APPLIED:")
    print("1. Proper scaling: lambda 50-150 (vs original 1.0)")
    print("2. Dense mode: Use ALL denoising steps (vs k0 only)")
    print("3. Conservative training: Lower LR, smaller batches")
    print("4. Disabled reward normalization (critical!)")
    print("5. Strict gradient clipping")
    print()
    
    # Create configs
    configs = create_calibrated_ebm_configs()
    
    # Create validation tools
    create_validation_script()
    
    print("=" * 60)
    print("TESTING STRATEGY")
    print("=" * 60)
    print("🧪 RECOMMENDED ORDER:")
    print()
    
    for i, (config_type, info) in enumerate(configs.items(), 1):
        print(f"{i}. {config_type.upper()} CONFIG:")
        print(f"   python script/run.py \\")
        print(f"     --config-name=ft_ppo_diffusion_ebm_mlp_FIXED_{config_type} \\")
        print(f"     --config-dir=cfg/gym/finetune/hopper-v2")
        print(f"   Expected EBM rewards: ±{info['lambda'] * 20:.0f}")
        print(f"   {info['description']}")
        print()
    
    print("📊 SUCCESS CRITERIA:")
    print("✅ EBM rewards in range ±500 to ±3000 (similar to env)")
    print("✅ Training rewards increase or stay stable")
    print("✅ No sudden reward collapses")
    print("✅ Policy doesn't become deterministic too quickly")
    print()
    print("🛑 STOP IMMEDIATELY IF:")
    print("❌ EBM rewards are consistently 0")
    print("❌ Training rewards drop >50%")  
    print("❌ Actions become NaN or extremely large")
    print("❌ EBM rewards outside ±5000 range")
    print()
    print("💡 MONITORING:")
    print("- Use ebm_reward_validator.py to check reward quality")
    print("- Watch both EBM and policy loss curves")
    print("- Monitor action standard deviation")

if __name__ == "__main__":
    main()