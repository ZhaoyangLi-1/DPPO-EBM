"""
Test script to compare EBM rewards vs Environment rewards.
"""

import sys
sys.path.append('/linting-slow-vol/DPPO-EBM')

import torch
import numpy as np
import matplotlib.pyplot as plt

def diagnose_ebm_reward_issues():
    """Diagnose common EBM reward replacement issues."""
    
    print("=" * 60)
    print("EBM REWARD DIAGNOSIS")
    print("=" * 60)
    
    print("\n🔍 CURRENT EBM SETTINGS:")
    print("- use_ebm_reward: True  ⚠️  REPLACING ENV REWARDS")
    print("- ebm_reward_mode: k0   ⚠️  ONLY USING FINAL DENOISING STEP") 
    print("- ebm_reward_clip_u_max: 30.0")
    print("- ebm_reward_lambda: 1.0")
    
    print("\n❌ POTENTIAL ISSUES:")
    print("1. EBM MODEL QUALITY")
    print("   - EBM may not have learned correct preferences")
    print("   - Training data might not cover current policy states")
    
    print("\n2. REWARD SCALE MISMATCH") 
    print("   - Hopper env rewards: ~1000-3000")
    print("   - EBM energies: typically -10 to +10")
    print("   - Scale mismatch can destabilize training")
    
    print("\n3. K0 MODE LIMITATION")
    print("   - Only uses final denoising step (k=0)")
    print("   - Loses information from intermediate steps")
    print("   - May give poor signal for learning")
    
    print("\n4. REWARD DISTRIBUTION SHAPE")
    print("   - EBM rewards may be too sparse/dense")
    print("   - Different reward landscape than env")
    
    print("\n✅ IMMEDIATE FIXES:")
    print("1. DISABLE EBM REWARDS (test hypothesis)")
    print("   use_ebm_reward: False")
    
    print("\n2. IF KEEPING EBM, TRY:")
    print("   - ebm_reward_mode: dense  (use all denoising steps)")
    print("   - ebm_reward_lambda: 100  (scale up EBM rewards)")
    print("   - ebm_reward_clip_u_max: 10.0  (more conservative clipping)")

def create_env_only_config():
    """Create config with EBM rewards disabled."""
    
    import yaml
    
    # Load current config
    config_path = "/linting-slow-vol/DPPO-EBM/cfg/gym/finetune/hopper-v2/ft_ppo_diffusion_ebm_mlp.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Disable EBM rewards
    config['model']['use_ebm_reward'] = False
    
    # Keep other EBM settings for potential future use
    config['name'] = "${env_name}_ppo_diffusion_mlp_ENV_ONLY_ta${horizon_steps}_td${denoising_steps}_tdf${ft_denoising_steps}"
    
    # Also apply some stability fixes
    config['train']['actor_lr'] = 5e-5
    config['train']['batch_size'] = 20000
    config['train']['update_epochs'] = 3
    if 'max_grad_norm' not in config['train']:
        config['train']['max_grad_norm'] = 0.5
    
    # Save config
    output_path = "/linting-slow-vol/DPPO-EBM/cfg/gym/finetune/hopper-v2/ft_ppo_diffusion_mlp_env_only.yaml"
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Environment-only config saved: {output_path}")
    return output_path

def create_improved_ebm_config():
    """Create config with improved EBM settings."""
    
    import yaml
    
    # Load current config  
    config_path = "/linting-slow-vol/DPPO-EBM/cfg/gym/finetune/hopper-v2/ft_ppo_diffusion_ebm_mlp.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Improve EBM settings
    config['model']['use_ebm_reward'] = True
    config['model']['ebm_reward_mode'] = 'dense'  # Use all denoising steps
    config['model']['ebm_reward_lambda'] = 50.0   # Scale up rewards
    config['model']['ebm_reward_clip_u_max'] = 10.0  # More conservative clipping
    config['model']['ebm_reward_baseline_M'] = 16   # Fewer baseline samples
    
    config['name'] = "${env_name}_ppo_diffusion_mlp_EBM_IMPROVED_ta${horizon_steps}_td${denoising_steps}_tdf${ft_denoising_steps}"
    
    # Apply stability fixes
    config['train']['actor_lr'] = 3e-5  # Even lower for EBM
    config['train']['batch_size'] = 15000
    config['train']['update_epochs'] = 2  # Very conservative
    if 'max_grad_norm' not in config['train']:
        config['train']['max_grad_norm'] = 0.3
    
    # Save config
    output_path = "/linting-slow-vol/DPPO-EBM/cfg/gym/finetune/hopper-v2/ft_ppo_diffusion_mlp_ebm_improved.yaml"
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Improved EBM config saved: {output_path}")
    return output_path

if __name__ == "__main__":
    diagnose_ebm_reward_issues()
    
    print("\n" + "=" * 60)
    print("CREATING TEST CONFIGURATIONS")
    print("=" * 60)
    
    env_config = create_env_only_config()
    ebm_config = create_improved_ebm_config()
    
    print("\n" + "=" * 60)
    print("TESTING STRATEGY")
    print("=" * 60)
    
    print("🧪 STEP 1: Test with environment rewards only")
    print("python script/run.py \\")
    print("  --config-name=ft_ppo_diffusion_mlp_env_only \\")
    print("  --config-dir=cfg/gym/finetune/hopper-v2")
    print("➡️  This will prove if EBM is the problem")
    
    print("\n🧪 STEP 2: If env-only works, try improved EBM")  
    print("python script/run.py \\")
    print("  --config-name=ft_ppo_diffusion_mlp_ebm_improved \\")
    print("  --config-dir=cfg/gym/finetune/hopper-v2")
    print("➡️  This tests better EBM settings")
    
    print("\n📊 EXPECTED RESULTS:")
    print("- Env-only: Rewards should be stable/increasing")
    print("- EBM improved: May work if EBM model is good")
    print("- If both fail: Issue is with DPPO hyperparameters")
    
    print("\n⚠️  KEY INSIGHT:")
    print("EBM rewards are fundamentally different from env rewards.")
    print("They measure 'energy' rather than task performance.")
    print("This mismatch can completely break RL training!")