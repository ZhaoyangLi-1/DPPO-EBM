"""
DPPO reward collapse fix script.
"""

import yaml
from pathlib import Path

def create_fixed_dppo_config():
    """Create a fixed DPPO configuration for stable training."""
    
    # Read the problematic config
    config_path = "/linting-slow-vol/DPPO-EBM/cfg/gym/finetune/hopper-v2/ft_ppo_diffusion_ebm_mlp.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Key fixes for DPPO reward collapse
    print("Applying DPPO stability fixes...")
    
    # 1. Reduce learning rates (critical for DPPO)
    config['train']['actor_lr'] = 5e-5  # Much lower for diffusion models
    config['train']['critic_lr'] = 3e-4
    print("✓ Reduced learning rates")
    
    # 2. Fix diffusion-specific parameters
    config['model']['ft_denoising_steps'] = 5  # Fewer denoising steps
    config['model']['min_sampling_denoising_std'] = 0.2  # Higher minimum noise
    config['model']['min_logprob_denoising_std'] = 0.2
    print("✓ Fixed denoising parameters")
    
    # 3. Reduce batch size and update frequency
    config['train']['batch_size'] = 10000  # Smaller batches
    config['train']['update_epochs'] = 3   # Fewer updates
    print("✓ Reduced batch size and update frequency")
    
    # 4. Add gradient clipping (critical for stability)
    if 'max_grad_norm' not in config['train']:
        config['train']['max_grad_norm'] = 0.3
    print("✓ Added gradient clipping")
    
    # 5. Increase entropy for exploration
    if 'ent_coef' not in config['train']:
        config['train']['ent_coef'] = 0.01
    print("✓ Added entropy coefficient")
    
    # 6. More conservative PPO clipping for diffusion
    config['model']['clip_ploss_coef'] = 0.05  # Very conservative
    config['model']['clip_ploss_coef_base'] = 0.01
    print("✓ More conservative PPO clipping")
    
    # 7. Disable reward scaling initially
    config['train']['reward_scale_running'] = False
    config['train']['reward_scale_const'] = 1.0
    print("✓ Disabled reward scaling")
    
    # 8. Reduce environment parallel workers
    config['env']['n_envs'] = 20  # Fewer envs for stability
    print("✓ Reduced parallel environments")
    
    # Save fixed config
    output_path = "/linting-slow-vol/DPPO-EBM/cfg/gym/finetune/hopper-v2/ft_ppo_diffusion_ebm_mlp_fixed.yaml"
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Fixed config saved to: {output_path}")
    return output_path

def print_dppo_diagnosis():
    """Print DPPO-specific diagnosis."""
    print("=" * 60)
    print("DPPO REWARD COLLAPSE DIAGNOSIS")
    print("=" * 60)
    
    print("\n🔍 DPPO-SPECIFIC ISSUES:")
    print("1. DIFFUSION NOISE TOO HIGH")
    print("   - Denoising steps create too much randomness")
    print("   - Solution: Reduce ft_denoising_steps to 5-10")
    
    print("\n2. LEARNING RATE TOO HIGH FOR DIFFUSION")
    print("   - Diffusion models are sensitive to LR")
    print("   - Solution: Use 5e-5 instead of 1e-4")
    
    print("\n3. BATCH SIZE TOO LARGE")
    print("   - Large batches amplify noise effects")
    print("   - Solution: Reduce batch_size to 10k-20k")
    
    print("\n4. INSUFFICIENT GRADIENT CLIPPING")
    print("   - Diffusion gradients can explode")
    print("   - Solution: max_grad_norm = 0.3")
    
    print("\n5. PPO CLIPPING TOO AGGRESSIVE")
    print("   - Standard PPO clipping hurts diffusion")
    print("   - Solution: clip_ploss_coef = 0.05")
    
    print("\n💡 MONITORING TIPS:")
    print("- Watch for sudden reward drops after policy updates")
    print("- Monitor action standard deviation (shouldn't go to 0)")
    print("- Check if denoising is producing reasonable actions")
    
    print("\n🚨 EMERGENCY SIGNS:")
    print("- Reward drops >50% in 1-2 iterations")
    print("- Actions become NaN or extremely large")
    print("- Policy loss increases suddenly")

if __name__ == "__main__":
    print_dppo_diagnosis()
    print("\n" + "=" * 60)
    print("APPLYING FIXES...")
    print("=" * 60)
    
    fixed_config = create_fixed_dppo_config()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Run with fixed config:")
    print("   python script/run.py \\")
    print("     --config-name=ft_ppo_diffusion_ebm_mlp_fixed \\")
    print("     --config-dir=cfg/gym/finetune/hopper-v2")
    
    print("\n2. Monitor training closely:")
    print("   - Stop if reward drops >30% suddenly")
    print("   - Check action distributions")
    print("   - Watch policy loss values")
    
    print("\n3. If still failing, try:")
    print("   - Even lower LR: 1e-5")
    print("   - Fixed noise schedule")
    print("   - Pretrain policy with BC first")