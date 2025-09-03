"""
EBM Reward Calibration and Scaling Tool.

This tool helps calibrate EBM rewards to match environment reward scales
for stable DPPO training.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import pickle
from typing import Dict, List, Tuple

class EBMRewardCalibrator:
    """Calibrate EBM rewards to environment reward scale."""
    
    def __init__(self, env_name="hopper-v2"):
        self.env_name = env_name
        
        # Environment reward statistics (from D4RL datasets)
        self.env_reward_stats = {
            "hopper-v2": {
                "random": {"mean": 50, "std": 100, "range": (-50, 200)},
                "medium": {"mean": 1500, "std": 500, "range": (500, 2500)},
                "expert": {"mean": 3000, "std": 300, "range": (2500, 3500)}
            },
            "walker2d-v2": {
                "random": {"mean": 20, "std": 80, "range": (-100, 150)},
                "medium": {"mean": 2000, "std": 600, "range": (800, 3000)},
                "expert": {"mean": 4000, "std": 400, "range": (3500, 4500)}
            },
            "halfcheetah-v2": {
                "random": {"mean": -200, "std": 300, "range": (-800, 200)},
                "medium": {"mean": 3000, "std": 800, "range": (1000, 4500)},
                "expert": {"mean": 8000, "std": 1000, "range": (6000, 10000)}
            }
        }
        
        # Typical EBM energy ranges
        self.ebm_energy_stats = {
            "typical_range": (-20, 20),
            "mean": 0,
            "std": 8
        }
    
    def calculate_optimal_lambda(self, target_policy="medium"):
        """Calculate optimal lambda for EBM reward scaling."""
        env_stats = self.env_reward_stats[self.env_name][target_policy]
        env_mean = abs(env_stats["mean"])
        env_std = env_stats["std"]
        
        ebm_std = self.ebm_energy_stats["std"]
        
        # Scale EBM to match environment reward magnitude
        lambda_mean = env_mean / (3 * ebm_std)  # 3-sigma rule
        lambda_std = env_std / ebm_std
        
        # Use the more conservative scaling
        optimal_lambda = min(lambda_mean, lambda_std)
        
        return {
            "optimal_lambda": optimal_lambda,
            "env_mean": env_mean,
            "env_std": env_std,
            "scaling_ratio": optimal_lambda,
            "expected_ebm_range": (
                -20 * optimal_lambda, 
                20 * optimal_lambda
            )
        }
    
    def create_calibrated_config(self, base_config_path, output_path, 
                               target_policy="medium", conservative=True):
        """Create calibrated EBM configuration."""
        
        # Load base config
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Calculate optimal scaling
        calibration = self.calculate_optimal_lambda(target_policy)
        optimal_lambda = calibration["optimal_lambda"]
        
        # Apply conservative factor if requested
        if conservative:
            optimal_lambda *= 0.5  # Start with half the optimal value
        
        # Update EBM reward parameters
        config['model']['use_ebm_reward'] = True
        config['model']['ebm_reward_mode'] = 'dense'  # Use all denoising steps
        config['model']['ebm_reward_lambda'] = float(optimal_lambda)
        config['model']['ebm_reward_clip_u_max'] = optimal_lambda * 10  # 10x lambda
        config['model']['ebm_reward_baseline_M'] = 8  # Fewer samples for speed
        config['model']['ebm_reward_baseline_use_mu_only'] = False  # Use stochastic baseline
        
        # Update training parameters for stability with EBM
        config['train']['actor_lr'] = 3e-5  # Lower LR for EBM
        config['train']['batch_size'] = 10000  # Smaller batches
        config['train']['update_epochs'] = 2  # Very conservative
        config['train']['ent_coef'] = 0.01  # Maintain exploration
        if 'max_grad_norm' not in config['train']:
            config['train']['max_grad_norm'] = 0.2  # Strict gradient clipping
        
        # Update name to reflect calibration
        config['name'] = f"${{env_name}}_ppo_diffusion_ebm_CALIBRATED_lambda{optimal_lambda:.0f}_ta${{horizon_steps}}"
        
        # Add calibration metadata
        config['calibration_info'] = {
            'target_policy': target_policy,
            'optimal_lambda': float(optimal_lambda),
            'conservative_factor': 0.5 if conservative else 1.0,
            'expected_reward_range': calibration["expected_ebm_range"],
            'env_stats': self.env_reward_stats[self.env_name][target_policy]
        }
        
        # Save calibrated config
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return config, calibration
    
    def validate_ebm_model(self, ebm_ckpt_path):
        """Validate EBM model quality before using it for rewards."""
        if not Path(ebm_ckpt_path).exists():
            return {
                "status": "file_not_found",
                "message": f"EBM checkpoint not found: {ebm_ckpt_path}"
            }
        
        try:
            # Load checkpoint
            checkpoint = torch.load(ebm_ckpt_path, map_location='cpu')
            
            # Check if it's a complete checkpoint
            required_keys = ['model']
            missing_keys = [k for k in required_keys if k not in checkpoint]
            
            if missing_keys:
                return {
                    "status": "invalid_checkpoint",
                    "message": f"Missing keys in checkpoint: {missing_keys}"
                }
            
            # TODO: Add more sophisticated validation
            # - Check model architecture compatibility
            # - Validate on known good/bad trajectories
            # - Check training metrics if available
            
            return {
                "status": "valid",
                "message": "EBM checkpoint appears valid",
                "checkpoint_keys": list(checkpoint.keys())
            }
            
        except Exception as e:
            return {
                "status": "load_error",
                "message": f"Failed to load checkpoint: {e}"
            }

def create_progressive_training_strategy():
    """Create a progressive strategy for EBM reward training."""
    
    strategy = {
        "phase_1": {
            "name": "Validation Phase",
            "duration": "50 iterations",
            "purpose": "Validate EBM rewards are reasonable",
            "config": {
                "use_ebm_reward": True,
                "ebm_reward_lambda": "calculated_optimal * 0.1",  # Very small
                "log_both_rewards": True,  # Log both EBM and env for comparison
                "early_stop_if": "ebm_rewards consistently 0 or exploding"
            }
        },
        "phase_2": {
            "name": "Gradual Scaling",
            "duration": "100 iterations", 
            "purpose": "Gradually increase EBM influence",
            "config": {
                "ebm_reward_lambda": "linearly increase to optimal * 0.5",
                "monitor": "reward correlation, policy stability"
            }
        },
        "phase_3": {
            "name": "Full EBM Training",
            "duration": "remaining iterations",
            "purpose": "Train with full EBM rewards",
            "config": {
                "ebm_reward_lambda": "optimal value",
                "strict_monitoring": "stop if rewards drop >30%"
            }
        }
    }
    
    return strategy

def main():
    """Main calibration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calibrate EBM rewards")
    parser.add_argument("--env", default="hopper-v2", 
                       choices=["hopper-v2", "walker2d-v2", "halfcheetah-v2"])
    parser.add_argument("--target-policy", default="medium",
                       choices=["random", "medium", "expert"])
    parser.add_argument("--base-config", 
                       default="/linting-slow-vol/DPPO-EBM/cfg/gym/finetune/hopper-v2/ft_ppo_diffusion_ebm_mlp.yaml")
    parser.add_argument("--ebm-ckpt", 
                       default="/linting-slow-vol/EBM-Guidance/outputs/gym/ebm_best_val_hopper-v2-medium.pt")
    args = parser.parse_args()
    
    print("=" * 60)
    print("EBM REWARD CALIBRATION TOOL")
    print("=" * 60)
    
    # Initialize calibrator
    calibrator = EBMRewardCalibrator(args.env)
    
    # Validate EBM model
    print(f"\n🔍 Validating EBM model: {args.ebm_ckpt}")
    validation = calibrator.validate_ebm_model(args.ebm_ckpt)
    print(f"   Status: {validation['status']}")
    print(f"   Message: {validation['message']}")
    
    if validation['status'] != 'valid':
        print("⚠️  EBM model validation failed!")
        print("   Consider using a different checkpoint or retraining the EBM.")
        return
    
    # Calculate calibration
    print(f"\n📏 Calculating optimal scaling for {args.env} ({args.target_policy} policy)")
    calibration = calibrator.calculate_optimal_lambda(args.target_policy)
    
    print(f"   Environment reward mean: {calibration['env_mean']:.0f}")
    print(f"   Environment reward std: {calibration['env_std']:.0f}")
    print(f"   Optimal lambda: {calibration['optimal_lambda']:.1f}")
    print(f"   Expected EBM reward range: {calibration['expected_ebm_range']}")
    
    # Create calibrated configs
    print(f"\n⚙️  Creating calibrated configurations...")
    
    # Conservative version
    output_conservative = args.base_config.replace('.yaml', '_ebm_calibrated_conservative.yaml')
    config_conservative, _ = calibrator.create_calibrated_config(
        args.base_config, output_conservative, args.target_policy, conservative=True
    )
    print(f"   Conservative config: {output_conservative}")
    print(f"   Lambda: {config_conservative['model']['ebm_reward_lambda']:.1f}")
    
    # Aggressive version  
    output_aggressive = args.base_config.replace('.yaml', '_ebm_calibrated_aggressive.yaml')
    config_aggressive, _ = calibrator.create_calibrated_config(
        args.base_config, output_aggressive, args.target_policy, conservative=False
    )
    print(f"   Aggressive config: {output_aggressive}")
    print(f"   Lambda: {config_aggressive['model']['ebm_reward_lambda']:.1f}")
    
    # Show progressive strategy
    print(f"\n📈 Progressive Training Strategy:")
    strategy = create_progressive_training_strategy()
    for phase, details in strategy.items():
        print(f"   {details['name']}: {details['purpose']}")
        print(f"     Duration: {details['duration']}")
    
    print(f"\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("🧪 RECOMMENDED TESTING ORDER:")
    print("1. Start with conservative config:")
    print(f"   python script/run.py --config-name={Path(output_conservative).stem}")
    print("   --config-dir=cfg/gym/finetune/hopper-v2")
    print()
    print("2. Monitor EBM rewards closely:")
    print("   python /linting-slow-vol/DPPO-EBM/ebm_reward_debugger.py")
    print()
    print("3. If stable, try aggressive config")
    print()
    print("⚠️  CRITICAL SUCCESS CRITERIA:")
    print("- EBM rewards should be in similar range as env rewards")
    print("- Training rewards should increase or stay stable") 
    print("- No sudden policy collapses")
    print()
    print("🛑 STOP TRAINING IF:")
    print("- Rewards drop >50% suddenly")
    print("- EBM rewards are consistently 0 or NaN")
    print("- Action distributions become degenerate")

if __name__ == "__main__":
    main()