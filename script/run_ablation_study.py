#!/usr/bin/env python3
"""
Ablation study script for SAC vs SAC + EBM.

This script runs both SAC and SAC + EBM to compare their performance
for ablation studies.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def run_experiment(config_name, config_dir, experiment_name, additional_args=None):
    """Run a single experiment."""
    cmd = [
        "python", "script/run.py",
        f"--config-name={config_name}",
        f"--config-dir={config_dir}",
    ]
    
    if additional_args:
        cmd.extend(additional_args)
    
    print(f"\n{'='*60}")
    print(f"Running {experiment_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        end_time = time.time()
        print(f"\n✅ {experiment_name} completed successfully in {end_time - start_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {experiment_name} failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️ {experiment_name} interrupted by user")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run SAC vs SAC + EBM ablation study")
    parser.add_argument(
        "--env", 
        type=str, 
        default="hopper-v2",
        choices=["hopper-v2", "walker2d-v2", "halfcheetah-v2"],
        help="Environment to test on"
    )
    parser.add_argument(
        "--ebm-ckpt", 
        type=str, 
        default=None,
        help="Path to EBM checkpoint for SAC + EBM"
    )
    parser.add_argument(
        "--lambda", 
        type=float, 
        default=0.5,
        help="EBM reward shaping weight"
    )
    parser.add_argument(
        "--beta", 
        type=float, 
        default=1.0,
        help="EBM inverse temperature parameter"
    )
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=0.1,
        help="EBM potential scaling factor"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--skip-sac", 
        action="store_true",
        help="Skip running pure SAC"
    )
    parser.add_argument(
        "--skip-sac-ebm", 
        action="store_true",
        help="Skip running SAC + EBM"
    )
    
    args = parser.parse_args()
    
    config_dir = f"cfg/gym/finetune/{args.env}"
    
    # Define experiments
    experiments = []
    
    # Pure SAC experiment
    if not args.skip_sac:
        experiments.append({
            "name": "SAC (Pure)",
            "config_name": "ft_sac_diffusion_mlp",
            "config_dir": config_dir,
            "additional_args": None
        })
    
    # SAC + EBM experiment
    if not args.skip_sac_ebm:
        sac_ebm_args = [
            f"model.use_ebm_reward_shaping=True",
            f"model.pbrs_lambda={args.lambda}",
            f"model.pbrs_beta={args.beta}",
            f"model.pbrs_alpha={args.alpha}",
        ]
        
        if args.ebm_ckpt:
            sac_ebm_args.append(f"model.ebm_ckpt_path={args.ebm_ckpt}")
        
        experiments.append({
            "name": "SAC + EBM",
            "config_name": "ft_sac_diffusion_ebm_mlp",
            "config_dir": config_dir,
            "additional_args": sac_ebm_args
        })
    
    # Print experiment plan
    print("🧪 Ablation Study Plan")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"EBM Checkpoint: {args.ebm_ckpt or 'Default'}")
    print(f"EBM Parameters: λ={args.lambda}, β={args.beta}, α={args.alpha}")
    print(f"Experiments to run: {len(experiments)}")
    print()
    
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['name']}")
        print(f"   Config: {exp['config_name']}")
        if exp['additional_args']:
            print(f"   Args: {' '.join(exp['additional_args'])}")
        print()
    
    if args.dry_run:
        print("🔍 Dry run mode - commands will be printed but not executed")
        for exp in experiments:
            cmd = [
                "python", "script/run.py",
                f"--config-name={exp['config_name']}",
                f"--config-dir={exp['config_dir']}",
            ]
            if exp['additional_args']:
                cmd.extend(exp['additional_args'])
            print(f"\n{exp['name']}:")
            print(" ".join(cmd))
        return
    
    # Run experiments
    results = []
    for exp in experiments:
        success = run_experiment(
            exp['config_name'],
            exp['config_dir'],
            exp['name'],
            exp['additional_args']
        )
        results.append((exp['name'], success))
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 Ablation Study Summary")
    print(f"{'='*60}")
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nOverall: {passed}/{total} experiments completed successfully")
    
    if passed == total:
        print("🎉 All experiments completed successfully!")
    else:
        print("⚠️ Some experiments failed. Check the logs above for details.")

if __name__ == "__main__":
    main()
