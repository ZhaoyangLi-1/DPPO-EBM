#!/usr/bin/env python3
"""
Example script for running SAC + EBM training.

This script demonstrates how to use the SAC + EBM integration for training
diffusion policies with energy-based model reward shaping.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run SAC + EBM training")
    parser.add_argument(
        "--env", 
        type=str, 
        default="hopper-v2",
        choices=["hopper-v2", "walker2d-v2", "halfcheetah-v2"],
        help="Environment to train on"
    )
    parser.add_argument(
        "--ebm-ckpt", 
        type=str, 
        default=None,
        help="Path to EBM checkpoint (optional)"
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
        "--no-ebm", 
        action="store_true",
        help="Disable EBM reward shaping"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Print command without executing"
    )
    
    args = parser.parse_args()
    
    # Build the command
    cmd = [
        "python", "script/run.py",
        "--config-name=ft_sac_diffusion_ebm_mlp",
        f"--config-dir=cfg/gym/finetune/{args.env}",
    ]
    
    # Add EBM-specific parameters
    if not args.no_ebm:
        cmd.extend([
            f"model.use_ebm_reward_shaping=True",
            f"model.pbrs_lambda={args.lambda}",
            f"model.pbrs_beta={args.beta}",
            f"model.pbrs_alpha={args.alpha}",
        ])
        
        if args.ebm_ckpt:
            cmd.append(f"model.ebm_ckpt_path={args.ebm_ckpt}")
    else:
        cmd.append("model.use_ebm_reward_shaping=False")
    
    # Print or execute command
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    if not args.dry_run:
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            sys.exit(0)

if __name__ == "__main__":
    main()
