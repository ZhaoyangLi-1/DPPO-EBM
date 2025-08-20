#!/usr/bin/env python3
"""
Parallel A/B comparison script for DPPO vs DPPO+EBM reward shaping.
Runs both experiments in parallel and compares their sample efficiency.
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
import pickle
import subprocess
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


def setup_environment():
    """Setup environment variables and check dependencies."""
    repo_root = Path(__file__).parent.parent
    os.environ.setdefault("DPPO_DATA_DIR", str(repo_root / "data"))
    os.environ.setdefault("DPPO_LOG_DIR", str(repo_root / "log"))
    
    # Check PyTorch version
    try:
        import torch
        version = torch.__version__
        major, minor = map(int, version.split('.')[:2])
        if major < 1 or (major == 1 and minor < 8):
            print(f"[WARNING] PyTorch {version} may be too old for torch.unravel_index")
        else:
            print(f"[INFO] PyTorch {version} is compatible")
    except ImportError:
        print("[ERROR] PyTorch not found")
        return False
    
    # Check matplotlib and pandas
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        print("[INFO] Plotting dependencies OK")
    except ImportError as e:
        print(f"[WARNING] Plotting dependencies missing: {e}")
    
    return True


def find_latest_result(experiment_name):
    """Find the latest result.pkl file for an experiment."""
    log_dir = Path(os.environ["DPPO_LOG_DIR"]) / "gym-finetune" / experiment_name
    if not log_dir.exists():
        return None
    
    # Find the most recent subdirectory
    subdirs = [d for d in log_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    
    latest_dir = max(subdirs, key=lambda x: x.stat().st_mtime)
    result_file = latest_dir / "result.pkl"
    return str(result_file) if result_file.exists() else None


def analyze_results(result_path):
    """Analyze results from a result.pkl file."""
    if not result_path or not Path(result_path).exists():
        return {
            "exists": False,
            "entries": 0,
            "best_eval": None,
            "best_train": None,
            "learning_curve": [],
            "steps_to_threshold": None
        }
    
    try:
        with open(result_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load {result_path}: {e}")
        return {
            "exists": False,
            "entries": 0,
            "best_eval": None,
            "best_train": None,
            "learning_curve": [],
            "steps_to_threshold": None
        }
    
    # Extract metrics
    best_eval = None
    best_train = None
    learning_curve = []
    
    for entry in data:
        # Learning curve
        if "step" in entry and "eval_episode_reward" in entry:
            learning_curve.append((entry["step"], entry["eval_episode_reward"]))
        elif "step" in entry and "train_episode_reward" in entry:
            learning_curve.append((entry["step"], entry["train_episode_reward"]))
        
        # Best rewards
        if "eval_episode_reward" in entry:
            val = entry["eval_episode_reward"]
            best_eval = val if best_eval is None else max(best_eval, val)
        if "train_episode_reward" in entry:
            val = entry["train_episode_reward"]
            best_train = val if best_train is None else max(best_train, val)
    
    # Calculate steps to 80% threshold
    steps_to_threshold = None
    if learning_curve and best_eval is not None:
        threshold = best_eval * 0.8
        for step, reward in learning_curve:
            if reward >= threshold:
                steps_to_threshold = step
                break
    
    return {
        "exists": True,
        "entries": len(data),
        "best_eval": best_eval,
        "best_train": best_train,
        "learning_curve": learning_curve,
        "steps_to_threshold": steps_to_threshold
    }


def run_experiment(config_name, name, overrides, log_file, cfg_dir):
    """Run a single experiment."""
    run_script = Path(__file__).parent / "run.py"
    cmd = [
        sys.executable, str(run_script),
        f"--config-name={config_name}",
        f"--config-dir={cfg_dir}",
        f"name={name}"
    ] + overrides
    
    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Run experiment
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        return proc


def format_number(value, fmt=".4f"):
    """Safely format a number."""
    if value is None:
        return "None"
    try:
        return format(value, fmt)
    except:
        return str(value)


def create_comparison_plot(baseline_data, ebm_data, output_dir, timestamp):
    """Create learning curve comparison plot."""
    if not baseline_data["learning_curve"] or not ebm_data["learning_curve"]:
        print("[WARN] Cannot create plot: missing learning curve data")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot curves
    base_steps, base_rewards = zip(*baseline_data["learning_curve"])
    ebm_steps, ebm_rewards = zip(*ebm_data["learning_curve"])
    
    plt.plot(base_steps, base_rewards, 'b-', linewidth=2, label='DPPO Baseline', alpha=0.8)
    plt.plot(ebm_steps, ebm_rewards, 'r-', linewidth=2, label='DPPO + EBM PBRS', alpha=0.8)
    
    # Add threshold lines
    if baseline_data["steps_to_threshold"]:
        plt.axvline(x=baseline_data["steps_to_threshold"], color='blue', 
                   linestyle='--', alpha=0.5, 
                   label=f'Baseline 80% ({baseline_data["steps_to_threshold"]:,} steps)')
    if ebm_data["steps_to_threshold"]:
        plt.axvline(x=ebm_data["steps_to_threshold"], color='red', 
                   linestyle='--', alpha=0.5,
                   label=f'EBM 80% ({ebm_data["steps_to_threshold"]:,} steps)')
    
    # Customize plot
    plt.xlabel('Environment Steps', fontsize=14)
    plt.ylabel('Evaluation Reward', fontsize=14)
    plt.title('DPPO vs DPPO + EBM: Learning Curves', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add performance metrics
    if baseline_data["best_eval"] and ebm_data["best_eval"]:
        delta = ebm_data["best_eval"] - baseline_data["best_eval"]
        plt.text(0.02, 0.98, f'Final Delta: {delta:+.4f}', 
                transform=plt.gca().transAxes, fontsize=12, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if baseline_data["steps_to_threshold"] and ebm_data["steps_to_threshold"]:
        speedup = baseline_data["steps_to_threshold"] / ebm_data["steps_to_threshold"]
        plt.text(0.02, 0.92, f'Speedup: {speedup:.2f}x', 
                transform=plt.gca().transAxes, fontsize=12, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"learning_curves_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
    
    # Save CSV data
    baseline_csv = output_dir / f"baseline_curve_{timestamp}.csv"
    ebm_csv = output_dir / f"ebm_curve_{timestamp}.csv"
    
    with open(baseline_csv, 'w') as f:
        f.write("steps,reward\n")
        for step, reward in baseline_data["learning_curve"]:
            f.write(f"{step},{reward}\n")
    
    with open(ebm_csv, 'w') as f:
        f.write("steps,reward\n")
        for step, reward in ebm_data["learning_curve"]:
            f.write(f"{step},{reward}\n")
    
    print(f"CSV data saved: {baseline_csv}, {ebm_csv}")
    plt.close()


def print_comparison(baseline_data, ebm_data, baseline_name, ebm_name):
    """Print comparison results."""
    print("\n" + "="*60)
    print("A/B COMPARISON RESULTS")
    print("="*60)
    
    # Baseline results
    print(f"\n📊 BASELINE ({baseline_name})")
    print(f"  Entries: {baseline_data['entries']}")
    print(f"  Best Eval: {format_number(baseline_data['best_eval'])}")
    print(f"  Best Train: {format_number(baseline_data['best_train'])}")
    print(f"  Steps to 80%: {baseline_data['steps_to_threshold']}")
    
    # EBM results
    print(f"\n🚀 EBM PBRS ({ebm_name})")
    print(f"  Entries: {ebm_data['entries']}")
    print(f"  Best Eval: {format_number(ebm_data['best_eval'])}")
    print(f"  Best Train: {format_number(ebm_data['best_train'])}")
    print(f"  Steps to 80%: {ebm_data['steps_to_threshold']}")
    
    # Performance comparison
    print(f"\n📈 PERFORMANCE ANALYSIS")
    baseline_best = baseline_data["best_eval"] or baseline_data["best_train"]
    ebm_best = ebm_data["best_eval"] or ebm_data["best_train"]
    
    if baseline_best and ebm_best:
        final_delta = ebm_best - baseline_best
        print(f"  Final Performance Delta: {final_delta:+.4f}")
        if final_delta > 0:
            print("  ✅ EBM improves final performance")
        elif final_delta < 0:
            print("  ❌ EBM hurts final performance")
        else:
            print("  ➖ No final performance difference")
    else:
        print("  ⚠️  Cannot compare final performance (missing data)")
    
    # Sample efficiency comparison
    if baseline_data["steps_to_threshold"] and ebm_data["steps_to_threshold"]:
        speedup = baseline_data["steps_to_threshold"] / ebm_data["steps_to_threshold"]
        print(f"  Sample Efficiency Speedup: {speedup:.2f}x")
        if speedup > 1.1:
            print("  ✅ EBM significantly improves sample efficiency")
        elif speedup < 0.9:
            print("  ❌ EBM hurts sample efficiency")
        else:
            print("  ➖ No significant sample efficiency difference")
    else:
        print("  ⚠️  Cannot analyze sample efficiency (missing threshold data)")
    
    # Overall conclusion
    print(f"\n🎯 OVERALL CONCLUSION")
    if baseline_best and ebm_best:
        if final_delta > 0 and (not baseline_data["steps_to_threshold"] or 
                               not ebm_data["steps_to_threshold"] or speedup > 1.05):
            print("  ✅ EBM reward shaping shows clear benefits")
        elif final_delta < 0 and (not baseline_data["steps_to_threshold"] or 
                                 not ebm_data["steps_to_threshold"] or speedup < 0.95):
            print("  ❌ EBM reward shaping shows clear drawbacks")
        else:
            print("  ➖ EBM reward shaping shows mixed results")
    else:
        print("  ⚠️  Insufficient data for conclusion")


def main():
    parser = argparse.ArgumentParser(description="Run DPPO vs DPPO+EBM parallel comparison")
    parser.add_argument("--ebm-ckpt", required=True, help="Path to EBM checkpoint")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--iters", type=int, default=30, help="Training iterations")
    parser.add_argument("--n-steps", type=int, default=500, help="Steps per iteration")
    parser.add_argument("--n-envs", type=int, default=16, help="Number of environments")
    parser.add_argument("--ft-denoising-steps", type=int, default=20, help="Fine-tuning denoising steps")
    parser.add_argument("--horizon-steps", type=int, default=4, help="Horizon steps")
    parser.add_argument("--act-steps", type=int, default=4, help="Action steps")
    parser.add_argument("--update-epochs", type=int, default=1, help="Update epochs")
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch size")
    parser.add_argument("--pbrs-k-use-mode", default="all", help="PBRS k-use mode")
    parser.add_argument("--cfg-dir", help="Config directory")
    
    args = parser.parse_args()
    
    # Setup environment
    if not setup_environment():
        print("[ERROR] Environment setup failed")
        sys.exit(1)
    
    # Validate EBM checkpoint
    if not os.path.isabs(args.ebm_ckpt):
        print("[ERROR] EBM checkpoint path must be absolute")
        sys.exit(1)
    if not os.path.isfile(args.ebm_ckpt):
        print(f"[ERROR] EBM checkpoint not found: {args.ebm_ckpt}")
        sys.exit(1)
    
    # Setup paths
    cfg_dir = args.cfg_dir or str(Path(__file__).parent.parent / "cfg" / "gym" / "finetune" / "hopper-v2")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_name = f"hopper_baseline_{timestamp}"
    ebm_name = f"hopper_ebm_{timestamp}"
    
    # Build overrides
    common_overrides = [
        f"device=cuda:{args.gpu}",
        f"env.n_envs={args.n_envs}",
        f"horizon_steps={args.horizon_steps}",
        f"act_steps={args.act_steps}",
        f"model.ft_denoising_steps={args.ft_denoising_steps}",
        f"train.n_train_itr={args.iters}",
        f"train.n_steps={args.n_steps}",
        f"train.update_epochs={args.update_epochs}",
        f"train.batch_size={args.batch_size}",
        "train.val_freq=5",
        "train.save_model_freq=1000",
        "seed=42"
    ]
    
    ebm_overrides = common_overrides + [
        f"model.ebm_ckpt_path={args.ebm_ckpt}",
        f"model.pbrs_k_use_mode={args.pbrs_k_use_mode}"
    ]
    
    # Setup logging
    log_dir = Path(os.environ["DPPO_LOG_DIR"]) / "ab_runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    baseline_log = log_dir / f"{baseline_name}.log"
    ebm_log = log_dir / f"{ebm_name}.log"
    
    # Launch experiments
    print("🚀 Launching parallel experiments...")
    print(f"Baseline: {baseline_name}")
    print(f"EBM: {ebm_name}")
    
    p_baseline = run_experiment("ft_ppo_diffusion_mlp", baseline_name, 
                               common_overrides, baseline_log, cfg_dir)
    p_ebm = run_experiment("ft_ppo_diffusion_ebm_mlp", ebm_name, 
                          ebm_overrides, ebm_log, cfg_dir)
    
    # Wait for completion
    print("⏳ Waiting for experiments to complete...")
    rc1 = p_baseline.wait()
    rc2 = p_ebm.wait()
    
    print(f"Baseline exited with code {rc1}")
    print(f"EBM exited with code {rc2}")
    
    # Handle failures
    if rc1 != 0 or rc2 != 0:
        print("\n❌ One or both experiments failed!")
        print(f"Baseline log: {baseline_log}")
        print(f"EBM log: {ebm_log}")
        
        # Show error details
        for name, log_file, rc in [("Baseline", baseline_log, rc1), ("EBM", ebm_log, rc2)]:
            if rc != 0 and log_file.exists():
                print(f"\nLast 5 lines of {name} log:")
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-5:]:
                        print(f"  {line.rstrip()}")
        
        print("\n💡 Common solutions:")
        print("  1. Check PyTorch version (need >= 1.8.0)")
        print("  2. Verify EBM checkpoint exists")
        print("  3. Check GPU memory")
        print("  4. Try smaller parameters: --iters=5 --n-steps=10 --n-envs=4")
    
    # Analyze results
    print("\n📊 Analyzing results...")
    baseline_result = find_latest_result(baseline_name)
    ebm_result = find_latest_result(ebm_name)
    
    baseline_data = analyze_results(baseline_result)
    ebm_data = analyze_results(ebm_result)
    
    # Print comparison
    print_comparison(baseline_data, ebm_data, baseline_name, ebm_name)
    
    # Create plot
    if baseline_data["learning_curve"] and ebm_data["learning_curve"]:
        print("\n📈 Creating comparison plot...")
        plot_dir = log_dir / "plots"
        create_comparison_plot(baseline_data, ebm_data, plot_dir, timestamp)
    else:
        print("\n⚠️  Cannot create plot: missing learning curve data")
    
    print(f"\n✅ Comparison complete! Check logs in: {log_dir}")


if __name__ == "__main__":
    main()


# python script/run_ab_compare.py --ebm-ckpt /linting-slow-vol/EBM-Guidance/outputs/gym/ebm_best_val_hopper-v2-medium.pt 