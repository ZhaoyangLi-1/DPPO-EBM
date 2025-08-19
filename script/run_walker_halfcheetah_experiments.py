#!/usr/bin/env python3
"""
运行walker2d和halfcheetah环境的PPO+EBM、SAC、SAC+EBM实验脚本
"""

import os
import subprocess
import argparse
from pathlib import Path

def run_experiment(env_name, algorithm, config_path, gpu_id=0):
    """运行单个实验"""
    print(f"🚀 开始运行 {env_name} - {algorithm}")
    
    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 构建命令
    cmd = [
        'python', 'main.py',
        '--config-path', config_path,
        '--config-name', f'ft_{algorithm}_diffusion_mlp.yaml'
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, env=env, cwd='/root/DPPO-EBM', check=True)
        print(f"✅ {env_name} - {algorithm} 实验完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {env_name} - {algorithm} 实验失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='运行walker2d和halfcheetah环境实验')
    parser.add_argument('--env', choices=['walker2d', 'halfcheetah', 'both'], 
                       default='both', help='选择环境')
    parser.add_argument('--algorithm', choices=['ppo_ebm', 'sac', 'sac_ebm', 'all'], 
                       default='all', help='选择算法')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--dry-run', action='store_true', help='只打印命令，不执行')
    
    args = parser.parse_args()
    
    # 实验配置
    experiments = {
        'walker2d': {
            'config_path': 'cfg/gym/finetune/walker2d-v2',
            'algorithms': {
                'ppo_ebm': 'ft_ppo_diffusion_ebm_mlp.yaml',
                'sac': 'ft_sac_diffusion_mlp.yaml', 
                'sac_ebm': 'ft_sac_diffusion_ebm_mlp.yaml'
            }
        },
        'halfcheetah': {
            'config_path': 'cfg/gym/finetune/halfcheetah-v2',
            'algorithms': {
                'ppo_ebm': 'ft_ppo_diffusion_ebm_mlp.yaml',
                'sac': 'ft_sac_diffusion_mlp.yaml',
                'sac_ebm': 'ft_sac_diffusion_ebm_mlp.yaml'
            }
        }
    }
    
    # 选择环境和算法
    envs_to_run = ['walker2d', 'halfcheetah'] if args.env == 'both' else [args.env]
    algorithms_to_run = ['ppo_ebm', 'sac', 'sac_ebm'] if args.algorithm == 'all' else [args.algorithm]
    
    print("🎯 实验配置:")
    print(f"   环境: {envs_to_run}")
    print(f"   算法: {algorithms_to_run}")
    print(f"   GPU: {args.gpu}")
    print(f"   模式: {'dry-run' if args.dry_run else '执行'}")
    print()
    
    # 检查配置文件是否存在
    for env_name in envs_to_run:
        for alg in algorithms_to_run:
            config_file = experiments[env_name]['algorithms'][alg]
            config_path = Path(f"DPPO-EBM/{experiments[env_name]['config_path']}/{config_file}")
            if not config_path.exists():
                print(f"❌ 配置文件不存在: {config_path}")
                return
    
    # 运行实验
    success_count = 0
    total_count = len(envs_to_run) * len(algorithms_to_run)
    
    for env_name in envs_to_run:
        for alg in algorithms_to_run:
            config_path = experiments[env_name]['config_path']
            config_file = experiments[env_name]['algorithms'][alg]
            
            if args.dry_run:
                print(f"📋 将运行: {env_name} - {alg}")
                print(f"   配置文件: {config_path}/{config_file}")
                print()
            else:
                success = run_experiment(env_name, alg, config_path, args.gpu)
                if success:
                    success_count += 1
                print()
    
    if not args.dry_run:
        print(f"📊 实验总结:")
        print(f"   成功: {success_count}/{total_count}")
        print(f"   失败: {total_count - success_count}/{total_count}")

if __name__ == "__main__":
    main()
