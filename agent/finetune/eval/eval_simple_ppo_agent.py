"""
Simple PPO evaluation agent.

This module provides an evaluation agent for simple MLP-based PPO models,
supporting both environment-only and EBM-guided policies.
"""

import os
import pickle
import numpy as np
import torch
import hydra
import logging
import random
from scipy.stats import pearsonr

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.eval.eval_agent import EvalAgent


class EvalSimplePPOAgent(EvalAgent):
    """
    Evaluation agent for Simple MLP PPO models.
    
    This agent can evaluate both environment-only and EBM-guided PPO policies,
    providing comprehensive performance metrics.
    """

    def __init__(self, cfg):
        """Initialize the evaluation agent with configuration."""
        super().__init__(cfg)
        
        # Model configuration
        self.model_path = cfg.model_path
        self.use_ebm_reward = cfg.get("use_ebm_reward", False)
        self.ebm_model = None
        
        # Evaluation configuration
        self.n_steps = cfg.n_steps
        self.render_num = cfg.get("render_num", 0)
        self.n_cond_step = cfg.cond_steps
        self.horizon_steps = cfg.horizon_steps
        self.act_steps = cfg.act_steps
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim
        
        # Load the trained model
        self._load_model(cfg)
        
        log.info(f"SimplePPO evaluation agent initialized")
        log.info(f"  Model path: {self.model_path}")
        log.info(f"  Use EBM reward: {self.use_ebm_reward}")
        log.info(f"  Evaluation steps: {self.n_steps}")

    def _load_model(self, cfg):
        """Load the trained SimplePPO model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            log.info(f"Loaded checkpoint from {self.model_path}")
            
            # Create model using hydra
            self.model = hydra.utils.instantiate(cfg.model)
            
            # Load model state
            if "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"])
                log.info("Loaded model state from checkpoint['model']")
            else:
                self.model.load_state_dict(checkpoint)
                log.info("Loaded model state directly from checkpoint")
            
            # Set model to evaluation mode
            self.model.eval()
            self.model = self.model.to(self.device)
            
            # Load EBM if specified
            if self.use_ebm_reward and hasattr(self.model, 'ebm_model') and self.model.ebm_model is not None:
                self.ebm_model = self.model.ebm_model
                log.info("EBM model found in loaded policy")
            
            log.info("✅ Model loaded successfully")
            
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            raise

    def reset_env_all(self, options_venv=None):
        """Reset all environments."""
        if options_venv is None:
            options_venv = [{} for _ in range(self.n_envs)]
        obs_venv = self.venv.reset_arg(options_list=options_venv)
        # convert to OrderedDict if obs_venv is a list of dict
        if isinstance(obs_venv, list):
            obs_venv = {
                key: np.stack([obs_venv[i][key] for i in range(self.n_envs)])
                for key in obs_venv[0].keys()
            }
        return obs_venv

    def run(self):
        """Run evaluation."""
        log.info(f"Starting evaluation for {self.n_steps} steps")
        
        # Initialize
        timer = Timer()
        prev_obs_venv = self.reset_env_all()
        
        # Trajectory storage
        all_episode_rewards = []
        all_episode_lengths = []
        episode_rewards = np.zeros(self.n_envs)
        episode_lengths = np.zeros(self.n_envs, dtype=int)
        
        # Statistics tracking
        step_count = 0
        total_episodes = 0
        
        # Main evaluation loop
        with torch.no_grad():
            for step in range(self.n_steps):
                # Select actions
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"])
                    .float()
                    .to(self.device)
                }
                
                # Use model to generate actions
                actions = self.model(cond=cond, deterministic=True)  # Deterministic for evaluation
                action_venv = actions.cpu().numpy()[:, : self.act_steps]
                
                # Execute actions
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )
                done_venv = terminated_venv | truncated_venv
                
                # Update statistics
                episode_rewards += reward_venv
                episode_lengths += 1
                step_count += 1
                
                # Handle episode completion
                for env_idx in range(self.n_envs):
                    if done_venv[env_idx]:
                        all_episode_rewards.append(episode_rewards[env_idx])
                        all_episode_lengths.append(episode_lengths[env_idx])
                        total_episodes += 1
                        
                        # Reset episode statistics
                        episode_rewards[env_idx] = 0
                        episode_lengths[env_idx] = 0
                
                prev_obs_venv = obs_venv
                
                # Progress logging
                if step % 100 == 0 or step == self.n_steps - 1:
                    log.info(f"Evaluation step {step}/{self.n_steps}, Episodes completed: {total_episodes}")
        
        # Compute final statistics
        if len(all_episode_rewards) > 0:
            episode_rewards_array = np.array(all_episode_rewards)
            episode_lengths_array = np.array(all_episode_lengths)
            
            stats = {
                "mean_episode_reward": float(np.mean(episode_rewards_array)),
                "std_episode_reward": float(np.std(episode_rewards_array)),
                "min_episode_reward": float(np.min(episode_rewards_array)),
                "max_episode_reward": float(np.max(episode_rewards_array)),
                "median_episode_reward": float(np.median(episode_rewards_array)),
                "mean_episode_length": float(np.mean(episode_lengths_array)),
                "std_episode_length": float(np.std(episode_lengths_array)),
                "total_episodes": total_episodes,
                "total_steps": step_count,
            }
            
            # Success rate (if threshold is defined)
            if hasattr(self, 'success_threshold'):
                success_rate = np.mean(episode_rewards_array >= self.success_threshold)
                stats["success_rate"] = float(success_rate)
            
            # Log results
            log.info("="*60)
            log.info("EVALUATION RESULTS")
            log.info("="*60)
            log.info(f"Total episodes: {stats['total_episodes']}")
            log.info(f"Total steps: {stats['total_steps']}")
            log.info(f"Mean episode reward: {stats['mean_episode_reward']:.3f} ± {stats['std_episode_reward']:.3f}")
            log.info(f"Episode reward range: [{stats['min_episode_reward']:.3f}, {stats['max_episode_reward']:.3f}]")
            log.info(f"Median episode reward: {stats['median_episode_reward']:.3f}")
            log.info(f"Mean episode length: {stats['mean_episode_length']:.1f} ± {stats['std_episode_length']:.1f}")
            if 'success_rate' in stats:
                log.info(f"Success rate: {stats['success_rate']:.3f}")
            log.info("="*60)
            
        else:
            log.warning("No episodes completed during evaluation!")
            stats = {
                "mean_episode_reward": 0.0,
                "total_episodes": 0,
                "total_steps": step_count,
            }
        
        # Save results
        results = {
            "stats": stats,
            "episode_rewards": all_episode_rewards,
            "episode_lengths": all_episode_lengths,
            "config": self.cfg,
            "model_path": self.model_path,
        }
        
        # Save to file
        result_path = os.path.join(self.cfg.logdir, "eval_results.pkl")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "wb") as f:
            pickle.dump(results, f)
        
        log.info(f"Results saved to: {result_path}")
        
        return stats