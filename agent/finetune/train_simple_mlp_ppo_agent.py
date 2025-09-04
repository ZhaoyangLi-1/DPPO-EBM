"""
Training agent for Simple MLP PPO baseline.

This module provides a training agent for simple MLP-based PPO that can use
both environment rewards and EBM rewards as a baseline comparison.
"""

import os
import pickle
import einops
import numpy as np
import torch
import logging
import wandb
import math
from scipy.stats import pearsonr
log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_ppo_agent import TrainPPOAgent


class DynamicEnergyNormalizer:
    """
    Dynamic normalization for EBM energy values using exponential moving average
    with temperature scaling for reward sensitivity control.
    """
    
    def __init__(self, momentum=0.99, temperature=1.0, epsilon=1e-8):
        """
        Args:
            momentum (float): Momentum for exponential moving average (beta in paper)
            temperature (float): Temperature scaling parameter (tau in paper)
            epsilon (float): Small constant for numerical stability
        """
        self.momentum = momentum
        self.temperature = temperature
        self.epsilon = epsilon
        
        # Running statistics
        self.running_mean = 0.0
        self.running_var = 1.0
        self.num_updates = 0
        
        log.info(f"DynamicEnergyNormalizer initialized: momentum={momentum}, temperature={temperature}")
    
    def update(self, energies):
        """
        Update running statistics with new batch of energies.
        
        Args:
            energies (np.ndarray): Batch of energy values
            
        Returns:
            np.ndarray: Normalized and temperature-scaled energies
        """
        if len(energies) == 0:
            return energies
            
        # Compute batch statistics
        batch_mean = np.mean(energies)
        batch_var = np.var(energies)
        
        # Update running statistics using exponential moving average
        if self.num_updates == 0:
            # First update: initialize with batch statistics
            self.running_mean = batch_mean
            self.running_var = batch_var
        else:
            # Exponential moving average update
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        
        self.num_updates += 1
        
        # Normalize energies: E_norm = (E - mu) / (sigma + epsilon)
        normalized_energies = (energies - self.running_mean) / (np.sqrt(self.running_var) + self.epsilon)
        
        # Apply temperature scaling: E_scaled = E_norm / tau
        scaled_energies = normalized_energies / self.temperature
        
        # Log statistics every 100 updates for debugging
        if self.num_updates % 100 == 0 or self.num_updates <= 10:
            log.info(f"EnergyNormalizer Update {self.num_updates}:")
            log.info(f"  Batch: mean={batch_mean:.4f}, var={batch_var:.4f}")
            log.info(f"  Running: mean={self.running_mean:.4f}, var={self.running_var:.4f}")
            log.info(f"  Normalized range: [{normalized_energies.min():.4f}, {normalized_energies.max():.4f}]")
            log.info(f"  Temperature-scaled range: [{scaled_energies.min():.4f}, {scaled_energies.max():.4f}]")
        
        return scaled_energies
    
    def normalize_without_update(self, energies):
        """
        Normalize energies using current running statistics without updating them.
        Useful for evaluation.
        
        Args:
            energies (np.ndarray): Energy values to normalize
            
        Returns:
            np.ndarray: Normalized and temperature-scaled energies
        """
        if len(energies) == 0:
            return energies
            
        normalized_energies = (energies - self.running_mean) / (np.sqrt(self.running_var) + self.epsilon)
        scaled_energies = normalized_energies / self.temperature
        return scaled_energies
    
    def get_stats(self):
        """Get current normalization statistics."""
        return {
            "running_mean": self.running_mean,
            "running_var": self.running_var,
            "running_std": np.sqrt(self.running_var),
            "num_updates": self.num_updates,
            "momentum": self.momentum,
            "temperature": self.temperature
        }


class TrainSimpleMLP_PPOAgent(TrainPPOAgent):
    """
    Training agent for Simple MLP PPO with optional EBM reward integration.
    
    This agent extends the base PPO training to support:
    1. Simple MLP actor/critic networks
    2. Environment reward training
    3. EBM reward training
    4. Runtime switching between reward types
    """

    def __init__(self, cfg):
        """Initialize the training agent with configuration."""
        # Store EBM configuration for later setup (after device is available)
        self.use_ebm_reward = cfg.model.get("use_ebm_reward", False)
        self.ebm_reward_scale = cfg.model.get("ebm_reward_scale", 1.0)
        self.ebm_reward_clip_max = cfg.model.get("ebm_reward_clip_max", 10.0)
        self.use_auto_scaling = cfg.model.get("use_auto_scaling", False)  # Disabled by default
        
        # Dynamic normalization parameters
        self.use_dynamic_normalization = cfg.model.get("use_dynamic_normalization", True)
        self.normalization_momentum = cfg.model.get("normalization_momentum", 0.99)
        self.temperature_scale = cfg.model.get("temperature_scale", 1.0)
        self.normalization_epsilon = cfg.model.get("normalization_epsilon", 1e-8)
        
        self.ebm_model = None
        self.energy_normalizer = None
        
        # Call parent initialization first - this sets up self.device and creates main model
        super().__init__(cfg)
        
        # NOW setup EBM model (after device is available)
        if self.use_ebm_reward:
            log.info("🔄 Setting up EBM model after parent initialization...")
            self._setup_ebm_model(cfg)
            
            # Initialize dynamic energy normalizer if enabled
            if self.use_dynamic_normalization:
                self.energy_normalizer = DynamicEnergyNormalizer(
                    momentum=self.normalization_momentum,
                    temperature=self.temperature_scale,
                    epsilon=self.normalization_epsilon
                )
                log.info("🌡️  Dynamic energy normalizer initialized")
            
            # Link EBM to main model
            if self.ebm_model is not None and hasattr(self.model, 'set_ebm_model'):
                self.model.set_ebm_model(self.ebm_model)
                log.info("🔗 EBM model linked to main PPO model")
        
        # Final status check
        if self.use_ebm_reward and self.ebm_model is None:
            log.error("💥 CRITICAL: use_ebm_reward=True but EBM model failed to load!")
            log.error("💥 This will cause the 'no EBM model provided' warning during training")
        
        log.info(f"SimpleMLP PPO Agent initialized with use_ebm_reward={self.use_ebm_reward}, ebm_model_loaded={self.ebm_model is not None}")

    def _setup_ebm_model(self, cfg):
        """Set up EBM model for reward computation."""
        log.info(f"🔧 Setting up EBM model with use_ebm_reward={self.use_ebm_reward}")
        
        try:
            # Import EBM model
            log.info("📦 Importing EBM model...")
            from model.ebm.ebm import EBM
            log.info("✅ EBM import successful")
            
            # Create EBM model
            ebm_cfg = cfg.model.ebm
            log.info(f"🏗️  Creating EBM with config: embed_dim={ebm_cfg.get('embed_dim', 256)}, state_dim={cfg.obs_dim}, action_dim={cfg.action_dim}")
            
            self.ebm_model = EBM(cfg=type('EBMConfig', (), {
                'ebm': type('EBMSubConfig', (), {
                    'embed_dim': ebm_cfg.get('embed_dim', 256),
                    'state_dim': cfg.obs_dim,
                    'action_dim': cfg.action_dim,
                    'nhead': ebm_cfg.get('nhead', 8),
                    'depth': ebm_cfg.get('depth', 4),
                    'dropout': ebm_cfg.get('dropout', 0.0),
                    'use_cls_token': ebm_cfg.get('use_cls_token', False),
                    'num_views': ebm_cfg.get('num_views', None),
                })()
            })()).to(self.device)
            log.info("✅ EBM model created successfully")
            
            # Load EBM checkpoint if provided
            ebm_ckpt_path = cfg.model.get('ebm_ckpt_path', None)
            log.info(f"🔍 Looking for EBM checkpoint at: {ebm_ckpt_path}")
            
            if ebm_ckpt_path and os.path.exists(ebm_ckpt_path):
                try:
                    log.info(f"📂 Loading checkpoint from {ebm_ckpt_path}")
                    checkpoint = torch.load(ebm_ckpt_path, map_location=self.device)
                    log.info(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
                    
                    if "model" in checkpoint:
                        self.ebm_model.load_state_dict(checkpoint["model"])
                        log.info("✅ Loaded EBM state dict from checkpoint['model']")
                    else:
                        self.ebm_model.load_state_dict(checkpoint)
                        log.info("✅ Loaded EBM state dict directly from checkpoint")
                    
                    log.info(f"🎯 Successfully loaded EBM checkpoint from {ebm_ckpt_path}")
                except Exception as e:
                    log.error(f"❌ Failed to load EBM checkpoint: {e}")
                    log.error(f"❌ Checkpoint path was: {ebm_ckpt_path}")
                    self.ebm_model = None
            else:
                log.error(f"❌ EBM checkpoint path not found or doesn't exist: {ebm_ckpt_path}")
                log.error(f"❌ Path exists check: {os.path.exists(ebm_ckpt_path) if ebm_ckpt_path else 'None path'}")
                self.ebm_model = None
            
            # Freeze EBM model
            if self.ebm_model is not None:
                for param in self.ebm_model.parameters():
                    param.requires_grad = False
                self.ebm_model.eval()
                log.info("🔒 EBM model frozen and set to eval mode")
                
                # Set EBM model in the PPO model
                if hasattr(self.model, 'set_ebm_model'):
                    self.model.set_ebm_model(self.ebm_model)
                    log.info("🔗 EBM model linked to PPO model")
                else:
                    log.warning("⚠️  PPO model doesn't have set_ebm_model method")
                
                log.info("🎉 EBM model setup completed successfully!")
            else:
                log.error("💥 EBM model setup failed - model is None")
                
        except ImportError as e:
            log.error(f"❌ Failed to import EBM model: {e}")
            log.error(f"❌ Make sure the EBM module is in the correct path")
            self.ebm_model = None
        except Exception as e:
            log.error(f"❌ Failed to setup EBM model: {e}")
            import traceback
            log.error(f"❌ Full traceback: {traceback.format_exc()}")
            self.ebm_model = None

    def _compute_ebm_rewards(self, obs_trajs, samples_trajs, firsts_trajs=None, env_reward_trajs=None, eval_mode=False):
        """
        Compute EBM rewards for trajectory data with dynamic normalization and temperature scaling.
        
        The method applies the following transformations:
        1. Compute raw EBM energies for state-action pairs
        2. Apply dynamic normalization using exponential moving average:
           E_norm = (E - μ_t) / (σ_t + ε)
        3. Apply temperature scaling: E_scaled = E_norm / τ
        4. Convert to rewards: reward = -E_scaled (lower energy = higher reward)
        
        Args:
            obs_trajs: Observation trajectories dict with 'state' key
            samples_trajs: Action trajectories
            firsts_trajs: [n_steps+1, n_envs] Episode start indicators (optional)
            env_reward_trajs: [n_steps, n_envs] Environment rewards (optional, for logging)
            eval_mode: If True, don't update running normalization statistics
            
        Returns:
            EBM rewards array [n_steps, n_envs]
        """
        if not self.use_ebm_reward or self.ebm_model is None:
            return np.zeros((self.n_steps, self.n_envs))
        
        ebm_rewards = np.zeros((self.n_steps, self.n_envs))
        
        # Initialize environment step counters for each environment
        # This tracks the actual timestep within each episode
        env_step_counters = np.zeros(self.n_envs, dtype=int)
        
        # If firsts_trajs is provided, use it to track episode resets
        if firsts_trajs is not None:
            # firsts_trajs[0] indicates which envs started fresh episodes
            env_step_counters = (1 - firsts_trajs[0]).astype(int) * env_step_counters
        
        # No auto-scaling - removed raw reward collection
        
        try:
            with torch.no_grad():
                for step in range(self.n_steps):
                    # Prepare batch data
                    batch_obs = {
                        'state': obs_trajs['state'][step].to(self.device)  # [n_envs, cond_steps, obs_dim]
                    }
                    batch_actions = torch.from_numpy(samples_trajs[step]).float().to(self.device)  # [n_envs, horizon_steps, action_dim]
                    
                    # Create default indices
                    B = self.n_envs
                    # k_idx=0 for Simple MLP PPO (no diffusion/denoising process)
                    k_idx = torch.zeros(B, device=self.device, dtype=torch.long)
                    
                    # Extract state features (take last observation if sequence)
                    states = batch_obs['state']  # [n_envs, cond_steps, obs_dim]
                    if states.dim() == 3:
                        states = states[:, -1]  # Take most recent state
                    
                    # Compute EBM energies
                    try:
                        # t_idx represents environment time step within current episode
                        # Use actual environment timestep counters for each environment
                        env_t_idx = torch.from_numpy(env_step_counters).long().to(self.device)
                        
                        # Update step counters for next iteration
                        # Reset counter when episode starts (firsts_trajs[step+1] == 1)
                        if firsts_trajs is not None and step < len(firsts_trajs) - 1:
                            # Reset counters for environments that will start new episodes
                            reset_mask = firsts_trajs[step + 1].astype(bool)
                            env_step_counters[reset_mask] = 0
                        # Increment all counters
                        env_step_counters += 1
                        
                        if step == 0:
                            log.info(f"t_idx range: {env_t_idx.min().item()} to {env_t_idx.max().item()}")
                        elif step < 3:
                            log.info(f"Step {step}: t_idx range [{env_t_idx.min().item()}, {env_t_idx.max().item()}], mean {env_t_idx.float().mean().item():.1f}")
                        
                        # Use EBM forward method for single action evaluation
                        if step == 0:
                            log.info(f"Using EBM forward: actions shape {batch_actions.shape}")
                        
                        ebm_output = self.ebm_model(
                            k_idx=k_idx,
                            t_idx=env_t_idx,
                            views=None,  # No vision for simple baseline
                            poses=states,  # [n_envs, obs_dim]
                            actions=batch_actions  # [n_envs, horizon_steps, action_dim]
                        )
                        
                        # EBM forward returns (energies, z, z_s, z_a)
                        if isinstance(ebm_output, tuple) and len(ebm_output) >= 1:
                            energies = ebm_output[0]
                            if step == 0:
                                log.info(f"EBM forward output: tuple with {len(ebm_output)} elements, energies shape {energies.shape}")
                        else:
                            log.error(f"Unexpected EBM output format: {type(ebm_output)}")
                            energies = torch.zeros(B, device=self.device)
                        
                        # Get raw energies (before sign flip)
                        raw_energies = energies.cpu().numpy()
                        
                        # Apply dynamic normalization if enabled
                        if self.use_dynamic_normalization and self.energy_normalizer is not None:
                            # Normalize energies using dynamic statistics
                            # Note: we normalize the raw energies, then flip sign for rewards
                            if eval_mode:
                                # During evaluation, don't update running statistics
                                normalized_energies = self.energy_normalizer.normalize_without_update(raw_energies)
                            else:
                                # During training, update running statistics
                                normalized_energies = self.energy_normalizer.update(raw_energies)
                            # Convert to rewards: Lower normalized energy = Higher reward
                            ebm_step_rewards = -normalized_energies
                        else:
                            # Fallback: simple sign flip without normalization
                            ebm_step_rewards = -raw_energies
                        
                        # Debug: log statistics for first few steps
                        if step < 3:
                            log.info(f"Step {step}: EBM energies (raw) range [{raw_energies.min():.3f}, {raw_energies.max():.3f}], mean {raw_energies.mean():.3f}")
                            if self.use_dynamic_normalization and self.energy_normalizer is not None:
                                log.info(f"Step {step}: EBM rewards (normalized, -E) range [{ebm_step_rewards.min():.3f}, {ebm_step_rewards.max():.3f}], mean {ebm_step_rewards.mean():.3f}")
                                # Log normalizer stats
                                if step == 0:
                                    stats = self.energy_normalizer.get_stats()
                                    log.info(f"Normalizer stats: mean={stats['running_mean']:.4f}, std={stats['running_std']:.4f}, temp={stats['temperature']:.4f}")
                            else:
                                log.info(f"Step {step}: EBM rewards (-energy, raw) range [{ebm_step_rewards.min():.3f}, {ebm_step_rewards.max():.3f}], mean {ebm_step_rewards.mean():.3f}")
                        
                        # Store rewards
                        ebm_rewards[step] = ebm_step_rewards
                        
                    except Exception as e:
                        log.warning(f"EBM computation failed at step {step}: {e}")
                        ebm_rewards[step] = 0.0
                        
        except Exception as e:
            log.error(f"Failed to compute EBM rewards: {e}")
            
        # Apply scaling based on configuration
        if self.use_dynamic_normalization and self.energy_normalizer is not None:
            log.info("🌡️  Using dynamic normalization (temperature scaling included)")
            # Dynamic normalization already includes temperature scaling, 
            # but we still apply manual scaling for fine-tuning
            ebm_rewards = ebm_rewards * self.ebm_reward_scale
        else:
            log.info("🔧 Using manual scaling only (no dynamic normalization)")
            ebm_rewards = ebm_rewards * self.ebm_reward_scale
        
        # Apply clipping
        ebm_rewards = np.clip(ebm_rewards, -self.ebm_reward_clip_max, self.ebm_reward_clip_max)
        
        log.info(f"📊 Final EBM rewards: range [{ebm_rewards.min():.3f}, {ebm_rewards.max():.3f}], mean {ebm_rewards.mean():.3f}")
            
        return ebm_rewards
    
    def _compute_auto_scale(self, ebm_rewards, env_rewards):
        """
        Compute automatic scaling factor to match EBM and environment reward scales.
        
        Args:
            ebm_rewards: List of raw EBM rewards (already sign-flipped)
            env_rewards: List of environment rewards
            
        Returns:
            float: Auto-scaling factor
        """
        ebm_array = np.array(ebm_rewards)
        env_array = np.array(env_rewards)
        
        # Method 1: Match standard deviations
        ebm_std = np.std(ebm_array)
        env_std = np.std(env_array)
        
        if ebm_std > 1e-8:  # Avoid division by zero
            std_scale = env_std / ebm_std
        else:
            std_scale = 1.0
        
        # Method 2: Match value ranges (more robust)
        ebm_range = np.max(ebm_array) - np.min(ebm_array)
        env_range = np.max(env_array) - np.min(env_array)
        
        if ebm_range > 1e-8:
            range_scale = env_range / ebm_range
        else:
            range_scale = 1.0
        
        # Method 3: Match 90th percentiles (most robust against outliers)
        ebm_90 = np.percentile(np.abs(ebm_array), 90)
        env_90 = np.percentile(np.abs(env_array), 90)
        
        if ebm_90 > 1e-8:
            percentile_scale = env_90 / ebm_90
        else:
            percentile_scale = 1.0
        
        # Use weighted average of the three methods
        # Favor percentile method as it's most robust
        auto_scale = 0.6 * percentile_scale + 0.3 * std_scale + 0.1 * range_scale
        
        # Apply conservative bounds to prevent extreme scaling
        auto_scale = np.clip(auto_scale, 0.01, 100.0)
        
        log.info(f"🔧 Auto-scale components: std={std_scale:.4f}, range={range_scale:.4f}, p90={percentile_scale:.4f}")
        
        return auto_scale

    def run(self):
        """Main training loop with EBM reward support."""
        
        # Start training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        
        while self.itr < self.n_train_itr:
            
            # Prepare video paths for each env
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )
            
            # Define train or eval mode
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode
            
            # Reset environment if needed
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                firsts_trajs[0] = done_venv
            
            # Initialize trajectory holders
            obs_trajs = {
                "state": np.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim)
                )
            }
            samples_trajs = np.zeros(
                (self.n_steps, self.n_envs, self.horizon_steps, self.action_dim)
            )
            reward_trajs = np.zeros((self.n_steps, self.n_envs))
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            
            if self.save_full_observations:
                obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
                obs_full_trajs = np.vstack(
                    (obs_full_trajs, prev_obs_venv["state"][:, -1][None])
                )
            
            # Initialize detailed reward tracking
            raw_env_rewards = []
            
            # Collect trajectories
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")
                
                # Select actions
                with torch.no_grad():
                    cond = {
                        "state": torch.from_numpy(prev_obs_venv["state"])
                        .float()
                        .to(self.device)
                    }
                    samples = self.model(
                        cond=cond,
                        deterministic=eval_mode,
                    )
                    output_venv = samples.cpu().numpy()
                action_venv = output_venv[:, : self.act_steps]
                
                # Apply actions in environment
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )
                done_venv = terminated_venv | truncated_venv
                
                # Store raw environment reward for detailed logging
                raw_env_rewards.append(reward_venv.copy())
                
                if self.save_full_observations:
                    obs_full_venv = np.array(
                        [info["full_obs"]["state"] for info in info_venv]
                    )
                    obs_full_trajs = np.vstack(
                        (obs_full_trajs, obs_full_venv.transpose(1, 0, 2))
                    )
                
                # Store trajectory data
                obs_trajs["state"][step] = prev_obs_venv["state"]
                samples_trajs[step] = output_venv
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                firsts_trajs[step + 1] = done_venv
                
                # Update for next step
                prev_obs_venv = obs_venv
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0
            
            # Store original environment rewards before any EBM modification (for logging)
            original_reward_trajs = reward_trajs.copy()
            
            # Compute episode rewards and statistics
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            
            # Compute detailed reward statistics
            raw_env_rewards_array = np.array(raw_env_rewards)  # [n_steps, n_envs]
            
            # Basic reward statistics for current iteration
            raw_reward_stats = {
                "mean": float(np.mean(raw_env_rewards_array)),
                "std": float(np.std(raw_env_rewards_array)),
                "min": float(np.min(raw_env_rewards_array)),
                "max": float(np.max(raw_env_rewards_array)),
                "median": float(np.median(raw_env_rewards_array)),
            }
            
            if len(episodes_start_end) > 0:
                # ALWAYS compute environment reward stats for logging (use pure environment rewards)
                env_reward_trajs_split = [
                    original_reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(env_reward_trajs_split)
                
                # Environment episode rewards (for wandb logging)
                env_episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in env_reward_trajs_split]
                )
                
                if self.furniture_sparse_reward:
                    episode_best_reward = env_episode_reward
                else:
                    episode_best_reward = np.array(
                        [
                            np.max(reward_traj) / self.act_steps
                            for reward_traj in env_reward_trajs_split
                        ]
                    )
                
                # For wandb logging - ALWAYS use environment rewards
                avg_episode_reward = np.mean(env_episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
                
                # Additional episode statistics
                episode_stats = {
                    "mean": float(avg_episode_reward),
                    "std": float(np.std(env_episode_reward)) if len(env_episode_reward) > 1 else 0.0,
                    "min": float(np.min(env_episode_reward)),
                    "max": float(np.max(env_episode_reward)),
                    "median": float(np.median(env_episode_reward)),
                    "count": num_episode_finished,
                    "avg_length": float(np.mean([end - start + 1 for _, start, end in episodes_start_end]))
                }
            else:
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                episode_stats = {
                    "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, 
                    "count": 0, "avg_length": 0.0
                }
                log.info("[WARNING] No episode completed within the iteration!")
            
            # Update models
            if not eval_mode:
                # Convert to tensors
                obs_trajs["state"] = (
                    torch.from_numpy(obs_trajs["state"]).float().to(self.device)
                )
                
                # Optionally replace environment rewards with EBM rewards
                ebm_reward_stats = None
                reward_correlation = None
                if self.use_ebm_reward:
                    # Quick EBM model health check
                    if self.ebm_model is None:
                        log.error("❌ EBM model is None - no EBM rewards will be computed!")
                        reward_trajs = original_reward_trajs  # Fall back to env rewards
                    elif not hasattr(self.ebm_model, 'forward') and not hasattr(self.ebm_model, '__call__'):
                        log.error("❌ EBM model doesn't have forward method!")
                        reward_trajs = original_reward_trajs
                    else:
                        log.info(f"Computing EBM rewards for iteration {self.itr} (eval_mode={eval_mode})")
                        ebm_rewards = self._compute_ebm_rewards(obs_trajs, samples_trajs, firsts_trajs, original_reward_trajs, eval_mode)
                        
                        # Store EBM reward stats for wandb logging
                        self._last_ebm_reward_mean = np.mean(ebm_rewards)
                        self._last_ebm_reward_std = np.std(ebm_rewards)
                        ebm_reward_stats = {
                            "mean": float(self._last_ebm_reward_mean),
                            "std": float(self._last_ebm_reward_std),
                            "min": float(np.min(ebm_rewards)),
                            "max": float(np.max(ebm_rewards)),
                            "median": float(np.median(ebm_rewards))
                        }
                        
                        # Analyze correlation between EBM and environment rewards
                        try:
                            env_flat = original_reward_trajs.flatten()
                            ebm_flat = ebm_rewards.flatten()
                            if len(env_flat) > 10:  # Need sufficient data for correlation
                                correlation, p_value = pearsonr(env_flat, ebm_flat)
                                reward_correlation = {
                                    "pearson_r": float(correlation),
                                    "p_value": float(p_value),
                                    "env_mean": float(np.mean(env_flat)),
                                    "ebm_mean": float(np.mean(ebm_flat)),
                                    "sample_size": len(env_flat)
                                }
                                log.info(f"EBM-ENV reward correlation: r={correlation:.4f} (p={p_value:.4f})")
                                
                                # Adaptive EBM weighting based on correlation
                                if correlation > 0.3 and p_value < 0.01:
                                    ebm_weight = 0.5  # Strong positive correlation: trust EBM more
                                    log.info("✅ Strong positive correlation - using 50% EBM weight")
                                elif correlation > 0.0:
                                    ebm_weight = 0.2  # Weak positive correlation: use EBM conservatively
                                    log.info("⚖️  Weak positive correlation - using 20% EBM weight")
                                else:
                                    ebm_weight = 0.0  # Negative correlation: ignore EBM
                                    log.warning("❌ Negative correlation - ignoring EBM rewards!")
                                
                                # Mix rewards based on adaptive weight
                                mixed_rewards = (1.0 - ebm_weight) * original_reward_trajs + ebm_weight * ebm_rewards
                                log.info(f"🔄 Mixed rewards: {ebm_weight*100:.0f}% EBM + {(1-ebm_weight)*100:.0f}% ENV")
                                
                                if abs(correlation) < 0.1:
                                    log.warning("⚠️  Very weak correlation between EBM and environment rewards!")
                                elif correlation < -0.3:
                                    log.warning("⚠️  Strong negative correlation - EBM may be counterproductive!")
                        except Exception as e:
                            log.warning(f"Failed to compute reward correlation: {e}")
                            mixed_rewards = original_reward_trajs  # Fallback to env rewards
                            ebm_weight = 0.0
                        
                        reward_trajs = mixed_rewards
                        log.info(f"EBM reward stats: mean={self._last_ebm_reward_mean:.4f}, std={self._last_ebm_reward_std:.4f}")
                
                # Calculate values and logprobs in batches
                num_split = math.ceil(
                    self.n_envs * self.n_steps / self.logprob_batch_size
                )
                obs_ts = [{} for _ in range(num_split)]
                obs_k = einops.rearrange(
                    obs_trajs["state"],
                    "s e ... -> (s e) ...",
                )
                obs_ts_k = torch.split(obs_k, self.logprob_batch_size, dim=0)
                for i, obs_t in enumerate(obs_ts_k):
                    obs_ts[i]["state"] = obs_t
                
                values_trajs = np.empty((0, self.n_envs))
                for obs in obs_ts:
                    values = self.model.critic(obs).detach().cpu().numpy().flatten()
                    values_trajs = np.vstack(
                        (values_trajs, values.reshape(-1, self.n_envs))
                    )
                
                samples_t = einops.rearrange(
                    torch.from_numpy(samples_trajs).float().to(self.device),
                    "s e h d -> (s e) h d",
                )
                samples_ts = torch.split(samples_t, self.logprob_batch_size, dim=0)
                logprobs_trajs = np.empty((0))
                for obs_t, samples_t in zip(obs_ts, samples_ts):
                    logprobs = (
                        self.model.get_logprobs(obs_t, samples_t)[0].detach().cpu().numpy()
                    )
                    logprobs_trajs = np.concatenate(
                        (logprobs_trajs, logprobs.reshape(-1))
                    )
                
                # Track rewards before scaling for comparison
                reward_before_scaling = reward_trajs.copy()
                scaling_stats = None
                
                # Normalize rewards with running variance if specified
                if self.reward_scale_running:
                    reward_trajs_transpose = self.running_reward_scaler(
                        reward=reward_trajs.T, first=firsts_trajs[:-1].T
                    )
                    reward_trajs = reward_trajs_transpose.T
                    
                    # Compute scaling statistics
                    scaling_factor = np.mean(reward_trajs) / (np.mean(reward_before_scaling) + 1e-8)
                    scaling_stats = {
                        "before_mean": float(np.mean(reward_before_scaling)),
                        "before_std": float(np.std(reward_before_scaling)),
                        "after_mean": float(np.mean(reward_trajs)),
                        "after_std": float(np.std(reward_trajs)),
                        "scaling_factor": float(scaling_factor),
                        "scaler_mean": float(getattr(self.running_reward_scaler, 'mean', 0.0)),
                        "scaler_var": float(getattr(self.running_reward_scaler, 'var', 1.0))
                    }
                
                # Bootstrap value with GAE
                obs_venv_ts = {
                    "state": torch.from_numpy(obs_venv["state"])
                    .float()
                    .to(self.device)
                }
                advantages_trajs = np.zeros_like(reward_trajs)
                lastgaelam = 0
                for t in reversed(range(self.n_steps)):
                    if t == self.n_steps - 1:
                        nextvalues = (
                            self.model.critic(obs_venv_ts)
                            .detach()
                            .reshape(1, -1)
                            .cpu()
                            .numpy()
                        )
                    else:
                        nextvalues = values_trajs[t + 1]
                    nonterminal = 1.0 - terminated_trajs[t]
                    
                    # GAE computation
                    delta = (
                        reward_trajs[t] * self.reward_scale_const
                        + self.gamma * nextvalues * nonterminal
                        - values_trajs[t]
                    )
                    advantages_trajs[t] = lastgaelam = (
                        delta
                        + self.gamma * self.gae_lambda * nonterminal * lastgaelam
                    )
                returns_trajs = advantages_trajs + values_trajs
                
                # Prepare training data
                obs_k = {
                    "state": einops.rearrange(
                        obs_trajs["state"],
                        "s e ... -> (s e) ...",
                    )
                }
                samples_k = einops.rearrange(
                    torch.tensor(samples_trajs, device=self.device).float(),
                    "s e h d -> (s e) h d",
                )
                returns_k = (
                    torch.tensor(returns_trajs, device=self.device).float().reshape(-1)
                )
                values_k = (
                    torch.tensor(values_trajs, device=self.device).float().reshape(-1)
                )
                advantages_k = (
                    torch.tensor(advantages_trajs, device=self.device).float().reshape(-1)
                )
                logprobs_k = torch.tensor(logprobs_trajs, device=self.device).float()
                
                # Update policy and critic
                total_steps = self.n_steps * self.n_envs
                clipfracs = []
                for update_epoch in range(self.update_epochs):
                    
                    # Shuffle and batch data
                    flag_break = False
                    inds_k = torch.randperm(total_steps, device=self.device)
                    num_batch = max(1, total_steps // self.batch_size)
                    for batch in range(num_batch):
                        start = batch * self.batch_size
                        end = start + self.batch_size
                        inds_b = inds_k[start:end]
                        
                        obs_b = {"state": obs_k["state"][inds_b]}
                        samples_b = samples_k[inds_b]
                        returns_b = returns_k[inds_b]
                        values_b = values_k[inds_b]
                        advantages_b = advantages_k[inds_b]
                        logprobs_b = logprobs_k[inds_b]
                        
                        # Compute loss
                        (
                            pg_loss,
                            entropy_loss,
                            v_loss,
                            clipfrac,
                            approx_kl,
                            ratio,
                            bc_loss,
                            std,
                        ) = self.model.loss(
                            obs_b,
                            samples_b,
                            returns_b,
                            values_b,
                            advantages_b,
                            logprobs_b,
                            use_bc_loss=self.use_bc_loss,
                        )
                        loss = (
                            pg_loss
                            + entropy_loss * self.ent_coef
                            + v_loss * self.vf_coef
                            + bc_loss * self.bc_loss_coeff
                        )
                        clipfracs += [clipfrac]
                        
                        # Update networks
                        self.actor_optimizer.zero_grad()
                        self.critic_optimizer.zero_grad()
                        loss.backward()
                        if self.itr >= self.n_critic_warmup_itr:
                            if self.max_grad_norm is not None:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.actor_ft.parameters(), self.max_grad_norm
                                )
                            self.actor_optimizer.step()
                        self.critic_optimizer.step()
                        
                        log.info(
                            f"approx_kl: {approx_kl}, update_epoch: {update_epoch}, num_batch: {num_batch}"
                        )
                        
                        # Early stopping on KL divergence
                        if self.target_kl is not None and approx_kl > self.target_kl:
                            flag_break = True
                            break
                    if flag_break:
                        break
                
                # Explained variance
                y_pred, y_true = values_k.cpu().numpy(), returns_k.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )
            
            # Plot state trajectories if needed
            if (
                self.itr % self.render_freq == 0
                and self.n_render > 0
                and self.traj_plotter is not None
            ):
                self.traj_plotter(
                    obs_full_trajs=obs_full_trajs,
                    n_render=self.n_render,
                    max_episode_steps=self.max_episode_steps,
                    render_dir=self.render_dir,
                    itr=self.itr,
                )
            
            # Update learning rates
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            
            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()
            
            # Log results
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )
            if self.save_trajs:
                run_results[-1]["obs_full_trajs"] = obs_full_trajs
                run_results[-1]["obs_trajs"] = obs_trajs
                run_results[-1]["action_trajs"] = samples_trajs
                run_results[-1]["reward_trajs"] = reward_trajs
            
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    log.info(
                        f"eval rewards: raw_mean {raw_reward_stats['mean']:.4f} ± {raw_reward_stats['std']:.4f} | episodes: {num_episode_finished} | avg_len: {episode_stats['avg_length']:.1f}"
                    )
                    if self.use_wandb:
                        log_dict = {
                            "success rate - eval": success_rate,
                            "avg episode reward - eval (ENV)": avg_episode_reward,  # Always environment reward
                            "avg best reward - eval": avg_best_reward,
                            "num episode - eval": num_episode_finished,
                            # Raw step reward statistics
                            "eval/raw_reward_mean": raw_reward_stats["mean"],
                            "eval/raw_reward_std": raw_reward_stats["std"],
                            "eval/raw_reward_min": raw_reward_stats["min"],
                            "eval/raw_reward_max": raw_reward_stats["max"],
                            "eval/raw_reward_median": raw_reward_stats["median"],
                            # Episode statistics
                            "eval/episode_reward_std": episode_stats["std"],
                            "eval/episode_reward_min": episode_stats["min"],
                            "eval/episode_reward_max": episode_stats["max"],
                            "eval/episode_reward_median": episode_stats["median"],
                            "eval/avg_episode_length": episode_stats["avg_length"],
                        }
                        if self.use_ebm_reward:
                            log_dict["evaluation_reward_type"] = "Environment_only"
                        else:
                            log_dict["evaluation_reward_type"] = "Environment"
                        wandb.log(log_dict, step=self.itr, commit=False)
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss {loss:8.4f} | pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} | ent {-entropy_loss:8.4f} | reward {avg_episode_reward:8.4f} | t:{time:8.4f}"
                    )
                    log.info(
                        f"train rewards: raw_mean {raw_reward_stats['mean']:.4f} ± {raw_reward_stats['std']:.4f} | episodes: {num_episode_finished} | avg_len: {episode_stats['avg_length']:.1f}"
                    )
                    if scaling_stats is not None:
                        log.info(
                            f"reward scaling: {scaling_stats['before_mean']:.4f} -> {scaling_stats['after_mean']:.4f} (factor: {scaling_stats['scaling_factor']:.4f})"
                        )
                    if self.use_ebm_reward and ebm_reward_stats is not None:
                        log.info(
                            f"EBM rewards: mean {ebm_reward_stats['mean']:.4f} ± {ebm_reward_stats['std']:.4f} (scale: {self.ebm_reward_scale})"
                        )
                    if self.use_wandb:
                        log_dict = {
                            "total env step": cnt_train_step,
                            "loss": loss,
                            "pg loss": pg_loss,
                            "value loss": v_loss,
                            "entropy": -entropy_loss,
                            "std": std,
                            "approx kl": approx_kl,
                            "ratio": ratio,
                            "clipfrac": np.mean(clipfracs),
                            "explained variance": explained_var,
                            "avg episode reward - train (ENV)": avg_episode_reward,  # Always environment reward
                            "num episode - train": num_episode_finished,
                            # Raw step reward statistics
                            "train/raw_reward_mean": raw_reward_stats["mean"],
                            "train/raw_reward_std": raw_reward_stats["std"],
                            "train/raw_reward_min": raw_reward_stats["min"],
                            "train/raw_reward_max": raw_reward_stats["max"],
                            "train/raw_reward_median": raw_reward_stats["median"],
                            # Episode statistics
                            "train/episode_reward_std": episode_stats["std"],
                            "train/episode_reward_min": episode_stats["min"],
                            "train/episode_reward_max": episode_stats["max"],
                            "train/episode_reward_median": episode_stats["median"],
                            "train/avg_episode_length": episode_stats["avg_length"],
                            # Reward scaling information
                            "train/reward_scale_const": self.reward_scale_const,
                        }
                        
                        # Add EBM reward statistics if using EBM rewards
                        if self.use_ebm_reward and ebm_reward_stats is not None:
                            reward_type = "EBM_dynamic_norm" if self.use_dynamic_normalization else "EBM_guided"
                            log_dict.update({
                                "training_reward_type": reward_type,
                                "logging_reward_type": "Environment",
                                "ebm_reward_scale": self.ebm_reward_scale,
                                "train/ebm_reward_mean": ebm_reward_stats["mean"],
                                "train/ebm_reward_std": ebm_reward_stats["std"],
                                "train/ebm_reward_min": ebm_reward_stats["min"],
                                "train/ebm_reward_max": ebm_reward_stats["max"],
                                "train/ebm_reward_median": ebm_reward_stats["median"],
                            })
                            
                            # Add dynamic normalization statistics
                            if self.use_dynamic_normalization and self.energy_normalizer is not None:
                                norm_stats = self.energy_normalizer.get_stats()
                                log_dict.update({
                                    "train/energy_running_mean": norm_stats["running_mean"],
                                    "train/energy_running_std": norm_stats["running_std"],
                                    "train/energy_num_updates": norm_stats["num_updates"],
                                    "train/temperature_scale": norm_stats["temperature"],
                                    "train/normalization_momentum": norm_stats["momentum"],
                                })
                            # Add reward correlation analysis
                            if reward_correlation is not None:
                                log_dict.update({
                                    "train/ebm_env_correlation": reward_correlation["pearson_r"],
                                    "train/correlation_p_value": reward_correlation["p_value"],
                                    "train/correlation_sample_size": reward_correlation["sample_size"],
                                })
                        else:
                            log_dict.update({
                                "training_reward_type": "Environment",
                                "logging_reward_type": "Environment",
                            })
                        
                        # Add reward scaling statistics if scaling is enabled
                        if scaling_stats is not None:
                            log_dict.update({
                                "train/reward_scaling_enabled": True,
                                "train/reward_before_scaling_mean": scaling_stats["before_mean"],
                                "train/reward_before_scaling_std": scaling_stats["before_std"],
                                "train/reward_after_scaling_mean": scaling_stats["after_mean"],
                                "train/reward_after_scaling_std": scaling_stats["after_std"],
                                "train/reward_scaling_factor": scaling_stats["scaling_factor"],
                                "train/running_scaler_mean": scaling_stats["scaler_mean"],
                                "train/running_scaler_var": scaling_stats["scaler_var"],
                            })
                        else:
                            log_dict["train/reward_scaling_enabled"] = False
                        
                        wandb.log(log_dict, step=self.itr, commit=True)
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                
                # Save results
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            
            self.itr += 1