"""
Enhanced DPPO fine-tuning agent with Energy-Based Model integration.

This module extends the standard PPODiffusionAgent with energy-based model
capabilities, including EnergyScalerPerK and potential-based reward shaping.
"""

import os
import pickle
import einops
import numpy as np
import torch
import logging
import wandb
import math
import copy

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_ppo_diffusion_agent import TrainPPODiffusionAgent
from model.diffusion.diffusion_ppo_ebm import PPODiffusionEBM
from model.diffusion.energy_utils import EnergyScalerPerK, build_k_use_indices


class TrainPPODiffusionEBMAgent(TrainPPODiffusionAgent):
    """
    Enhanced DPPO training agent with Energy-Based Model integration.
    
    This class extends TrainPPODiffusionAgent with:
    1. EnergyScalerPerK integration for per-step energy normalization
    2. Potential-based reward shaping using EBMs
    3. Enhanced logging and monitoring of energy-based components
    """
    
    def __init__(self, cfg):
        # Initialize parent class
        super().__init__(cfg)
        
        # Check if we should use EBM features
        self.use_energy_scaling = cfg.model.get("use_energy_scaling", False)
        self.use_pbrs = cfg.model.get("use_pbrs", False)
        
        # EBM model setup (if PBRS is enabled)
        self.ebm_model = None
        if self.use_pbrs:
            self._setup_ebm_model(cfg)
        
        # Energy scaling configuration
        if self.use_energy_scaling:
            log.info("Energy scaling enabled")
            self.energy_scaling_config = {
                "momentum": cfg.model.get("energy_scaling_momentum", 0.99),
                "eps": cfg.model.get("energy_scaling_eps", 1e-6),
                "use_mad": cfg.model.get("energy_scaling_use_mad", True),
            }
        
        # PBRS configuration
        if self.use_pbrs:
            log.info("Potential-based reward shaping enabled")
            self.pbrs_config = {
                "lambda": cfg.model.get("pbrs_lambda", 1.0),
                "beta": cfg.model.get("pbrs_beta", 1.0),
                "alpha": cfg.model.get("pbrs_alpha", 1.0),
                "M": cfg.model.get("pbrs_M", 1),
                "eta": cfg.model.get("pbrs_eta", 0.3),
                "use_mu_only": cfg.model.get("pbrs_use_mu_only", True),
                "k_use_mode": cfg.model.get("pbrs_k_use_mode", "tail:6"),
            }
        
        # Setup potential function if PBRS is enabled
        if self.use_pbrs and hasattr(self.model, 'setup_potential_function'):
            self._setup_potential_function()
    
    def _setup_ebm_model(self, cfg):
        """Set up the energy-based model for PBRS."""
        ebm_config = cfg.model.get("ebm", None)
        if ebm_config is None:
            raise ValueError("EBM configuration required when use_pbrs=True")
        
        # Import and instantiate EBM model
        from hydra.utils import instantiate
        self.ebm_model = instantiate(ebm_config)
        
        # Load EBM checkpoint if specified
        ebm_ckpt_path = cfg.model.get("ebm_ckpt_path", None)
        if ebm_ckpt_path is not None:
            checkpoint = torch.load(ebm_ckpt_path, map_location=self.device)
            self.ebm_model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
            log.info(f"Loaded EBM from {ebm_ckpt_path}")
        
        self.ebm_model.to(self.device)
        self.ebm_model.eval()
        
        # Freeze EBM parameters
        for param in self.ebm_model.parameters():
            param.requires_grad = False
    
    def _setup_potential_function(self):
        """Set up the potential function for reward shaping."""
        if not self.use_pbrs or not hasattr(self.model, 'setup_potential_function'):
            return
        
        # Load normalization statistics
        normalization_path = self.cfg.env.wrappers.mujoco_locomotion_lowdim.normalization_path
        stats = self._load_normalization_stats(normalization_path)
        
        # Setup potential function
        self.model.setup_potential_function(stats)
    
    def _load_normalization_stats(self, normalization_path):
        """Load normalization statistics from file."""
        if normalization_path.endswith('.npz'):
            data = np.load(normalization_path)
            return {
                "obs_min": data["obs_min"],
                "obs_max": data["obs_max"],
                "act_min": data["action_min"],
                "act_max": data["action_max"],
            }
        elif normalization_path.endswith('.pkl'):
            with open(normalization_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported normalization file format: {normalization_path}")
    
    def _compute_energy_values(self, obs, chains, denoising_inds):
        """
        Compute energy values for each denoising step.
        
        Args:
            obs: Observations [B, obs_dim]
            chains: Action chains [B, K+1, H, A]
            denoising_inds: Denoising step indices [B]
            
        Returns:
            Energy values [B, K] or None if EBM not available
        """
        if not self.use_pbrs or self.ebm_model is None:
            return None
        
        B, K_plus_1, H, A = chains.shape
        K = K_plus_1 - 1
        
        # Initialize energy values tensor
        energy_values = torch.zeros(B, K, device=self.device)
        
        # Compute energy for each denoising step
        for k in range(1, K + 1):
            # Get actions at denoising step k
            actions_k = chains[:, K - k, :, :]  # [B, H, A]
            
            # Normalize actions
            act_min = torch.tensor(self.stats["act_min"], dtype=torch.float32, device=self.device)
            act_max = torch.tensor(self.stats["act_max"], dtype=torch.float32, device=self.device)
            if act_min.ndim == 2:
                act_min = act_min.min(dim=0).values
            if act_max.ndim == 2:
                act_max = act_max.max(dim=0).values
            act_min = act_min.view(1, 1, -1)
            act_max = act_max.view(1, 1, -1)
            
            # Normalize to [-1, 1] range
            actions_norm = torch.clamp((actions_k - act_min) / (act_max - act_min + 1e-6) * 2 - 1, -1.0, 1.0)
            
            # Compute energy using EBM
            with torch.no_grad():
                k_vec = torch.full((B,), k, dtype=torch.long, device=self.device)
                E_out = self.ebm_model(k_idx=k_vec, views=None, poses=obs, actions=actions_norm)
                E = E_out[0] if isinstance(E_out, (tuple, list)) else E_out
                
                # Average over additional dimensions if present
                if E.dim() >= 2 and E.size(0) == B:
                    E = E.mean(dim=tuple(range(1, E.dim())))
                
                energy_values[:, k - 1] = E
        
        return energy_values
    
    def _compute_pbrs_reward(self, obs_t, obs_tp1, gamma=0.99):
        """
        Compute potential-based reward shaping reward.
        
        Args:
            obs_t: Current observations [B, obs_dim]
            obs_tp1: Next observations [B, obs_dim]
            gamma: Discount factor
            
        Returns:
            Shaped reward values [B]
        """
        if not self.use_pbrs or not hasattr(self.model, 'compute_pbrs_reward'):
            return torch.zeros(obs_t.size(0), device=self.device)
        
        return self.model.compute_pbrs_reward(obs_t, obs_tp1, gamma)
    
    def _log_energy_scaling_stats(self, step):
        """Log energy scaling statistics to wandb."""
        if not self.use_energy_scaling or not hasattr(self.model, 'get_energy_scaler_stats'):
            return
        
        if self.use_wandb and step % 100 == 0:
            # Log statistics for a few key denoising steps
            key_steps = [1, self.ft_denoising_steps // 2, self.ft_denoising_steps]
            for k in key_steps:
                if k <= self.ft_denoising_steps:
                    stats = self.model.get_energy_scaler_stats(k)
                    wandb.log({
                        f"energy_scaling/step_{k}/mean": stats["mean"],
                        f"energy_scaling/step_{k}/std": stats["std"],
                        f"energy_scaling/step_{k}/count": stats["count"],
                    }, step=step)
    
    def run(self):
        """Enhanced training loop with EBM integration."""
        # Start training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        
        while self.itr < self.n_train_itr:
            # Prepare video paths for each envs
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            # Define train or eval - all envs restart
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode

            # Reset env before iteration starts
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                firsts_trajs[0] = done_venv

            # Holder for trajectories
            obs_trajs = {
                "state": np.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim)
                )
            }
            chains_trajs = np.zeros(
                (
                    self.n_steps,
                    self.n_envs,
                    self.model.ft_denoising_steps + 1,
                    self.horizon_steps,
                    self.action_dim,
                )
            )
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            reward_trajs = np.zeros((self.n_steps, self.n_envs))
            pbrs_reward_trajs = np.zeros((self.n_steps, self.n_envs))  # PBRS rewards
            
            if self.save_full_observations:
                obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
                obs_full_trajs = np.vstack(
                    (obs_full_trajs, prev_obs_venv["state"][:, -1][None])
                )

            # Collect trajectories
            for step in range(self.n_steps):
                # Store observations
                obs_trajs["state"][step] = prev_obs_venv["state"]

                # Sample actions using diffusion model
                with torch.no_grad():
                    samples = self.model.forward(
                        cond=prev_obs_venv,
                        deterministic=eval_mode,
                        return_chain=True,
                    )
                    chains_trajs[step] = samples.chains.cpu().numpy()

                # Execute actions in environment
                actions = samples.actions.cpu().numpy()
                next_obs_venv, rewards, dones, infos = self.venv.step(actions)

                # Compute PBRS reward if enabled
                if self.use_pbrs:
                    obs_t = torch.from_numpy(prev_obs_venv["state"][:, -1]).float().to(self.device)
                    obs_tp1 = torch.from_numpy(next_obs_venv["state"][:, -1]).float().to(self.device)
                    pbrs_rewards = self._compute_pbrs_reward(obs_t, obs_tp1, self.gamma)
                    pbrs_reward_trajs[step] = pbrs_rewards.cpu().numpy()
                    
                    # Add PBRS reward to environment reward
                    rewards = rewards + pbrs_rewards.cpu().numpy()

                # Store results
                reward_trajs[step] = rewards
                terminated_trajs[step] = dones
                prev_obs_venv = next_obs_venv

                # Update done status
                done_venv = dones

            # Compute advantages and returns
            with torch.no_grad():
                last_obs = {"state": prev_obs_venv["state"]}
                last_values = self.model.critic(last_obs).cpu().numpy()

            # Compute GAE
            advantages = np.zeros((self.n_steps, self.n_envs))
            returns = np.zeros((self.n_steps, self.n_envs))
            gae = 0
            for step in reversed(range(self.n_steps)):
                if step == self.n_steps - 1:
                    next_values = last_values
                else:
                    next_values = self.model.critic({"state": obs_trajs["state"][step + 1]}).cpu().numpy()

                delta = (
                    reward_trajs[step]
                    + self.gamma * next_values * (1 - terminated_trajs[step])
                    - self.model.critic({"state": obs_trajs["state"][step]}).cpu().numpy()
                )
                gae = delta + self.gamma * self.gae_lambda * (1 - terminated_trajs[step]) * gae
                advantages[step] = gae
                returns[step] = gae + self.model.critic({"state": obs_trajs["state"][step]}).cpu().numpy()

            # Prepare data for training
            obs_batch = obs_trajs["state"].reshape(-1, self.n_cond_step, self.obs_dim)
            chains_batch = chains_trajs.reshape(-1, self.model.ft_denoising_steps + 1, self.horizon_steps, self.action_dim)
            advantages_batch = advantages.reshape(-1)
            returns_batch = returns.reshape(-1)
            old_values_batch = self.model.critic({"state": obs_batch}).detach()

            # Compute energy values if enabled
            energy_values = None
            if self.use_energy_scaling or self.use_pbrs:
                obs_flat = torch.from_numpy(obs_batch[:, -1]).float().to(self.device)
                chains_flat = torch.from_numpy(chains_batch).float().to(self.device)
                denoising_inds = torch.arange(self.model.ft_denoising_steps, device=self.device)
                energy_values = self._compute_energy_values(obs_flat, chains_flat, denoising_inds)

            # Training loop
            for epoch in range(self.update_epochs):
                # Shuffle data
                indices = np.random.permutation(len(obs_batch))
                
                for start_idx in range(0, len(obs_batch), self.batch_size):
                    end_idx = start_idx + self.batch_size
                    batch_indices = indices[start_idx:end_idx]
                    
                    # Prepare batch data
                    obs_batch_subset = {"state": obs_batch[batch_indices]}
                    chains_batch_subset = chains_batch[batch_indices]
                    advantages_batch_subset = advantages_batch[batch_indices]
                    returns_batch_subset = returns_batch[batch_indices]
                    old_values_batch_subset = old_values_batch[batch_indices]
                    
                    # Get old log probabilities
                    with torch.no_grad():
                        old_logprobs = self.model.get_logprobs(
                            obs_batch_subset,
                            torch.from_numpy(chains_batch_subset).float().to(self.device),
                            get_ent=False,
                        )
                    
                    # Prepare next observations for PBRS if enabled
                    next_obs_batch_subset = None
                    if self.use_pbrs and start_idx + self.batch_size < len(obs_batch):
                        next_batch_indices = indices[start_idx + self.batch_size:start_idx + self.batch_size + self.batch_size]
                        next_obs_batch_subset = {"state": obs_batch[next_batch_indices]}
                    
                    # Compute loss with EBM integration
                    loss_tuple = self.model.loss(
                        obs=obs_batch_subset,
                        chains_prev=torch.from_numpy(chains_batch_subset).float().to(self.device),
                        chains_next=torch.from_numpy(chains_batch_subset).float().to(self.device),
                        denoising_inds=torch.arange(self.model.ft_denoising_steps, device=self.device),
                        returns=torch.from_numpy(returns_batch_subset).float().to(self.device),
                        oldvalues=old_values_batch_subset,
                        advantages=torch.from_numpy(advantages_batch_subset).float().to(self.device),
                        oldlogprobs=old_logprobs,
                        use_bc_loss=False,
                        reward_horizon=self.reward_horizon,
                        energy_values=energy_values[batch_indices] if energy_values is not None else None,
                        next_obs=next_obs_batch_subset,
                        gamma=self.gamma,
                    )
                    
                    # Unpack loss components
                    if len(loss_tuple) >= 9:  # Enhanced version with PBRS reward
                        (pg_loss, entropy_loss, v_loss, clipfrac, approx_kl, 
                         ratio, bc_loss, eta, pbrs_reward) = loss_tuple
                    else:  # Standard version
                        (pg_loss, entropy_loss, v_loss, clipfrac, approx_kl, 
                         ratio, bc_loss, eta) = loss_tuple
                        pbrs_reward = 0.0
                    
                    # Backward pass
                    total_loss = pg_loss + self.vf_coef * v_loss + self.ent_coef * entropy_loss
                    if bc_loss > 0:
                        total_loss += bc_loss
                    
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                    # Update eta if learning it
                    if self.learn_eta:
                        self.eta_optimizer.zero_grad()
                        eta_loss = -eta
                        eta_loss.backward()
                        self.eta_optimizer.step()
                        self.eta_lr_scheduler.step()
                    
                    cnt_train_step += 1
                    
                    # Logging
                    if self.use_wandb and cnt_train_step % 10 == 0:
                        log_dict = {
                            "train/global_step": cnt_train_step,
                            "train/policy_loss": pg_loss.item(),
                            "train/value_loss": v_loss.item(),
                            "train/entropy_loss": entropy_loss.item(),
                            "train/clipfrac": clipfrac,
                            "train/approx_kl": approx_kl,
                            "train/ratio": ratio,
                            "train/eta": eta.item(),
                        }
                        
                        if bc_loss > 0:
                            log_dict["train/bc_loss"] = bc_loss.item()
                        
                        if self.use_pbrs:
                            log_dict["train/pbrs_reward"] = pbrs_reward
                            log_dict["train/mean_pbrs_reward"] = pbrs_reward_trajs.mean()
                        
                        wandb.log(log_dict, step=cnt_train_step)
                    
                    # Log energy scaling statistics
                    self._log_energy_scaling_stats(cnt_train_step)
            
            # Evaluation
            if self.itr % self.val_freq == 0:
                eval_results = self.evaluate_policy()
                run_results.append(eval_results)
                
                if self.use_wandb:
                    wandb.log({
                        "eval/return_mean": eval_results["return_mean"],
                        "eval/return_std": eval_results["return_std"],
                        "eval/success_rate": eval_results.get("success_rate", 0.0),
                    }, step=cnt_train_step)
            
            # Save model
            if self.itr % self.save_model_freq == 0:
                self.save_model()
            
            self.itr += 1
        
        return run_results
