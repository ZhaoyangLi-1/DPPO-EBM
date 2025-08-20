"""
Soft Actor-Critic (SAC) with Energy-Based Model integration for diffusion policy training.

This module extends the standard SAC training with energy-based model capabilities,
including potential-based reward shaping.
"""

import os
import pickle
import einops
import numpy as np
import torch
import torch.nn as nn
import logging
import wandb
import math
import copy
from typing import Dict, Any, Optional, List
from collections import deque

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_agent import TrainAgent
from model.diffusion.diffusion_sac_ebm import SACDiffusionEBM
from model.diffusion.energy_utils import build_k_use_indices


class TrainSACDiffusionEBMAgent(TrainAgent):
    """
    Enhanced SAC training agent with Energy-Based Model integration.
    
    This class extends the standard SAC training with potential-based reward shaping
    using energy-based models.
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # SAC parameters
        self.gamma = cfg.train.gamma
        self.tau = cfg.train.tau
        self.alpha = cfg.train.alpha
        # Support both keys: automatic_entropy_tuning (preferred) and auto_alpha (legacy)
        self.automatic_entropy_tuning = getattr(cfg.train, "automatic_entropy_tuning", getattr(cfg.train, "auto_alpha", True))
        self.target_entropy = cfg.train.target_entropy
        
        # EBM parameters
        self.use_ebm_reward_shaping = cfg.model.use_ebm_reward_shaping
        self.ebm_config = {
            "lambda": cfg.model.pbrs_lambda,
            "beta": cfg.model.pbrs_beta,
            "alpha": cfg.model.pbrs_alpha,
            "M": cfg.model.pbrs_M,
            "use_mu_only": cfg.model.pbrs_use_mu_only,
            "k_use_mode": cfg.model.pbrs_k_use_mode,
        }
        
        # Initialize EBM model if enabled
        self.ebm_model = None
        if self.use_ebm_reward_shaping:
            self.ebm_model = hydra.utils.instantiate(cfg.model.ebm)
            if hasattr(cfg.model, 'ebm_ckpt_path') and cfg.model.ebm_ckpt_path:
                checkpoint = torch.load(cfg.model.ebm_ckpt_path, map_location=self.device)
                self.ebm_model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
                log.info(f"Loaded EBM from {cfg.model.ebm_ckpt_path}")
        
        # Optimizers
        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor.parameters(),
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.actor_weight_decay,
        )
        self.critic_q1_optimizer = torch.optim.AdamW(
            self.model.critic_q1.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )
        self.critic_q2_optimizer = torch.optim.AdamW(
            self.model.critic_q2.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )
        self.critic_v_optimizer = torch.optim.AdamW(
            self.model.critic_v.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )
        
        # Alpha optimizer for entropy tuning
        if self.automatic_entropy_tuning:
            self.alpha_optimizer = torch.optim.Adam(
                [self.model.log_alpha],
                lr=cfg.train.critic_lr,
            )
        
        # Buffer parameters
        self.buffer_size = cfg.train.buffer_size
        self.batch_size = cfg.train.batch_size
        self.replay_ratio = cfg.train.replay_ratio
        
        # Update frequencies
        self.critic_update_freq = int(self.batch_size / self.replay_ratio)
        self.actor_update_freq = int(self.batch_size / self.replay_ratio)
        
        # Evaluation parameters
        self.n_eval_episode = cfg.train.n_eval_episode
        self.n_explore_steps = cfg.train.n_explore_steps
        
        # Setup potential function if EBM is enabled
        if self.use_ebm_reward_shaping:
            self._setup_potential_function(cfg)
        
        log.info(f"Initialized SAC EBM agent with buffer_size={self.buffer_size}, batch_size={self.batch_size}")
    
    def _setup_potential_function(self, cfg):
        """Set up the potential function for reward shaping."""
        if not self.use_ebm_reward_shaping or self.ebm_model is None:
            return

        # Load normalization statistics
        normalization_path = cfg.env.wrappers.mujoco_locomotion_lowdim.normalization_path
        stats = self._load_normalization_stats(normalization_path)
        
        # Setup potential function in the model
        self.model.setup_potential_function(stats)
        
        log.info("Setup potential function for EBM reward shaping")
    
    def _load_normalization_stats(self, normalization_path):
        """Load normalization statistics from file."""
        if normalization_path.endswith(".npz"):
            data = np.load(normalization_path)
            return {
                "obs_min": data["obs_min"],
                "obs_max": data["obs_max"],
                "act_min": data["action_min"],
                "act_max": data["action_max"],
            }
        elif normalization_path.endswith(".pkl"):
            with open(normalization_path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported normalization file format: {normalization_path}")
    
    def run(self):
        """Main training loop."""
        # Initialize replay buffer
        obs_buffer = deque(maxlen=self.buffer_size)
        next_obs_buffer = deque(maxlen=self.buffer_size)
        action_buffer = deque(maxlen=self.buffer_size)
        reward_buffer = deque(maxlen=self.buffer_size)
        done_buffer = deque(maxlen=self.buffer_size)
        
        # Training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        done_venv = np.zeros((1, self.n_envs))
        
        while self.itr < self.n_train_itr:
            if self.itr % 1000 == 0:
                print(f"Finished training iteration {self.itr} of {self.n_train_itr}")
            
            # Prepare video paths for rendering
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )
            
            # Define train or eval mode
            eval_mode = (
                self.itr % self.val_freq == 0
                and self.itr > self.n_explore_steps
                and not self.force_train
            )
            
            # Initialize episode data
            obs_trajs = {"state": np.zeros((self.n_steps, self.n_envs, self.obs_dim))}
            action_trajs = np.zeros((self.n_steps, self.n_envs, self.action_dim))
            reward_trajs = np.zeros((self.n_steps, self.n_envs))
            pbrs_reward_trajs = np.zeros((self.n_steps, self.n_envs))  # EBM reward shaping
            done_trajs = np.zeros((self.n_steps, self.n_envs))
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            
            # Reset environments
            obs_venv = self.venv.reset()
            firsts_trajs[0] = 1
            prev_obs_venv = obs_venv
            
            # Collect trajectories
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")
                
                # Sample actions using diffusion model
                with torch.no_grad():
                    cond = {
                        "state": torch.from_numpy(prev_obs_venv["state"])
                        .float()
                        .to(self.device)
                    }
                    samples = self.model.forward(
                        cond=cond,
                        deterministic=eval_mode,
                        return_chain=True,
                    )
                    output_venv = samples.trajectories.cpu().numpy()
                    chains_venv = samples.chains.cpu().numpy()
                
                action_venv = output_venv[:, :self.act_steps]
                
                # Apply multi-step action
                (
                    obs_venv,
                    reward_venv,
                    terminated_venv,
                    truncated_venv,
                    info_venv,
                ) = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv
                
                # Store trajectory data
                obs_trajs["state"][step] = prev_obs_venv["state"]
                action_trajs[step] = action_venv[:, 0]  # Store first action
                reward_trajs[step] = reward_venv
                done_trajs[step] = done_venv
                firsts_trajs[step + 1] = done_venv
                
                # Compute EBM reward shaping if enabled
                if self.use_ebm_reward_shaping:
                    try:
                        # Convert observations to tensors for potential computation
                        obs_t_tensor = torch.from_numpy(prev_obs_venv["state"][:, -1]).float().to(self.device)
                        obs_tp1_tensor = torch.from_numpy(obs_venv["state"][:, -1]).float().to(self.device)
                        
                        # Compute shaped reward: r_shape = γ * φ(s_{t+1}) - φ(s_t)
                        shaped_reward = self.model.compute_shaped_reward(obs_t_tensor, obs_tp1_tensor)
                        
                        # Scale by lambda and add to environment reward
                        pbrs_reward = self.ebm_config["lambda"] * shaped_reward.cpu().numpy()
                        pbrs_reward_trajs[step] = pbrs_reward
                        
                        # Add shaped reward to environment reward for training
                        reward_venv += pbrs_reward
                        reward_trajs[step] = reward_venv
                        
                    except Exception as e:
                        log.warning(f"Error in EBM reward shaping at step {step}: {e}")
                        pbrs_reward_trajs[step] = np.zeros(self.n_envs)
                else:
                    reward_trajs[step] = reward_venv
                    pbrs_reward_trajs[step] = np.zeros(self.n_envs)
                
                # Update for next step
                prev_obs_venv = obs_venv
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0
            
            # Summarize episode rewards
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start:end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                pbrs_reward_trajs_split = [
                    pbrs_reward_trajs[start:end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array([np.sum(reward_traj) for reward_traj in reward_trajs_split])
                pbrs_episode_reward = np.array([np.sum(pbrs_traj) for pbrs_traj in pbrs_reward_trajs_split])
                
                avg_episode_reward = np.mean(episode_reward)
                avg_pbrs_reward = np.mean(pbrs_episode_reward)
                success_rate = np.mean(
                    episode_reward >= self.best_reward_threshold_for_success
                )
            else:
                episode_reward = np.array([])
                pbrs_episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_pbrs_reward = 0
                success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")
            
            # Update models if not in eval mode
            if not eval_mode:
                # Store data in replay buffer
                for step in range(self.n_steps):
                    for env_ind in range(self.n_envs):
                        if not done_trajs[step, env_ind]:
                            obs_buffer.append(obs_trajs["state"][step, env_ind])
                            next_obs_buffer.append(obs_trajs["state"][min(step + 1, self.n_steps - 1), env_ind])
                            action_buffer.append(action_trajs[step, env_ind])
                            reward_buffer.append(reward_trajs[step, env_ind])
                            done_buffer.append(done_trajs[step, env_ind])
                
                # Update networks if buffer has enough samples
                if len(obs_buffer) >= self.batch_size:
                    self._update_networks(obs_buffer, next_obs_buffer, action_buffer, reward_buffer, done_buffer)
            
            # Update target networks
            self.model.update_target_networks()
            
            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()
            
            # Log results
            run_results.append({
                "itr": self.itr,
                "step": cnt_train_step,
            })
            
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg PBR reward {avg_pbrs_reward:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log({
                            "success rate - eval": success_rate,
                            "avg episode reward - eval": avg_episode_reward,
                            "avg PBR reward - eval": avg_pbrs_reward,
                            "num episode - eval": num_episode_finished,
                        }, step=self.itr, commit=False)
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_pbr_reward"] = avg_pbrs_reward
                else:
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | reward {avg_episode_reward:8.4f} | PBR {avg_pbrs_reward:8.4f} | t:{time:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log({
                            "total env step": cnt_train_step,
                            "avg episode reward - train": avg_episode_reward,
                            "avg PBR reward - train": avg_pbrs_reward,
                            "num episode - train": num_episode_finished,
                            "alpha": self.model.get_alpha().item(),
                        }, step=self.itr, commit=True)
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                    run_results[-1]["train_pbr_reward"] = avg_pbrs_reward
                
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            
            self.itr += 1
    
    def _update_networks(self, obs_buffer, next_obs_buffer, action_buffer, reward_buffer, done_buffer):
        """Update SAC networks."""
        # Sample batch from replay buffer
        batch_size = min(self.batch_size, len(obs_buffer))
        indices = np.random.choice(len(obs_buffer), batch_size, replace=False)
        
        obs_batch = torch.from_numpy(np.array([obs_buffer[i] for i in indices])).float().to(self.device)
        next_obs_batch = torch.from_numpy(np.array([next_obs_buffer[i] for i in indices])).float().to(self.device)
        action_batch = torch.from_numpy(np.array([action_buffer[i] for i in indices])).float().to(self.device)
        reward_batch = torch.from_numpy(np.array([reward_buffer[i] for i in indices])).float().to(self.device)
        done_batch = torch.from_numpy(np.array([done_buffer[i] for i in indices])).float().to(self.device)
        
        # Update critics
        q1_loss, q2_loss, v_loss = self.model.compute_critic_loss(
            obs_batch, action_batch, reward_batch, next_obs_batch, done_batch
        )
        
        self.critic_q1_optimizer.zero_grad()
        q1_loss.backward()
        self.critic_q1_optimizer.step()
        
        self.critic_q2_optimizer.zero_grad()
        q2_loss.backward()
        self.critic_q2_optimizer.step()
        
        self.critic_v_optimizer.zero_grad()
        v_loss.backward()
        self.critic_v_optimizer.step()
        
        # Update actor (ensure gradients flow into the actor by sampling without no_grad)
        # Sample actions from current policy for actor loss
        cond = {"state": obs_batch}
        samples = self.model.forward(cond, deterministic=False)
        actions_policy = samples.trajectories[:, 0]  # Take first action
        
        # Approximate log probabilities (kept as zeros; acts as no-entropy SAC/DPG)
        log_probs = torch.zeros(actions_policy.size(0), device=self.device)

        actor_loss = self.model.compute_actor_loss(obs_batch, actions_policy, log_probs)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha if automatic entropy tuning is enabled
        if self.automatic_entropy_tuning:
            # If log_probs are zeros (no-entropy variant), skip alpha update to avoid collapsing alpha->0
            if torch.allclose(log_probs, torch.zeros_like(log_probs)):
                alpha_loss = torch.tensor(0.0, device=self.device)
            else:
                alpha_loss = self.model.compute_alpha_loss(obs_batch, actions_policy, log_probs)
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.model.update_alpha(self.alpha_optimizer)
        
        # Log losses
        if self.use_wandb and self.itr % self.log_freq == 0:
            wandb.log({
                "loss/q1": q1_loss.item(),
                "loss/q2": q2_loss.item(),
                "loss/v": v_loss.item(),
                "loss/actor": actor_loss.item(),
                "loss/alpha": alpha_loss.item() if self.automatic_entropy_tuning else 0.0,
            }, step=self.itr, commit=False)


# Import hydra for configuration instantiation
import hydra
