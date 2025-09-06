"""
FDPP (Fine-tune Diffusion Policy with Human Preference) implementation.
Based on the paper: "FDPP: Fine-tune Diffusion Policy with Human Preference"

This module implements fine-tuning of diffusion policies using preference-based
reward learning with KL regularization.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import logging
import wandb
from typing import Dict, Optional

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_ppo_diffusion_agent import TrainPPODiffusionAgent
from model.rl.preference_reward_model import (
    PreferenceRewardModel,
    TrajectoryRewardModel,
    PreferenceDataset,
)


class TrainFDPPDiffusionAgent(TrainPPODiffusionAgent):
    """
    FDPP agent that fine-tunes diffusion policy with human preferences.
    Incorporates preference-based reward learning and KL regularization.
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # FDPP specific configurations
        self.use_preference_reward = cfg.get("use_preference_reward", True)
        self.kl_weight = cfg.get("kl_weight", 0.01)  # Alpha in the paper
        self.preference_buffer_size = cfg.get("preference_buffer_size", 10000)
        self.preference_batch_size = cfg.get("preference_batch_size", 256)
        self.preference_learning_rate = cfg.get("preference_lr", 1e-4)
        self.preference_update_freq = cfg.get("preference_update_freq", 10)
        self.n_preference_epochs = cfg.get("n_preference_epochs", 3)
        
        # Whether to use trajectory-level or state-level rewards
        self.use_trajectory_reward = cfg.get("use_trajectory_reward", False)
        
        # Initialize preference reward model
        if self.use_preference_reward:
            self._init_preference_model(cfg)
        
        # Store reference policy for KL regularization
        self._init_reference_policy()
        
        # Preference dataset
        self.preference_dataset = PreferenceDataset(
            max_size=self.preference_buffer_size
        )
        
        # Tracking
        self.preference_loss_history = []
        self.kl_divergence_history = []
    
    def _init_preference_model(self, cfg):
        """Initialize the preference-based reward model."""
        if self.use_trajectory_reward:
            self.preference_model = TrajectoryRewardModel(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dims=cfg.get("preference_hidden_dims", [256, 256]),
                activation=cfg.get("preference_activation", "relu"),
                output_activation=cfg.get("preference_output_activation", "tanh"),
                use_action=cfg.get("use_action_in_reward", False),
                device=self.device,
            )
        else:
            self.preference_model = PreferenceRewardModel(
                obs_dim=self.obs_dim,
                hidden_dims=cfg.get("preference_hidden_dims", [256, 256]),
                activation=cfg.get("preference_activation", "relu"),
                output_activation=cfg.get("preference_output_activation", "tanh"),
                device=self.device,
            )
        
        # Optimizer for preference model
        self.preference_optimizer = torch.optim.Adam(
            self.preference_model.parameters(),
            lr=self.preference_learning_rate,
        )
    
    def _init_reference_policy(self):
        """
        Initialize reference policy for KL regularization.
        This is a frozen copy of the pre-trained policy.
        """
        # Create a copy of the model for reference
        import copy
        self.reference_model = copy.deepcopy(self.model)
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        self.reference_model.eval()
    
    def compute_preference_reward(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reward using the preference model.
        
        Args:
            obs: Observation tensor
            actions: Optional action tensor for trajectory rewards
        
        Returns:
            Reward tensor
        """
        if not self.use_preference_reward:
            return torch.zeros(obs.shape[0], device=self.device)
        
        with torch.no_grad():
            if self.use_trajectory_reward and actions is not None:
                rewards = self.preference_model.compute_trajectory_reward(
                    obs, actions
                )
            else:
                # For state-level rewards
                if len(obs.shape) == 3:  # (batch, seq, dim)
                    batch_size, seq_len = obs.shape[:2]
                    obs_flat = obs.reshape(-1, obs.shape[-1])
                    rewards_flat = self.preference_model(obs_flat)
                    rewards = rewards_flat.reshape(batch_size, seq_len).sum(dim=1)
                else:
                    rewards = self.preference_model(obs).squeeze(-1)
        
        return rewards
    
    def compute_kl_divergence(
        self,
        obs: torch.Tensor,
        actions_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between current and reference policy.
        This implements the upper bound as described in the FDPP paper.
        
        Args:
            obs: Observation conditioning
            actions_pred: Predicted actions from current policy
        
        Returns:
            KL divergence tensor
        """
        kl_div = 0.0
        
        # Get the denoising trajectory from both policies
        with torch.no_grad():
            # For diffusion models, we compute KL at each denoising step
            # This is the upper bound mentioned in the paper
            
            # Get reference policy predictions
            ref_output = self.reference_model(obs, actions_pred)
            
            # Compute KL divergence between Gaussian distributions
            # Assuming both models output mean and log_std
            if hasattr(self.model, 'get_distribution_params'):
                curr_mean, curr_log_std = self.model.get_distribution_params(
                    obs, actions_pred
                )
                ref_mean, ref_log_std = self.reference_model.get_distribution_params(
                    obs, actions_pred
                )
                
                # KL divergence for Gaussians
                curr_var = torch.exp(2 * curr_log_std)
                ref_var = torch.exp(2 * ref_log_std)
                
                kl_div = 0.5 * (
                    (ref_var / curr_var).sum(dim=-1) +
                    ((curr_mean - ref_mean).pow(2) / curr_var).sum(dim=-1) -
                    curr_mean.shape[-1] +
                    (curr_log_std - ref_log_std).sum(dim=-1) * 2
                )
            else:
                # Simplified KL using MSE as proxy
                curr_output = self.model(obs, actions_pred)
                kl_div = F.mse_loss(curr_output, ref_output, reduction='none').mean(dim=-1)
        
        return kl_div
    
    def update_preference_model(self):
        """
        Update the preference reward model using collected preference data.
        """
        if len(self.preference_dataset) < self.preference_batch_size:
            return
        
        total_loss = 0
        for _ in range(self.n_preference_epochs):
            # Sample batch from preference dataset
            batch = self.preference_dataset.sample(self.preference_batch_size)
            
            # Convert to tensors
            obs_0 = torch.FloatTensor(batch['obs_0']).to(self.device)
            obs_1 = torch.FloatTensor(batch['obs_1']).to(self.device)
            labels = torch.LongTensor(batch['labels']).to(self.device)
            
            # Compute loss
            loss = self.preference_model.compute_preference_loss(
                obs_0, obs_1, labels
            )
            
            # Update model
            self.preference_optimizer.zero_grad()
            loss.backward()
            self.preference_optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / self.n_preference_epochs
        self.preference_loss_history.append(avg_loss)
        
        return avg_loss
    
    def collect_preference_labels(
        self,
        obs_batch: np.ndarray,
        simulated: bool = True,
    ):
        """
        Collect preference labels for observation pairs.
        In practice, this would query human feedback.
        For now, we simulate preferences based on task-specific criteria.
        
        Args:
            obs_batch: Batch of observations to create pairs from
            simulated: Whether to use simulated preferences
        """
        batch_size = len(obs_batch)
        
        if batch_size < 2:
            return
        
        # Create random pairs
        n_pairs = min(batch_size // 2, 100)  # Limit number of pairs
        
        for _ in range(n_pairs):
            idx1, idx2 = np.random.choice(batch_size, size=2, replace=False)
            obs_0 = obs_batch[idx1]
            obs_1 = obs_batch[idx2]
            
            if simulated:
                # Simulate preference based on some criteria
                # This should be replaced with actual human feedback
                label = self._simulate_preference(obs_0, obs_1)
            else:
                # In real implementation, query human for preference
                label = self._query_human_preference(obs_0, obs_1)
            
            # Add to dataset
            self.preference_dataset.add(obs_0, obs_1, label)
    
    def _simulate_preference(self, obs_0: np.ndarray, obs_1: np.ndarray) -> int:
        """
        Simulate preference label based on task-specific criteria.
        This is a placeholder that should be replaced with actual human feedback.
        
        Returns:
            -1: equal preference
            0: obs_0 preferred
            1: obs_1 preferred
        """
        # Example: prefer observations with lower magnitude (closer to origin)
        # This is task-specific and should be customized
        norm_0 = np.linalg.norm(obs_0)
        norm_1 = np.linalg.norm(obs_1)
        
        if abs(norm_0 - norm_1) < 0.1:
            return -1  # Equal preference
        elif norm_0 < norm_1:
            return 0  # obs_0 preferred
        else:
            return 1  # obs_1 preferred
    
    def _query_human_preference(self, obs_0: np.ndarray, obs_1: np.ndarray) -> int:
        """
        Query human for preference between two observations.
        This would typically involve a GUI or web interface.
        """
        # Placeholder for actual human feedback collection
        raise NotImplementedError("Human feedback collection not implemented")
    
    def compute_returns(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        obs: np.ndarray,
        actions: Optional[np.ndarray] = None,
    ):
        """
        Compute returns with preference rewards and KL regularization.
        Overrides parent method to incorporate FDPP-specific rewards.
        """
        # Get preference rewards if enabled
        if self.use_preference_reward:
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            actions_tensor = None
            if actions is not None:
                actions_tensor = torch.FloatTensor(actions).to(self.device)
            
            pref_rewards = self.compute_preference_reward(
                obs_tensor, actions_tensor
            ).cpu().numpy()
            
            # Combine task rewards with preference rewards
            rewards = rewards + pref_rewards
        
        # Compute returns using parent method
        returns, advantages = super().compute_returns(
            rewards, dones, values, obs
        )
        
        return returns, advantages
    
    def update_policy(
        self,
        obs_trajs: Dict,
        act_trajs: np.ndarray,
        ret_trajs: np.ndarray,
        adv_trajs: np.ndarray,
        logp_trajs: np.ndarray,
    ):
        """
        Update policy with KL regularization.
        Overrides parent method to add KL divergence term.
        """
        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs_trajs['state']).to(self.device)
        act_tensor = torch.FloatTensor(act_trajs).to(self.device)
        adv_tensor = torch.FloatTensor(adv_trajs).to(self.device)
        logp_old_tensor = torch.FloatTensor(logp_trajs).to(self.device)
        
        # Flatten batch and time dimensions
        obs_flat = obs_tensor.reshape(-1, *obs_tensor.shape[2:])
        act_flat = act_tensor.reshape(-1, *act_tensor.shape[2:])
        
        # Get current policy output
        policy_output = self.model(obs_flat, act_flat)
        logp_new = self.model.get_log_prob(act_flat, policy_output)
        
        # Compute policy ratio
        ratio = torch.exp(logp_new - logp_old_tensor.reshape(-1))
        
        # Compute PPO loss
        adv_flat = adv_tensor.reshape(-1)
        clip_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        loss_pi = -torch.min(ratio * adv_flat, clip_ratio * adv_flat).mean()
        
        # Add KL regularization
        if self.kl_weight > 0:
            kl_div = self.compute_kl_divergence(obs_flat, act_flat)
            kl_loss = self.kl_weight * kl_div.mean()
            total_loss = loss_pi + kl_loss
            
            # Track KL divergence
            self.kl_divergence_history.append(kl_div.mean().item())
        else:
            total_loss = loss_pi
            kl_loss = torch.tensor(0.0)
        
        # Update policy
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.grad_norm_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_norm_clip
            )
        self.optimizer.step()
        
        # Log metrics
        if self.use_wandb:
            wandb.log({
                "loss/policy": loss_pi.item(),
                "loss/kl_regularization": kl_loss.item(),
                "loss/total": total_loss.item(),
                "metrics/kl_divergence": kl_div.mean().item() if self.kl_weight > 0 else 0,
            })
        
        return total_loss.item()
    
    def run(self):
        """
        Main training loop for FDPP.
        Extends parent run method with preference learning updates.
        """
        # Start training loop
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
            
            # Define train or eval mode
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
            
            # Collect trajectories (simplified version)
            obs_trajs = {"state": []}
            act_trajs = []
            rew_trajs = []
            done_trajs = []
            
            for _ in range(self.n_steps):
                # Get action from policy
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(prev_obs_venv).to(self.device)
                    actions = self.model.sample(obs_tensor)
                    actions_np = actions.cpu().numpy()
                
                # Step environment
                obs_venv, rew_venv, done_venv, _ = self.step_env_all(
                    actions_np
                )
                
                # Store transitions
                obs_trajs["state"].append(prev_obs_venv)
                act_trajs.append(actions_np)
                rew_trajs.append(rew_venv)
                done_trajs.append(done_venv)
                
                prev_obs_venv = obs_venv
            
            # Convert lists to arrays
            obs_trajs["state"] = np.array(obs_trajs["state"])
            act_trajs = np.array(act_trajs)
            rew_trajs = np.array(rew_trajs)
            done_trajs = np.array(done_trajs)
            
            # Collect preference labels periodically
            if self.itr % self.preference_update_freq == 0:
                # Flatten observations for preference collection
                obs_flat = obs_trajs["state"].reshape(-1, self.obs_dim)
                self.collect_preference_labels(obs_flat, simulated=True)
                
                # Update preference model
                if len(self.preference_dataset) >= self.preference_batch_size:
                    pref_loss = self.update_preference_model()
                    log.info(f"Preference model loss: {pref_loss:.4f}")
            
            # Compute returns with preference rewards
            values = self.compute_values(obs_trajs["state"])
            returns, advantages = self.compute_returns(
                rew_trajs, done_trajs, values, obs_trajs["state"], act_trajs
            )
            
            # Update policy with KL regularization
            if not eval_mode:
                # Compute log probabilities for old policy
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs_trajs["state"]).to(self.device)
                    act_tensor = torch.FloatTensor(act_trajs).to(self.device)
                    logp_old = self.model.get_log_prob(act_tensor, self.model(obs_tensor))
                    logp_old_np = logp_old.cpu().numpy()
                
                # Perform policy updates
                for _ in range(self.n_updates):
                    self.update_policy(
                        obs_trajs, act_trajs, returns, advantages, logp_old_np
                    )
                
                cnt_train_step += 1
            
            # Logging
            avg_reward = rew_trajs.mean()
            log.info(
                f"Iteration {self.itr}: "
                f"Avg Reward = {avg_reward:.3f}, "
                f"KL Weight = {self.kl_weight:.4f}"
            )
            
            if self.use_wandb:
                wandb.log({
                    "iteration": self.itr,
                    "reward/average": avg_reward,
                    "reward/preference_dataset_size": len(self.preference_dataset),
                })
            
            # Save model periodically
            if self.itr % self.save_freq == 0:
                self.save(self.itr)
            
            self.itr += 1
        
        # Final save
        self.save(self.itr)
        log.info("FDPP training completed!")
    
    def save(self, itr: int):
        """
        Save model, preference model, and training state.
        """
        save_dict = {
            "iteration": itr,
            "model_state": self.model.state_dict(),
            "preference_model_state": self.preference_model.state_dict() if self.use_preference_reward else None,
            "reference_model_state": self.reference_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "preference_optimizer_state": self.preference_optimizer.state_dict() if self.use_preference_reward else None,
            "kl_weight": self.kl_weight,
            "preference_dataset_size": len(self.preference_dataset),
        }
        
        save_path = os.path.join(self.model_dir, f"fdpp_checkpoint_{itr}.pt")
        torch.save(save_dict, save_path)
        log.info(f"Saved FDPP checkpoint to {save_path}")
    
    def load(self, checkpoint_path: str):
        """
        Load model and training state from checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.itr = checkpoint["iteration"]
        self.model.load_state_dict(checkpoint["model_state"])
        
        if self.use_preference_reward and checkpoint["preference_model_state"] is not None:
            self.preference_model.load_state_dict(checkpoint["preference_model_state"])
            self.preference_optimizer.load_state_dict(checkpoint["preference_optimizer_state"])
        
        self.reference_model.load_state_dict(checkpoint["reference_model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.kl_weight = checkpoint["kl_weight"]
        
        log.info(f"Loaded FDPP checkpoint from {checkpoint_path}")