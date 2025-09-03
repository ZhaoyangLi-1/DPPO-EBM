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
from copy import deepcopy

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_ppo_agent import TrainPPOAgent


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
        super().__init__(cfg)
        
        # EBM reward configuration
        self.use_ebm_reward = cfg.model.get("use_ebm_reward", False)
        self.ebm_reward_scale = cfg.model.get("ebm_reward_scale", 1.0)
        self.ebm_reward_clip_max = cfg.model.get("ebm_reward_clip_max", 10.0)
        
        # EBM model setup
        self.ebm_model = None
        if self.use_ebm_reward:
            self._setup_ebm_model(cfg)
        
        log.info(f"SimpleMLP PPO Agent initialized with use_ebm_reward={self.use_ebm_reward}")

    def _setup_ebm_model(self, cfg):
        """Set up EBM model for reward computation."""
        try:
            # Import EBM model
            from model.ebm.ebm import EBM
            
            # Create EBM model
            ebm_cfg = cfg.model.ebm
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
            
            # Load EBM checkpoint if provided
            ebm_ckpt_path = cfg.model.get('ebm_ckpt_path', None)
            if ebm_ckpt_path and os.path.exists(ebm_ckpt_path):
                try:
                    checkpoint = torch.load(ebm_ckpt_path, map_location=self.device)
                    if "model" in checkpoint:
                        self.ebm_model.load_state_dict(checkpoint["model"])
                    else:
                        self.ebm_model.load_state_dict(checkpoint)
                    log.info(f"Loaded EBM checkpoint from {ebm_ckpt_path}")
                except Exception as e:
                    log.warning(f"Failed to load EBM checkpoint: {e}")
                    self.ebm_model = None
            else:
                log.warning(f"EBM checkpoint path not found: {ebm_ckpt_path}")
                self.ebm_model = None
            
            # Freeze EBM model
            if self.ebm_model is not None:
                for param in self.ebm_model.parameters():
                    param.requires_grad = False
                self.ebm_model.eval()
                
                # Set EBM model in the PPO model
                self.model.set_ebm_model(self.ebm_model)
                
        except ImportError as e:
            log.error(f"Failed to import EBM model: {e}")
            self.ebm_model = None
        except Exception as e:
            log.error(f"Failed to setup EBM model: {e}")
            self.ebm_model = None

    def _compute_ebm_rewards(self, obs_trajs, samples_trajs):
        """
        Compute EBM rewards for trajectory data.
        
        Args:
            obs_trajs: Observation trajectories dict with 'state' key
            samples_trajs: Action trajectories
            
        Returns:
            EBM rewards array [n_steps, n_envs]
        """
        if not self.use_ebm_reward or self.ebm_model is None:
            return np.zeros((self.n_steps, self.n_envs))
        
        ebm_rewards = np.zeros((self.n_steps, self.n_envs))
        
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
                    k_idx = torch.zeros(B, device=self.device, dtype=torch.long)
                    t_idx = torch.full((B,), step, device=self.device, dtype=torch.long)
                    
                    # Extract state features (take last observation if sequence)
                    states = batch_obs['state']  # [n_envs, cond_steps, obs_dim]
                    if states.dim() == 3:
                        states = states[:, -1]  # Take most recent state
                    
                    # Compute EBM energies
                    try:
                        energies, _ = self.ebm_model(
                            k_idx=k_idx,
                            t_idx=t_idx,
                            views=None,  # No vision for simple baseline
                            poses=states,  # [n_envs, obs_dim]
                            actions=batch_actions  # [n_envs, horizon_steps, action_dim]
                        )
                        
                        # Convert to rewards and apply scaling
                        rewards = energies.cpu().numpy() * self.ebm_reward_scale
                        
                        # Clip rewards for stability
                        rewards = np.clip(rewards, -self.ebm_reward_clip_max, self.ebm_reward_clip_max)
                        
                        ebm_rewards[step] = rewards
                        
                    except Exception as e:
                        log.warning(f"EBM computation failed at step {step}: {e}")
                        ebm_rewards[step] = 0.0
                        
        except Exception as e:
            log.error(f"Failed to compute EBM rewards: {e}")
            
        return ebm_rewards

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
            
            # Compute episode rewards and statistics
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            
            if len(episodes_start_end) > 0:
                # ALWAYS compute environment reward stats for logging (before EBM replacement)
                env_reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
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
            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")
            
            # Update models
            if not eval_mode:
                # Convert to tensors
                obs_trajs["state"] = (
                    torch.from_numpy(obs_trajs["state"]).float().to(self.device)
                )
                
                # Optionally replace environment rewards with EBM rewards
                if self.use_ebm_reward:
                    log.info(f"Computing EBM rewards for iteration {self.itr}")
                    ebm_rewards = self._compute_ebm_rewards(obs_trajs, samples_trajs)
                    # Store EBM reward stats for wandb logging
                    self._last_ebm_reward_mean = np.mean(ebm_rewards)
                    self._last_ebm_reward_std = np.std(ebm_rewards)
                    reward_trajs = ebm_rewards
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
                
                # Normalize rewards with running variance if specified
                if self.reward_scale_running:
                    reward_trajs_transpose = self.running_reward_scaler(
                        reward=reward_trajs.T, first=firsts_trajs[:-1].T
                    )
                    reward_trajs = reward_trajs_transpose.T
                
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
                    if self.use_wandb:
                        log_dict = {
                            "success rate - eval": success_rate,
                            "avg episode reward - eval (ENV)": avg_episode_reward,  # Always environment reward
                            "avg best reward - eval": avg_best_reward,
                            "num episode - eval": num_episode_finished,
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
                        }
                        if self.use_ebm_reward:
                            log_dict["training_reward_type"] = "EBM_guided"
                            log_dict["logging_reward_type"] = "Environment"
                            log_dict["ebm_reward_scale"] = self.ebm_reward_scale
                            # Optionally log EBM reward stats if available
                            if hasattr(self, '_last_ebm_reward_mean'):
                                log_dict["ebm_reward_mean"] = self._last_ebm_reward_mean
                                log_dict["ebm_reward_std"] = self._last_ebm_reward_std
                        else:
                            log_dict["training_reward_type"] = "Environment"
                            log_dict["logging_reward_type"] = "Environment"
                        wandb.log(log_dict, step=self.itr, commit=True)
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                
                # Save results
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            
            self.itr += 1