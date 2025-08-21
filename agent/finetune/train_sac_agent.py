"""
SAC trainer for Gaussian policy (non-diffusion).

Structure mirrors TrainPPOAgent and other trainers in this repo, adapted to SAC:
  - Online data collection with optional exploration phase
  - FIFO replay buffer
  - Separate replay ratios for critic and actor updates
  - Temperature auto-tuning towards target entropy
"""

import os
import pickle
import numpy as np
import torch
import logging
import wandb
from collections import deque

from agent.finetune.train_agent import TrainAgent
from util.timer import Timer

log = logging.getLogger(__name__)


class TrainSACAgent(TrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        # Hyperparameters
        self.gamma = cfg.train.gamma
        self.target_ema_rate = cfg.train.target_ema_rate
        self.scale_reward_factor = cfg.train.get("scale_reward_factor", 1.0)
        self.critic_replay_ratio = cfg.train.critic_replay_ratio
        self.actor_replay_ratio = cfg.train.actor_replay_ratio

        # Optimizers
        self.actor_optimizer = torch.optim.AdamW(
            self.model.network.parameters(),  # Gaussian actor network
            lr=cfg.train.actor_lr,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.model.critic.parameters(),
            lr=cfg.train.critic_lr,
        )

        # Temperature (entropy coefficient)
        init_temperature = cfg.train.init_temperature
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = cfg.train.target_entropy
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.train.critic_lr)

        # Replay buffer
        self.buffer_size = cfg.train.buffer_size
        self.n_explore_steps = cfg.train.n_explore_steps
        self.n_eval_episode = cfg.train.n_eval_episode

    def run(self):
        # FIFO buffers
        obs_buffer = deque(maxlen=self.buffer_size)
        next_obs_buffer = deque(maxlen=self.buffer_size)
        action_buffer = deque(maxlen=self.buffer_size)
        reward_buffer = deque(maxlen=self.buffer_size)
        terminated_buffer = deque(maxlen=self.buffer_size)

        timer = Timer()
        run_results = []
        cnt_train_step = 0
        done_venv = np.zeros((1, self.n_envs))

        while self.itr < self.n_train_itr:
            # Train/eval mode selection
            eval_mode = self.itr % self.val_freq == 0 and self.itr >= self.n_explore_steps and not self.force_train
            self.model.eval() if eval_mode else self.model.train()

            # Reset environments when needed
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or self.itr == 0:
                prev_obs_venv = self.reset_env_all(options_venv=[{} for _ in range(self.n_envs)])
                firsts_trajs[0] = 1
            else:
                firsts_trajs[0] = done_venv

            reward_trajs = np.zeros((self.n_steps, self.n_envs))

            # Collect trajectories
            for step in range(self.n_steps):
                # Select action
                if self.itr < self.n_explore_steps:
                    action_venv = self.venv.action_space.sample()
                else:
                    with torch.no_grad():
                        cond = {"state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)}
                        sampled = self.model(cond=cond, deterministic=eval_mode)
                        action_venv = sampled.cpu().numpy()[:, : self.act_steps]

                # Step vectorized env
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv
                reward_trajs[step] = reward_venv
                firsts_trajs[step + 1] = done_venv

                # Push transitions into replay buffer
                if not eval_mode:
                    for i in range(self.n_envs):
                        obs_buffer.append(prev_obs_venv["state"][i])
                        next_obs = info_venv[i]["final_obs"]["state"] if truncated_venv[i] else obs_venv["state"][i]
                        next_obs_buffer.append(next_obs)
                        action_buffer.append(action_venv[i])
                    reward_buffer.extend((reward_venv * self.scale_reward_factor).tolist())
                    terminated_buffer.extend(terminated_venv.tolist())

                # Prepare for next step
                prev_obs_venv = obs_venv
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0

            # Model updates
            if not eval_mode and self.itr >= self.n_explore_steps:
                num_batch_critic = int(self.n_steps * self.n_envs / self.batch_size * self.critic_replay_ratio)
                num_batch_actor = int(self.n_steps * self.n_envs / self.batch_size * self.actor_replay_ratio)

                # Convert to arrays once
                obs_array = np.array(obs_buffer)
                next_obs_array = np.array(next_obs_buffer)
                action_array = np.array(action_buffer)
                reward_array = np.array(reward_buffer)
                terminated_array = np.array(terminated_buffer)

                # Critic updates
                for _ in range(max(1, num_batch_critic)):
                    inds = np.random.choice(len(obs_array), self.batch_size)
                    obs_b = torch.from_numpy(obs_array[inds]).float().to(self.device)
                    next_obs_b = torch.from_numpy(next_obs_array[inds]).float().to(self.device)
                    actions_b = torch.from_numpy(action_array[inds]).float().to(self.device)
                    rewards_b = torch.from_numpy(reward_array[inds]).float().to(self.device)
                    terminated_b = torch.from_numpy(terminated_array[inds]).float().to(self.device)

                    obs_b = {"state": obs_b}
                    next_obs_b = {"state": next_obs_b}
                    alpha_val = self.log_alpha.exp().item()

                    loss_critic = self.model.loss_critic(
                        obs_b, next_obs_b, actions_b, rewards_b, terminated_b, self.gamma, alpha_val
                    )
                    self.critic_optimizer.zero_grad(); loss_critic.backward(); self.critic_optimizer.step()
                    self.model.update_target_critic(self.target_ema_rate)

                # Actor updates
                for _ in range(max(1, num_batch_actor)):
                    inds = np.random.choice(len(obs_array), self.batch_size)
                    obs_b = torch.from_numpy(obs_array[inds]).float().to(self.device)
                    obs_b = {"state": obs_b}
                    alpha_val = self.log_alpha.exp().item()

                    loss_actor = self.model.loss_actor(obs_b, alpha_val)
                    self.actor_optimizer.zero_grad(); loss_actor.backward(); self.actor_optimizer.step()

                # Temperature update (once using a fresh batch)
                inds = np.random.choice(len(obs_array), self.batch_size)
                obs_b = torch.from_numpy(obs_array[inds]).float().to(self.device)
                obs_b = {"state": obs_b}
                self.log_alpha_optimizer.zero_grad()
                loss_alpha = self.model.loss_temperature(obs_b, self.log_alpha.exp(), self.target_entropy)
                loss_alpha.backward(); self.log_alpha_optimizer.step()

            # Save
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Logging
            run_results.append({"itr": self.itr, "step": cnt_train_step})
            if self.itr % self.log_freq == 0 and self.itr >= self.n_explore_steps:
                time = timer()
                avg_episode_reward = float(np.mean(reward_trajs.sum(axis=0)))
                alpha_val = self.log_alpha.exp().item()
                log.info(
                    f"{self.itr}: step {cnt_train_step:8d} | avg episode reward {avg_episode_reward:8.4f} | alpha {alpha_val:8.4f} | t:{time:8.4f}"
                )
                if self.use_wandb:
                    wandb.log(
                        {
                            "total env step": cnt_train_step,
                            "avg episode reward - train": avg_episode_reward,
                            "entropy coeff": alpha_val,
                        },
                        step=self.itr,
                        commit=True,
                    )
            self.itr += 1


