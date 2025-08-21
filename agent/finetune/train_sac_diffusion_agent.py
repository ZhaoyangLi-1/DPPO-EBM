"""
SAC training loop for diffusion policy actor.

Implements standard SAC with value target (V-network) and automatic entropy
temperature tuning using the diffusion actor's approximate log-probability.
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
from util.scheduler import CosineAnnealingWarmupRestarts

log = logging.getLogger(__name__)


class TrainSACDiffusionAgent(TrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        # Discount factor (applied per env step across action chunk)
        self.gamma = cfg.train.gamma

        # Optimizers
        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor_ft.parameters(),  # finetuned head
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.actor_weight_decay,
        )
        # Optional scheduler if provided in cfg
        self.actor_lr_scheduler = None
        if "actor_lr_scheduler" in cfg.train:
            self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.actor_optimizer,
                first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.train.actor_lr,
                min_lr=cfg.train.actor_lr_scheduler.min_lr,
                warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
                gamma=1.0,
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

        # No critic scheduler by default (multiple optimizers). Add if needed.
        self.critic_lr_scheduler = None

        # Replay buffer
        self.buffer_size = cfg.train.buffer_size
        self.batch_size = cfg.train.batch_size
        self.replay_ratio = cfg.train.replay_ratio

        # Target update rate
        self.tau = cfg.train.tau

        # Scale reward if needed
        self.scale_reward_factor = cfg.train.get("scale_reward_factor", 1.0)

        # Temperature (entropy regularization)
        self.automatic_entropy_tuning = cfg.train.automatic_entropy_tuning
        self.target_entropy = cfg.train.target_entropy
        if self.automatic_entropy_tuning:
            init_alpha = cfg.train.alpha
            self.log_alpha = torch.tensor(np.log(init_alpha), device=self.device, requires_grad=True)
            self.log_alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=cfg.train.critic_lr
            )
        else:
            self.log_alpha = torch.tensor(np.log(cfg.train.alpha), device=self.device, requires_grad=False)

        # Exploration steps and eval
        self.n_explore_steps = cfg.train.n_explore_steps
        self.n_eval_episode = cfg.train.n_eval_episode

    def run(self):
        # FIFO replay buffers
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
            # Eval mode scheduling
            eval_mode = self.itr % self.val_freq == 0 and self.itr >= self.n_explore_steps and not self.force_train
            self.model.eval() if eval_mode else self.model.train()

            # Reset per-iteration
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or self.itr == 0:
                prev_obs_venv = self.reset_env_all(options_venv=[{} for _ in range(self.n_envs)])
                firsts_trajs[0] = 1
            else:
                firsts_trajs[0] = done_venv

            reward_trajs = np.zeros((self.n_steps, self.n_envs))

            # Collect rollouts
            for step in range(self.n_steps):
                # Action selection
                if self.itr < self.n_explore_steps:
                    action_venv = self.venv.action_space.sample()
                else:
                    with torch.no_grad():
                        cond = {"state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)}
                        samples = self.model(cond=cond, deterministic=eval_mode, return_chain=False)
                        output_venv = samples.trajectories.cpu().numpy()
                        action_venv = output_venv[:, : self.act_steps]

                # Step envs
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv
                reward_trajs[step] = reward_venv
                firsts_trajs[step + 1] = done_venv

                # Store in buffer (online)
                if not eval_mode:
                    for i in range(self.n_envs):
                        obs_buffer.append(prev_obs_venv["state"][i])
                        next_obs = info_venv[i]["final_obs"]["state"] if truncated_venv[i] else obs_venv["state"][i]
                        next_obs_buffer.append(next_obs)
                        action_buffer.append(action_venv[i])
                    reward_buffer.extend((reward_venv * self.scale_reward_factor).tolist())
                    terminated_buffer.extend(terminated_venv.tolist())

                prev_obs_venv = obs_venv
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0

            # Summarize episode reward --- only count episodes that finish within the iteration.
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
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                )
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
                avg_episode_reward = float(np.mean(episode_reward))
                avg_best_reward = float(np.mean(episode_best_reward))
                success_rate = float(
                    np.mean(
                        episode_best_reward >= self.best_reward_threshold_for_success
                    )
                )
            else:
                num_episode_finished = 0
                avg_episode_reward = 0.0
                avg_best_reward = 0.0
                success_rate = 0.0

            # Training updates
            if not eval_mode and self.itr >= self.n_explore_steps:
                num_batch = int(self.n_steps * self.n_envs / self.batch_size * self.replay_ratio)

                obs_array = np.array(obs_buffer)
                next_obs_array = np.array(next_obs_buffer)
                action_array = np.array(action_buffer)
                reward_array = np.array(reward_buffer)
                terminated_array = np.array(terminated_buffer)

                # Accumulators for logging
                last_loss_q = None
                last_loss_v = None
                last_loss_pi = None
                last_loss_alpha = None

                for _ in range(max(1, num_batch)):
                    inds = np.random.choice(len(obs_array), self.batch_size)
                    obs_b = torch.from_numpy(obs_array[inds]).float().to(self.device)
                    next_obs_b = torch.from_numpy(next_obs_array[inds]).float().to(self.device)
                    actions_b = torch.from_numpy(action_array[inds]).float().to(self.device)
                    rewards_b = torch.from_numpy(reward_array[inds]).float().to(self.device)
                    terminated_b = torch.from_numpy(terminated_array[inds]).float().to(self.device)

                    obs_b = {"state": obs_b}
                    next_obs_b = {"state": next_obs_b}

                    # Critic Q update
                    loss_q = self.model.loss_critic(
                        obs_b, next_obs_b, actions_b, rewards_b, terminated_b, self.gamma
                    )
                    self.critic_q1_optimizer.zero_grad(); self.critic_q2_optimizer.zero_grad()
                    loss_q.backward()
                    self.critic_q1_optimizer.step(); self.critic_q2_optimizer.step()
                    last_loss_q = float(loss_q.detach().item())

                    # Value update
                    alpha = self.log_alpha.exp().item()
                    loss_v = self.model.loss_value(obs_b, alpha)
                    self.critic_v_optimizer.zero_grad()
                    loss_v.backward()
                    self.critic_v_optimizer.step()
                    last_loss_v = float(loss_v.detach().item())

                    # Actor update
                    self.actor_optimizer.zero_grad()
                    loss_pi = self.model.loss_actor(obs_b, alpha)
                    loss_pi.backward()
                    self.actor_optimizer.step()
                    last_loss_pi = float(loss_pi.detach().item())

                    # Temperature update
                    if self.automatic_entropy_tuning:
                        self.log_alpha_optimizer.zero_grad()
                        loss_alpha = self.model.loss_temperature(obs_b, self.log_alpha.exp(), self.target_entropy)
                        loss_alpha.backward()
                        self.log_alpha_optimizer.step()
                        last_loss_alpha = float(loss_alpha.detach().item())

                    # Target update
                    self.model.update_target_value(self.tau)

            # LR schedulers
            if self.actor_lr_scheduler is not None:
                self.actor_lr_scheduler.step()

            # Save
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Logging
            run_results.append({"itr": self.itr, "step": cnt_train_step})
            if self.itr % self.log_freq == 0:
                time = timer()
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "success rate - eval": success_rate,
                                "avg episode reward - eval": avg_episode_reward,
                                "avg best reward - eval": avg_best_reward,
                                "num episode - eval": num_episode_finished,
                            },
                            step=self.itr,
                            commit=True,
                        )
                else:
                    # Optional diagnostic batch for Q/V/logpi
                    diag = {}
                    if self.itr >= self.n_explore_steps and len(obs_buffer) >= self.batch_size:
                        inds = np.random.choice(len(obs_array), self.batch_size)
                        obs_b = torch.from_numpy(obs_array[inds]).float().to(self.device)
                        next_obs_b = torch.from_numpy(next_obs_array[inds]).float().to(self.device)
                        obs_diag = {"state": obs_b}
                        next_obs_diag = {"state": next_obs_b}
                        with torch.no_grad():
                            actions_d, chains_d = self.model._sample_action_and_chain(obs_diag, deterministic=False)
                            log_pi_d = self.model._chain_logprob(obs_diag, chains_d)
                            q1_d = self.model.critic_q1(obs_diag, actions_d).view(-1)
                            q2_d = self.model.critic_q2(obs_diag, actions_d).view(-1)
                            v_d = self.model.critic_v(obs_diag).view(-1)
                            target_v_d = self.model.target_v(next_obs_diag).view(-1)
                            y_d = torch.from_numpy(reward_array[inds]).float().to(self.device) + self.gamma * target_v_d * (1.0 - torch.from_numpy(terminated_array[inds]).float().to(self.device))
                        diag = {
                            "logpi mean": float(log_pi_d.mean().item()),
                            "logpi std": float(log_pi_d.std().item()),
                            "q1 mean": float(q1_d.mean().item()),
                            "q2 mean": float(q2_d.mean().item()),
                            "v mean": float(v_d.mean().item()),
                            "q_target mean": float(y_d.mean().item()),
                        }

                    alpha_val = float(self.log_alpha.exp().item())
                    line = f"{self.itr}: step {cnt_train_step:8d} | reward {avg_episode_reward:8.4f} | alpha {alpha_val:8.4f} | t:{time:8.4f}"
                    if self.itr >= self.n_explore_steps and last_loss_q is not None:
                        line += f" | loss_q {last_loss_q:8.4f} | loss_v {last_loss_v:8.4f} | loss_pi {last_loss_pi:8.4f}"
                        if last_loss_alpha is not None:
                            line += f" | loss_alpha {last_loss_alpha:8.4f}"
                    log.info(line)
                    if self.use_wandb:
                        payload = {
                            "total env step": cnt_train_step,
                            "avg episode reward - train": avg_episode_reward,
                            "entropy coeff": alpha_val,
                            "buffer size": len(obs_buffer),
                        }
                        if self.itr >= self.n_explore_steps and last_loss_q is not None:
                            payload.update(
                                {
                                    "loss - critic": last_loss_q,
                                    "loss - value": last_loss_v,
                                    "loss - actor": last_loss_pi,
                                }
                            )
                            if last_loss_alpha is not None:
                                payload["loss - alpha"] = last_loss_alpha
                        payload.update(diag)
                        wandb.log(payload, step=self.itr, commit=True)
            self.itr += 1


