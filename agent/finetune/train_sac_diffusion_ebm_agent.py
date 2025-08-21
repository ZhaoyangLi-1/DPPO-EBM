"""
SAC training loop for diffusion policy actor with EBM integration.

Differences vs TrainSACDiffusionAgent:
  - Replace actor's log pi with -E_psi(s,a) in actor objective (handled by model).
  - Optional potential-based reward shaping applied at rollout using KFreePotential.
  - Optional per-k energy normalization (EnergyScalerPerK) handled here if needed.
"""

import os
import pickle
import numpy as np
import torch
import logging
import wandb
from collections import deque
from copy import deepcopy

from agent.finetune.train_agent import TrainAgent
from util.timer import Timer
from util.scheduler import CosineAnnealingWarmupRestarts
from model.diffusion.energy_utils import KFreePotential, build_k_use_indices, EnergyScalerPerK

log = logging.getLogger(__name__)


class TrainSACDiffusionEBMAgent(TrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        # SAC hyperparams
        self.gamma = cfg.train.gamma
        self.tau = cfg.train.tau

        # Optimizers
        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor_ft.parameters(),
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.actor_weight_decay,
        )
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

        # Buffer
        self.buffer_size = cfg.train.buffer_size
        self.batch_size = cfg.train.batch_size
        self.replay_ratio = cfg.train.replay_ratio

        # Temperature
        self.automatic_entropy_tuning = cfg.train.automatic_entropy_tuning
        self.target_entropy = cfg.train.target_entropy
        if self.automatic_entropy_tuning:
            init_alpha = cfg.train.alpha
            self.log_alpha = torch.tensor(np.log(init_alpha), device=self.device, requires_grad=True)
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.train.critic_lr)
        else:
            self.log_alpha = torch.tensor(np.log(cfg.train.alpha), device=self.device, requires_grad=False)

        # Exploration/eval
        self.n_explore_steps = cfg.train.n_explore_steps
        self.n_eval_episode = cfg.train.n_eval_episode

        # EBM reward shaping configuration
        self.use_ebm_reward_shaping = cfg.model.get("use_ebm_reward_shaping", False)
        self.pbrs_lambda = cfg.model.get("pbrs_lambda", 1.0)
        self.pbrs_beta = cfg.model.get("pbrs_beta", 1.0)
        self.pbrs_alpha = cfg.model.get("pbrs_alpha", 1.0)
        self.pbrs_M = cfg.model.get("pbrs_M", 1)
        self.pbrs_use_mu_only = cfg.model.get("pbrs_use_mu_only", True)
        self.pbrs_k_use_mode = cfg.model.get("pbrs_k_use_mode", "tail:6")

        # Attach EBM to model if provided by Hydra config
        if hasattr(cfg.model, "ebm") and cfg.model.ebm is not None:
            from hydra.utils import instantiate
            ebm_model = instantiate(cfg.model.ebm)
            ebm_ckpt_path = cfg.model.get("ebm_ckpt_path", None)
            if hasattr(self.model, "set_ebm"):
                self.model.set_ebm(ebm_model, ebm_ckpt_path)
            else:
                self.model.ebm_model = ebm_model
                if ebm_ckpt_path is not None:
                    state = torch.load(ebm_ckpt_path, map_location=self.device)
                    self.model.ebm_model.load_state_dict(state.get("model", state), strict=False)
                for p in self.model.ebm_model.parameters():
                    p.requires_grad = False

        # Load normalization stats and provide to model for energy inputs
        normalization_path = self.cfg.env.wrappers.mujoco_locomotion_lowdim.normalization_path
        stats = self._load_normalization_stats(normalization_path)
        if hasattr(self.model, "setup_action_norm_stats"):
            self.model.setup_action_norm_stats(stats)
        self._stats = stats

        # PBRS potential
        self.potential = None
        if self.use_ebm_reward_shaping and getattr(self.model, "ebm_model", None) is not None:
            pi0 = deepcopy(self.model).eval()
            for p in pi0.parameters():
                p.requires_grad = False
            K = self.model.ft_denoising_steps
            k_use = build_k_use_indices(self.pbrs_k_use_mode, K)
            self.potential = KFreePotential(
                ebm_model=self.model.ebm_model,
                pi0=pi0,
                K=K,
                stats=stats,
                k_use=k_use,
                beta_k=self.pbrs_beta,
                alpha=self.pbrs_alpha,
                M=self.pbrs_M,
                use_mu_only=self.pbrs_use_mu_only,
                device=self.device,
            )
            log.info(f"Initialized PBRS with k_use={k_use}")

        # Reward scaling
        self.scale_reward_factor = cfg.train.get("scale_reward_factor", 1.0)

    def _load_normalization_stats(self, normalization_path):
        if normalization_path.endswith(".npz"):
            data = np.load(normalization_path)
            return {
                "obs_min": data["obs_min"],
                "obs_max": data["obs_max"],
                "act_min": data["action_min"],
                "act_max": data["action_max"],
            }
        elif normalization_path.endswith(".pkl"):
            import pickle as pkl
            with open(normalization_path, "rb") as f:
                return pkl.load(f)
        else:
            raise ValueError(f"Unsupported normalization file format: {normalization_path}")

    def run(self):
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
            eval_mode = self.itr % self.val_freq == 0 and self.itr >= self.n_explore_steps and not self.force_train
            self.model.eval() if eval_mode else self.model.train()

            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or self.itr == 0:
                prev_obs_venv = self.reset_env_all(options_venv=[{} for _ in range(self.n_envs)])
                firsts_trajs[0] = 1
            else:
                firsts_trajs[0] = done_venv

            reward_trajs = np.zeros((self.n_steps, self.n_envs))

            for step in range(self.n_steps):
                # Select action
                if self.itr < self.n_explore_steps:
                    action_venv = self.venv.action_space.sample()
                else:
                    with torch.no_grad():
                        cond = {"state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)}
                        samples = self.model(cond=cond, deterministic=eval_mode, return_chain=False)
                        output_venv = samples.trajectories.cpu().numpy()
                        action_venv = output_venv[:, : self.act_steps]

                # Step env
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv

                # PBRS at rollout
                if self.potential is not None:
                    with torch.no_grad():
                        s_t = torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)
                        s_tp1 = torch.from_numpy(obs_venv["state"]).float().to(self.device)
                        r_shape = self.potential.shape_reward(s_t, s_tp1, self.gamma).cpu().numpy()
                    reward_venv = reward_venv + self.pbrs_lambda * r_shape

                reward_trajs[step] = reward_venv
                firsts_trajs[step + 1] = done_venv

                # Buffer add
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

            # Summarize episode reward for eval/report
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

            # Updates
            if not eval_mode and self.itr >= self.n_explore_steps:
                num_batch = int(self.n_steps * self.n_envs / self.batch_size * self.replay_ratio)

                obs_array = np.array(obs_buffer)
                next_obs_array = np.array(next_obs_buffer)
                action_array = np.array(action_buffer)
                reward_array = np.array(reward_buffer)
                terminated_array = np.array(terminated_buffer)

                # Accumulate metrics
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
                    loss_q = self.model.loss_critic(obs_b, next_obs_b, actions_b, rewards_b, terminated_b, self.gamma)
                    self.critic_q1_optimizer.zero_grad(); self.critic_q2_optimizer.zero_grad()
                    loss_q.backward()
                    self.critic_q1_optimizer.step(); self.critic_q2_optimizer.step()
                    last_loss_q = float(loss_q.detach().item())

                    # Value update
                    alpha = self.log_alpha.exp().item()
                    loss_v = self.model.loss_value(obs_b, alpha)
                    self.critic_v_optimizer.zero_grad(); loss_v.backward(); self.critic_v_optimizer.step()
                    last_loss_v = float(loss_v.detach().item())

                    # Actor update (uses EBM internally)
                    self.actor_optimizer.zero_grad()
                    loss_pi = self.model.loss_actor(obs_b, alpha)
                    loss_pi.backward(); self.actor_optimizer.step()
                    last_loss_pi = float(loss_pi.detach().item())

                    # Temperature update
                    if self.automatic_entropy_tuning:
                        self.log_alpha_optimizer.zero_grad()
                        loss_alpha = self.model.loss_temperature(obs_b, self.log_alpha.exp(), self.target_entropy)
                        loss_alpha.backward(); self.log_alpha_optimizer.step()
                        last_loss_alpha = float(loss_alpha.detach().item())

                    # Target update
                    self.model.update_target_value(self.tau)

            # LR
            if self.actor_lr_scheduler is not None:
                self.actor_lr_scheduler.step()

            # Save
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Log
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
                    # Diagnostics for Q/V/E and shaped reward
                    diag = {}
                    if self.itr >= self.n_explore_steps and len(obs_buffer) >= self.batch_size:
                        inds = np.random.choice(len(obs_array), self.batch_size)
                        obs_b = torch.from_numpy(obs_array[inds]).float().to(self.device)
                        next_obs_b = torch.from_numpy(next_obs_array[inds]).float().to(self.device)
                        obs_diag = {"state": obs_b}
                        next_obs_diag = {"state": next_obs_b}
                        with torch.no_grad():
                            actions_d, chains_d = self.model._sample_action_and_chain(obs_diag, deterministic=False)
                            q1_d = self.model.critic_q1(obs_diag, actions_d).view(-1)
                            q2_d = self.model.critic_q2(obs_diag, actions_d).view(-1)
                            v_d = self.model.critic_v(obs_diag).view(-1)
                            target_v_d = self.model.target_v(next_obs_diag).view(-1)
                            if getattr(self.model, "ebm_model", None) is not None:
                                E_d = self.model._compute_energy(obs_diag, actions_d)
                                diag["energy mean"] = float(E_d.mean().item())
                                diag["energy std"] = float(E_d.std().item())
                            if self.potential is not None:
                                r_shape_d = self.potential.shape_reward(obs_b, next_obs_b, self.gamma)
                                diag["pbrs mean"] = float(r_shape_d.mean().item())
                                diag["pbrs std"] = float(r_shape_d.std().item())
                        diag.update({
                            "q1 mean": float(q1_d.mean().item()),
                            "q2 mean": float(q2_d.mean().item()),
                            "v mean": float(v_d.mean().item()),
                            "v_target mean": float(target_v_d.mean().item()),
                        })

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
                            payload.update({
                                "loss - critic": last_loss_q,
                                "loss - value": last_loss_v,
                                "loss - actor": last_loss_pi,
                            })
                            if last_loss_alpha is not None:
                                payload["loss - alpha"] = last_loss_alpha
                        payload.update(diag)
                        wandb.log(payload, step=self.itr, commit=True)

            self.itr += 1


