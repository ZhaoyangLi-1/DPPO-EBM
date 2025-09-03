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
from model.diffusion.energy_utils import EnergyScalerPerK

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

        # EBM scalar reward replacement configuration
        self.use_ebm_reward = cfg.model.get("use_ebm_reward", False)
        self.ebm_reward_mode = cfg.model.get("ebm_reward_mode", "k0")  # "k0" or "dense"
        self.ebm_reward_clip_u_max = cfg.model.get("ebm_reward_clip_u_max", 5.0)
        self.ebm_reward_use_mad = cfg.model.get("ebm_reward_use_mad", True)
        self.ebm_reward_lambda = cfg.model.get("ebm_reward_lambda", 1.0)
        self.ebm_reward_lambda_per_k = cfg.model.get("ebm_reward_lambda_per_k", None)
        self.ebm_reward_baseline_M = cfg.model.get("ebm_reward_baseline_M", 4)
        self.ebm_reward_baseline_use_mu_only = cfg.model.get("ebm_reward_baseline_use_mu_only", False)

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

        # Macro time index per environment (episode-relative t_idx)
        self._macro_t_idx = np.zeros((self.n_envs,), dtype=np.int64)

        # No PBRS shaping: keep only EBM reward replacement

        # Reward scaling
        self.scale_reward_factor = cfg.train.get("scale_reward_factor", 1.0)

        # ------------------------------------------------------------------
        # Optional: EBM reward calibration state (per-k scaler + frozen BC)
        # ------------------------------------------------------------------
        self._reward_scaler = None
        self._pi0_for_reward = None
        if self.use_ebm_reward and getattr(self.model, "ebm_model", None) is not None:
            from model.diffusion.energy_utils import EnergyScalerPerK
            K = self.model.ft_denoising_steps
            self._reward_scaler = EnergyScalerPerK(
                K=K,
                momentum=cfg.model.get("energy_scaling_momentum", 0.99),
                eps=cfg.model.get("energy_scaling_eps", 1e-6),
                use_mad=self.ebm_reward_use_mad,
            )
            self._pi0_for_reward = deepcopy(self.model).eval()
            for p in self._pi0_for_reward.parameters():
                p.requires_grad = False
            log.info("Initialized SAC EBM reward calibration modules")

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
                # reset macro t_idx at episode starts
                self._macro_t_idx[:] = 0
            else:
                firsts_trajs[0] = done_venv

            reward_trajs = np.zeros((self.n_steps, self.n_envs))
            env_reward_trajs = np.zeros((self.n_steps, self.n_envs))

            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")
                # Select action
                if self.itr < self.n_explore_steps:
                    action_venv = self.venv.action_space.sample()
                else:
                    with torch.no_grad():
                        cond = {"state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)}
                        samples = self.model(cond=cond, deterministic=eval_mode, return_chain=True)
                        output_venv = samples.trajectories.cpu().numpy()
                        action_venv = output_venv[:, : self.act_steps]

                # Step env
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv

                # Replace env reward with calibrated EBM utility if enabled
                if self.use_ebm_reward and getattr(self.model, "ebm_model", None) is not None:
                    # Build obs tensor and generate current chain and BC baseline chain
                    with torch.no_grad():
                        s_t = torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)
                        cond = {"state": s_t}
                        # Current actions (deterministic=False for exploration)
                        samples_cur = self.model(cond=cond, deterministic=eval_mode, return_chain=True)
                        chains_cur = samples_cur.chains  # [n_env, K+1, H, A]
                        # Baseline from frozen BC
                        samples_bc = self._pi0_for_reward(cond=cond, deterministic=True, return_chain=True)
                        chains_bc = samples_bc.chains

                        # Normalize and compute energies per k
                        act_min = torch.tensor(self._stats["act_min"], dtype=torch.float32, device=self.device)
                        act_max = torch.tensor(self._stats["act_max"], dtype=torch.float32, device=self.device)
                        if act_min.ndim == 2:
                            act_min = act_min.min(dim=0).values
                        if act_max.ndim == 2:
                            act_max = act_max.max(dim=0).values
                        # Match [B, H, A]
                        act_min = act_min.view(1, 1, -1)
                        act_max = act_max.view(1, 1, -1)

                        # Build macro t_idx tensor per env (episode-relative)
                        t_vec = torch.from_numpy(self._macro_t_idx).to(self.device).long()
                        # print(f"t_vec: {t_vec}")

                        def energies_per_k(chains):
                            B, Kp1, H, A = chains.shape
                            E = torch.zeros(B, Kp1, device=self.device)
                            for k in range(Kp1):
                                # Map k=0 to the final (fully denoised) action
                                idx = Kp1 - 1 - k
                                actions_k = chains[:, idx]
                                actions_norm = torch.clamp((actions_k - act_min) / (act_max - act_min + 1e-6) * 2 - 1, -1.0, 1.0)
                                # Guard: ensure 3D [B,H,A]
                                if actions_norm.dim() == 4:
                                    actions_norm = actions_norm[:, idx]
                                k_vec = torch.full((B,), k, dtype=torch.long, device=self.device)
                                poses = s_t[:, -1, :] if s_t.dim() == 3 else s_t
                                E_out = self.model.ebm_model(k_idx=k_vec, t_idx=t_vec, views=None, poses=poses, actions=actions_norm)
                                e = E_out[0] if isinstance(E_out, (tuple, list)) else E_out
                                if e.dim() >= 2 and e.size(0) == B:
                                    e = e.mean(dim=tuple(range(1, e.dim())))
                                E[:, k] = e
                            return E

                        E_cur = energies_per_k(chains_cur)
                        # Robust baseline over M samples
                        E_base_list = []
                        M = max(1, int(self.ebm_reward_baseline_M))
                        for m in range(M):
                            det = bool(self.ebm_reward_baseline_use_mu_only)
                            samples_bc_m = self._pi0_for_reward(cond={"state": s_t}, deterministic=det, return_chain=True)
                            chains_bc_m = samples_bc_m.chains
                            E_base_list.append(energies_per_k(chains_bc_m).unsqueeze(0))
                        E_base = torch.cat(E_base_list, dim=0).median(dim=0).values

                        # Update scaler on baseline energies and compute utilities
                        B, Kp1 = E_cur.shape
                        for k in range(Kp1):
                            k_vec = torch.full((B,), k, dtype=torch.long)
                            self._reward_scaler.update(k_vec, E_base[:, k].detach().cpu())

                        s_vec = self._reward_scaler.s.to(self.device)
                        u = torch.zeros(B, Kp1, device=self.device)
                        eps = 1e-6
                        for k in range(Kp1):
                            tau_k = torch.clamp(s_vec[k], min=1e-3)
                            u[:, k] = -(E_cur[:, k] - E_base[:, k]) / (tau_k + eps)
                            u[:, k] = torch.clamp(u[:, k], -float(self.ebm_reward_clip_u_max), float(self.ebm_reward_clip_u_max))

                        if self.ebm_reward_mode == "dense":
                            if isinstance(self.ebm_reward_lambda_per_k, (list, tuple)):
                                lam = torch.tensor(list(self.ebm_reward_lambda_per_k)[:Kp1], dtype=torch.float32, device=self.device)
                                if lam.numel() < Kp1:
                                    lam = torch.nn.functional.pad(lam, (0, Kp1 - lam.numel()), value=float(self.ebm_reward_lambda))
                            else:
                                lam = torch.full((Kp1,), float(self.ebm_reward_lambda), device=self.device)
                            gamma_dn = float(getattr(self.model, "gamma_denoising", 1.0))
                            w = torch.tensor([gamma_dn ** k for k in range(Kp1)], dtype=torch.float32, device=self.device)
                            ebm_rew = (u * lam.view(1, -1) * w.view(1, -1)).sum(dim=1)
                        else:
                            ebm_rew = u[:, 0]
                        # Log raw env reward separately for fair comparison
                        env_reward_trajs[step] = reward_venv
                        reward_venv = ebm_rew.detach().cpu().numpy()

                # Store rewards
                if not (self.use_ebm_reward and getattr(self.model, "ebm_model", None) is not None):
                    env_reward_trajs[step] = reward_venv
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

                # Update macro t_idx per env: increment, and reset to 0 on done
                self._macro_t_idx += 1
                self._macro_t_idx[done_venv.astype(bool)] = 0

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
                env_reward_trajs_split = [
                    env_reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                )
                env_episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in env_reward_trajs_split]
                )
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
                avg_episode_reward = float(np.mean(episode_reward))
                avg_env_episode_reward = float(np.mean(env_episode_reward)) if num_episode_finished > 0 else 0.0
                avg_best_reward = float(np.mean(episode_best_reward))
                success_rate = float(
                    np.mean(
                        episode_best_reward >= self.best_reward_threshold_for_success
                    )
                )
            else:
                num_episode_finished = 0
                avg_episode_reward = 0.0
                avg_env_episode_reward = 0.0
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
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg episode env reward {avg_env_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "success rate - eval": success_rate,
                                "avg episode reward - eval": avg_episode_reward,
                                "avg episode env reward - eval": avg_env_episode_reward,
                                "avg best reward - eval": avg_best_reward,
                                "num episode - eval": num_episode_finished,
                            },
                            step=self.itr,
                            commit=True,
                        )
                else:
                    alpha_val = float(self.log_alpha.exp().item())
                    line = f"{self.itr}: step {cnt_train_step:8d} | reward {avg_episode_reward:8.4f} | env reward {avg_env_episode_reward:8.4f} | alpha {alpha_val:8.4f} | t:{time:8.4f}"
                    log.info(line)
                    if self.use_wandb:
                        payload = {
                            "total env step": cnt_train_step,
                            "avg episode reward - train": avg_episode_reward,
                            "avg episode env reward - train": avg_env_episode_reward,
                            "entropy coeff": alpha_val,
                            # keep minimal, remove verbose diagnostics
                            "method": "SAC+EBM",
                        }
                        wandb.log(payload, step=self.itr, commit=True)

            self.itr += 1


