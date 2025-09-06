import os
import pickle
import einops
import numpy as np
import torch
import logging
import wandb
import math
from scipy.stats import pearsonr, spearmanr, kendalltau
log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_ppo_agent import TrainPPOAgent


class DynamicEnergyNormalizer:
    def __init__(self, momentum=0.99, temperature=1.0, epsilon=1e-8):
        self.momentum = momentum
        self.temperature = temperature
        self.epsilon = epsilon
        self.running_mean = 0.0
        self.running_var = 1.0
        self.num_updates = 0

    def update(self, energies):
        if len(energies) == 0:
            return energies
        batch_mean = np.mean(energies)
        batch_var = np.var(energies)
        if self.num_updates == 0:
            self.running_mean = batch_mean
            self.running_var = batch_var
        else:
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        self.num_updates += 1
        normalized_energies = (energies - self.running_mean) / (np.sqrt(self.running_var) + self.epsilon)
        scaled_energies = normalized_energies / self.temperature
        return scaled_energies

    def normalize_without_update(self, energies):
        if len(energies) == 0:
            return energies
        normalized_energies = (energies - self.running_mean) / (np.sqrt(self.running_var) + self.epsilon)
        scaled_energies = normalized_energies / self.temperature
        return scaled_energies

    def get_stats(self):
        return {
            "running_mean": self.running_mean,
            "running_var": self.running_var,
            "running_std": np.sqrt(self.running_var),
            "num_updates": self.num_updates,
            "momentum": self.momentum,
            "temperature": self.temperature
        }


class TrainSimpleMLP_PPOAgent(TrainPPOAgent):
    def __init__(self, cfg):
        self.use_ebm_reward = cfg.model.get("use_ebm_reward", False)
        self.ebm_reward_scale = cfg.model.get("ebm_reward_scale", 1.0)
        self.ebm_reward_clip_max = cfg.model.get("ebm_reward_clip_max", 10.0)
        self.use_auto_scaling = cfg.model.get("use_auto_scaling", False)
        self.use_dynamic_normalization = cfg.model.get("use_dynamic_normalization", True)
        self.normalization_momentum = cfg.model.get("normalization_momentum", 0.99)
        self.temperature_scale = cfg.model.get("temperature_scale", 1.0)
        self.normalization_epsilon = cfg.model.get("normalization_epsilon", 1e-8)
        self.adv_std_target = cfg.model.get("adv_std_target", 1.0)
        self.adv_std_tol = cfg.model.get("adv_std_tol", 0.3)
        self.temp_lr = cfg.model.get("temp_lr", 0.10)
        self.scale_lr = cfg.model.get("scale_lr", 0.05)
        self.min_temp = cfg.model.get("min_temp", 0.3)
        self.max_temp = cfg.model.get("max_temp", 3.0)
        self.ebm_model = None
        self.energy_normalizer = None
        super().__init__(cfg)
        if self.use_ebm_reward:
            self._setup_ebm_model(cfg)
            if self.use_dynamic_normalization:
                self.energy_normalizer = DynamicEnergyNormalizer(
                    momentum=self.normalization_momentum,
                    temperature=self.temperature_scale,
                    epsilon=self.normalization_epsilon
                )
            if self.ebm_model is not None and hasattr(self.model, 'set_ebm_model'):
                self.model.set_ebm_model(self.ebm_model)
        if self.use_ebm_reward and self.ebm_model is None:
            log.error("EBM model failed to load")

    def _setup_ebm_model(self, cfg):
        try:
            from model.ebm.ebm import EBM
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
            ebm_ckpt_path = cfg.model.get('ebm_ckpt_path', None)
            if ebm_ckpt_path and os.path.exists(ebm_ckpt_path):
                checkpoint = torch.load(ebm_ckpt_path, map_location=self.device)
                if "model" in checkpoint:
                    self.ebm_model.load_state_dict(checkpoint["model"])
                else:
                    self.ebm_model.load_state_dict(checkpoint)
            else:
                self.ebm_model = None
            if self.ebm_model is not None:
                for p in self.ebm_model.parameters():
                    p.requires_grad = False
                self.ebm_model.eval()
        except Exception as e:
            log.error(f"EBM setup error: {e}")
            self.ebm_model = None

    def _compute_ebm_rewards(self, obs_trajs, samples_trajs, firsts_trajs=None, env_reward_trajs=None, eval_mode=False):
        if not self.use_ebm_reward or self.ebm_model is None:
            return np.zeros((self.n_steps, self.n_envs))
        ebm_rewards = np.zeros((self.n_steps, self.n_envs))
        env_step_counters = np.zeros(self.n_envs, dtype=int)
        try:
            with torch.no_grad():
                for step in range(self.n_steps):
                    batch_obs = {'state': obs_trajs['state'][step].to(self.device)}
                    batch_actions = torch.from_numpy(samples_trajs[step]).float().to(self.device)
                    B = self.n_envs
                    k_idx = torch.zeros(B, device=self.device, dtype=torch.long)
                    states = batch_obs['state']
                    if states.dim() == 3:
                        states = states[:, -1]
                    if firsts_trajs is not None and firsts_trajs.shape[0] >= step:
                        resets = firsts_trajs[step].astype(bool)
                        env_step_counters[resets] = 0
                    env_t_idx = torch.from_numpy(env_step_counters).long().to(self.device)
                    max_t = int(getattr(self, "max_episode_steps", 1000) - 1)
                    env_t_idx = torch.clamp(env_t_idx, 0, max_t)
                    env_step_counters += 1
                    ebm_output = self.ebm_model(
                        k_idx=k_idx,
                        t_idx=env_t_idx,
                        views=None,
                        poses=states,
                        actions=batch_actions
                    )
                    if isinstance(ebm_output, tuple) and len(ebm_output) >= 1:
                        energies = ebm_output[0]
                    else:
                        energies = torch.zeros(B, device=self.device)
                    raw_energies = energies.cpu().numpy()
                    if self.use_dynamic_normalization and self.energy_normalizer is not None:
                        if eval_mode:
                            normalized_energies = self.energy_normalizer.normalize_without_update(raw_energies)
                        else:
                            normalized_energies = self.energy_normalizer.update(raw_energies)
                        ebm_step_rewards = -normalized_energies
                    else:
                        ebm_step_rewards = -raw_energies
                    ebm_rewards[step] = ebm_step_rewards
        except Exception as e:
            log.error(f"EBM reward compute error: {e}")
        if self.use_dynamic_normalization and self.energy_normalizer is not None:
            ebm_rewards = ebm_rewards * self.ebm_reward_scale
        else:
            ebm_rewards = ebm_rewards * self.ebm_reward_scale
        log.info(f"Final EBM rewards: range [{ebm_rewards.min():.3f}, {ebm_rewards.max():.3f}], mean {ebm_rewards.mean():.3f}")
        return ebm_rewards

    def _compute_auto_scale(self, ebm_rewards, env_rewards):
        ebm_array = np.array(ebm_rewards)
        env_array = np.array(env_rewards)
        ebm_std = np.std(ebm_array)
        env_std = np.std(env_array)
        std_scale = env_std / ebm_std if ebm_std > 1e-8 else 1.0
        ebm_range = np.max(ebm_array) - np.min(ebm_array)
        env_range = np.max(env_array) - np.min(env_array)
        range_scale = env_range / ebm_range if ebm_range > 1e-8 else 1.0
        ebm_90 = np.percentile(np.abs(ebm_array), 90)
        env_90 = np.percentile(np.abs(env_array), 90)
        percentile_scale = env_90 / ebm_90 if ebm_90 > 1e-8 else 1.0
        auto_scale = 0.6 * percentile_scale + 0.3 * std_scale + 0.1 * range_scale
        auto_scale = np.clip(auto_scale, 0.01, 100.0)
        return auto_scale

    def run(self):
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        while self.itr < self.n_train_itr:
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4")
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                firsts_trajs[0] = done_venv
            obs_trajs = {"state": np.zeros((self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim))}
            samples_trajs = np.zeros((self.n_steps, self.n_envs, self.horizon_steps, self.action_dim))
            reward_trajs = np.zeros((self.n_steps, self.n_envs))
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            if self.save_full_observations:
                obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
                obs_full_trajs = np.vstack((obs_full_trajs, prev_obs_venv["state"][:, -1][None]))
            raw_env_rewards = []
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")
                with torch.no_grad():
                    cond = {"state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)}
                    samples = self.model(cond=cond, deterministic=eval_mode)
                    output_venv = samples.cpu().numpy()
                action_venv = output_venv[:, : self.act_steps]
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv
                raw_env_rewards.append(reward_venv.copy())
                if self.save_full_observations:
                    obs_full_venv = np.array([info["full_obs"]["state"] for info in info_venv])
                    obs_full_trajs = np.vstack((obs_full_trajs, obs_full_venv.transpose(1, 0, 2)))
                obs_trajs["state"][step] = prev_obs_venv["state"]
                samples_trajs[step] = output_venv
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                firsts_trajs[step + 1] = done_venv
                prev_obs_venv = obs_venv
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0
            original_reward_trajs = reward_trajs.copy()
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            raw_env_rewards_array = np.array(raw_env_rewards)
            raw_reward_stats = {
                "mean": float(np.mean(raw_env_rewards_array)),
                "std": float(np.std(raw_env_rewards_array)),
                "min": float(np.min(raw_env_rewards_array)),
                "max": float(np.max(raw_env_rewards_array)),
                "median": float(np.median(raw_env_rewards_array)),
            }
            if len(episodes_start_end) > 0:
                env_reward_trajs_split = [original_reward_trajs[start : end + 1, env_ind] for env_ind, start, end in episodes_start_end]
                num_episode_finished = len(env_reward_trajs_split)
                env_episode_reward = np.array([np.sum(reward_traj) for reward_traj in env_reward_trajs_split])
                if self.furniture_sparse_reward:
                    episode_best_reward = env_episode_reward
                else:
                    episode_best_reward = np.array([np.max(reward_traj) / self.act_steps for reward_traj in env_reward_trajs_split])
                avg_episode_reward = np.mean(env_episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(episode_best_reward >= self.best_reward_threshold_for_success)
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
                episode_stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "count": 0, "avg_length": 0.0}
            if not eval_mode:
                obs_trajs["state"] = torch.from_numpy(obs_trajs["state"]).float().to(self.device)
                ebm_reward_stats = None
                ebm_episode_stats = None
                avg_ebm_episode_reward = 0.0
                reward_correlation = None
                if self.use_ebm_reward:
                    if self.ebm_model is None:
                        reward_trajs = original_reward_trajs
                    elif not hasattr(self.ebm_model, 'forward') and not hasattr(self.ebm_model, '__call__'):
                        reward_trajs = original_reward_trajs
                    else:
                        ebm_rewards = self._compute_ebm_rewards(obs_trajs, samples_trajs, firsts_trajs, original_reward_trajs, eval_mode)
                        self._last_ebm_reward_mean = np.mean(ebm_rewards)
                        self._last_ebm_reward_std = np.std(ebm_rewards)
                        ebm_reward_stats = {
                            "mean": float(self._last_ebm_reward_mean),
                            "std": float(self._last_ebm_reward_std),
                            "min": float(np.min(ebm_rewards)),
                            "max": float(np.max(ebm_rewards)),
                            "median": float(np.median(ebm_rewards))
                        }
                        ebm_episode_stats = None
                        avg_ebm_episode_reward = 0.0
                        if len(episodes_start_end) > 0:
                            ebm_reward_trajs_split = [ebm_rewards[start : end + 1, env_ind] for env_ind, start, end in episodes_start_end]
                            ebm_episode_reward = np.array([np.sum(reward_traj) for reward_traj in ebm_reward_trajs_split])
                            avg_ebm_episode_reward = np.mean(ebm_episode_reward)
                            ebm_episode_stats = {
                                "mean": float(avg_ebm_episode_reward),
                                "std": float(np.std(ebm_episode_reward)) if len(ebm_episode_reward) > 1 else 0.0,
                                "min": float(np.min(ebm_episode_reward)),
                                "max": float(np.max(ebm_episode_reward)),
                                "median": float(np.median(ebm_episode_reward)),
                                "count": len(ebm_episode_reward)
                            }
                        try:
                            env_flat = original_reward_trajs.flatten()
                            ebm_flat = ebm_rewards.flatten()
                            if len(env_flat) > 10:
                                r, p = pearsonr(env_flat, ebm_flat)
                                sr, _ = spearmanr(env_flat, ebm_flat)
                                kt, _ = kendalltau(env_flat, ebm_flat)
                                reward_correlation = {"pearson_r": float(r), "p_value": float(p), "spearman": float(sr), "kendall": float(kt), "sample_size": len(env_flat)}
                        except Exception:
                            pass
                        reward_trajs = ebm_rewards
                num_split = math.ceil(self.n_envs * self.n_steps / self.logprob_batch_size)
                obs_ts = [{} for _ in range(num_split)]
                obs_k = einops.rearrange(obs_trajs["state"], "s e ... -> (s e) ...")
                obs_ts_k = torch.split(obs_k, self.logprob_batch_size, dim=0)
                for i, obs_t in enumerate(obs_ts_k):
                    obs_ts[i]["state"] = obs_t
                values_trajs = np.empty((0, self.n_envs))
                for obs in obs_ts:
                    values = self.model.critic(obs).detach().cpu().numpy().flatten()
                    values_trajs = np.vstack((values_trajs, values.reshape(-1, self.n_envs)))
                samples_t = einops.rearrange(torch.from_numpy(samples_trajs).float().to(self.device), "s e h d -> (s e) h d")
                samples_ts = torch.split(samples_t, self.logprob_batch_size, dim=0)
                logprobs_trajs = np.empty((0))
                for obs_t, samples_t in zip(obs_ts, samples_ts):
                    logprobs = self.model.get_logprobs(obs_t, samples_t)[0].detach().cpu().numpy()
                    logprobs_trajs = np.concatenate((logprobs_trajs, logprobs.reshape(-1)))
                reward_before_scaling = reward_trajs.copy()
                scaling_stats = None
                if self.reward_scale_running:
                    reward_trajs_transpose = self.running_reward_scaler(reward=reward_trajs.T, first=firsts_trajs[:-1].T)
                    reward_trajs = reward_trajs_transpose.T
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
                obs_venv_ts = {"state": torch.from_numpy(obs_venv["state"]).float().to(self.device)}
                advantages_trajs = np.zeros_like(reward_trajs)
                lastgaelam = 0
                for t in reversed(range(self.n_steps)):
                    if t == self.n_steps - 1:
                        nextvalues = self.model.critic(obs_venv_ts).detach().reshape(1, -1).cpu().numpy()
                    else:
                        nextvalues = values_trajs[t + 1]
                    nonterminal = 1.0 - terminated_trajs[t]
                    delta = reward_trajs[t] * self.reward_scale_const + self.gamma * nextvalues * nonterminal - values_trajs[t]
                    advantages_trajs[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nonterminal * lastgaelam
                returns_trajs = advantages_trajs + values_trajs
                if self.use_ebm_reward:
                    adv_std = float(np.std(advantages_trajs))
                    low, high = self.adv_std_target * (1 - self.adv_std_tol), self.adv_std_target * (1 + self.adv_std_tol)
                    if adv_std < low or adv_std > high:
                        ratio = np.clip(adv_std / (self.adv_std_target + 1e-8), 0.2, 5.0)
                        if self.energy_normalizer is not None:
                            new_temp = float(self.energy_normalizer.temperature * (ratio ** self.temp_lr))
                            new_temp = float(np.clip(new_temp, self.min_temp, self.max_temp))
                            self.energy_normalizer.temperature = new_temp
                            if self.use_wandb:
                                wandb.log({"train/temperature_scale": new_temp, "train/adv_std": adv_std}, step=self.itr, commit=False)
                        self.ebm_reward_scale = float(self.ebm_reward_scale * (1.0 / (ratio ** self.scale_lr)))
                        self.ebm_reward_scale = float(np.clip(self.ebm_reward_scale, 0.05, 20.0))
                        if self.use_wandb:
                            wandb.log({"train/ebm_reward_scale_runtime": self.ebm_reward_scale}, step=self.itr, commit=False)
                obs_k = {"state": einops.rearrange(obs_trajs["state"], "s e ... -> (s e) ...")}
                samples_k = einops.rearrange(torch.tensor(samples_trajs, device=self.device).float(), "s e h d -> (s e) h d")
                returns_k = torch.tensor(returns_trajs, device=self.device).float().reshape(-1)
                values_k = torch.tensor(values_trajs, device=self.device).float().reshape(-1)
                advantages_k = torch.tensor(advantages_trajs, device=self.device).float().reshape(-1)
                logprobs_k = torch.tensor(logprobs_trajs, device=self.device).float()
                total_steps = self.n_steps * self.n_envs
                clipfracs = []
                for update_epoch in range(self.update_epochs):
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
                        (pg_loss, entropy_loss, v_loss, clipfrac, approx_kl, ratio, bc_loss, std) = self.model.loss(
                            obs_b, samples_b, returns_b, values_b, advantages_b, logprobs_b, use_bc_loss=self.use_bc_loss
                        )
                        loss = pg_loss + entropy_loss * self.ent_coef + v_loss * self.vf_coef + bc_loss * self.bc_loss_coeff
                        clipfracs += [clipfrac]
                        self.actor_optimizer.zero_grad()
                        self.critic_optimizer.zero_grad()
                        loss.backward()
                        if self.itr >= self.n_critic_warmup_itr:
                            if self.max_grad_norm is not None:
                                torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), self.max_grad_norm)
                            self.actor_optimizer.step()
                        self.critic_optimizer.step()
                        log.info(f"approx_kl: {approx_kl}, update_epoch: {update_epoch}, num_batch: {num_batch}")
                        if self.target_kl is not None and approx_kl > self.target_kl:
                            flag_break = True
                            break
                    if flag_break:
                        break
                y_pred, y_true = values_k.cpu().numpy(), returns_k.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            if self.itr % self.render_freq == 0 and self.n_render > 0 and self.traj_plotter is not None:
                self.traj_plotter(obs_full_trajs=obs_full_trajs, n_render=self.n_render, max_episode_steps=self.max_episode_steps, render_dir=self.render_dir, itr=self.itr)
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()
            run_results.append({"itr": self.itr, "step": cnt_train_step})
            if self.save_trajs:
                run_results[-1]["obs_full_trajs"] = obs_full_trajs
                run_results[-1]["obs_trajs"] = obs_trajs
                run_results[-1]["action_trajs"] = samples_trajs
                run_results[-1]["reward_trajs"] = reward_trajs
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    log.info(f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}")
                    log.info(f"eval rewards: raw_mean {raw_reward_stats['mean']:.4f} ± {raw_reward_stats['std']:.4f} | episodes: {num_episode_finished} | avg_len: {episode_stats['avg_length']:.1f}")
                    if self.use_wandb:
                        log_dict = {
                            "success rate - eval": success_rate,
                            "avg episode reward - eval (ENV)": avg_episode_reward,
                            "avg best reward - eval": avg_best_reward,
                            "num episode - eval": num_episode_finished,
                            "eval/raw_reward_mean": raw_reward_stats["mean"],
                            "eval/raw_reward_std": raw_reward_stats["std"],
                            "eval/raw_reward_min": raw_reward_stats["min"],
                            "eval/raw_reward_max": raw_reward_stats["max"],
                            "eval/raw_reward_median": raw_reward_stats["median"],
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
                    log.info(f"{self.itr}: step {cnt_train_step:8d} | loss {loss:8.4f} | pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} | ent {-entropy_loss:8.4f} | reward {avg_episode_reward:8.4f} | t:{time:8.4f}")
                    log.info(f"train rewards: raw_mean {raw_reward_stats['mean']:.4f} ± {raw_reward_stats['std']:.4f} | episodes: {num_episode_finished} | avg_len: {episode_stats['avg_length']:.1f}")
                    if scaling_stats is not None:
                        log.info(f"reward scaling: {scaling_stats['before_mean']:.4f} -> {scaling_stats['after_mean']:.4f} (factor: {scaling_stats['scaling_factor']:.4f})")
                    if self.use_ebm_reward and ebm_reward_stats is not None:
                        log.info(f"EBM rewards: mean {ebm_reward_stats['mean']:.4f} ± {ebm_reward_stats['std']:.4f} (scale: {self.ebm_reward_scale})")
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
                            "avg episode reward - train (ENV)": avg_episode_reward,
                            "num episode - train": num_episode_finished,
                            "train/raw_reward_mean": raw_reward_stats["mean"],
                            "train/raw_reward_std": raw_reward_stats["std"],
                            "train/raw_reward_min": raw_reward_stats["min"],
                            "train/raw_reward_max": raw_reward_stats["max"],
                            "train/raw_reward_median": raw_reward_stats["median"],
                            "train/avg_episode_length": episode_stats["avg_length"],
                            "train/reward_scale_const": self.reward_scale_const,
                        }
                        if self.use_ebm_reward and ebm_reward_stats is not None:
                            reward_type = "EBM_only" if self.use_dynamic_normalization else "EBM_only_raw"
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
                            if ebm_episode_stats is not None:
                                log_dict.update({
                                    "avg episode reward - train (EBM)": avg_ebm_episode_reward,
                                    "train/ebm_episode_reward_mean": ebm_episode_stats["mean"],
                                    "train/ebm_episode_reward_std": ebm_episode_stats["std"],
                                    "train/ebm_episode_reward_min": ebm_episode_stats["min"],
                                    "train/ebm_episode_reward_max": ebm_episode_stats["max"],
                                    "train/ebm_episode_reward_median": ebm_episode_stats["median"],
                                    "train/ebm_episode_count": ebm_episode_stats["count"],
                                })
                            if self.use_dynamic_normalization and self.energy_normalizer is not None:
                                norm_stats = self.energy_normalizer.get_stats()
                                log_dict.update({
                                    "train/energy_running_mean": norm_stats["running_mean"],
                                    "train/energy_running_std": norm_stats["running_std"],
                                    "train/energy_num_updates": norm_stats["num_updates"],
                                    "train/temperature_scale": norm_stats["temperature"],
                                    "train/normalization_momentum": norm_stats["momentum"],
                                })
                            if reward_correlation is not None:
                                log_dict.update({
                                    "train/ebm_env_correlation": reward_correlation["pearson_r"],
                                    "train/correlation_p_value": reward_correlation["p_value"],
                                    "train/spearman": reward_correlation.get("spearman", 0.0),
                                    "train/kendall": reward_correlation.get("kendall", 0.0),
                                    "train/correlation_sample_size": reward_correlation["sample_size"],
                                })
                        else:
                            log_dict.update({"training_reward_type": "Environment", "logging_reward_type": "Environment"})
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
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
