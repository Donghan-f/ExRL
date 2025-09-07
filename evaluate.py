#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-model evaluation (with explicit Uniform Baseline)
- Parses model filename to reconstruct env
- Evaluates the model and a uniform baseline on the same trajectories
- Prints metrics, saves a CSV and a comparison plot

Expected filename pattern (robust/optional fields):
  {algo}_pred-{Predictor}_H{H}_k{K}_{reward_mode}
  [_alpha{alpha}][_beta{beta}][ _base{baseline_mode}_bw{bw}_cd{cd}]
  _steps{timesteps}_{timestamp}.zip

Examples:
  ppo_pred-Drift_H15_k15_relative_alpha0.001_basepredictive_bw20_cd0.1_steps1000000_20250901-101010.zip
  a2c_pred-NaiveRW_H5_k5_original_steps200000_20250901-101010.zip
  ddpg_pred-LocalLevel_H25_k25_burn_beta0.05_baseRolling_bw20_cd0.1_steps500000_20250901-101010.zip
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, A2C, DDPG
from utility import ExchangeRateData
from envs.gymnasium_env.envs.exchange_env import ExchangeEnv

# --- predictors (NO ARIMA/PROPHET) ---
from predictor.fx_predictor import (
    NaiveRWPredictor,
    DriftPredictor,
    HoltWintersPredictor,
    ThetaPredictor,
    LocalLevelPredictor,
    SARIMAXPredictor,
)

PREDICTOR_MAP = {
    "NaiveRW": NaiveRWPredictor,
    "Drift": DriftPredictor,
    "HoltWinters": HoltWintersPredictor,
    "Theta": ThetaPredictor,
    "LocalLevel": LocalLevelPredictor,
    "SARIMAX": SARIMAXPredictor,
}

ALGO_MAP = {"ppo": PPO, "a2c": A2C, "ddpg": DDPG}


# =========================
# Filename parsing
# =========================
@dataclass
class RunConfig:
    algo: str
    predictor_name: str
    horizon: int
    reward_mode: str
    alpha: float = 0.0
    beta: float = 0.0
    baseline_mode: str = "predictive"
    baseline_window: int = 20
    clip_diff: float = 0.1
    timesteps: Optional[int] = None
    # you can extend if needed


_FILENAME_RE = re.compile(
    r"^(?P<algo>ppo|a2c|ddpg)"
    r"_pred-(?P<pred>[A-Za-z0-9]+)"
    r"_H(?P<H>\d+)_k(?P<K>\d+)"
    r"_(?P<mode>original|relative|burn|relative_burn|relative_final)"
    r"(?:_alpha(?P<alpha>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?"
    r"(?:_beta(?P<beta>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?"
    r"(?:_base(?P<base>(?:predictive|rolling)))?"
    r"(?:_bw(?P<bw>\d+))?"
    r"(?:_cd(?P<cd>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?"
    r"(?:_steps(?P<steps>\d+))?"
    r"_\d{8}-\d{6}\.zip$"
)


def parse_model_filename(path: str) -> RunConfig:
    fname = os.path.basename(path)
    m = _FILENAME_RE.match(fname)
    if not m:
        raise ValueError(f"Unrecognized model filename format: {fname}")

    gd = m.groupdict()
    algo = gd["algo"]
    pred = gd["pred"]
    H = int(gd["H"])
    K = int(gd["K"])
    mode = gd["mode"]

    # defaults; only meaningful params are used by each reward_mode (env内部会忽略无关参数)
    alpha = float(gd["alpha"]) if gd["alpha"] is not None else 0.0
    beta = float(gd["beta"]) if gd["beta"] is not None else 0.0
    base = gd["base"] if gd["base"] is not None else "predictive"
    bw = int(gd["bw"]) if gd["bw"] is not None else 20
    cd = float(gd["cd"]) if gd["cd"] is not None else 0.1
    steps = int(gd["steps"]) if gd["steps"] is not None else None

    if H != K:
        # 我们的训练约定 k=horizon，这里保持一致
        print(f"[WARN] H ({H}) != K ({K}) in filename; forcing k = H.")

    return RunConfig(
        algo=algo,
        predictor_name=pred,
        horizon=H,
        reward_mode=mode,
        alpha=alpha,
        beta=beta,
        baseline_mode=base,
        baseline_window=bw,
        clip_diff=cd,
        timesteps=steps
    )


# =========================
# Env factory
# =========================
def make_env(
    data: ExchangeRateData,
    cfg: RunConfig,
    total_amount: float = 10_000_000.0,
    history_window: int = 250,
):
    if cfg.predictor_name not in PREDICTOR_MAP:
        raise ValueError(f"Unknown predictor: {cfg.predictor_name}")
    predictor_cls = PREDICTOR_MAP[cfg.predictor_name]

    env = ExchangeEnv(
        sampler=data,
        predictor_class=predictor_cls,
        horizon=cfg.horizon,
        total_amount=total_amount,
        k=cfg.horizon,                 # force k = horizon
        history_window=history_window,
        reward_mode=cfg.reward_mode,   # "original" | "relative" | "burn" | "relative_burn" | "relative_final"
        alpha=cfg.alpha,
        beta=cfg.beta,
        baseline_mode=cfg.baseline_mode,
        baseline_window=cfg.baseline_window,
        clip_diff=cfg.clip_diff,
    )
    return env


# =========================
# Evaluation (same as你的加强版)
# =========================
def evaluate_model(model, env, n_episodes: int = 1000, deterministic=True, verbose=True):
    episode_rewards = []
    episode_lengths = []
    all_actions = []
    metrics = {
        "realized_cost": [],
        "avg_exec_rate": [],
        "leftover_final": [],
        "final_spent_amount": [],
        "effective_cost_incl_leftover": [],
        "effective_avg_rate_incl_leftover": [],
        "benchmark_cost": [],
        "improvement_vs_benchmark": [],
        "spent_ratio": [],
    }

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        episode_actions = []

        realized_cost = 0.0
        spent_sum = 0.0
        # Uniform benchmark (cap-aware)
        bench_left = float(env.total_amount)
        bench_cost = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)

            # pre-accounting for metric
            fraction = float(np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0))
            money_left_now = float(env.money_left)
            max_cap = float(getattr(env, "max_daily_exchange", env.total_amount))
            amount = fraction * min(max_cap, money_left_now)

            cur_idx = int(env.current_step)
            rate = float(env.true_rates[cur_idx])

            realized_cost += amount * rate
            spent_sum += amount

            # benchmark for today
            remaining_days_including_today = max(env.horizon - cur_idx, 1)
            bench_today_target = bench_left / remaining_days_including_today
            bench_today = min(bench_today_target, max_cap, bench_left)
            bench_cost += bench_today * rate
            bench_left -= bench_today

            # env step
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            step_count += 1
            episode_actions.append(action)

        last_rate = float(env.true_rates[env.current_step - 1])
        leftover_final = float(env.money_left)
        final_spent_amount = float(env.total_amount) - leftover_final

        effective_cost_incl_leftover = realized_cost + leftover_final * last_rate
        effective_avg_rate_incl_leftover = effective_cost_incl_leftover / float(env.total_amount)

        if spent_sum > 0:
            avg_exec_rate = realized_cost / spent_sum
            spent_ratio = spent_sum / float(env.total_amount)
        else:
            avg_exec_rate = last_rate
            spent_ratio = 0.0

        if bench_left > 0:
            bench_cost += bench_left * last_rate
            bench_left = 0.0

        if bench_cost > 0:
            improvement_vs_benchmark = (bench_cost - effective_cost_incl_leftover) / bench_cost
        else:
            improvement_vs_benchmark = 0.0

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        all_actions.append(episode_actions)

        metrics["realized_cost"].append(realized_cost)
        metrics["avg_exec_rate"].append(avg_exec_rate)
        metrics["leftover_final"].append(leftover_final)
        metrics["final_spent_amount"].append(final_spent_amount)
        metrics["effective_cost_incl_leftover"].append(effective_cost_incl_leftover)
        metrics["effective_avg_rate_incl_leftover"].append(effective_avg_rate_incl_leftover)
        metrics["benchmark_cost"].append(bench_cost)
        metrics["improvement_vs_benchmark"].append(improvement_vs_benchmark)
        metrics["spent_ratio"].append(spent_ratio)

    # aggregate
    ep_rewards = np.asarray(episode_rewards, dtype=np.float64)
    mean_reward = float(np.mean(ep_rewards)) if len(ep_rewards) else 0.0
    std_reward = float(np.std(ep_rewards)) if len(ep_rewards) else 0.0
    min_reward = float(np.min(ep_rewards)) if len(ep_rewards) else 0.0
    max_reward = float(np.max(ep_rewards)) if len(ep_rewards) else 0.0

    # actions
    if len(all_actions) and len(all_actions[0]):
        flat_actions = np.array([float(np.asarray(a, dtype=np.float64)) for ep in all_actions for a in ep], dtype=np.float64)
        mean_action = float(np.mean(flat_actions))
        std_action = float(np.std(flat_actions))
    else:
        mean_action = 0.0
        std_action = 0.0

    agg = {k: np.asarray(v, dtype=np.float64) for k, v in metrics.items()}
    results = {
        "episode_rewards": ep_rewards,
        "episode_lengths": np.asarray(episode_lengths, dtype=np.int32),
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "min_reward": min_reward,
        "max_reward": max_reward,
        "mean_action": mean_action,
        "std_action": std_action,
        "all_actions": all_actions,

        "realized_cost": agg["realized_cost"],
        "avg_exec_rate": agg["avg_exec_rate"],
        "leftover_final": agg["leftover_final"],
        "final_spent_amount": agg["final_spent_amount"],
        "effective_cost_incl_leftover": agg["effective_cost_incl_leftover"],
        "effective_avg_rate_incl_leftover": agg["effective_avg_rate_incl_leftover"],
        "benchmark_cost": agg["benchmark_cost"],
        "improvement_vs_benchmark": agg["improvement_vs_benchmark"],
        "spent_ratio": agg["spent_ratio"],

        "mean_final_spent_amount": float(np.mean(agg["final_spent_amount"])),
        "mean_final_leftover": float(np.mean(agg["leftover_final"])),
        "mean_spent_ratio": float(np.mean(agg["spent_ratio"])),
        "mean_effective_total_cost": float(np.mean(agg["effective_cost_incl_leftover"])),
        "mean_effective_avg_rate": float(np.mean(agg["effective_avg_rate_incl_leftover"])),
        "mean_improvement_vs_benchmark": float(np.mean(agg["improvement_vs_benchmark"])),
    }
    if verbose:
        print("\n===== Evaluation Results =====")
        print(f"Env Reward:  {mean_reward:.2f} ± {std_reward:.2f}   [{min_reward:.2f}, {max_reward:.2f}]")
        print(f"Avg Steps:   {np.mean(results['episode_lengths']):.2f}")
        print(f"Avg Action:  {mean_action:.6f} ± {std_action:.6f}")
        print("--- Task-native metrics ---")
        print(f"Mean Final Spent Amount:     {results['mean_final_spent_amount']:,.2f}")
        print(f"Mean Final Leftover:         {results['mean_final_leftover']:,.2f}")
        print(f"Mean Spent Ratio:            {results['mean_spent_ratio']:.3f}")
        print(f"Mean Effective Total Cost:   {results['mean_effective_total_cost']:,.2f}")
        print(f"Mean Eff. Avg Rate (incl L): {results['mean_effective_avg_rate']:.6f}")
        print(f"Mean Improvement vs Uniform: {100.0*results['mean_improvement_vs_benchmark']:.2f}%")
        print("=============================\n")

    return mean_reward, std_reward, results


# =========================
# Explicit uniform baseline model
# =========================
class UniformBaselinePolicy:
    """Spend equally over remaining days, respecting max_daily_exchange."""
    def __init__(self, env: ExchangeEnv):
        self.env = env

    def predict(self, obs, deterministic=True):
        money_left = float(self.env.money_left)
        days_left = max(self.env.horizon - self.env.current_step, 1)
        cap = float(getattr(self.env, "max_daily_exchange", self.env.total_amount))
        target_today = min(money_left / days_left, cap, money_left)
        frac = 0.0 if money_left <= 0 else target_today / money_left
        return np.array([frac], dtype=np.float32), None


# =========================
# Main: single run
# =========================
if __name__ == "__main__":
    # ==== 1) Set your model path here ====
    MODEL_PATH = "/Users/fandonghan/Desktop/Exchange_RL/models/ppo_pred-Drift_H15_k15_relative_alpha0.1_basepredictive_bw20_cd0.1_steps1000000_20250901-101010.zip"

    # ==== 2) Data sources ====
    csv_paths = [
        "/Users/fandonghan/Desktop/Exchange_RL/data/USD_CNY Historical Data 2000.9.7-2019.11.11.csv",
        "/Users/fandonghan/Desktop/Exchange_RL/data/USD_CNY Historical Data 2019.11.12-2025.7.1.csv",
    ]
    TOTAL_AMOUNT = 10_000_000.0
    HISTORY_WINDOW = 250
    N_EPISODES = 1000

    # ---- parse model filename ----
    cfg = parse_model_filename(MODEL_PATH)
    print("[INFO] Parsed:", cfg)

    # ---- build env (shared for both model & baseline) ----
    data = ExchangeRateData(csv_paths)
    env = make_env(
        data=data,
        cfg=cfg,
        total_amount=TOTAL_AMOUNT,
        history_window=HISTORY_WINDOW,
    )

    # ---- load model ----
    if cfg.algo not in ALGO_MAP:
        raise ValueError(f"Unsupported algo in filename: {cfg.algo}")
    ModelCls = ALGO_MAP[cfg.algo]
    model = ModelCls.load(MODEL_PATH)

    # ---- evaluate trained model ----
    _, _, res_model = evaluate_model(model, env, n_episodes=N_EPISODES, verbose=True)

    # ---- explicit baseline evaluation on same env/config ----
    baseline_env = make_env(data=data, cfg=cfg, total_amount=TOTAL_AMOUNT, history_window=HISTORY_WINDOW)
    baseline_model = UniformBaselinePolicy(baseline_env)
    _, _, res_base = evaluate_model(baseline_model, baseline_env, n_episodes=N_EPISODES, verbose=False)

    # ---- summarize into a table ----
    rows = []
    rows.append({
        "who": "MODEL",
        "algo": cfg.algo,
        "predictor": cfg.predictor_name,
        "horizon": cfg.horizon,
        "reward_mode": cfg.reward_mode,
        "alpha": cfg.alpha,
        "beta": cfg.beta,
        "baseline_mode": cfg.baseline_mode,
        "bw": cfg.baseline_window,
        "cd": cfg.clip_diff,
        "mean_effective_total_cost_CNY": res_model["mean_effective_total_cost"],
        "mean_effective_avg_rate": res_model["mean_effective_avg_rate"],
        "mean_final_spent_amount": res_model["mean_final_spent_amount"],
        "mean_final_leftover": res_model["mean_final_leftover"],
        "mean_spent_ratio": res_model["mean_spent_ratio"],
        "mean_improvement_vs_uniform_%": 100.0 * res_model["mean_improvement_vs_benchmark"],
    })
    rows.append({
        "who": "BASELINE_UNIFORM",
        "algo": "baseline",
        "predictor": "N/A",
        "horizon": cfg.horizon,
        "reward_mode": "N/A",
        "alpha": np.nan,
        "beta": np.nan,
        "baseline_mode": "N/A",
        "bw": np.nan,
        "cd": np.nan,
        "mean_effective_total_cost_CNY": res_base["mean_effective_total_cost"],
        "mean_effective_avg_rate": res_base["mean_effective_avg_rate"],
        "mean_final_spent_amount": res_base["mean_final_spent_amount"],
        "mean_final_leftover": res_base["mean_final_leftover"],
        "mean_spent_ratio": res_base["mean_spent_ratio"],
        "mean_improvement_vs_uniform_%": 100.0 * res_base["mean_improvement_vs_benchmark"],  # ~0%
    })
    df = pd.DataFrame(rows)

    os.makedirs("./eval_outputs", exist_ok=True)
    csv_out = os.path.join("./eval_outputs", "single_eval_summary.csv")
    df.to_csv(csv_out, index=False)
    print(f"[SAVED] {csv_out}")

    # ---- plot: effective total cost comparison ----
    plt.figure(figsize=(8, 5))
    x = np.arange(len(df))
    plt.bar(x, df["mean_effective_total_cost_CNY"].values)
    plt.xticks(x, df["who"].values)
    plt.ylabel("Mean Effective Total Cost (CNY)")
    plt.title("Model vs Uniform Baseline — Effective Total Cost")
    plt.tight_layout()
    fig_out = os.path.join("./eval_outputs", "single_eval_cost_compare.png")
    plt.savefig(fig_out, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"[SAVED] {fig_out}")

    # ---- print table nicely ----
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print("\n=== Summary Table ===")
        print(df.round(6))
