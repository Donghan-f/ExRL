#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 05:21:24 2025

@author: fandonghan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch evaluation for trained models.

- Auto-detect models from a folder by filename pattern produced in training:
  {algo}_pred-{predictor}_H{H}_k{H}_{reward_mode}[_alpha{a}][_beta{b}]
  [ _base{baseline_mode}_bw{bw}_cd{cd} ]_steps{timesteps}_{stamp}.zip

- Rebuild ExchangeEnv for each model based on parsed params
- Evaluate with N episodes (default 1000) on total_amount=1e7
- Report:
    * Env reward stats (mean ± std)
    * Mean effective total cost (CNY, incl. leftover at last-day rate)
    * Mean effective average rate (CNY/unit)
    * Mean improvement vs uniform benchmark (%)
    * Mean final leftover and spent ratio
- Save CSV summary and plots
"""

import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3 import PPO, A2C, DDPG
from envs.gymnasium_env.envs.exchange_env import ExchangeEnv
from utility import ExchangeRateData

# predictors 
from predictor.fx_predictor import (
    NaiveRWPredictor,
    DriftPredictor,
    HoltWintersPredictor,
    ThetaPredictor,
    LocalLevelPredictor,
    SARIMAXPredictor,
)

# ------------------------------
# Config (edit here)
# ------------------------------
MODELS_DIR = "./models"  # folder containing .zip models
CSV_PATHS = [
    "/Users/fandonghan/Desktop/Exchange_RL/data/USD_CNY Historical Data 2000.9.7-2019.11.11.csv",
    "/Users/fandonghan/Desktop/Exchange_RL/data/USD_CNY Historical Data 2019.11.12-2025.7.1.csv",
]
N_EPISODES = 1000
TOTAL_AMOUNT = 10_000_000.0
HISTORY_WINDOW = 250
DETERMINISTIC = True
RENDER = False

OUT_DIR = "./eval_reports"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------
# Helpers: env factory + filename parser + evaluator
# -------------------------------------------------
PRED_MAP = {
    "NaiveRW": NaiveRWPredictor,
    "Drift": DriftPredictor,
    "HoltWinters": HoltWintersPredictor,
    "Theta": ThetaPredictor,
    "LocalLevel": LocalLevelPredictor,
    "SARIMAX": SARIMAXPredictor,
}
ALGO_LOADERS = {"ppo": PPO, "a2c": A2C, "ddpg": DDPG}

FNAME_RE = re.compile(
    r"^(?P<algo>ppo|a2c|ddpg)"
    r"_pred-(?P<predictor>[A-Za-z0-9]+)"
    r"_H(?P<h>\d+)_k(?P<k>\d+)"
    r"_(?P<mode>original|relative|burn|relative_burn|relative_final)"
    r"(?:_alpha(?P<alpha>[\d\.eE\-]+))?"
    r"(?:_beta(?P<beta>[\d\.eE\-]+))?"
    r"(?:_base(?P<base>(predictive|rolling))_bw(?P<bw>\d+)_cd(?P<cd>[\d\.eE\-]+))?"
    r"_steps(?P<steps>\d+)"
    r"_(?P<stamp>\d{8}-\d{6})$"
)

def parse_model_path(path: str):
    name = os.path.basename(path)
    if name.endswith(".zip"):
        name = name[:-4]
    m = FNAME_RE.match(name)
    if not m:
        return None
    g = m.groupdict()
    out = {
        "algo": g["algo"],
        "predictor": g["predictor"],
        "h": int(g["h"]),
        "k": int(g["k"]),
        "reward_mode": g["mode"],
        "alpha": float(g["alpha"]) if g["alpha"] is not None else 0.0,
        "beta": float(g["beta"]) if g["beta"] is not None else 0.0,
        "baseline_mode": g["base"] if g["base"] is not None else "predictive",
        "baseline_window": int(g["bw"]) if g["bw"] is not None else 20,
        "clip_diff": float(g["cd"]) if g["cd"] is not None else 0.1,
        "steps": int(g["steps"]),
        "stamp": g["stamp"],
        "fname": os.path.basename(path),
        "path": path,
    }
    return out

def make_env_for_cfg(data: ExchangeRateData, cfg: dict):
    # NOTE: we force k=h (as saved in model names)
    env = ExchangeEnv(
        sampler=data,
        predictor_class=PRED_MAP[cfg["predictor"]],
        horizon=cfg["h"],
        total_amount=TOTAL_AMOUNT,
        k=cfg["k"],
        history_window=HISTORY_WINDOW,
        reward_mode=cfg["reward_mode"],
        alpha=cfg["alpha"],
        beta=cfg["beta"],
        baseline_mode=cfg["baseline_mode"],
        baseline_window=cfg["baseline_window"],
        clip_diff=cfg["clip_diff"],
    )
    return env

# ---------- evaluation (same core as你当前 evaluate_model，轻度整合打印) ----------
def evaluate_model(model, env, n_episodes=1000, deterministic=True, render=False, verbose=False):
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
        bench_left = float(env.total_amount)
        bench_cost = 0.0

        while not done:
            if render:
                env.render()
                time.sleep(0.005)

            action, _ = model.predict(obs, deterministic=deterministic)

            try:
                fraction = float(np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0))
            except Exception:
                fraction = float(np.clip(action, 0.0, 1.0))

            money_left_now = float(env.money_left)
            max_cap = float(getattr(env, "max_daily_exchange", env.total_amount))
            amount = fraction * min(max_cap, money_left_now)

            cur_idx = int(env.current_step)
            rate = float(env.true_rates[cur_idx])

            realized_cost += amount * rate
            spent_sum += amount

            remaining_days_including_today = max(env.horizon - cur_idx, 1)
            bench_today_target = bench_left / remaining_days_including_today
            bench_today = min(bench_today_target, max_cap, bench_left)
            bench_cost += bench_today * rate
            bench_left -= bench_today

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

    episode_rewards = np.array(episode_rewards, dtype=np.float64)
    res = {
        "env_mean_reward": float(np.mean(episode_rewards)) if len(episode_rewards) else 0.0,
        "env_std_reward": float(np.std(episode_rewards)) if len(episode_rewards) else 0.0,
        "env_min_reward": float(np.min(episode_rewards)) if len(episode_rewards) else 0.0,
        "env_max_reward": float(np.max(episode_rewards)) if len(episode_rewards) else 0.0,
        "mean_final_spent_amount": float(np.mean(metrics["final_spent_amount"])),
        "mean_final_leftover": float(np.mean(metrics["leftover_final"])),
        "mean_spent_ratio": float(np.mean(metrics["spent_ratio"])),
        "mean_effective_total_cost": float(np.mean(metrics["effective_cost_incl_leftover"])),
        "mean_effective_avg_rate": float(np.mean(metrics["effective_avg_rate_incl_leftover"])),
        "mean_improvement_vs_benchmark": float(np.mean(metrics["improvement_vs_benchmark"])),
    }
    return res

# ------------------------------
# Main: scan, evaluate, summarize
# ------------------------------
def main():
    # preload data once
    data = ExchangeRateData(CSV_PATHS)

    # gather models
    model_files = [os.path.join(MODELS_DIR, f) for f in os.listdir(MODELS_DIR) if f.endswith(".zip")]
    model_files.sort()

    rows = []
    for path in model_files:
        meta = parse_model_path(path)
        if meta is None:
            print(f"[SKIP] Unrecognized filename: {os.path.basename(path)}")
            continue

        # 可选：如果你“本批次只测某个 horizon”，在这里加筛选：
        # if meta["h"] != 15: continue

        try:
            env = make_env_for_cfg(data, meta)
            ModelCls = ALGO_LOADERS[meta["algo"]]
            model = ModelCls.load(path)

            res = evaluate_model(
                model=model,
                env=env,
                n_episodes=N_EPISODES,
                deterministic=DETERMINISTIC,
                render=RENDER,
                verbose=False
            )

            row = {
                **meta,
                **res,
            }
            rows.append(row)
            env.close()
            print(f"[OK] {meta['fname']}  →  improv={100*res['mean_improvement_vs_benchmark']:.2f}%  "
                  f"eff_rate={res['mean_effective_avg_rate']:.6f}  leftover={row['mean_final_leftover']:,.0f}")
        except Exception as e:
            print(f"[ERR] {meta['fname']}: {e}")

    if not rows:
        print("No valid models evaluated. Check MODELS_DIR or filename patterns.")
        return

    df = pd.DataFrame(rows)
    # 排序：按 horizon、reward_mode、predictor、algo
    df.sort_values(by=["h", "reward_mode", "predictor", "algo"], inplace=True)

    # 保存表格
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(OUT_DIR, f"eval_summary_{stamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[SAVED] Summary CSV -> {csv_path}")

    # ---------------- Plots ----------------
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # 1) Improvement vs uniform (top-N)
    topN = min(20, len(df))
    df_top = df.nlargest(topN, "mean_improvement_vs_benchmark").copy()
    df_top["label"] = (
        df_top["algo"] + " | " + df_top["predictor"] + " | H" + df_top["h"].astype(str)
        + " | " + df_top["reward_mode"]
    )
    plt.figure(figsize=(10, max(6, topN*0.4)))
    sns.barplot(
        data=df_top,
        y="label",
        x=df_top["mean_improvement_vs_benchmark"]*100.0
    )
    plt.xlabel("Improvement vs Uniform (%)")
    plt.ylabel("Run")
    plt.title("Top runs by improvement vs uniform")
    plt.tight_layout()
    plot1 = os.path.join(OUT_DIR, f"plot_top_improvement_{stamp}.png")
    plt.savefig(plot1, dpi=200)
    print(f"[SAVED] {plot1}")

    # 2) Effective average rate (lower is better)
    plt.figure(figsize=(10, max(6, topN*0.4)))
    df_low_rate = df.nsmallest(topN, "mean_effective_avg_rate").copy()
    df_low_rate["label"] = (
        df_low_rate["algo"] + " | " + df_low_rate["predictor"] + " | H" + df_low_rate["h"].astype(str)
        + " | " + df_low_rate["reward_mode"]
    )
    sns.barplot(
        data=df_low_rate,
        y="label",
        x="mean_effective_avg_rate"
    )
    plt.xlabel("Mean Effective Avg Rate (CNY/unit, incl leftover)")
    plt.ylabel("Run")
    plt.title("Top runs by lowest effective average rate")
    plt.tight_layout()
    plot2 = os.path.join(OUT_DIR, f"plot_low_eff_rate_{stamp}.png")
    plt.savefig(plot2, dpi=200)
    print(f"[SAVED] {plot2}")

    # 3) Spent ratio (how much of the budget is used)
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=df,
        x="h",
        y="mean_spent_ratio",
        hue="reward_mode"
    )
    plt.xlabel("Horizon (days)")
    plt.ylabel("Mean Spent Ratio")
    plt.title("Spent ratio by horizon and reward_mode")
    plt.tight_layout()
    plot3 = os.path.join(OUT_DIR, f"plot_spent_ratio_{stamp}.png")
    plt.savefig(plot3, dpi=200)
    print(f"[SAVED] {plot3}")

    # 打印 Top 10 摘要（改进率）
    top10 = df.nlargest(10, "mean_improvement_vs_benchmark")[
        ["fname","algo","predictor","h","reward_mode",
         "mean_improvement_vs_benchmark","mean_effective_avg_rate",
         "mean_final_leftover","env_mean_reward","env_std_reward"]
    ]
    print("\n=== TOP 10 by improvement vs uniform ===")
    print(top10.to_string(index=False))


if __name__ == "__main__":
    main()
