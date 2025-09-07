"""
Grid training 

- RL algos: PPO / A2C / DDPG
- predictors: NaiveRW / Drift / HoltWinters / Theta / LocalLevel / SARIMAX
- horizons: 5 / 15 / 25  (force k = horizon)
- reward_modes: original / relative / burn / relative_burn / relative_final
- Only traverse parameters that are relevant to the chosen reward_mode.
"""

import argparse
import os
from datetime import datetime

from stable_baselines3 import PPO, A2C, DDPG
from envs.gymnasium_env.envs.exchange_env import ExchangeEnv
from utility import ExchangeRateData

# predictors (NO ARIMA/PROPHET)
from predictor.fx_predictor import (
    NaiveRWPredictor,
    DriftPredictor,
    HoltWintersPredictor,
    ThetaPredictor,
    LocalLevelPredictor,
    SARIMAXPredictor,
)

# ------------------------------
# Parameter grid (base)
# ------------------------------
def base_grid():
    """Base ranges before filtering by reward_mode relevance."""
    predictors = {
        "NaiveRW": NaiveRWPredictor,
        "Drift": DriftPredictor,
        "HoltWinters": HoltWintersPredictor,
        "Theta": ThetaPredictor,
        "LocalLevel": LocalLevelPredictor,
        "SARIMAX": SARIMAXPredictor,
    }
    return {
        "algos": ["ppo", "a2c", "ddpg"],
        "horizons": [5, 15, 25],                        # k = horizon
        "predictors": predictors,
        "reward_modes": ["original", "relative", "burn", "relative_burn", "relative_final"], 
        # alpha only used by: relative / relative_burn / relative_final
        "alphas": [0.0, 1e-4, 1e-3],
        # beta only used by: burn / relative_burn
        "betas": [0.0, 0.01, 0.05],
        # baseline settings used by: relative / burn / relative_burn
        "baseline_modes": ["predictive", "rolling"],
        "baseline_windows": [20],
        "clip_diffs": [0.1],
    }

# ------------------------------
# Keep only meaningful combinations per reward_mode
# ------------------------------
def is_relevant_combo(reward_mode: str, alpha: float, beta: float,
                      baseline_mode: str, bw: int, cd: float) -> bool:
    """
    Filter grid points so we only train combinations that actually matter
    for the chosen reward_mode.
    """
    if reward_mode == "original":
        # No alpha/beta/baseline are used. Keep a single trivial combo.
        return (alpha == 0.0) and (beta == 0.0) and (baseline_mode == "predictive")

    if reward_mode == "relative":
        # Uses alpha + baseline params; beta must be 0
        return (beta == 0.0)

    if reward_mode == "burn":
        # Uses beta + baseline params; alpha must be 0
        return (alpha == 0.0)

    if reward_mode == "relative_burn":
        # Uses alpha + beta + baseline params; all ranges valid
        return True

    if reward_mode == "relative_final":
        # Uses alpha only; beta must be 0; baseline params ignored (keep a single baseline_mode)
        return (beta == 0.0) and (baseline_mode == "predictive")

    return True

def build_run_name(algo: str, predictor_name: str, horizon: int, reward_mode: str,
                   alpha: float, beta: float, baseline_mode: str, bw: int, cd: float,
                   timesteps: int, stamp: str) -> str:
    """
    Filename containing only parameters relevant to the chosen reward_mode.
    """
    parts = [f"{algo}", f"pred-{predictor_name}", f"H{horizon}_k{horizon}", reward_mode]

    if reward_mode in {"relative", "relative_burn", "relative_final"}:
        parts.append(f"alpha{alpha}")

    if reward_mode in {"burn", "relative_burn"}:
        parts.append(f"beta{beta}")

    if reward_mode in {"relative", "burn", "relative_burn"}:
        parts.append(f"base{baseline_mode}")
        parts.append(f"bw{bw}")
        parts.append(f"cd{cd}")

    parts.append(f"steps{timesteps}")
    parts.append(stamp)
    return "_".join(parts)

# ------------------------------
# Env factory (reuses preloaded data)
# ------------------------------
def make_env_preloaded(
    data: ExchangeRateData,
    predictor_class,
    horizon: int,
    total_amount: float,
    history_window: int,
    # reward config
    reward_mode: str,
    alpha: float,
    beta: float,
    baseline_mode: str,
    baseline_window: int,
    clip_diff: float,
):
    env = ExchangeEnv(
        sampler=data,
        predictor_class=predictor_class,
        horizon=horizon,
        total_amount=total_amount,
        k=horizon,                      # force k = horizon
        history_window=history_window,
        reward_mode=reward_mode,        # "original" | "relative" | "burn" | "relative_burn" | "relative_final"
        alpha=alpha,
        beta=beta,
        baseline_mode=baseline_mode,    # "predictive" | "rolling"
        baseline_window=baseline_window,
        clip_diff=clip_diff,
    )
    return env

# ------------------------------
# Single run
# ------------------------------
def train_one(
    algo_name: str,
    predictor_name: str,
    predictor_class,
    horizon: int,
    data: ExchangeRateData,
    total_amount: float,
    history_window: int,
    timesteps: int,
    # reward config
    reward_mode: str,
    alpha: float,
    beta: float,
    baseline_mode: str,
    baseline_window: int,
    clip_diff: float,
    base_logdir: str = "./tensorboard_logs",
    models_dir: str = "./models",
):
    algo_map = {"ppo": PPO, "a2c": A2C, "ddpg": DDPG}
    ModelCls = algo_map[algo_name]

    env = make_env_preloaded(
        data=data,
        predictor_class=predictor_class,
        horizon=horizon,
        total_amount=total_amount,
        history_window=history_window,
        reward_mode=reward_mode,
        alpha=alpha,
        beta=beta,
        baseline_mode=baseline_mode,
        baseline_window=baseline_window,
        clip_diff=clip_diff,
    )

    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_dir = os.path.join(
        base_logdir,
        algo_name, predictor_name,
        f"H{horizon}", reward_mode,
        run_stamp,
    )
    os.makedirs(tb_dir, exist_ok=True)

    common_kwargs = dict(verbose=1, tensorboard_log=tb_dir, gamma=1.0)
    if algo_name in ["ppo", "a2c"]:
        model = ModelCls("MlpPolicy", env, gae_lambda=1.0, **common_kwargs)
    else:  # ddpg
        model = ModelCls("MlpPolicy", env, **common_kwargs)

    model.learn(total_timesteps=timesteps)

    os.makedirs(models_dir, exist_ok=True)
    fname = build_run_name(
        algo=algo_name,
        predictor_name=predictor_name,
        horizon=horizon,
        reward_mode=reward_mode,
        alpha=alpha,
        beta=beta,
        baseline_mode=baseline_mode,
        bw=baseline_window,
        cd=clip_diff,
        timesteps=timesteps,
        stamp=run_stamp,
    )
    path = os.path.join(models_dir, fname)
    model.save(path)
    print(f"[SAVED] {path}.zip")

    try:
        env.close()
    except Exception:
        pass

# ------------------------------
# Orchestrator
# ------------------------------
def train_grid(args):
    grid = base_grid()

    # Preload data once
    data = ExchangeRateData(args.csv_paths)

    for algo in grid["algos"]:
        for horizon in grid["horizons"]:
            for predictor_name, predictor_class in grid["predictors"].items():
                for reward_mode in grid["reward_modes"]:
                    for alpha in grid["alphas"]:
                        for beta in grid["betas"]:
                            for baseline_mode in grid["baseline_modes"]:
                                for bw in grid["baseline_windows"]:
                                    for cd in grid["clip_diffs"]:

                                        if not is_relevant_combo(
                                            reward_mode=reward_mode,
                                            alpha=alpha,
                                            beta=beta,
                                            baseline_mode=baseline_mode,
                                            bw=bw,
                                            cd=cd
                                        ):
                                            continue

                                        print(
                                            f"\n=== RUN === "
                                            f"algo={algo}  pred={predictor_name}  H={horizon} "
                                            f"mode={reward_mode}  alpha={alpha}  beta={beta}  "
                                            f"base={baseline_mode}  bw={bw}  cd={cd}"
                                        )
                                        try:
                                            train_one(
                                                algo_name=algo,
                                                predictor_name=predictor_name,
                                                predictor_class=predictor_class,
                                                horizon=horizon,
                                                data=data,
                                                total_amount=args.total_amount,
                                                history_window=args.history_window,
                                                timesteps=args.timesteps,
                                                reward_mode=reward_mode,
                                                alpha=alpha,
                                                beta=beta,
                                                baseline_mode=baseline_mode,
                                                baseline_window=bw,
                                                clip_diff=cd,
                                            )
                                        except Exception as e:
                                            print(f"[SKIP] Failed run due to: {e}")

# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_paths",
        nargs="+",
        type=str,
        default=[
            "/Users/fandonghan/Desktop/Exchange_RL/data/USD_CNY Historical Data 2000.9.7-2019.11.11.csv",
            "/Users/fandonghan/Desktop/Exchange_RL/data/USD_CNY Historical Data 2019.11.12-2025.7.1.csv",
        ],
    )
    parser.add_argument("--total_amount", type=float, default=10_000_000.0)
    parser.add_argument("--history_window", type=int, default=250)
    parser.add_argument("--timesteps", type=int, default=1_000_00)
    args = parser.parse_args()

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./tensorboard_logs", exist_ok=True)

    train_grid(args)
