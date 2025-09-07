
"""


"""


"""
Single-run training 

- Pick ONE combo below and run.
- RL algos:       "ppo" | "a2c" | "ddpg"
- Predictors:     NaiveRW | Drift | HoltWinters | Theta | LocalLevel | SARIMAX
- Horizons:       e.g. 5 / 15 / 25   (k is forced to horizon)
- Reward modes:   "original" | "relative" | "burn" | "relative_burn" | "relative_final"
"""

import os
from datetime import datetime

from stable_baselines3 import PPO, A2C, DDPG
from envs.gymnasium_env.envs.exchange_env import ExchangeEnv
from utility import ExchangeRateData

# --- predictors (NO ARIMA/PROPHET) ---
from predictor.fx_predictor import (
    NaiveRWPredictor,
    DriftPredictor,
    HoltWintersPredictor,
    ThetaPredictor,
    LocalLevelPredictor,
    SARIMAXPredictor,
)

# ============ CHOOSE YOUR SETTINGS HERE ============
# Data
CSV_PATHS = [
    "/Users/fandonghan/Desktop/Exchange_RL/data/USD_CNY Historical Data 2000.9.7-2019.11.11.csv",
    "/Users/fandonghan/Desktop/Exchange_RL/data/USD_CNY Historical Data 2019.11.12-2025.7.1.csv",
]

TOTAL_AMOUNT   = 10_000_000.0
HISTORY_WINDOW = 250
HORIZON        = 15                # k will be set to this value
TIMESTEPS      = 1_000_000

# RL algo: "ppo" | "a2c" | "ddpg"
ALGO = "ppo"

# Predictor choice (pick ONE):
PREDICTOR_NAME  = "Drift"          # "NaiveRW"|"Drift"|"HoltWinters"|"Theta"|"LocalLevel"|"SARIMAX"
PREDICTOR_CLASS = {
    "NaiveRW":     NaiveRWPredictor,
    "Drift":       DriftPredictor,
    "HoltWinters": HoltWintersPredictor,
    "Theta":       ThetaPredictor,
    "LocalLevel":  LocalLevelPredictor,
    "SARIMAX":     SARIMAXPredictor,
}[PREDICTOR_NAME]

# Reward config
REWARD_MODE     = "relative"       # "original"|"relative"|"burn"|"relative_burn"|"relative_final"
ALPHA           = 1e-3             # used by: relative / relative_burn / relative_final
BETA            = 0.01             # used by: burn / relative_burn
BASELINE_MODE   = "predictive"     # used by: relative / burn / relative_burn; ("predictive"|"rolling")
BASELINE_WINDOW = 20               # used by: baseline_mode="rolling"
CLIP_DIFF       = 0.1              # used by: relative / burn / relative_burn / relative_final
# ===================================================


def _assert_relevance_or_coerce():
    """
    Keep settings meaningful per reward_mode.
    - original:        ignore alpha/beta/baseline (we set alpha=beta=0, baseline_mode="predictive")
    - relative:        uses alpha + baseline; force beta=0
    - burn:            uses beta + baseline; force alpha=0
    - relative_burn:   uses alpha + beta + baseline; keep all
    - relative_final:  uses alpha only; force beta=0; baseline ignored (keep "predictive")
    """
    global ALPHA, BETA, BASELINE_MODE, BASELINE_WINDOW, CLIP_DIFF

    if REWARD_MODE == "original":
        ALPHA = 0.0
        BETA  = 0.0
        BASELINE_MODE = "predictive"  # dummy
    elif REWARD_MODE == "relative":
        BETA  = 0.0
        # uses baseline_mode/window/clip_diff
    elif REWARD_MODE == "burn":
        ALPHA = 0.0
        # uses baseline_mode/window/clip_diff
    elif REWARD_MODE == "relative_burn":
        # uses alpha + beta + baseline; keep all
        pass
    elif REWARD_MODE == "relative_final":
        BETA  = 0.0
        BASELINE_MODE = "predictive"  # ignored by env in this mode
        # CLIP_DIFF still used (for diff clipping)
    else:
        raise ValueError(f"Unknown REWARD_MODE: {REWARD_MODE}")


def build_run_name(stamp: str) -> str:
    """
    Filename containing only parameters relevant to the chosen reward_mode.
    """
    parts = [ALGO, f"pred-{PREDICTOR_NAME}", f"H{HORIZON}_k{HORIZON}", REWARD_MODE]

    if REWARD_MODE in {"relative", "relative_burn", "relative_final"}:
        parts.append(f"alpha{ALPHA}")

    if REWARD_MODE in {"burn", "relative_burn"}:
        parts.append(f"beta{BETA}")

    if REWARD_MODE in {"relative", "burn", "relative_burn"}:
        parts.append(f"base{BASELINE_MODE}")
        parts.append(f"bw{BASELINE_WINDOW}")
        parts.append(f"cd{CLIP_DIFF}")

    parts.append(f"steps{TIMESTEPS}")
    parts.append(stamp)
    return "_".join(parts)


def make_env_preloaded(data: ExchangeRateData):
    env = ExchangeEnv(
        sampler=data,
        predictor_class=PREDICTOR_CLASS,
        horizon=HORIZON,
        total_amount=TOTAL_AMOUNT,
        k=HORIZON,                 # force k = horizon
        history_window=HISTORY_WINDOW,
        reward_mode=REWARD_MODE,   # "original" | "relative" | "burn" | "relative_burn" | "relative_final"
        alpha=ALPHA,
        beta=BETA,
        baseline_mode=BASELINE_MODE,    # "predictive" | "rolling"
        baseline_window=BASELINE_WINDOW,
        clip_diff=CLIP_DIFF,
    )
    return env


def main():
    _assert_relevance_or_coerce()

    # Preload data once
    data = ExchangeRateData(CSV_PATHS)

    env = make_env_preloaded(data)

    # Choose algo
    algo_map = {"ppo": PPO, "a2c": A2C, "ddpg": DDPG}
    ModelCls = algo_map[ALGO]

    # TB log & save dirs
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_dir = os.path.join("./tensorboard_logs", ALGO, PREDICTOR_NAME, f"H{HORIZON}", REWARD_MODE, run_stamp)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    # Construct model
    common_kwargs = dict(verbose=1, tensorboard_log=tb_dir, gamma=1.0)
    if ALGO in ["ppo", "a2c"]:
        model = ModelCls("MlpPolicy", env, gae_lambda=1.0, **common_kwargs)
    else:  # ddpg
        model = ModelCls("MlpPolicy", env, **common_kwargs)

    # Train
    model.learn(total_timesteps=TIMESTEPS)

    # Save
    fname = build_run_name(run_stamp)
    path = os.path.join("./models", fname)
    model.save(path)
    print(f"[SAVED] {path}.zip")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
