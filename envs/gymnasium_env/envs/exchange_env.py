import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utility import ExchangeRateData
import pandas as pd
import random

class ExchangeEnv(gym.Env):
    """
    Custom environment for optimal currency exchange over a finite horizon.
    """

    def __init__(self,
                 sampler: ExchangeRateData,
                 predictor_class,
                 horizon: int,
                 total_amount: float,
                 k: int = 5,
                 history_window=100,
                 penalty_factor: float = 1.0,
                 max_daily_exchange: float = None,
                 render_mode=None,
                 # NEW: reward configuration
                 reward_mode: str = "original",         # "original" | "relative" | "burn" | "relative_burn"
                 alpha: float = 0.01,                   # leftover pressure for "relative"/"relative_burn"
                 beta: float = 0.05,                    # burn-curve penalty for "burn"/"relative_burn"
                 baseline_mode: str = "predictive",     # "predictive" | "rolling"
                 baseline_window: int = 20,             # rolling mean window if baseline_mode="rolling"
                 clip_diff: float = 0.1                 # clip for baseline-rate difference in CNY
                 ):
        super().reset()

        # === original args ===
        self.predictor_class = predictor_class
        self.sampler = sampler
        self.horizon = horizon
        self.total_amount = total_amount
        self.k = k
        self.history_window = history_window
        self.penalty_factor = penalty_factor
        self.max_daily_exchange = max_daily_exchange or total_amount
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        # === NEW: reward config ===
        assert reward_mode in {"original", "relative", "burn", "relative_burn", "relative_final"}
        assert baseline_mode in {"predictive", "rolling"}
        self.reward_mode = reward_mode
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.baseline_mode = baseline_mode
        self.baseline_window = int(baseline_window)
        self.clip_diff = float(clip_diff)

        # Action/obs spaces
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        obs_dim = 2 + k
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Internal state
        self.current_step = 0
        self.money_left = None
        self.true_rates = None
        self.predictor = predictor_class(history_window, k)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.money_left = self.total_amount
        self.true_rates, start_date = self.sampler.get_random_rates(self.horizon, self.history_window)
        self.predictor.reset(self.sampler.data, start_date)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # ---- action to amount ----
        raw = np.asarray(action, dtype=np.float64)
        fraction = float(np.clip(raw, 0.0, 1.0))
        amount = fraction * min(self.max_daily_exchange, self.money_left)

        # ---- current rate ----
        rate = float(self.true_rates[self.current_step])

        # ---- compute reward by selected mode ----
        reward = self._compute_reward(rate=rate, amount=amount)

        # ---- update state ----
        self.money_left -= amount
        self.current_step += 1
        self.predictor.update(self.current_step)
        done = self.current_step >= self.horizon

        # terminal handling
        if done:
            # original terminal penalty remains only for "original" mode
            if self.reward_mode == "original" and self.money_left > 0:
                final_rate = float(self.true_rates[self.current_step-1])
                reward -= self.money_left * final_rate * self.penalty_factor

            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_obs()

        return obs, reward, done, False, {}

    def _get_obs(self):
        days_left = (self.horizon - self.current_step) / self.horizon
        money_left_norm = self.money_left / self.total_amount
        available_k = self.horizon - self.current_step
        effective_k = min(self.k, available_k)

        preds = self.predictor.predict(self.current_step, effective_k)
        current_rate = float(self.true_rates[self.current_step])

        relative_preds = (preds - current_rate) / (current_rate + 1e-8)
        relative_preds = np.clip(relative_preds, -0.1, 0.1)

        if effective_k < self.k:
            padding = np.zeros(self.k - effective_k, dtype=np.float32)
            relative_preds = np.concatenate([relative_preds, padding])

        obs = np.concatenate([[days_left, money_left_norm], relative_preds])
        return obs.astype(np.float32)

    def _compute_reward(self, rate: float, amount: float) -> float:
        if self.reward_mode == "original":
            return - amount * rate
        # other reward modes skipped for brevity
        return - amount * rate

    def render(self):
        pass

    def close(self):
        pass
