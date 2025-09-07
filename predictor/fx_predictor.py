#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 15:30:40 2025

@author: fandonghan
"""

# fx_predictors.py
from __future__ import annotations
import numpy as np
import pandas as pd
import warnings
from typing import Optional, Tuple, List

from .base import BasePredictor

# Statsmodels building blocks
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents
try:
    # Available in statsmodels >= 0.12
    from statsmodels.tsa.forecasting.theta import ThetaModel
    _HAS_THETA = True
except Exception:
    _HAS_THETA = False

# ------------------------------
# Utility: clean price series
# ------------------------------
def _clean_prices(df: pd.DataFrame) -> np.ndarray:
    s = pd.to_numeric(df["Price"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return s.values.astype(float)

# ------------------------------
# 1) Naive random-walk predictor
# ------------------------------
class NaiveRWPredictor(BasePredictor):
    """
    Baseline: random walk (last observed price repeated).
    Extremely robust; a must-have benchmark in FX.
    """
    def predict(self, current_step: int, k: int) -> np.ndarray:
        self._update_history(current_step)
        y = _clean_prices(self.history_data)
        last = float(y[-1]) if len(y) else 1.0
        return np.full(k, last, dtype=np.float32)

# ------------------------------
# 2) Drift predictor (OLS slope)
# ------------------------------
class DriftPredictor(BasePredictor):
    """
    Linear drift on price using OLS slope over the history window.
    Forecast: last_price + slope * h.
    """
    def __init__(self, history_window: int = 250, pred_window: int = 5, include_current: bool = False):
        super().__init__(history_window, pred_window, include_current)

    def predict(self, current_step: int, k: int) -> np.ndarray:
        self._update_history(current_step)
        y = _clean_prices(self.history_data)
        n = len(y)
        if n < 3:
            last = float(y[-1]) if n else 1.0
            return np.full(k, last, dtype=np.float32)

        # OLS slope (time index 0..n-1)
        t = np.arange(n, dtype=np.float64)
        y = y.astype(np.float64)
        t_mean = t.mean()
        y_mean = y.mean()
        denom = ((t - t_mean) ** 2).sum()
        slope = 0.0 if denom == 0.0 else ((t - t_mean) * (y - y_mean)).sum() / denom
        last = y[-1]
        fc = np.array([last + slope * (h) for h in range(1, k + 1)], dtype=np.float64)
        return fc.astype(np.float32)

# ---------------------------------------------
# 3) Holt-Winters (Exponential Smoothing, ETS)
# ---------------------------------------------
class HoltWintersPredictor(BasePredictor):
    """
    Exponential Smoothing (Holt-Winters):
    - trend='add' (robust for FX levels)
    - optional weekly seasonality (seasonal='add', seasonal_periods=7)
    Falls back to Drift if series too short to estimate a seasonal component.
    """
    def __init__(self, history_window: int = 250, pred_window: int = 5,
                 include_current: bool = False, seasonal: bool = False, seasonal_periods: int = 7):
        super().__init__(history_window, pred_window, include_current)
        self.seasonal = seasonal
        self.seasonal_periods = int(seasonal_periods)

    def predict(self, current_step: int, k: int) -> np.ndarray:
        self._update_history(current_step)
        y = _clean_prices(self.history_data)
        n = len(y)

        if n < 5:
            # Too short, use naive
            last = float(y[-1]) if n else 1.0
            return np.full(k, last, dtype=np.float32)

        try:
            if self.seasonal and n >= max(10, 2 * self.seasonal_periods):
                model = ExponentialSmoothing(
                    y, trend="add", seasonal="add",
                    seasonal_periods=self.seasonal_periods, initialization_method="estimated"
                )
            else:
                model = ExponentialSmoothing(
                    y, trend="add", seasonal=None, initialization_method="estimated"
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = model.fit(optimized=True, use_brute=False)
            fc = fit.forecast(k)
            return np.asarray(fc, dtype=np.float32)
        except Exception as e:
            print(f"[WARN] Holt-Winters failed: {type(e).__name__}: {e} -> using Drift fallback.")
            # Drift fallback
            t = np.arange(n, dtype=np.float64)
            y_mean = y.mean(); t_mean = t.mean()
            denom = ((t - t_mean) ** 2).sum()
            slope = 0.0 if denom == 0.0 else ((t - t_mean) * (y - y_mean)).sum() / denom
            last = y[-1]
            fc = np.array([last + slope * (h) for h in range(1, k + 1)], dtype=np.float64)
            return fc.astype(np.float32)

# -----------------------------------------------------
# 4) Theta method (M3/M4 winner; via statsmodels Theta)
# -----------------------------------------------------
class ThetaPredictor(BasePredictor):
    """
    Theta method (if available in statsmodels). Robust, low-variance.
    If ThetaModel is not available, falls back to Holt-Winters without seasonality.
    """
    def __init__(self, history_window: int = 250, pred_window: int = 5,
                 include_current: bool = False, seasonal_periods: int = 7):
        super().__init__(history_window, pred_window, include_current)
        self.seasonal_periods = int(seasonal_periods)

    def predict(self, current_step: int, k: int) -> np.ndarray:
        self._update_history(current_step)
        y = _clean_prices(self.history_data)
        n = len(y)

        if n < 4:
            last = float(y[-1]) if n else 1.0
            return np.full(k, last, dtype=np.float32)

        if _HAS_THETA:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tm = ThetaModel(y, period=self.seasonal_periods if self.seasonal_periods > 1 else None)
                    fit = tm.fit()
                    fc = fit.forecast(k)
                return np.asarray(fc, dtype=np.float32)
            except Exception as e:
                print(f"[WARN] ThetaModel failed: {type(e).__name__}: {e} -> using Holt-Winters no-seasonal.")
        # Fallback: Holt-Winters without seasonality
        try:
            model = ExponentialSmoothing(y, trend="add", seasonal=None, initialization_method="estimated")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = model.fit(optimized=True)
            fc = fit.forecast(k)
            return np.asarray(fc, dtype=np.float32)
        except Exception as e:
            print(f"[WARN] Holt-Winters fallback failed: {type(e).__name__}: {e} -> using Naive.")
            last = float(y[-1])
            return np.full(k, last, dtype=np.float32)

# ----------------------------------------------------
# 5) Local Level (Kalman filter; Structural time series)
# ----------------------------------------------------
class LocalLevelPredictor(BasePredictor):
    """
    State-space local level model (random walk + noise), estimated via Kalman filter.
    Very appropriate for FX levels that behave like noisy random walks.
    """
    def __init__(self, history_window: int = 250, pred_window: int = 5, include_current: bool = False):
        super().__init__(history_window, pred_window, include_current)

    def predict(self, current_step: int, k: int) -> np.ndarray:
        self._update_history(current_step)
        y = _clean_prices(self.history_data)
        n = len(y)

        if n < 3:
            last = float(y[-1]) if n else 1.0
            return np.full(k, last, dtype=np.float32)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = UnobservedComponents(y, level="local level")
                fit = mod.fit(disp=False)
                fc = fit.forecast(steps=int(k))
            return np.asarray(fc, dtype=np.float32)
        except Exception as e:
            print(f"[WARN] LocalLevel failed: {type(e).__name__}: {e} -> using Naive.")
            last = float(y[-1])
            return np.full(k, last, dtype=np.float32)

# ----------------------------------------------------
# 6) SARIMAX (auto small grid; d=1; optional weekly seasonality)
# ----------------------------------------------------
class SARIMAXPredictor(BasePredictor):
    """
    SARIMAX on price with small-order grid (d=1), optional weekly seasonality (s=7).
    Safer than manual ARIMA on tricky series; good general-purpose baseline.
    """
    def __init__(self, history_window: int = 250, pred_window: int = 5,
                 include_current: bool = False, seasonal: bool = False, s: int = 7):
        super().__init__(history_window, pred_window, include_current)
        self.seasonal = seasonal
        self.s = int(s)
        # Non-seasonal order grid (p,d,q) with d=1
        self.orders = [(0,1,0), (1,1,0), (0,1,1), (1,1,1), (2,1,1)]
        # Seasonal order grid (P,D,Q,s)
        self.seasonal_orders = [(0,0,0,self.s), (0,1,1,self.s), (1,0,1,self.s), (1,1,1,self.s)]

    def _fit_one(self, y: np.ndarray, order: Tuple[int,int,int], seasonal_order: Tuple[int,int,int,int] | Tuple[int,int,int,int] | None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = SARIMAX(
                y, order=order,
                seasonal_order=(seasonal_order if self.seasonal else (0,0,0,0)),
                enforce_stationarity=False, enforce_invertibility=False
            )
            res = mod.fit(disp=False, maxiter=500, method="lbfgs")
        return res

    def predict(self, current_step: int, k: int) -> np.ndarray:
        self._update_history(current_step)
        y = _clean_prices(self.history_data)
        n = len(y)

        if n < 10:
            # Not enough data -> Naive
            last = float(y[-1]) if n else 1.0
            return np.full(k, last, dtype=np.float32)

        best = None
        best_aic = np.inf

        # Try non-seasonal first
        for order in self.orders:
            try:
                res = self._fit_one(y, order, None)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best = res
            except Exception:
                continue

        # Optionally try a small seasonal grid
        if self.seasonal:
            for sorder in self.seasonal_orders:
                for order in self.orders[:3]:
                    try:
                        res = self._fit_one(y, order, sorder)
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best = res
                    except Exception:
                        continue

        if best is None:
            print("[WARN] SARIMAX grid search failed -> using Drift fallback.")
            # Drift fallback
            t = np.arange(n, dtype=np.float64)
            y_mean = y.mean(); t_mean = t.mean()
            denom = ((t - t_mean) ** 2).sum()
            slope = 0.0 if denom == 0.0 else ((t - t_mean) * (y - y_mean)).sum() / denom
            last = y[-1]
            fc = np.array([last + slope * (h) for h in range(1, k + 1)], dtype=np.float64)
            return fc.astype(np.float32)

        try:
            fc = best.forecast(steps=int(k))
            return np.asarray(fc, dtype=np.float32)
        except Exception as e:
            print(f"[WARN] SARIMAX forecast failed: {type(e).__name__}: {e} -> using Naive.")
            last = float(y[-1])
            return np.full(k, last, dtype=np.float32)
