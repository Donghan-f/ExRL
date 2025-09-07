#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 18:11:06 2025

@author: fandonghan
"""
# base.py
# Author: fandonghan (revised)


from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Union

class BasePredictor:
    """
    Generic base class for predictors.

    Parameters
    ----------
    history_window : int
        Number of past rows to keep in history for modeling.
    pred_window : int
        Default number of steps to forecast (can be overridden in predict()).
    include_current : bool
        Whether to include the current index row in the history slice.
        - If your environment increments current_step after one full day,
          then set include_current=False (default). This way the "just finished day"
          is included in the history, and the current day (not finished yet)
          is excluded.
        - If you want to include the current row as well, set True.

    Notes
    -----
    full_data must contain at least columns ["Date", "Price"].
    """

    def __init__(self, history_window: int = 100, pred_window: int = 5, include_current: bool = False):
        self.history_window = int(history_window)
        self.pred_window = int(pred_window)
        self.include_current = bool(include_current)

        self.full_data: Optional[pd.DataFrame] = None
        self.start_date: Optional[pd.Timestamp] = None
        self.start_idx: Optional[int] = None
        self.history_data: Optional[pd.DataFrame] = None

    # ------------ main flow ------------
    def reset(self, full_data: pd.DataFrame, start_date: Union[str, pd.Timestamp]):
        """
        Set the full dataset and the starting date (for backtest/rolling forecast),
        and initialize the history window.
        """
        if not isinstance(full_data, pd.DataFrame):
            raise TypeError("full_data must be a pandas DataFrame.")
        if "Date" not in full_data.columns or "Price" not in full_data.columns:
            raise ValueError("full_data must contain columns ['Date', 'Price'].")

        # Copy and normalize
        df = full_data.copy()
        df = df[["Date", "Price"]].reset_index(drop=True)

        # Standardize types
        df["Date"] = pd.to_datetime(df["Date"])
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

        # Drop rows with invalid values
        df = df.dropna(subset=["Date", "Price"]).reset_index(drop=True)

        self.full_data = df
        self.start_date = pd.to_datetime(start_date)

        self._init_start_index()
        self._update_history(0)

    def update(self, current_step: int):
        """
        Called after the environment advances.
        Adds the most recent day to the history (or also the current row,
        depending on include_current).
        """
        self._update_history(current_step)

    def predict(self, current_step: int, k: int) -> np.ndarray:
        """
        Must be implemented by subclasses.
        Forecast k steps ahead using the current history_data.
        """
        raise NotImplementedError

    # ------------ internals ------------
    def _init_start_index(self):
        assert self.full_data is not None and self.start_date is not None

        # Find position index of the start_date
        mask = (self.full_data["Date"] == self.start_date)
        pos = np.flatnonzero(mask.values)
        if len(pos) == 0:
            raise ValueError(f"{self.start_date} not found in full_data['Date'].")
        self.start_idx = int(pos[0])

    def _update_history(self, current_step: int = 0):
        """
        Slice history window based on position index.

        end_pos = start_idx + current_step (+1 if include_current)
        history = [max(0, end_pos - history_window) : end_pos]
        """
        assert self.full_data is not None and self.start_idx is not None

        idx_pos = int(self.start_idx + int(current_step))
        end_pos = idx_pos + (1 if self.include_current else 0)
        end_pos = max(0, min(end_pos, len(self.full_data)))
        start_pos = max(0, end_pos - self.history_window)

        self.history_data = self.full_data.iloc[start_pos:end_pos].reset_index(drop=True)
