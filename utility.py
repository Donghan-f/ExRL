#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 14:52:11 2025

@author: fandonghan
"""


import pandas as pd
import numpy as np
import random  

class ExchangeRateData:
    def __init__(self, csv_paths, fill_method='ffill'):
        """
        Initialize exchange rate data processor.

        Parameters:
        - csv_paths: List of CSV file paths
        - fill_method: 'ffill' (forward fill), 'bfill' (backward fill), or 'linear' (interpolate)
        """
        # Read and concatenate
        dfs = [pd.read_csv(path) for path in csv_paths]
        df = pd.concat(dfs, axis=0)

        # Keep only the columns you need; drop obviously unused or problematic ones
        # (Vol. is all NaN in your file; Change % is a string with '%')
        keep_cols = [c for c in df.columns if c in ("Date", "Price")]
        df = df[keep_cols].copy()

        # Parse dates and sort ascending
        # Your file uses mm/dd/YYYY; pandas can infer, but explicit is safer:
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df = df.dropna(subset=["Date", "Price"]).sort_values("Date").drop_duplicates(subset="Date", keep="last").reset_index(drop=True)

        # Ensure strictly positive prices for log-returns models
        # (clip tiny/negative anomalies if any)
        df["Price"] = df["Price"].clip(lower=1e-12)

        self.data = df
        self._create_complete_date_range(fill_method)

    def _create_complete_date_range(self, fill_method):
        """Create a complete daily date range and fill missing dates for Price."""
        start_date = self.data["Date"].min()
        end_date = self.data["Date"].max()
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Use DatetimeIndex during reindex for cleaner lookups
        self.data = self.data.set_index("Date").reindex(full_date_range)

        if fill_method == 'ffill':
            self.data["Price"] = self.data["Price"].ffill()
        elif fill_method == 'bfill':
            self.data["Price"] = self.data["Price"].bfill()
        else:
            # linear interpolation on calendar days
            self.data["Price"] = self.data["Price"].interpolate(method='linear')

        # Backstops to remove any edge NaN
        self.data["Price"] = self.data["Price"].bfill().ffill()

        # Keep dates as a column again
        self.data = self.data.rename_axis("Date").reset_index()

    def get_rates(self, date_input, horizon):
        """
        Get 'horizon' days of Price starting from the given date,
        using an as-of (previous available date) to avoid lookahead bias.
        """
        date = pd.to_datetime(date_input)

        # data is sorted ascending and unique on Date
        dates = self.data["Date"].values
        pos = self.data["Date"].searchsorted(date, side="right") - 1
        if pos < 0:
            raise ValueError(f"No available date on or before {date}.")

        if pos + horizon > len(self.data):
            raise ValueError(f"Not enough data to get {horizon} days from {self.data.loc[pos, 'Date']} (idx={pos}).")

        return self.data.loc[pos:pos + horizon - 1, "Price"].reset_index(drop=True)

    def get_random_start_date(self, horizon, history_window):
        """
        Get a random start date ensuring enough history (history_window) before it
        and enough future data (horizon) after it.
        """
        n = len(self.data)
        if n < history_window + horizon:
            raise ValueError("Insufficient data to satisfy history_window + horizon.")

        min_start_idx = history_window
        max_start_idx = n - horizon
        if max_start_idx <= min_start_idx:
            raise ValueError("Horizon too long or data too short for the given history_window.")

        start_idx = random.randint(min_start_idx, max_start_idx)
        return self.data.loc[start_idx, "Date"]

    def get_random_rates(self, horizon, history_window=100):
        """Return a (prices, start_date) tuple for a random window."""
        date = self.get_random_start_date(horizon, history_window)
        rates = self.get_rates(date, horizon)
        return rates.to_numpy(), date

    def plot_missing_dates(self):
        """Quick visualization of completeness after reindex (optional)."""
        start_date = self.data["Date"].min()
        end_date = self.data["Date"].max()
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        exists = pd.Series(False, index=full_date_range)
        exists[self.data["Date"].values] = True

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(exists.index, exists.values)
        plt.title("Date completeness (1=exists)")
        plt.xlabel("Date")
        plt.yticks([0, 1])
        plt.grid(True, axis='x', alpha=0.3)
        plt.show()
