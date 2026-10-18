"""
fx.py
-----
Fixed, dated currency conversion using only dataset/exchange_rates.csv.
No live rates, no invented rates. Every foreign-currency event in this
dataset settles on a date that has an exact matching rate row for its
(from_currency -> to_currency) pair, so exact-match lookup is the primary
path; a small nearest-neighbour fallback exists only for robustness.
"""
from __future__ import annotations

import pandas as pd


class RateTable:
    def __init__(self, rates_df: pd.DataFrame):
        self._df = rates_df.copy()
        self._df["rate_date"] = pd.to_datetime(self._df["rate_date"])
        self._index = {}
        for row in self._df.itertuples(index=False):
            key = (row.from_currency, row.to_currency, row.rate_date)
            self._index[key] = float(row.rate)
        # per-pair sorted date list for fallback nearest lookup
        self._pairs = {}
        for (f, t), g in self._df.groupby(["from_currency", "to_currency"]):
            self._pairs[(f, t)] = g.sort_values("rate_date")

    def rate(self, from_ccy: str, to_ccy: str, on_date) -> float:
        if from_ccy == to_ccy:
            return 1.0
        on_date = pd.to_datetime(on_date)
        key = (from_ccy, to_ccy, on_date)
        if key in self._index:
            return self._index[key]
        # try inverse direction exact match
        inv_key = (to_ccy, from_ccy, on_date)
        if inv_key in self._index:
            return 1.0 / self._index[inv_key]
        # fallback: nearest available date for that pair (defensive only;
        # not expected to trigger on the supplied dataset).
        if (from_ccy, to_ccy) in self._pairs:
            g = self._pairs[(from_ccy, to_ccy)]
            nearest = (g["rate_date"] - on_date).abs().idxmin()
            return float(g.loc[nearest, "rate"])
        if (to_ccy, from_ccy) in self._pairs:
            g = self._pairs[(to_ccy, from_ccy)]
            nearest = (g["rate_date"] - on_date).abs().idxmin()
            return 1.0 / float(g.loc[nearest, "rate"])
        raise ValueError(f"No exchange rate available for {from_ccy}->{to_ccy} near {on_date}")

    def convert(self, amount: float, from_ccy: str, to_ccy: str, on_date) -> float:
        return amount * self.rate(from_ccy, to_ccy, on_date)
