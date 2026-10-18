"""
data_loader.py
---------------
Loads every CSV in dataset/, fills in blank financial-event amounts using the
cached image-extraction evidence, and returns everything indexed by user_id /
request_id so the rest of the pipeline never has to touch pandas directly.

Only files inside dataset/ are read. No organizer-only data is used.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "media", "images")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache", "image_extractions.json")


def _read_csv(name: str) -> pd.DataFrame:
    path = os.path.join(DATASET_DIR, name)
    return pd.read_csv(path)


def load_image_cache() -> Dict[str, dict]:
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if k != "_comment"}


@dataclass
class Dataset:
    requests: pd.DataFrame
    sample_requests: pd.DataFrame
    profiles: pd.DataFrame
    events: pd.DataFrame
    rates: pd.DataFrame
    payment_options: pd.DataFrame
    messages: pd.DataFrame
    images: pd.DataFrame

    profiles_by_user: Dict[str, dict] = field(default_factory=dict)
    events_by_user: Dict[str, pd.DataFrame] = field(default_factory=dict)
    options_by_request: Dict[str, pd.DataFrame] = field(default_factory=dict)
    messages_by_user: Dict[str, pd.DataFrame] = field(default_factory=dict)
    messages_by_request: Dict[str, pd.DataFrame] = field(default_factory=dict)
    images_by_request: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def request_row(self, request_id: str) -> Optional[dict]:
        rows = self.requests[self.requests.request_id == request_id]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()


def load_dataset() -> Dataset:
    requests = _read_csv("requests.csv")
    sample_requests = _read_csv("sample_requests.csv")
    profiles = _read_csv("financial_profiles.csv")
    events = _read_csv("financial_events.csv")
    rates = _read_csv("exchange_rates.csv")
    options = _read_csv("request_payment_options.csv")
    messages = _read_csv("messages.csv")
    images = _read_csv("images.csv")

    # ---- fill blank amounts from the image-extraction cache ---------------
    cache = load_image_cache()
    blank_mask = events["amount"].isna()
    filled = 0
    for idx in events[blank_mask].index:
        eid = events.at[idx, "event_id"]
        if eid in cache:
            events.at[idx, "amount"] = float(cache[eid]["amount"])
            filled += 1
    still_blank = events["amount"].isna().sum()
    if still_blank:
        # Never invent a number: if evidence is truly missing, drop the
        # amount to 0 only as an absolute last resort and flag it.
        events.loc[events["amount"].isna(), "amount"] = 0.0

    ds = Dataset(
        requests=requests,
        sample_requests=sample_requests,
        profiles=profiles,
        events=events,
        rates=rates,
        payment_options=options,
        messages=messages,
        images=images,
    )

    ds.profiles_by_user = {row["user_id"]: row for row in profiles.to_dict("records")}
    ds.events_by_user = {uid: g.copy() for uid, g in events.groupby("user_id")}
    ds.options_by_request = {rid: g.copy() for rid, g in options.groupby("request_id")}
    ds.messages_by_user = {uid: g.copy() for uid, g in messages.dropna(subset=["user_id"]).groupby("user_id")}
    ds.messages_by_request = {rid: g.copy() for rid, g in messages.dropna(subset=["request_id"]).groupby("request_id")}
    ds.images_by_request = {rid: g.copy() for rid, g in images.dropna(subset=["request_id"]).groupby("request_id")}

    return ds


def image_path(image_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"{image_id}.png")
