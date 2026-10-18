"""
payment_planner.py
-------------------
Builds every safe, eligible candidate plan for a request:
  full_payment (with or without up to 3 spending changes), partial_payment,
  installments (one candidate per matching supplied payment option), and
  wait. decision_engine.py then ranks the candidates per the challenge's
  six tie-break rules.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from financial_state import UserFinancialModel
from forecast_engine import (
    Forecast,
    amount_safe_today,
    compute_forecast,
    earliest_full_payment_date,
    schedule_is_safe,
)
from fx import RateTable

EPS = 1e-6


@dataclass
class Candidate:
    method: str
    total_paid: float
    payments: List[Tuple[date, float]]
    uses_changes: bool
    start_date: date
    num_payments: int
    meets_deadline: bool
    payment_option_id: Optional[str] = None
    changes: List[str] = field(default_factory=list)
    changes_detail: List[dict] = field(default_factory=list)
    amount_today: float = 0.0  # what this plan actually pays on request_date


def _find_spending_changes(model: UserFinancialModel, rates: RateTable, request_date: date,
                            requested_amount: float):
    candidates = []
    all_cats = (model.reduce_categories | model.stop_categories) - model.protect
    for cat in all_cats:
        item = model.recurring_bills.get(cat) or model.variable_bills.get(cat)
        if item is None:
            continue
        flexibility = item.flexibility
        if (cat in model.reduce_categories and flexibility in ("reducible", "reducible_or_stoppable")
                and item.minimum_allowed_amount is not None and item.minimum_allowed_amount < item.amount):
            saving = item.amount - item.minimum_allowed_amount
            candidates.append({"type": "reduce", "category": cat, "event_id": item.representative_event_id,
                                "new_amount": item.minimum_allowed_amount, "saving": saving,
                                "description": item.description})
        if cat in model.stop_categories and flexibility in ("stoppable", "reducible_or_stoppable"):
            candidates.append({"type": "stop", "category": cat, "event_id": item.representative_event_id,
                                "new_amount": None, "saving": item.amount, "description": item.description})

    if not candidates:
        return None, None, None

    ordered = sorted(candidates, key=lambda c: (0 if c["type"] == "reduce" else 1, c["saving"]))
    for k in (1, 2, 3):
        for combo in itertools.combinations(ordered, k):
            cats = [c["category"] for c in combo]
            if len(set(cats)) != len(cats):
                continue
            exclude = {c["category"]: c["new_amount"] for c in combo if c["type"] == "reduce"}
            stop_set = {c["category"] for c in combo if c["type"] == "stop"}
            fc = compute_forecast(model, request_date, rates, exclude_categories=exclude, stop_categories=stop_set)
            safe = amount_safe_today(fc, model.min_balance, requested_amount)
            if safe >= requested_amount - EPS:
                changes = []
                for c in combo:
                    if c["type"] == "stop":
                        changes.append(f"stop:{c['event_id']}")
                    else:
                        changes.append(f"reduce_to:{c['event_id']}:{round(c['new_amount'], 2)}")
                return changes, list(combo), fc
    return None, None, None


def build_candidates(model: UserFinancialModel, request_row: dict, rates: RateTable,
                      payment_options: Optional[pd.DataFrame]) -> Dict:
    request_date = pd.to_datetime(request_row["request_date"]).date()
    desired_completion_date = pd.to_datetime(request_row["desired_completion_date"]).date()
    requested_amount = float(request_row["requested_amount"])
    allows_partial = bool(request_row["allows_partial_payment"])

    baseline = compute_forecast(model, request_date, rates)
    amount_safe0 = amount_safe_today(baseline, model.min_balance, requested_amount)
    earliest0 = earliest_full_payment_date(baseline, model.min_balance, requested_amount, request_date)

    candidates: List[Candidate] = []
    methods = model.payment_methods

    if "full_payment" in methods and amount_safe0 >= requested_amount - EPS:
        candidates.append(Candidate(
            method="full_payment", total_paid=requested_amount,
            payments=[(request_date, requested_amount)], uses_changes=False,
            start_date=request_date, num_payments=1,
            meets_deadline=request_date <= desired_completion_date,
            amount_today=requested_amount,
        ))

    changes_used, changes_detail, forecast_with_changes = (None, None, None)
    if "full_payment" in methods and amount_safe0 < requested_amount - EPS:
        changes_used, changes_detail, forecast_with_changes = _find_spending_changes(
            model, rates, request_date, requested_amount)
        if changes_used:
            candidates.append(Candidate(
                method="full_payment", total_paid=requested_amount,
                payments=[(request_date, requested_amount)], uses_changes=True,
                start_date=request_date, num_payments=1,
                meets_deadline=request_date <= desired_completion_date,
                changes=changes_used, changes_detail=changes_detail, amount_today=requested_amount,
            ))

    if (allows_partial and "partial_payment" in methods and earliest0 is not None
            and 0 < amount_safe0 < requested_amount - EPS):
        remainder = requested_amount - amount_safe0
        payments = [(request_date, amount_safe0), (earliest0, remainder)]
        candidates.append(Candidate(
            method="partial_payment", total_paid=requested_amount, payments=payments,
            uses_changes=False, start_date=request_date, num_payments=2,
            meets_deadline=earliest0 <= desired_completion_date, amount_today=amount_safe0,
        ))

    if "installments" in methods and pd.notna(model.max_installment_months) and payment_options is not None:
        inst = payment_options[payment_options.payment_method == "installments"]
        for _, row in inst.iterrows():
            n = int(row["number_of_payments"])
            if n > float(model.max_installment_months):
                continue
            freq = int(row["payment_frequency_days"]) if pd.notna(row["payment_frequency_days"]) else 30
            start = pd.to_datetime(row["first_payment_date"]).date()
            amt = float(row["payment_amount"])
            payments = [(start + timedelta(days=freq * i), amt) for i in range(n)]
            last_date = payments[-1][0]
            if not schedule_is_safe(baseline, model.min_balance, request_date, payments):
                continue
            candidates.append(Candidate(
                method="installments", total_paid=float(row["total_payable_amount"]), payments=payments,
                uses_changes=False, start_date=start, num_payments=n,
                meets_deadline=last_date <= desired_completion_date,
                payment_option_id=row["payment_option_id"],
                amount_today=amt if start == request_date else 0.0,
            ))

    if "full_payment" in methods and earliest0 is not None:
        candidates.append(Candidate(
            method="wait", total_paid=requested_amount, payments=[(earliest0, requested_amount)],
            uses_changes=False, start_date=earliest0, num_payments=1,
            meets_deadline=earliest0 <= desired_completion_date, amount_today=0.0,
        ))

    return {
        "candidates": candidates,
        "baseline": baseline,
        "amount_safe0": amount_safe0,
        "earliest0": earliest0,
        "request_date": request_date,
        "desired_completion_date": desired_completion_date,
        "requested_amount": requested_amount,
        "forecast_with_changes": forecast_with_changes,
    }
