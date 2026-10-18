"""
forecast_engine.py
-------------------
Projects a UserFinancialModel forward for 90 days from request_date and
answers the two safety questions the challenge is built around:

  * amount_safe_to_pay        -- the most that can be paid on request_date
                                   without the balance ever dropping below
                                   minimum_balance_to_keep in the next 90 days
  * earliest_date_for_full_payment -- the first day a single full payment
                                   of requested_amount would stay safe

Key property exploited here: a same-day cash payment shifts every later
point of the balance curve down by exactly that amount (no interest, no
re-borrowing). That means both questions reduce to one forward scan of the
90-day balance curve and one backward "suffix minimum" pass -- no search
or trial-and-error is needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional, Tuple

from financial_state import FutureFlow, UserFinancialModel
from fx import RateTable

HORIZON_DAYS = 90


@dataclass
class Forecast:
    request_date: date
    horizon_end: date
    daily_balance: List[float]           # index 0..HORIZON_DAYS, end-of-day balance
    suffix_min: List[float]              # suffix_min[d] = min(daily_balance[d:])
    flows: List[Tuple[date, float, str]]  # (date, signed amount, label) actually applied


def _day_index(d: date, request_date: date) -> Optional[int]:
    idx = (d - request_date).days
    if 0 <= idx <= HORIZON_DAYS:
        return idx
    return None


def build_flows(model: UserFinancialModel, request_date: date, rates: RateTable,
                 exclude_categories: Optional[dict] = None,
                 stop_categories: Optional[set] = None) -> List[Tuple[date, float, str]]:
    """Materialise every dated cash flow for the horizon, in home currency.

    `exclude_categories`: {category: new_amount} to apply a reduce_to change
    to every future occurrence of that recurring/variable category.
    `stop_categories`: categories whose future occurrences are dropped
    entirely (a stop: change).
    """
    exclude_categories = exclude_categories or {}
    stop_categories = stop_categories or set()
    horizon_end = request_date + timedelta(days=HORIZON_DAYS)
    flows: List[Tuple[date, float, str]] = []

    def add(d: date, amt: float, label: str, currency: str):
        if d < request_date or d > horizon_end:
            return
        home_amt = rates.convert(amt, currency, model.home_currency, d) if currency != model.home_currency else amt
        flows.append((d, home_amt, label))

    # recurring fixed bills
    for cat, item in model.recurring_bills.items():
        if cat in stop_categories:
            continue
        amount = exclude_categories.get(cat, item.amount)
        d = item.anchor_date
        while d <= request_date:
            d = d + timedelta(days=item.frequency_days)
        while d <= horizon_end:
            add(d, -amount, f"{item.description} ({cat})", item.currency)
            d = d + timedelta(days=item.frequency_days)

    # variable essential spend, spread mid-month across the horizon
    for cat, item in model.variable_bills.items():
        if cat in stop_categories:
            continue
        amount = exclude_categories.get(cat, item.amount)
        for offset in (15, 45, 75):
            d = request_date + timedelta(days=offset)
            if d <= horizon_end:
                add(d, -amount, f"{item.description} ({cat})", item.currency)

    # salary
    if model.salary is not None:
        item = model.salary
        d = item.anchor_date
        # first future occurrence
        if d <= request_date:
            while d <= request_date:
                d = d + timedelta(days=item.frequency_days)
        explicit_dates = {f.date for f in model.explicit_future if f.category == "salary"}
        while d <= horizon_end:
            if not any(abs((d - ed).days) <= 5 for ed in explicit_dates):
                add(d, item.amount, "Recurring salary", item.currency)
            d = d + timedelta(days=item.frequency_days)

    # explicit future rows already resolved into home-currency-agnostic FutureFlow
    for f in model.explicit_future:
        cat = f.category
        if cat in stop_categories:
            continue
        amt = f.amount
        if cat in exclude_categories and amt < 0:
            amt = -exclude_categories[cat]
        add(f.date, amt, f.label, model.home_currency)

    flows.sort(key=lambda x: x[0])
    return flows


def compute_forecast(model: UserFinancialModel, request_date: date, rates: RateTable,
                      exclude_categories: Optional[dict] = None,
                      stop_categories: Optional[set] = None) -> Forecast:
    flows = build_flows(model, request_date, rates, exclude_categories, stop_categories)
    horizon_end = request_date + timedelta(days=HORIZON_DAYS)
    daily_delta = [0.0] * (HORIZON_DAYS + 1)
    for d, amt, _label in flows:
        idx = _day_index(d, request_date)
        if idx is not None:
            daily_delta[idx] += amt

    daily_balance = [0.0] * (HORIZON_DAYS + 1)
    running = model.balance
    for i in range(HORIZON_DAYS + 1):
        running += daily_delta[i]
        daily_balance[i] = running

    suffix_min = [0.0] * (HORIZON_DAYS + 1)
    suffix_min[HORIZON_DAYS] = daily_balance[HORIZON_DAYS]
    for i in range(HORIZON_DAYS - 1, -1, -1):
        suffix_min[i] = min(daily_balance[i], suffix_min[i + 1])

    return Forecast(request_date=request_date, horizon_end=horizon_end,
                     daily_balance=daily_balance, suffix_min=suffix_min, flows=flows)


def amount_safe_today(forecast: Forecast, min_balance: float, requested_amount: float) -> float:
    safe = forecast.suffix_min[0] - min_balance
    return max(0.0, min(safe, requested_amount))


def earliest_full_payment_date(forecast: Forecast, min_balance: float, requested_amount: float,
                                request_date: date) -> Optional[date]:
    for i in range(HORIZON_DAYS + 1):
        if forecast.suffix_min[i] - min_balance >= requested_amount - 1e-6:
            return request_date + timedelta(days=i)
    return None


def schedule_is_safe(forecast: Forecast, min_balance: float, request_date: date,
                      payments: List[Tuple[date, float]]) -> bool:
    """Check a multi-payment schedule against the (unmodified) baseline curve."""
    cum_by_day = [0.0] * (HORIZON_DAYS + 1)
    for d, amt in payments:
        idx = _day_index(d, request_date)
        if idx is None:
            # a payment scheduled outside the 90-day window cannot be
            # verified safe within the forecast horizon.
            return False
        cum_by_day[idx] += amt
    running_extra = 0.0
    for i in range(HORIZON_DAYS + 1):
        running_extra += cum_by_day[i]
        if forecast.daily_balance[i] - running_extra < min_balance - 1e-6:
            return False
    return True
