"""
financial_state.py
-------------------
Reconstructs a user's forward-looking cash-flow model from:
  * financial_profiles.csv   (balance, minimum balance, preferences)
  * financial_events.csv     (history used to detect recurring patterns,
                               plus any explicit future-dated rows)
  * messages.csv              (evidence that revises the salary forecast,
                               reinstates a failed-but-still-owed bill, or
                               bumps up a recurring rent)

Historical settled events are NEVER replayed into the balance -- the
`current_available_balance` in the profile already reflects everything
that happened before `request_date`. History is used only to detect
*patterns* that are projected forward.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd

from ai.message_parser import (
    NEVER_TRUST_AS_INCOME,
    SALARY_KINDS,
    MessageEvidence,
    classify_message,
)

FIXED_BILL_CATEGORIES = {
    "rent", "utilities", "debt_repayment", "insurance", "housing",
    "education", "gym", "family_support", "music_subscription",
    "delivery_membership", "cloud_storage", "streaming",
}
VARIABLE_CATEGORIES = {
    "groceries", "dining", "transport", "shopping", "entertainment",
    "healthcare", "work_expense",
}
NON_CASH_STATUSES = {"cancelled", "failed", "unrealized"}


@dataclass
class RecurringItem:
    category: str
    description: str
    amount: float
    currency: str
    anchor_date: date
    frequency_days: int
    flexibility: str
    minimum_allowed_amount: Optional[float]
    representative_event_id: str
    direction: str  # "debit" or "credit"


@dataclass
class FutureFlow:
    date: date
    amount: float          # signed, home currency
    category: str
    label: str
    event_id: Optional[str] = None
    changeable: bool = False
    flexibility: Optional[str] = None
    minimum_allowed_amount: Optional[float] = None


@dataclass
class UserFinancialModel:
    user_id: str
    home_currency: str
    balance: float
    min_balance: float
    protect: set
    reduce_categories: set
    stop_categories: set
    payment_methods: set
    max_installment_months: Optional[float]
    priorities: List[str]
    recurring_bills: Dict[str, RecurringItem] = field(default_factory=dict)
    variable_bills: Dict[str, RecurringItem] = field(default_factory=dict)
    salary: Optional[RecurringItem] = None
    explicit_future: List[FutureFlow] = field(default_factory=list)
    representative_event: Dict[str, dict] = field(default_factory=dict)  # category -> most recent event row


def _pipe_set(val) -> set:
    if not isinstance(val, str) or not val.strip():
        return set()
    return {v.strip() for v in val.split("|") if v.strip()}


def _cv(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = statistics.mean(values)
    if m == 0:
        return 0.0
    return statistics.pstdev(values) / abs(m)


def _to_date(x) -> date:
    if isinstance(x, date):
        return x
    return pd.to_datetime(x).date()


def build_user_model(user_id: str, profile_row: dict, user_events: pd.DataFrame,
                      user_messages: pd.DataFrame, request_date: date) -> UserFinancialModel:
    model = UserFinancialModel(
        user_id=user_id,
        home_currency=profile_row["home_currency"],
        balance=float(profile_row["current_available_balance"]),
        min_balance=float(profile_row["minimum_balance_to_keep"]),
        protect=_pipe_set(profile_row.get("expense_categories_to_protect")),
        reduce_categories=_pipe_set(profile_row.get("expense_categories_user_is_willing_to_reduce")),
        stop_categories=_pipe_set(profile_row.get("expense_categories_user_is_willing_to_stop")),
        payment_methods=_pipe_set(profile_row.get("payment_methods_user_will_consider")),
        max_installment_months=profile_row.get("max_installment_months"),
        priorities=list(_pipe_set(profile_row.get("financial_priorities"))),
    )

    ev = user_events.copy()
    ev["event_date"] = pd.to_datetime(ev["event_date"]).dt.date
    ev["settlement_date"] = pd.to_datetime(ev["settlement_date"]).dt.date
    hist = ev[(ev.status == "settled") & (ev.event_date < request_date)].sort_values("event_date")

    # most-recent event per category (used as the reference id + flexibility
    # source for spending-change candidates, exactly mirroring how the
    # dataset's own solved examples reference the latest occurrence).
    for cat, g in hist.groupby("category"):
        model.representative_event[cat] = g.sort_values("event_date").iloc[-1].to_dict()

    _detect_recurring_bills(model, hist)
    _detect_salary(model, hist, ev, request_date)
    _apply_message_evidence(model, user_messages, request_date)
    _collect_explicit_future_and_reserved(model, ev, request_date)

    return model


def _detect_recurring_bills(model: UserFinancialModel, hist: pd.DataFrame):
    debit = hist[hist.direction == "debit"]

    # (a) description-level fixed recurring bills: rent, subscriptions, debt,
    #     insurance, housing/education/gym/family_support recurring charges.
    for (cat, desc), g in debit.groupby(["category", "description"]):
        if cat not in FIXED_BILL_CATEGORIES:
            continue
        g = g.sort_values("event_date")
        if len(g) < 2:
            continue
        dates = list(g["event_date"])
        intervals = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        avg_interval = statistics.mean(intervals)
        if not (24 <= avg_interval <= 36):
            continue
        amounts = list(g["amount"].astype(float))
        cv = _cv(amounts)
        last = g.iloc[-1]
        # conservative amount: stable bills use the latest value; a bill
        # with some observed variance is projected at the recent max so we
        # never under-reserve for it.
        amount = float(last["amount"]) if cv <= 0.05 else max(amounts[-3:])
        model.recurring_bills[cat] = RecurringItem(
            category=cat, description=desc, amount=amount, currency=last["currency"],
            anchor_date=last["event_date"], frequency_days=round(avg_interval),
            flexibility=last["flexibility"] if isinstance(last["flexibility"], str) else "fixed",
            minimum_allowed_amount=(float(last["minimum_allowed_amount"])
                                     if pd.notna(last["minimum_allowed_amount"]) else None),
            representative_event_id=last["event_id"], direction="debit",
        )

    # (b) category-level variable essential spending (groceries, dining,
    #     transport, shopping, entertainment, healthcare, work_expense):
    #     many small transactions a month with different descriptions.
    for cat, g in debit.groupby("category"):
        if cat not in VARIABLE_CATEGORIES:
            continue
        g = g.sort_values("event_date")
        g["month"] = g["event_date"].apply(lambda d: (d.year, d.month))
        monthly_totals = g.groupby("month")["amount"].sum().astype(float).tolist()
        if not monthly_totals:
            continue
        # conservative: use the higher of the last up-to-3 months so we
        # never understate essential variable spending.
        recent = monthly_totals[-3:]
        conservative_monthly = max(recent)
        last = g.iloc[-1]
        model.variable_bills[cat] = RecurringItem(
            category=cat, description=f"Average monthly {cat}", amount=conservative_monthly,
            currency=last["currency"], anchor_date=last["event_date"], frequency_days=30,
            flexibility=last["flexibility"] if isinstance(last["flexibility"], str) else "fixed",
            minimum_allowed_amount=(float(last["minimum_allowed_amount"])
                                     if pd.notna(last["minimum_allowed_amount"]) else None),
            representative_event_id=last["event_id"], direction="debit",
        )


def _detect_salary(model: UserFinancialModel, hist: pd.DataFrame, ev: pd.DataFrame, request_date: date):
    salary_hist = hist[(hist.event_type == "income") & (hist.category == "salary")].sort_values("event_date")
    avg_interval = 30

    if len(salary_hist) == 1:
        # A single settled payslip (e.g. a brand-new job) is still evidence
        # of an ongoing monthly salary -- there just isn't enough history
        # yet to measure variance, so it isn't penalised for that.
        last = salary_hist.iloc[-1]
        model.salary = RecurringItem(
            category="salary", description="Recurring salary", amount=float(last["amount"]),
            currency=last["currency"], anchor_date=last["event_date"], frequency_days=30,
            flexibility="fixed", minimum_allowed_amount=None,
            representative_event_id=last["event_id"], direction="credit",
        )
    elif len(salary_hist) >= 2:
        # Prefer detecting a stable repeated base amount (regular payroll
        # that occasionally comes with a one-off bonus/arrears/duplicate
        # payslip alongside it) -- this is far less sensitive to a single
        # unusual month than averaging every credit together.
        amounts = salary_hist["amount"].astype(float)
        counts = amounts.value_counts()
        mode_amount, mode_count = counts.index[0], counts.iloc[0]
        last = salary_hist.iloc[-1]
        if mode_count >= 2:
            modal_rows = salary_hist[amounts == mode_amount].sort_values("event_date")
            m_dates = list(modal_rows["event_date"])
            interval = 30
            if len(m_dates) >= 2:
                gaps = [(m_dates[i + 1] - m_dates[i]).days for i in range(len(m_dates) - 1)]
                avg_gap = statistics.mean(gaps)
                if 20 <= avg_gap <= 40:
                    interval = round(avg_gap)
            anchor_row = modal_rows.iloc[-1]
            model.salary = RecurringItem(
                category="salary", description="Recurring salary", amount=float(mode_amount),
                currency=anchor_row["currency"], anchor_date=anchor_row["event_date"],
                frequency_days=interval, flexibility="fixed", minimum_allowed_amount=None,
                representative_event_id=anchor_row["event_id"], direction="credit",
            )
        else:
            # No repeated exact amount (e.g. freelance/gig income): fall
            # back to a monthly-total aggregate, which smooths out an
            # irregular payment cadence within a month.
            h = salary_hist.copy()
            h["month"] = h["event_date"].apply(lambda d: (d.year, d.month))
            monthly_totals = h.groupby("month")["amount"].sum().astype(float).tolist()
            cv = _cv(monthly_totals)
            if len(monthly_totals) == 1 or cv <= 0.30:
                conservative_amount = min(monthly_totals[-3:]) if len(monthly_totals) > 1 else monthly_totals[0]
                model.salary = RecurringItem(
                    category="salary", description="Recurring salary", amount=conservative_amount,
                    currency=last["currency"], anchor_date=last["event_date"], frequency_days=30,
                    flexibility="fixed", minimum_allowed_amount=None,
                    representative_event_id=last["event_id"], direction="credit",
                )

    # An explicit "next confirmed salary" (status=scheduled) is the
    # strongest evidence available -- a scheduled paycheck implies ongoing
    # employment, so it both confirms the very next payment and anchors all
    # subsequent monthly occurrences within the forecast horizon. It
    # supersedes a pure historical-average anchor/amount when present.
    scheduled_income = ev[(ev.status == "scheduled") & (ev.event_type == "income")
                          & (ev.category == "salary") & (ev.event_date >= request_date)]
    if not scheduled_income.empty:
        last = scheduled_income.sort_values("event_date").iloc[-1]
        model.salary = RecurringItem(
            category="salary", description="Recurring salary (confirmed)", amount=float(last["amount"]),
            currency=last["currency"], anchor_date=last["event_date"], frequency_days=avg_interval,
            flexibility="fixed", minimum_allowed_amount=None,
            representative_event_id=last["event_id"], direction="credit",
        )


def _apply_message_evidence(model: UserFinancialModel, user_messages: pd.DataFrame, request_date: date):
    if user_messages is None or user_messages.empty:
        return
    msgs = [classify_message(r) for r in user_messages.to_dict("records")]
    # apply in message_id order (roughly chronological) so the latest
    # instruction wins, per "prefer a newer record from the same source".
    msgs.sort(key=lambda m: m.message_id or "")

    for m in msgs:
        if m.kind in ("employment_ended", "seasonal_contract_ended"):
            model.salary = None
            continue
        if m.kind == "household_ended_remaining" and m.amount is not None:
            base = model.salary
            model.salary = RecurringItem(
                category="salary", description="Recurring salary (adjusted)",
                amount=m.amount, currency=m.currency or (base.currency if base else model.home_currency),
                anchor_date=base.anchor_date if base else request_date,
                frequency_days=base.frequency_days if base else 30,
                flexibility="fixed", minimum_allowed_amount=None,
                representative_event_id=base.representative_event_id if base else "",
                direction="credit",
            )
            continue
        if m.kind in SALARY_KINDS and m.amount is not None:
            base = model.salary
            anchor = base.anchor_date if base else request_date
            if m.date:
                try:
                    anchor = _to_date(m.date)
                except Exception:
                    pass
            model.salary = RecurringItem(
                category="salary", description="Recurring salary (revised)",
                amount=m.amount, currency=m.currency or (base.currency if base else model.home_currency),
                anchor_date=anchor, frequency_days=base.frequency_days if base else 30,
                flexibility="fixed", minimum_allowed_amount=None,
                representative_event_id=base.representative_event_id if base else "",
                direction="credit",
            )
            continue
        if m.kind == "salary_date_change" and m.date and model.salary is not None:
            try:
                model.salary.anchor_date = _to_date(m.date)
            except Exception:
                pass
            continue
        if m.kind == "rent_increase_pct" and m.percent and "rent" in model.recurring_bills:
            model.recurring_bills["rent"].amount *= (1.0 + m.percent / 100.0)
            continue
        if m.kind == "debit_failed_still_outstanding" and m.related_event_id:
            model._pending_reinstate = getattr(model, "_pending_reinstate", [])
            model._pending_reinstate.append(m.related_event_id)
            continue
        # everything in NEVER_TRUST_AS_INCOME, invoice_confirmed (handled via
        # explicit future rows / one-time evidence) and unclassified
        # messages are intentionally not allowed to change the forecast.


def _collect_explicit_future_and_reserved(model: UserFinancialModel, ev: pd.DataFrame, request_date: date):
    future_income = ev[(ev.status == "scheduled") & (ev.event_type == "income")]
    for _, row in future_income.iterrows():
        model.explicit_future.append(FutureFlow(
            date=max(row["event_date"], request_date), amount=float(row["amount"]),
            category="salary", label=row["description"], event_id=row["event_id"],
        ))

    future_debt = ev[(ev.status == "scheduled") & (ev.event_type == "debt_payment")]
    for _, row in future_debt.iterrows():
        model.explicit_future.append(FutureFlow(
            date=max(row["event_date"], request_date), amount=-float(row["amount"]),
            category="debt_repayment", label=row["description"], event_id=row["event_id"],
        ))

    pending_debit = ev[(ev.status == "pending") & (ev.direction == "debit")]
    for _, row in pending_debit.iterrows():
        model.explicit_future.append(FutureFlow(
            date=request_date, amount=-float(row["amount"]), category=row["category"],
            label=f"Reserved pending: {row['description']}", event_id=row["event_id"],
        ))
    # pending credits are intentionally never added (do not count pending
    # credits until they settle).

    reinstate_ids = set(getattr(model, "_pending_reinstate", []))
    if reinstate_ids:
        failed = ev[(ev.status == "failed") & (ev.event_id.isin(reinstate_ids))]
        for _, row in failed.iterrows():
            model.explicit_future.append(FutureFlow(
                date=request_date + timedelta(days=2), amount=-float(row["amount"]),
                category=row["category"], label=f"Re-attempted debit: {row['description']}",
                event_id=row["event_id"],
            ))
