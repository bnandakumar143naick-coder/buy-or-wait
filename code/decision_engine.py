"""
decision_engine.py
-------------------
Ranks the candidate plans built by payment_planner.py using the challenge's
six tie-break rules, decides the final affordability_status /
recommended_payment_method, and writes a short grounded explanation.

Ranking (best first):
  1. Completes the full request by desired_completion_date
  2. Requires no spending changes
  3. Minimizes the total amount paid
  4. Starts payment earlier
  5. Uses fewer payments
  6. Lowest payment_option_id (final tie-breaker)
"""
from __future__ import annotations

import re
from datetime import date
from typing import Dict, List, Optional

from forecast_engine import amount_safe_today
from payment_planner import Candidate

CCY_FMT = {
    "IDR": 0, "ZAR": 2, "EUR": 2, "USD": 2, "INR": 2,
}


def _fmt_amount(amount: float, currency: str) -> str:
    decimals = CCY_FMT.get(currency, 2)
    val = round(amount, decimals)
    if decimals == 0:
        return f"{currency} {val:,.0f}"
    return f"{currency} {val:,.2f}"


def _fmt_date(d: date) -> str:
    return d.strftime("%-d %B %Y") if hasattr(d, "strftime") else str(d)


def _option_sort_key(pid: Optional[str]):
    if not pid:
        return (1, 0)
    m = re.search(r"(\d+)$", pid)
    return (0, int(m.group(1))) if m else (0, pid)


def _rank_key(c: Candidate):
    return (
        0 if c.meets_deadline else 1,
        0 if not c.uses_changes else 1,
        round(c.total_paid, 6),
        c.start_date,
        c.num_payments,
        _option_sort_key(c.payment_option_id),
    )


def request_row_deadline_str(plan_ctx: Dict) -> str:
    return plan_ctx["desired_completion_date"].isoformat()


def choose_best(candidates: List[Candidate]) -> Optional[Candidate]:
    if not candidates:
        return None
    return sorted(candidates, key=_rank_key)[0]


def _format_number(val: float) -> str:
    s = f"{round(val, 2):.2f}"
    if s.endswith(".00"):
        s = s[:-3]
    return s


def _format_plan(payments, currency: str) -> str:
    if not payments:
        return "none"
    parts = []
    for d, amt in sorted(payments, key=lambda p: p[0]):
        parts.append(f"{d.isoformat()}:{_format_number(amt)}")
    return "|".join(parts)


def _describe_changes(changes_detail: List[dict], currency: str) -> str:
    labels = []
    for c in changes_detail or []:
        desc = c.get("description", c.get("category", "expense"))
        if c["type"] == "stop":
            labels.append(f"stop the {desc.lower()}")
        else:
            labels.append(f"reduce the {desc.lower()} to {_fmt_amount(c['new_amount'], currency)}")
    if not labels:
        return "Adjust flexible spending"
    text = " and ".join(labels)
    return text[0].upper() + text[1:]


def assemble_output(model, plan_ctx: Dict, request_row: dict) -> Dict:
    currency = model.home_currency
    requested_amount = plan_ctx["requested_amount"]
    request_date = plan_ctx["request_date"]
    amount_safe0 = plan_ctx["amount_safe0"]
    earliest0 = plan_ctx["earliest0"]
    best = choose_best(plan_ctx["candidates"])

    earliest_str = earliest0.isoformat() if earliest0 is not None else ""

    if best is None:
        safe_amt = round(max(0.0, min(amount_safe0, requested_amount)), 2)
        deadline_str = request_row_deadline_str(plan_ctx)
        if safe_amt > 0:
            explanation = (
                f"Do not proceed with the {_fmt_amount(requested_amount, currency)} request. Although "
                f"{_fmt_amount(safe_amt, currency)} is available today, the full amount cannot be "
                "completed safely within 90 days."
            )
        else:
            explanation = (
                f"Do not make this payment by {deadline_str}. None of the available options keeps the "
                f"{_fmt_amount(model.min_balance, currency)} minimum protected."
            )
        return {
            "amount_safe_to_pay": safe_amt,
            "affordability_status": "not_affordable",
            "recommended_payment_method": "not_recommended",
            "payment_plan": "none",
            "earliest_date_for_full_payment": earliest_str,
            "spending_changes_needed": "none",
            "decision_explanation": explanation,
        }

    if best.method == "full_payment" and not best.uses_changes and best.start_date == request_date:
        status = "affordable_now"
    elif best.method == "wait":
        status = "affordable_later"
    else:
        status = "affordable_with_plan"

    if best.uses_changes:
        effective_safe = requested_amount
    else:
        effective_safe = amount_safe0
    amount_safe_to_pay = round(max(0.0, min(effective_safe, requested_amount)), 2)

    plan_str = _format_plan(best.payments, currency)
    changes_str = "|".join(best.changes) if best.changes else "none"

    if status == "affordable_now":
        explanation = (
            f"Pay {_fmt_amount(requested_amount, currency)} today. This leaves at least "
            f"{_fmt_amount(model.min_balance, currency)} available over the next 90 days."
        )
    elif best.method == "full_payment" and best.uses_changes:
        explanation = (
            f"{_describe_changes(best.changes_detail, currency)}, then pay "
            f"{_fmt_amount(requested_amount, currency)} today. This leaves at least "
            f"{_fmt_amount(model.min_balance, currency)} available."
        )
    elif best.method == "partial_payment":
        first_amt, second_amt = best.payments[0][1], best.payments[1][1]
        explanation = (
            f"Pay {_fmt_amount(first_amt, currency)} today and the remaining "
            f"{_fmt_amount(second_amt, currency)} on {best.payments[1][0].isoformat()}. "
            f"This completes the full request and keeps the {_fmt_amount(model.min_balance, currency)} "
            "minimum protected."
        )
    elif best.method == "installments":
        per = best.payments[0][1]
        explanation = (
            f"Use {best.num_payments} installments of {_fmt_amount(per, currency)}, starting "
            f"{best.start_date.isoformat()}. This leaves at least {_fmt_amount(model.min_balance, currency)} available."
        )
    elif best.method == "wait":
        explanation = (
            f"Pay {_fmt_amount(requested_amount, currency)} in full on {best.start_date.isoformat()}. "
            f"Paying earlier would take the balance below the {_fmt_amount(model.min_balance, currency)} minimum."
        )
    else:
        explanation = "Recommended plan computed from the 90-day forecast."

    return {
        "amount_safe_to_pay": amount_safe_to_pay,
        "affordability_status": status,
        "recommended_payment_method": best.method,
        "payment_plan": plan_str,
        "earliest_date_for_full_payment": earliest_str,
        "spending_changes_needed": changes_str,
        "decision_explanation": explanation,
    }

# Public alias -- other modules (e.g. ai/chat_assistant.py) reuse this exact
# formatter so amounts are worded identically everywhere in the system.
fmt_amount = _fmt_amount
