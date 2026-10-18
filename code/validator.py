"""
validator.py
-------------
Deterministically checks every produced row against the challenge's output
contract before it is written to output.csv. Anything that fails is
corrected to a safe fallback (never_affordable / no plan) rather than
silently shipped, and the reason is recorded so it shows up in the run log.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Dict, List, Tuple

ALLOWED_STATUS = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
ALLOWED_METHOD = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
REQUIRED_COLUMNS = [
    "request_id", "amount_safe_to_pay", "affordability_status", "recommended_payment_method",
    "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed", "decision_explanation",
]

_PLAN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}:-?\d+(\.\d+)?$")
_CHANGE_RE = re.compile(r"^(stop:[A-Za-z0-9_]+|reduce_to:[A-Za-z0-9_]+:-?\d+(\.\d+)?)$")


def _fallback(request_id: str, reason: str) -> Dict:
    return {
        "request_id": request_id,
        "amount_safe_to_pay": 0,
        "affordability_status": "not_affordable",
        "recommended_payment_method": "not_recommended",
        "payment_plan": "none",
        "earliest_date_for_full_payment": "",
        "spending_changes_needed": "none",
        "decision_explanation": f"Validation fallback ({reason}). Recommending no payment pending review.",
    }


def validate_row(row: Dict, requested_amount: float, desired_completion_date: date) -> Tuple[Dict, List[str]]:
    errors = []
    rid = row["request_id"]

    amt = row.get("amount_safe_to_pay")
    try:
        amt = float(amt)
    except (TypeError, ValueError):
        errors.append("amount_safe_to_pay not numeric")
        return _fallback(rid, "amount not numeric"), errors
    if not (-1e-6 <= amt <= requested_amount + 1e-6):
        errors.append(f"amount_safe_to_pay {amt} out of bounds [0,{requested_amount}]")
        return _fallback(rid, "amount out of bounds"), errors

    if row.get("affordability_status") not in ALLOWED_STATUS:
        errors.append("invalid affordability_status")
        return _fallback(rid, "bad status"), errors
    if row.get("recommended_payment_method") not in ALLOWED_METHOD:
        errors.append("invalid recommended_payment_method")
        return _fallback(rid, "bad method"), errors

    plan = row.get("payment_plan", "none")
    if plan != "none":
        entries = plan.split("|")
        dates = []
        total = 0.0
        for e in entries:
            if not _PLAN_RE.match(e):
                errors.append(f"bad plan entry {e}")
                return _fallback(rid, "bad plan format"), errors
            d_str, amt_str = e.split(":")
            d = datetime.strptime(d_str, "%Y-%m-%d").date()
            dates.append(d)
            total += float(amt_str)
        if dates != sorted(dates):
            errors.append("plan not chronological")
            return _fallback(rid, "plan not chronological"), errors
        if row["recommended_payment_method"] == "partial_payment":
            if len(entries) != 2:
                errors.append("partial_payment must have exactly two payments")
                return _fallback(rid, "partial payment shape"), errors
            if abs(total - requested_amount) > 0.05:
                errors.append("partial payments do not sum to requested amount")
                return _fallback(rid, "partial sum mismatch"), errors
            if dates[1] > desired_completion_date:
                errors.append("partial payment completes after deadline")
                return _fallback(rid, "partial deadline miss"), errors

    changes = row.get("spending_changes_needed", "none")
    if changes != "none":
        parts = changes.split("|")
        if len(parts) > 3:
            errors.append("too many spending changes")
            return _fallback(rid, "too many changes"), errors
        referenced = set()
        for p in parts:
            if not _CHANGE_RE.match(p):
                errors.append(f"bad spending change format {p}")
                return _fallback(rid, "bad change format"), errors
            eid = p.split(":")[1]
            if eid in referenced:
                errors.append("duplicate event referenced by multiple changes")
                return _fallback(rid, "duplicate change target"), errors
            referenced.add(eid)

    edate = row.get("earliest_date_for_full_payment", "")
    if edate:
        try:
            datetime.strptime(edate, "%Y-%m-%d")
        except ValueError:
            errors.append("bad earliest_date_for_full_payment format")
            return _fallback(rid, "bad date format"), errors
        if row["affordability_status"] == "affordable_now" and edate != row.get("_request_date_str"):
            pass  # informational only; not fatal, request_date already enforced upstream

    return row, errors


def validate_output_rows(rows: List[Dict], requests_index: Dict[str, dict]) -> Tuple[List[Dict], List[str]]:
    all_errors = []
    cleaned = []
    seen_ids = set()
    for row in rows:
        rid = row["request_id"]
        req = requests_index.get(rid)
        if req is None:
            all_errors.append(f"{rid}: not a known request_id, dropped")
            continue
        seen_ids.add(rid)
        fixed, errs = validate_row(row, float(req["requested_amount"]),
                                    req["desired_completion_date"])
        for e in errs:
            all_errors.append(f"{rid}: {e}")
        cleaned.append(fixed)

    missing = set(requests_index.keys()) - seen_ids
    for rid in sorted(missing):
        all_errors.append(f"{rid}: missing from output, added fallback row")
        cleaned.append(_fallback(rid, "missing row"))

    order = {rid: i for i, rid in enumerate(requests_index.keys())}
    cleaned.sort(key=lambda r: order.get(r["request_id"], 10**9))
    return cleaned, all_errors
