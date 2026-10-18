"""
main.py
--------
Entry point for the "Buy or Wait?" financial decision agent.

Usage:
    python3 code/main.py                  # run the full dataset, write output.csv
    python3 code/main.py --request request_26   # print the decision for one request
    python3 code/main.py --sample         # run against dataset/sample_requests.csv
                                             and report simple match stats

The engine is fully deterministic: given the same dataset/ contents it
always produces the same output.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import date

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd  # noqa: E402

from data_loader import REPO_ROOT, load_dataset  # noqa: E402
from decision_engine import assemble_output  # noqa: E402
from financial_state import build_user_model  # noqa: E402
from fx import RateTable  # noqa: E402
from payment_planner import build_candidates  # noqa: E402
from validator import REQUIRED_COLUMNS, validate_output_rows  # noqa: E402

OUTPUT_PATH = os.path.join(REPO_ROOT, "output.csv")


def run_one(ds, rates, request_row: dict) -> dict:
    user_id = request_row["user_id"]
    profile = ds.profiles_by_user.get(user_id)
    if profile is None:
        raise ValueError(f"No financial profile for {user_id}")

    user_events = ds.events_by_user.get(user_id, pd.DataFrame(columns=ds.events.columns))
    user_messages = ds.messages_by_user.get(user_id, pd.DataFrame(columns=ds.messages.columns))
    request_date = pd.to_datetime(request_row["request_date"]).date()

    model = build_user_model(user_id, profile, user_events, user_messages, request_date)
    payment_options = ds.options_by_request.get(request_row["request_id"])

    plan_ctx = build_candidates(model, request_row, rates, payment_options)
    result = assemble_output(model, plan_ctx, request_row)
    result["request_id"] = request_row["request_id"]
    return result


def run_all(ds, rates, request_ids=None) -> list:
    results = []
    requests = ds.requests
    if request_ids:
        requests = requests[requests.request_id.isin(request_ids)]
    for row in requests.to_dict("records"):
        try:
            results.append(run_one(ds, rates, row))
        except Exception as exc:  # keep the batch going; validator will fill a safe fallback
            results.append({
                "request_id": row["request_id"],
                "amount_safe_to_pay": 0,
                "affordability_status": "not_affordable",
                "recommended_payment_method": "not_recommended",
                "payment_plan": "none",
                "earliest_date_for_full_payment": "",
                "spending_changes_needed": "none",
                "decision_explanation": f"Engine error, recommending no payment pending review: {exc}",
            })
    return results


def write_output_csv(rows: list, path: str = OUTPUT_PATH):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in REQUIRED_COLUMNS})


def main():
    parser = argparse.ArgumentParser(description="Buy or Wait? financial decision agent")
    parser.add_argument("--request", help="Run a single request_id and print the result as JSON")
    parser.add_argument("--sample", action="store_true", help="Run against sample_requests.csv instead")
    parser.add_argument("--out", default=OUTPUT_PATH, help="Where to write output.csv")
    args = parser.parse_args()

    ds = load_dataset()
    rates = RateTable(ds.rates)

    if args.request:
        row = ds.request_row(args.request)
        if row is None:
            print(json.dumps({"error": f"unknown request_id {args.request}"}))
            return
        result = run_one(ds, rates, row)
        print(json.dumps(result, indent=2, default=str))
        return

    if args.sample:
        rows = []
        for row in ds.sample_requests.to_dict("records"):
            try:
                rows.append(run_one(ds, rates, row))
            except Exception as exc:
                rows.append({"request_id": row["request_id"], "affordability_status": "error",
                             "recommended_payment_method": "error", "_error": str(exc)})
        sample_by_id = {r["request_id"]: r for r in ds.sample_requests.to_dict("records")}
        matches = {"affordability_status": 0, "recommended_payment_method": 0, "total": 0}
        for r in rows:
            gold = sample_by_id.get(r["request_id"])
            if not gold:
                continue
            matches["total"] += 1
            if gold["affordability_status"] == r["affordability_status"]:
                matches["affordability_status"] += 1
            if gold["recommended_payment_method"] == r["recommended_payment_method"]:
                matches["recommended_payment_method"] += 1
        print(json.dumps(matches, indent=2))
        return

    start = time.time()
    requests_index = {row["request_id"]: {
        "requested_amount": row["requested_amount"],
        "desired_completion_date": pd.to_datetime(row["desired_completion_date"]).date(),
    } for row in ds.requests.to_dict("records")}

    rows = run_all(ds, rates)
    cleaned_rows, errors = validate_output_rows(rows, requests_index)
    write_output_csv(cleaned_rows, args.out)

    elapsed = time.time() - start
    print(f"Wrote {len(cleaned_rows)} rows to {args.out} in {elapsed:.2f}s")
    if errors:
        print(f"{len(errors)} validation notes (auto-corrected):")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
    else:
        print("All rows passed validation with no corrections.")


if __name__ == "__main__":
    main()
