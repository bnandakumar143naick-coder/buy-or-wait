"""
app.py
-------
Flask backend for the "Buy or Wait?" financial decision agent.

Required endpoints (per problem_statement.md / AGENTS.md)
-----------------------------------------------------------
POST /api/affordability
    body: {"request_id": "request_26"}
    -> the decision engine's output for that request (the same 8 fields
       written to output.csv, plus request_id).

POST /api/chat
    body: {"message": "...", "request_id"?: "...", "user_id"?: "...",
           "session_id"?: "..."}
    -> a natural-language answer, computed by running the SAME decision
       engine (ai/chat_assistant.py never duplicates or overrides the
       financial logic). Response: {answer, recommendation, facts,
       payment_plan, reason, request_id, user_id, intent}.

Dashboard + its supporting endpoints
-------------------------------------
GET  /                     the dashboard (static/index.html)
GET  /api/requests         list every request, for the browser panel
GET  /api/forecast/<id>    90-day balance curve, for the chart
POST /api/run-batch        re-run the full 250-request pipeline -> output.csv
GET  /api/run-batch/status poll batch progress
GET  /api/output-summary   aggregate stats from the last generated output.csv

All of these read from the same engine as main.py / /api/affordability --
nothing here recomputes a financial figure independently.
"""
from __future__ import annotations

import os
import sys
import threading

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd  # noqa: E402
from flask import Flask, jsonify, request, send_from_directory  # noqa: E402

from ai.chat_assistant import handle_chat  # noqa: E402
from data_loader import REPO_ROOT, load_dataset  # noqa: E402
from financial_state import build_user_model  # noqa: E402
from fx import RateTable  # noqa: E402
from main import OUTPUT_PATH, run_all, run_one, write_output_csv  # noqa: E402
from payment_planner import build_candidates  # noqa: E402
from validator import validate_output_rows  # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")

_state = {"ds": None, "rates": None, "batch_running": False, "batch_result": None}
_chat_sessions: dict = {}  # session_id -> {"request_id":..., "user_id":...}, in-memory only


def _get_ds():
    if _state["ds"] is None:
        _state["ds"] = load_dataset()
        _state["rates"] = RateTable(_state["ds"].rates)
    return _state["ds"], _state["rates"]


# --------------------------------------------------------------------- page
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


# --------------------------------------------------------------- required
@app.route("/api/affordability", methods=["POST"])
def api_affordability():
    payload = request.get_json(force=True, silent=True) or {}
    request_id = payload.get("request_id")
    if not request_id:
        return jsonify({"error": "request_id is required"}), 400

    ds, rates = _get_ds()
    row = ds.request_row(request_id)
    if row is None:
        return jsonify({"error": f"unknown request_id {request_id}"}), 404

    try:
        result = run_one(ds, rates, row)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(result)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(force=True, silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400

    session_id = payload.get("session_id")
    session = _chat_sessions.get(session_id, {}) if session_id else {}

    ds, rates = _get_ds()
    try:
        result = handle_chat(
            message, ds, rates,
            request_id=payload.get("request_id"),
            user_id=payload.get("user_id"),
            session=session,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    if session_id and result.get("request_id"):
        _chat_sessions[session_id] = {"request_id": result["request_id"], "user_id": result.get("user_id")}

    return jsonify(result)


# ------------------------------------------------------------- dashboard
@app.route("/api/requests")
def api_requests():
    ds, _ = _get_ds()
    profiles = ds.profiles_by_user
    out = []
    for r in ds.requests.to_dict("records"):
        profile = profiles.get(r["user_id"], {})
        out.append({
            "request_id": r["request_id"], "user_id": r["user_id"], "request_type": r["request_type"],
            "requested_amount": r["requested_amount"], "currency": profile.get("home_currency"),
            "request_date": r["request_date"], "desired_completion_date": r["desired_completion_date"],
            "allows_partial_payment": bool(r["allows_partial_payment"]), "request_text": r["request_text"],
        })
    return jsonify(out)


@app.route("/api/forecast/<request_id>")
def api_forecast(request_id):
    ds, rates = _get_ds()
    row = ds.request_row(request_id)
    if row is None:
        return jsonify({"error": "not found"}), 404

    user_id = row["user_id"]
    profile = ds.profiles_by_user[user_id]
    user_events = ds.events_by_user.get(user_id, pd.DataFrame(columns=ds.events.columns))
    user_messages = ds.messages_by_user.get(user_id, pd.DataFrame(columns=ds.messages.columns))
    request_date = pd.to_datetime(row["request_date"]).date()

    model = build_user_model(user_id, profile, user_events, user_messages, request_date)
    payment_options = ds.options_by_request.get(request_id)
    ctx = build_candidates(model, row, rates, payment_options)
    baseline = ctx["baseline"]

    return jsonify({
        "request_date": request_date.isoformat(), "currency": model.home_currency,
        "min_balance": model.min_balance, "daily_balance": baseline.daily_balance,
        "amount_safe0": ctx["amount_safe0"],
        "earliest0": ctx["earliest0"].isoformat() if ctx["earliest0"] else None,
        "requested_amount": ctx["requested_amount"],
    })


def _run_batch_async():
    ds, rates = _get_ds()
    requests_index = {row["request_id"]: {
        "requested_amount": row["requested_amount"],
        "desired_completion_date": pd.to_datetime(row["desired_completion_date"]).date(),
    } for row in ds.requests.to_dict("records")}
    rows = run_all(ds, rates)
    cleaned_rows, errors = validate_output_rows(rows, requests_index)
    write_output_csv(cleaned_rows, OUTPUT_PATH)
    _state["batch_result"] = {"rows": len(cleaned_rows), "errors": errors}
    _state["batch_running"] = False


@app.route("/api/run-batch", methods=["POST"])
def api_run_batch():
    if _state["batch_running"]:
        return jsonify({"status": "already running"}), 202
    _state["batch_running"] = True
    _state["batch_result"] = None
    threading.Thread(target=_run_batch_async, daemon=True).start()
    return jsonify({"status": "started"}), 202


@app.route("/api/run-batch/status")
def api_run_batch_status():
    return jsonify({"running": _state["batch_running"], "result": _state["batch_result"]})


@app.route("/api/output-summary")
def api_output_summary():
    if not os.path.exists(OUTPUT_PATH):
        return jsonify({"error": "output.csv has not been generated yet"}), 404
    out = pd.read_csv(OUTPUT_PATH)
    return jsonify({
        "row_count": len(out),
        "affordability_status": out["affordability_status"].value_counts().to_dict(),
        "recommended_payment_method": out["recommended_payment_method"].value_counts().to_dict(),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
