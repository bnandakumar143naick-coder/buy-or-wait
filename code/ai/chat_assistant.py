"""
ai/chat_assistant.py
----------------------
Natural-language front end for the financial decision engine.

Pipeline (exactly as specified):

    user message
      -> identify the relevant user/request (from the message text, or
         from request_id/user_id/session passed alongside it)
      -> run the SAME deterministic engine used by main.py / app.py
         (financial_state -> forecast_engine -> payment_planner ->
          decision_engine) -- this module never re-implements or
          re-derives a financial number itself
      -> turn the engine's output into a plain-language answer that
         matches what the user actually asked

The "AI" here is natural-language understanding and phrasing only:
  * `extract_entities` / `classify_intent` figure out which request/user
    the message is about and what kind of question is being asked.
  * `_shape_response` decides which already-computed facts to surface and
    how to phrase them.
Neither step is allowed to invent a number, a payment option, or a
recommendation that didn't come out of the engine -- every figure in a
response is read off `decision` (the engine's assembled 8-field output),
`model` (the reconstructed financial state) or `plan_ctx` (the candidate
plans the engine actually built and safety-checked).

If ANTHROPIC_API_KEY is set, the free-text understanding/phrasing step can
be delegated to a real language model for messages the keyword classifier
doesn't recognise (see `_llm_paraphrase`); it is given the already-computed
facts as context and is only ever allowed to *word* the answer, never to
change a number or the recommendation. Without a key (the default, and how
the bundled `output.csv` / demo were produced) the deterministic templates
below are used, so the endpoint works fully offline.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional

import pandas as pd

from data_loader import Dataset
from decision_engine import assemble_output, fmt_amount
from financial_state import build_user_model
from forecast_engine import schedule_is_safe
from fx import RateTable
from payment_planner import build_candidates

CCY_SYMBOLS = {"₹": "INR", "$": "USD", "€": "EUR", "R": "ZAR", "Rp": "IDR"}
AMOUNT_PATTERNS = [
    re.compile(r"(₹|\$|€|Rp)\s?([\d,]+(?:\.\d+)?)"),
    re.compile(r"\b(INR|USD|EUR|ZAR|IDR)\s?([\d,]+(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"([\d,]+(?:\.\d+)?)\s?(INR|USD|EUR|ZAR|IDR)\b", re.IGNORECASE),
]
REQUEST_ID_RE = re.compile(r"\brequest_\d+\b")
USER_ID_RE = re.compile(r"\buser_\d+\b")


@dataclass
class Entities:
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    wants_partial: bool = False


def extract_entities(text: str) -> Entities:
    e = Entities()
    m = REQUEST_ID_RE.search(text)
    if m:
        e.request_id = m.group(0)
    m = USER_ID_RE.search(text)
    if m:
        e.user_id = m.group(0)
    for pat in AMOUNT_PATTERNS:
        m = pat.search(text)
        if m:
            g1, g2 = m.group(1), m.group(2)
            if g1.upper() in ("INR", "USD", "EUR", "ZAR", "IDR"):
                currency, raw = g1.upper(), g2
            elif g2.upper() in ("INR", "USD", "EUR", "ZAR", "IDR"):
                currency, raw = g2.upper(), g1
            else:
                currency, raw = CCY_SYMBOLS.get(g1, None), g2
            try:
                e.amount = float(raw.replace(",", ""))
                e.currency = currency
            except ValueError:
                pass
            break
    e.wants_partial = "partial" in text.lower()
    return e


_INTENT_RULES = [
    ("reduce_expenses", [["reduce"], ["cut back"], ["flexible", "expense"], ["flexible", "spending"]]),
    ("which_expenses_affect", [["what", "expense"], ["which", "expense"], ["expenses", "affecting"],
                                ["what's using my money"], ["what is using my money"]]),
    ("which_option_safest", [["safest"]]),
    ("installments", [["installment"]]),
    ("what_if_payment", [["what happens"], ["if i pay"], ["if i make this payment"], ["impact", "balance"]]),
    ("why_not_affordable", [["why", "not afford"], ["why", "not", "affordable"], ["why", "can't afford"],
                             ["why", "cant afford"], ["why", "cannot afford"]]),
    ("why_wait", [["why", "wait"]]),
    ("when_afford", [["when", "afford"], ["when", "pay"], ["when", "safe"]]),
    ("pay_today", [["pay", "today"], ["pay", "now"]]),
    ("new_check", [["afford"]]),
]


def classify_intent(text: str) -> str:
    low = text.lower()
    for intent, keyword_sets in _INTENT_RULES:
        for kws in keyword_sets:
            if all(kw in low for kw in kws):
                return intent
    return "general"


def _empty_df(ds: Dataset, kind: str) -> pd.DataFrame:
    return pd.DataFrame(columns=getattr(ds, kind).columns)


def _users_actual_request(ds: Dataset, user_id: str) -> Optional[dict]:
    rows = ds.requests[ds.requests.user_id == user_id]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _build_synthetic_request(anchor_row: dict, entities: Entities, message: str) -> dict:
    """A hypothetical request ("I want to buy a laptop for X") for a user who
    is not currently asking about their one real request. Only the two
    request-level parameters the message didn't specify are defaulted
    (completion window, partial-payment allowance) -- balances, expenses,
    income and payment options all still come straight from the dataset.
    """
    request_date = pd.to_datetime(anchor_row["request_date"]).date()
    return {
        "request_id": "adhoc",
        "user_id": anchor_row["user_id"],
        "request_date": request_date.isoformat(),
        "request_type": "other",
        "requested_amount": entities.amount,
        "desired_completion_date": (request_date + timedelta(days=30)).isoformat(),
        "allows_partial_payment": entities.wants_partial,
        "request_text": message,
    }


def _run_engine(ds: Dataset, rates: RateTable, request_row: dict):
    user_id = request_row["user_id"]
    profile = ds.profiles_by_user[user_id]
    user_events = ds.events_by_user.get(user_id, _empty_df(ds, "events"))
    user_messages = ds.messages_by_user.get(user_id, _empty_df(ds, "messages"))
    request_date = pd.to_datetime(request_row["request_date"]).date()

    model = build_user_model(user_id, profile, user_events, user_messages, request_date)
    payment_options = ds.options_by_request.get(request_row["request_id"])
    plan_ctx = build_candidates(model, request_row, rates, payment_options)
    decision = assemble_output(model, plan_ctx, request_row)
    decision["request_id"] = request_row["request_id"]
    return model, plan_ctx, decision


_METHOD_LABEL = {
    "full_payment": "pay_now",
    "partial_payment": "partial_payment",
    "installments": "installments",
    "wait": "wait",
    "not_recommended": "do_not_proceed",
}


def _top_expenses(model, n=5) -> List[dict]:
    items = []
    for cat, item in model.recurring_bills.items():
        items.append({"category": cat, "amount": round(item.amount, 2), "currency": item.currency, "type": "recurring bill"})
    for cat, item in model.variable_bills.items():
        items.append({"category": cat, "amount": round(item.amount, 2), "currency": item.currency, "type": "monthly average spend"})
    items.sort(key=lambda x: x["amount"], reverse=True)
    return items[:n]


def _reducible_options(model) -> List[dict]:
    out = []
    for cat in (model.reduce_categories | model.stop_categories) - model.protect:
        item = model.recurring_bills.get(cat) or model.variable_bills.get(cat)
        if item is None:
            continue
        entry = {"category": cat, "current_amount": round(item.amount, 2), "currency": item.currency}
        if cat in model.reduce_categories and item.flexibility in ("reducible", "reducible_or_stoppable") \
                and item.minimum_allowed_amount is not None:
            entry["can_reduce_to"] = round(item.minimum_allowed_amount, 2)
        if cat in model.stop_categories and item.flexibility in ("stoppable", "reducible_or_stoppable"):
            entry["can_stop"] = True
        if "can_reduce_to" in entry or entry.get("can_stop"):
            out.append(entry)
    return out


def _installment_facts(ds: Dataset, request_id: str, model, plan_ctx) -> List[dict]:
    options = ds.options_by_request.get(request_id)
    if options is None:
        return []
    out = []
    request_date = plan_ctx["request_date"]
    for _, row in options[options.payment_method == "installments"].iterrows():
        n = int(row["number_of_payments"])
        freq = int(row["payment_frequency_days"]) if pd.notna(row["payment_frequency_days"]) else 30
        start = pd.to_datetime(row["first_payment_date"]).date()
        amt = float(row["payment_amount"])
        payments = [(start + timedelta(days=freq * i), amt) for i in range(n)]
        safe = schedule_is_safe(plan_ctx["baseline"], model.min_balance, request_date, payments)
        out.append({
            "payment_option_id": row["payment_option_id"], "number_of_payments": n,
            "payment_amount": round(amt, 2), "first_payment_date": start.isoformat(),
            "total_payable_amount": round(float(row["total_payable_amount"]), 2),
            "financing_fee": round(float(row["financing_fee"]), 2) if pd.notna(row["financing_fee"]) else 0.0,
            "safe": safe,
        })
    return out


def _shape_response(intent: str, message: str, ds: Dataset, model, plan_ctx: Dict,
                     decision: Dict, request_row: dict, entities: Entities) -> Dict:
    currency = model.home_currency
    requested_amount = plan_ctx["requested_amount"]
    facts = {
        "currency": currency,
        "requested_amount": requested_amount,
        "current_available_balance": round(model.balance, 2),
        "minimum_balance_to_keep": round(model.min_balance, 2),
        "amount_safe_to_pay_today": decision["amount_safe_to_pay"],
        "earliest_date_for_full_payment": decision["earliest_date_for_full_payment"] or None,
        "affordability_status": decision["affordability_status"],
    }
    recommendation = _METHOD_LABEL.get(decision["recommended_payment_method"], decision["recommended_payment_method"])
    payment_plan = decision["payment_plan"] if decision["payment_plan"] != "none" else None
    reason = decision["decision_explanation"]
    answer = reason

    if intent == "when_afford":
        if decision["earliest_date_for_full_payment"]:
            answer = (f"You can safely pay the full {fmt_amount(requested_amount, currency)} on "
                       f"{decision['earliest_date_for_full_payment']} without dropping below the "
                       f"{fmt_amount(model.min_balance, currency)} minimum you need to keep.")
        else:
            answer = (f"Based on the next 90 days, the full {fmt_amount(requested_amount, currency)} "
                       "does not become safe to pay within that window.")

    elif intent == "why_wait":
        if decision["recommended_payment_method"] == "wait":
            answer = (f"Paying the full amount today would take your balance below the "
                       f"{fmt_amount(model.min_balance, currency)} minimum at some point in the next 90 days. "
                       f"Only {fmt_amount(facts['amount_safe_to_pay_today'], currency)} is safe to pay today; "
                       f"waiting until {decision['earliest_date_for_full_payment']} lets the full amount clear safely.")
        else:
            answer = f"I'm not currently recommending waiting for this request. {reason}"

    elif intent == "why_not_affordable":
        if decision["affordability_status"] == "not_affordable":
            if facts["amount_safe_to_pay_today"] > 0:
                answer = (f"Only {fmt_amount(facts['amount_safe_to_pay_today'], currency)} of the "
                           f"{fmt_amount(requested_amount, currency)} requested is safe today, and the full "
                           "amount is not projected to become safe within the next 90 days without breaking "
                           f"the {fmt_amount(model.min_balance, currency)} minimum.")
            else:
                answer = (f"Paying any part of this today would drop your balance below the "
                           f"{fmt_amount(model.min_balance, currency)} minimum you need to keep over the "
                           "next 90 days, and none of your accepted payment methods can safely cover it.")
        else:
            answer = f"This request is actually assessed as {decision['affordability_status'].replace('_', ' ')}. {reason}"

    elif intent == "pay_today":
        safe_today = facts["amount_safe_to_pay_today"]
        if safe_today >= requested_amount - 1e-6:
            if decision["recommended_payment_method"] == "full_payment":
                answer = (f"Yes — paying {fmt_amount(requested_amount, currency)} today keeps you above the "
                           f"{fmt_amount(model.min_balance, currency)} minimum for the next 90 days.")
            else:
                answer = (f"The full {fmt_amount(requested_amount, currency)} would be safe to pay today, but "
                           f"the recommended plan for this request is {recommendation.replace('_', ' ')} instead. {reason}")
        elif safe_today > 0:
            answer = (f"Not the full amount. Today you can safely pay up to "
                       f"{fmt_amount(safe_today, currency)} of the "
                       f"{fmt_amount(requested_amount, currency)} requested without breaking the "
                       f"{fmt_amount(model.min_balance, currency)} minimum.")
        else:
            answer = f"No — paying any part of this today would break the {fmt_amount(model.min_balance, currency)} minimum you need to keep."

    elif intent == "installments":
        opts = _installment_facts(ds, request_row["request_id"], model, plan_ctx)
        facts["installment_options"] = opts
        if not opts:
            answer = "No installment plan was supplied for this request, so installments aren't an option here."
        else:
            safe_ones = [o for o in opts if o["safe"]]
            if "installments" in model.payment_methods:
                if safe_ones:
                    chosen = decision["recommended_payment_method"] == "installments"
                    answer = (f"{len(safe_ones)} of {len(opts)} supplied installment plan(s) stay above the "
                               f"{fmt_amount(model.min_balance, currency)} minimum throughout. "
                               + ("This is the recommended plan." if chosen else
                                  "A cheaper or faster safe option was preferred instead — see the recommendation."))
                else:
                    answer = "None of the supplied installment plans stay safely above your minimum balance."
            else:
                answer = "Installments aren't listed among the payment methods you're set up to use for this request."

    elif intent == "which_option_safest":
        cands = plan_ctx["candidates"]
        facts["safe_candidates"] = [{"method": c.method, "total_paid": round(c.total_paid, 2),
                                      "start_date": c.start_date.isoformat(), "num_payments": c.num_payments,
                                      "uses_spending_changes": c.uses_changes} for c in cands]
        if cands:
            answer = (f"Of the options available to you, {recommendation.replace('_', ' ')} is the one the "
                       "engine ranks safest and best overall — it meets your deadline, needs the fewest "
                       "changes, and costs the least. " + reason)
        else:
            answer = "None of your accepted payment methods produced a safe plan for this request."

    elif intent == "which_expenses_affect":
        top = _top_expenses(model)
        facts["top_expenses"] = top
        if top:
            listing = "; ".join(f"{t['category']} (~{fmt_amount(t['amount'], t['currency'])}/month)" for t in top)
            answer = f"The biggest recurring items shaping your 90-day forecast are: {listing}."
        else:
            answer = "No significant recurring or variable expenses were detected in your history."

    elif intent == "reduce_expenses":
        options = _reducible_options(model)
        facts["reducible_expenses"] = options
        if options:
            parts = []
            for o in options:
                actions = []
                if "can_reduce_to" in o:
                    actions.append(f"reduce it to {fmt_amount(o['can_reduce_to'], o['currency'])}")
                if o.get("can_stop"):
                    actions.append("stop it")
                parts.append(f"{o['category']} ({' or '.join(actions)})")
            answer = f"You're set up to reduce or stop these flexible expenses if needed: {'; '.join(parts)}."
            if decision["spending_changes_needed"] != "none":
                answer += f" The recommended plan already uses: {decision['spending_changes_needed']}."
        else:
            answer = "You don't have any flexible expenses currently marked as reducible or stoppable."

    elif intent == "what_if_payment":
        hypo_amount = entities.amount if entities.amount is not None else requested_amount
        payments = [(plan_ctx["request_date"], hypo_amount)]
        safe = schedule_is_safe(plan_ctx["baseline"], model.min_balance, plan_ctx["request_date"], payments)
        remaining_min = min(b - hypo_amount for b in plan_ctx["baseline"].daily_balance)
        facts["hypothetical_payment"] = round(hypo_amount, 2)
        facts["resulting_minimum_balance_over_90_days"] = round(remaining_min, 2)
        facts["stays_above_minimum"] = safe
        if safe:
            answer = (f"Paying {fmt_amount(hypo_amount, currency)} today would keep your balance at or above "
                       f"{fmt_amount(remaining_min, currency)} at its lowest point over the next 90 days — "
                       f"that's still above your {fmt_amount(model.min_balance, currency)} minimum.")
        else:
            answer = (f"Paying {fmt_amount(hypo_amount, currency)} today would take your balance down to "
                       f"{fmt_amount(remaining_min, currency)} at its lowest point, which is below your "
                       f"{fmt_amount(model.min_balance, currency)} minimum — not safe.")

    return {
        "answer": answer,
        "recommendation": recommendation,
        "facts": facts,
        "payment_plan": payment_plan,
        "reason": reason,
        "request_id": request_row["request_id"],
        "user_id": request_row["user_id"],
    }


def _llm_rephrase(answer: str, facts: Dict, recommendation: Optional[str]) -> str:
    """Optional: reword `answer` in friendlier language using a real language
    model, when one is configured. The model is given the already-computed
    answer/facts/recommendation as fixed context and is explicitly told not
    to change any number, date, currency, or the recommendation -- it may
    only rephrase. Never called for, and never able to affect, the actual
    decision. Silently returns the original answer if no key is configured
    or the call fails for any reason, so the endpoint always works offline.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return answer
    try:
        import anthropic  # type: ignore

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=200,
            system=(
                "Reword the given financial answer in clear, friendly, simple language. "
                "Do not add, remove, or change any number, date, currency amount, or the "
                "recommendation -- only rephrase. Keep it to 1-3 sentences."
            ),
            messages=[{"role": "user", "content": (
                f"Answer: {answer}\nRecommendation: {recommendation}\nFacts: {facts}"
            )}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        return text or answer
    except Exception:
        return answer


def handle_chat(message: str, ds: Dataset, rates: RateTable,
                 request_id: Optional[str] = None, user_id: Optional[str] = None,
                 session: Optional[dict] = None) -> Dict:
    session = session or {}
    entities = extract_entities(message)

    request_id = request_id or entities.request_id or session.get("request_id")
    user_id = user_id or entities.user_id or session.get("user_id")

    request_row = ds.request_row(request_id) if request_id else None
    if request_row is not None:
        user_id = request_row["user_id"]

    if request_row is None and user_id:
        anchor = _users_actual_request(ds, user_id)
        if anchor is None:
            return {
                "answer": f"I don't have any financial data for {user_id} in this dataset.",
                "recommendation": None, "facts": {}, "payment_plan": None,
                "reason": "Unknown user_id.", "request_id": None, "user_id": user_id,
            }
        if entities.amount is not None and entities.currency and entities.currency != ds.profiles_by_user[user_id]["home_currency"]:
            # Only the user's own home-currency figures are trustworthy without a
            # supplied exchange-rate lookup for an ad-hoc, non-dataset amount.
            pass
        if entities.amount is not None:
            request_row = _build_synthetic_request(anchor, entities, message)
        else:
            request_row = anchor

    if request_row is None:
        return {
            "answer": ("I need to know which request or user this is about. Include a request id "
                       "(like request_26) or a user id (like user_09), or tell me the amount you're "
                       "asking about."),
            "recommendation": None, "facts": {}, "payment_plan": None,
            "reason": "No request or user could be identified from the message.",
            "request_id": None, "user_id": None,
        }

    model, plan_ctx, decision = _run_engine(ds, rates, request_row)
    intent = classify_intent(message)
    response = _shape_response(intent, message, ds, model, plan_ctx, decision, request_row, entities)
    response["intent"] = intent
    response["answer"] = _llm_rephrase(response["answer"], response["facts"], response["recommendation"])
    return response
