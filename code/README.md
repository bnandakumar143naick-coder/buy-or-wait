# Solution notes — "Buy or Wait?"

This documents the implementation in this `code/` directory. The
organizer's own `README.md` and `problem_statement.md` at the repo root are
left untouched.

## Run it

```bash
pip install -r code/requirements.txt

# Full 250-request batch -> writes ./output.csv at the repo root
python3 code/main.py

# One request, printed as JSON (useful for debugging)
python3 code/main.py --request request_26

# Sanity-check against the 25 public sample requests
python3 code/main.py --sample

# Flask API + dashboard (http://localhost:5000)
python3 code/app.py
curl -X POST http://localhost:5000/api/affordability \
     -H "Content-Type: application/json" \
     -d '{"request_id": "request_26"}'

# AI chatbot over the same engine
curl -X POST http://localhost:5000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Why did you recommend waiting?", "request_id": "request_26"}'
```

Open `http://localhost:5000` for the dashboard: browse all 250 requests,
see each one's decision statement and 90-day forecast chart, and ask the
built-in chat assistant about the open request.

No API key is required. `ANTHROPIC_API_KEY` is optional and only used as a
live fallback for image extraction if the pre-computed cache is ever
missing an entry (see `code/evaluation/usage_report.md`).

## Architecture

```
code/
  data_loader.py       load every dataset/*.csv, fill blank amounts from the
                        image cache, index everything by user/request
  fx.py                 dated currency conversion (exact-match lookup)
  ai/
    message_parser.py   bilingual template classifier for messages.csv
    image_parser.py      resolves blank amounts via the image cache (or a
                          live vision call if ANTHROPIC_API_KEY is set)
    chat_assistant.py     natural-language front end for the chatbot: figures
                          out which request/user a message is about and
                          what's being asked, runs the SAME engine below,
                          and words the answer (see "Chatbot" section)
  financial_state.py    reconstructs each user's forward cash-flow model:
                          recurring bills, variable essential spend, salary
                          detection + message-driven overrides
  forecast_engine.py    builds the 90-day balance curve and answers the two
                          safety questions in O(n) (no search/backtracking)
  payment_planner.py    builds every safe candidate plan (full / partial /
                          installments / wait), including a bounded search
                          over spending changes (<=3) that could unlock an
                          otherwise-unsafe full payment today
  decision_engine.py    ranks candidates per the challenge's 6 tie-break
                          rules, assembles the 8 output columns + explanation
  validator.py           re-checks every row against the output contract
                          before it is written; auto-corrects to a safe
                          fallback rather than shipping a bad row
  main.py                CLI entry point / orchestrator (batch -> output.csv)
  app.py                 Flask API: POST /api/affordability, POST /api/chat,
                          plus the dashboard (GET /) and its read-only
                          supporting endpoints (/api/requests, /api/forecast,
                          /api/run-batch, /api/output-summary)
  static/                dashboard: request browser, decision statement,
                          90-day forecast chart, and the chat panel
                          (plain HTML/CSS/JS -- no build step, no framework)
  cache/image_extractions.json   the 16 image-derived amounts (see below)
  evaluation/usage_report.md      AI-usage and cost accounting
  requirements.txt
```

Only what the brief asks for plus a thin read-only viewer over it: the
dashboard calls the same engine through the same endpoints described
above -- it never computes or stores a financial number of its own.

## Chatbot (`POST /api/chat`)

```
user message
  -> identify the request/user (from message text, or an explicit
     request_id / user_id / session_id in the body)
  -> run the SAME engine as /api/affordability and main.py
     (financial_state -> forecast_engine -> payment_planner -> decision_engine)
  -> turn the already-computed result into a plain-language answer
```

`ai/chat_assistant.py` never re-implements or re-derives a financial
number: `extract_entities` / `classify_intent` only decide *which*
request/user the message is about and *what kind* of question is being
asked (why wait, when can I afford it, installments, which expenses are
affecting me, what-if, safest option, reduce expenses, ...); `_shape_response`
then picks which already-computed facts (from `decision`, `model`,
`plan_ctx`) to surface and how to phrase them. The recommendation and every
number in the response are always read straight off the engine's output —
the language layer can only choose *which* facts to mention and *how* to
word them, never invent or override one.

A follow-up question with no `request_id` (e.g. "Why is this not
affordable?") is resolved via `session_id`: the server remembers, in
memory only, which request/user the last message in that session
resolved to. A question with a new amount and no request_id at all
(e.g. "I want to buy a laptop for ₹80,000, can I afford it?") is run as a
one-off hypothetical against that user's real balance/history/messages —
only the two request-level details the message didn't specify (a 30-day
completion window, no partial payments) are defaulted; every financial
fact still comes from the dataset.

By default everything above runs with zero API calls (keyword/regex
classification, templated answers) so the endpoint works fully offline.
If `ANTHROPIC_API_KEY` is set, the final answer text is optionally reworded
by `claude-sonnet-4-6` for a friendlier tone -- it is given the already-
computed answer/facts/recommendation as fixed context and is explicitly
instructed not to change any number, date, or recommendation, only the
wording (see `ai/chat_assistant._llm_rephrase`).

## Key design decisions (and why)

**Balance is not replayed.** `current_available_balance` in
`financial_profiles.csv` already reflects every settled transaction before
`request_date`. History is only used to *detect patterns* (a recurring
rent payment, a stable salary), never summed into the starting balance —
otherwise everything would be double-counted.

**A same-day payment shifts the whole future curve down by a constant.**
There's no interest or re-borrowing in this problem, so paying `X` today
only ever subtracts `X` from every later point on the balance curve. That
means both `amount_safe_to_pay` and `earliest_date_for_full_payment` reduce
to one forward pass (build the daily balance) and one backward pass
(suffix-minimum) — no binary search or trial-and-error needed, and the two
metrics are computed once and reused across every candidate plan.

**Recurring bills are detected at the `(category, description)` level**
(rent, subscriptions, debt, insurance): >=2 historical occurrences ~30 days
apart with low amount variance. **Variable essential spend** (groceries,
dining, transport, shopping, entertainment, healthcare) is aggregated to a
**category-level monthly total** instead, since individual transactions
don't repeat but the monthly burn does — and the *maximum* of the last few
months is used, so essential spending is never understated.

**Salary detection** tries, in order: (1) an explicit `scheduled` next-salary
row (the strongest evidence — a scheduled paycheck implies ongoing
employment, so it anchors every later monthly occurrence too, not just the
one row); (2) the most frequently repeated exact historical amount (robust
to a single month containing an extra bonus/arrears/duplicate payslip
line); (3) a monthly-total aggregate for genuinely variable/freelance
income, using the *minimum* of recent months (income is estimated
conservatively — low, not high). Messages that confirm a raise, a
temporary reduction, a new job, or an end of employment then override this
baseline, applied in message order so the latest instruction wins.

**Spending changes always target the most recent historical event** in the
category being changed, and a `reduce_to` change always targets that
event's `minimum_allowed_amount` — this is what every worked example in
`sample_requests.csv` does, so the engine matches that convention rather
than inventing a different reduction amount.

**`earliest_date_for_full_payment` is always the baseline (no-change)
figure**, independent of the final recommended method or any spending
changes applied — exactly as the problem statement specifies ("measures
financial capacity independently of the user's payment-method
preferences"). `amount_safe_to_pay` is the baseline figure too, *unless*
spending changes are part of the chosen plan, in which case it's the
post-change figure (this reconciles every worked example, including the
ones where a spending change unlocks paying the full amount today).

**Untrusted message content never expands what's affordable.** Pending
payouts, unresolved disputes, unrealized investment gains, and a lottery
/ "pay a release charge to receive your prize" phishing attempt are all
recognised and explicitly never treated as available cash or as an
instruction to spend — regardless of how they're phrased in the message
text.

## Known limitations

This is a from-scratch reconstruction of the grading logic from 25 worked
examples, not the organizer's actual scoring code, so it will not match
every hidden test case exactly. In particular: the monthly-lump placement
for variable spending (mid-month, evenly spaced) and the exact day-of-month
salary lands on are reasonable approximations, not always reproductions of
a specific hidden convention. Where the specification was ambiguous, the
engine consistently prefers the financially safer interpretation.
