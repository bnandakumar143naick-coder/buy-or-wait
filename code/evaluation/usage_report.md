# AI Usage Report — "Buy or Wait?" (HackerRank Orchestrate, September 2026)

This report documents every place an AI model touched this submission, and
why the final, scored run (`python3 code/main.py` → `output.csv`) makes
**zero live model calls**.

## 1. Where AI is used in this system

| Task | Module | AI involved? |
|---|---|---|
| Final financial arithmetic, forecasting, ranking, validation | `financial_state.py`, `forecast_engine.py`, `payment_planner.py`, `decision_engine.py`, `validator.py` | **No** — pure deterministic Python, as required by the challenge. |
| Extracting the missing amount from the 16 receipt/payslip images in `dataset/media/images/` | `ai/image_parser.py` + `code/cache/image_extractions.json` | **Yes** — see §2. |
| Interpreting `dataset/messages.csv` (raises, cancellations, disputes, scams, etc.) | `ai/message_parser.py` | **Yes, in spirit** — see §3. |
| Understanding a chatbot question and wording its answer | `ai/chat_assistant.py` | **Yes, optionally** — see §4. |

## 2. Image extraction (vision model usage)

`dataset/images.csv` links exactly **16** financial events (across the whole
250-request dataset) to a scanned receipt/payslip/invoice because their
`amount` column is blank. Each of the 16 images was read once, during
development, by a vision-capable Claude model (Claude Sonnet 4.6, accessed
through this build environment) with a single prompt per image: *"identify
the one financial amount most relevant to the linked transaction's
description."* The extracted amount, the document field it came from, and a
short justification were recorded in `code/cache/image_extractions.json`.

At run time, `ai/image_parser.extract_amount()`:
1. Looks the event up in that cache first (this is what produced the
   submitted `output.csv`).
2. Only if the cache misses **and** `ANTHROPIC_API_KEY` is set in the
   environment, falls back to a live call to `claude-sonnet-4-6` via the
   `anthropic` Python SDK, asking for a single plain number.

This keeps the amount-extraction step genuinely AI-driven (it is not a
hand-typed lookup table pretending to be one — the 16 values were read
directly off the document images, several requiring judgement about which
of several printed totals is the "right" one — see the code comments in
`image_extractions.json` for the reasoning on each) while making the batch
run reproducible and free to execute in an offline grading sandbox.

**Estimated cost of the one-time extraction (16 images, development-time only):**

| | per image | × 16 |
|---|---:|---:|
| Input tokens (image + short instruction) | ≈1,500 | ≈24,000 |
| Output tokens (one number + brief note) | ≈40 | ≈640 |
| Cost @ Claude Sonnet pricing (~$3 / M input, ~$15 / M output) | ≈$0.0051 | **≈$0.08** |

Total one-time image-extraction cost: **under $0.10**. Zero cost at
evaluation time (cache hit).

## 3. Message interpretation

`dataset/messages.csv` contains 216 messages, but they are drawn from only
~25 fixed bilingual (English / Indonesian) templates issued by employers,
banks, merchants, and financial-service apps (a raise notice, a payroll
delay, a card dispute, a lottery-style phishing attempt, and so on).
Because the template set is small and fixed, `ai/message_parser.py`
implements the same classification a language model would perform, using
keyword/regex matching per template, with **zero marginal cost** and (unlike
a model call) perfectly reproducible output — which matters directly for
the challenge's requirement that *"the deterministic Python engine, not an
LLM, must control the final financial decision."*

If this dataset is extended with genuinely free-form, non-templated
messages, `classify_message()` degrades safely: unrecognised text is
tagged `unclassified` and is **never** used to unlock extra spending —
only ever to *restrict* it (the same fail-safe posture applied to the
scam / phishing message in the dataset, which is recognised and explicitly
never actioned no matter how it is phrased).

## 4. Chatbot (`POST /api/chat`)

The chatbot in `ai/chat_assistant.py` is natural-language understanding and
explanation *around* the decision engine, never a replacement for it:

1. `extract_entities` / `classify_intent` (regex + keyword rules, the same
   style as `message_parser.py`) figure out which request/user a question
   refers to and what's being asked — they never touch a financial number.
2. The identified request is run through the exact same
   `financial_state -> forecast_engine -> payment_planner -> decision_engine`
   pipeline used by `/api/affordability` and `main.py`. No number is
   recalculated or duplicated for the chatbot.
3. `_shape_response` selects which already-computed facts to show and
   phrases the answer from a template for the recognised question type.

By default (no API key) this entire pipeline runs with **zero model
calls** — the same offline guarantee as the rest of the system.

If `ANTHROPIC_API_KEY` is set, `_llm_rephrase` optionally sends the
already-produced `answer` / `facts` / `recommendation` to `claude-sonnet-4-6`
with an explicit instruction to reword only, never to add, remove, or
change a number, date, or the recommendation. This is a bounded,
single-call-per-message convenience layer, not a source of financial
facts, and it fails silently back to the deterministic answer if the call
errors or no key is present — the chatbot is never non-functional for lack
of a key.

## 5. Full-dataset run cost

Running `python3 code/main.py` over all 250 requests:

* API calls made: **0**
* Tokens consumed: **0**
* Cost: **$0.00**
* Wall-clock time: ≈5 seconds on a single CPU core.

| Provider | Model | Calls | Input tokens | Output tokens | Total tokens | Cost |
|---|---|---:|---:|---:|---:|---:|
| Anthropic (dev-time only, not part of the scored run) | claude-sonnet-4-6 | 16 | ~24,000 | ~640 | ~24,640 | ~$0.08 |
| — (scored run: `python3 code/main.py`) | — | 0 | 0 | 0 | 0 | $0.00 |

## 6. Secrets

No API key is required to reproduce `output.csv`. If a key is present as
`ANTHROPIC_API_KEY`, it is read from the environment only
(`os.environ.get(...)`) and is never hardcoded, logged, or written to any
file in this repository.
