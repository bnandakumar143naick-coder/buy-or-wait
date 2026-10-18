"""
ai/message_parser.py
---------------------
Turns dataset/messages.csv into structured financial evidence.

Every message in this dataset follows one of ~25 fixed bilingual
(English / Indonesian) templates emitted by employers, banks, merchants,
service providers, or financial-service apps. Because the templates are
fixed, a deterministic keyword + regex classifier extracts the same
structured facts a language model would, at zero marginal cost and with
perfectly reproducible output -- which matters because the deterministic
engine, not an LLM, must own the final financial decision.

If a future dataset introduces free-form messages outside these templates,
`classify_message` falls through to a generic amount/date extraction and
flags the message as `unclassified` so the caller can decide whether to
use it (conservatively, unclassified messages are never used to unlock
extra spending -- only to *restrict* it).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

CURRENCY_RE = re.compile(r"\b(EUR|USD|ZAR|INR|IDR)\s*([\d.,]+)\b")
DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _first_amount(text: str):
    m = CURRENCY_RE.search(text)
    if not m:
        return None, None
    currency = m.group(1)
    amount = float(m.group(2).replace(",", ""))
    return currency, amount


def _first_date(text: str) -> Optional[str]:
    m = DATE_RE.search(text)
    return m.group(1) if m else None


def _any(text: str, phrases: List[str]) -> bool:
    low = text.lower()
    return any(p.lower() in low for p in phrases)


# Each rule: (kind, [trigger phrases across en/id templates])
_RULES = [
    ("salary_increase", [
        "has increased to", "naik menjadi",
    ]),
    ("salary_temporary", [
        "temporary monthly pay", "gaji bulanan sementara",
    ]),
    ("salary_reduced_leave", [
        "is reduced to", "dikurangi menjadi",
    ]),
    ("salary_date_change", [
        "confirmed salary is now expected on", "kini diperkirakan masuk pada",
    ]),
    ("salary_resumed", [
        "resumes on",
    ]),
    ("salary_first", [
        "first salary will be", "first salary from the new employer",
        "first salary of", "gaji pertama",
    ]),
    ("salary_base_commission_pending", [
        "confirmed base salary is", "gaji pokok yang dikonfirmasi adalah",
    ]),
    ("household_ended_remaining", [
        "household employment record has ended",
        "sumber pendapatan kerja rumah tangga telah berakhir",
    ]),
    ("employment_ended", [
        "employment has ended", "hubungan kerja anda telah berakhir",
    ]),
    ("seasonal_contract_ended", [
        "seasonal contract has ended", "kontrak musiman saat ini telah berakhir",
    ]),
    ("bonus_pending", [
        "quarterly bonus", "bonus kuartalan",
    ]),
    ("reimbursement_not_salary", [
        "reimbursement for your earlier work expense",
        "penggantian atas biaya kerja",
    ]),
    ("invoice_confirmed", [
        "client approved an invoice payment of",
        "klien menyetujui pembayaran faktur sebesar",
    ]),
    ("gig_payout_pending", [
        "payout is still pending", "masih tertunda", "not withdrawable",
        "belum dapat ditarik",
    ]),
    ("rent_increase_pct", [
        "increases monthly rent by", "menaikkan biaya sewa bulanan sebesar",
        "menaikkan.*sewa",
    ]),
    ("internal_transfer", [
        "transfer between your two accounts", "transfer antara dua rekening",
    ]),
    ("card_dispute_pending", [
        "still being investigated", "masih dalam penyelidikan",
    ]),
    ("min_payments_separate_cards", [
        "minimum payments due on two separate card accounts",
    ]),
    ("debit_failed_still_outstanding", [
        "previous debit attempt failed",
    ]),
    ("merchant_refund_pending", [
        "refund has been initiated but has not reached",
        "pengembalian dana sudah diproses, tetapi belum masuk",
        "foreign-currency refund is still processing",
    ]),
    ("investment_valuation_change", [
        "displayed market value has increased",
        "displayed value of the investment has fallen",
        "nilai investasi yang ditampilkan telah",
    ]),
    ("investment_sale_settled", [
        "proceeds from your investment sale have settled",
        "hasil penjualan investasi anda sudah masuk",
    ]),
    ("prize_pending", [
        "prize claim has been verified and is still in payment processing",
        "klaim hadiah anda sudah diverifikasi dan masih dalam proses",
    ]),
    ("prize_settled", [
        "prize proceeds have reached your account after withholding",
    ]),
    ("scam_advance_fee", [
        "pay the release charge", "pay the processing charge",
        "bayar biaya pencairan", "bayar biaya pemrosesan",
    ]),
    ("fx_settlement_note", [
        "will convert it using the rate", "mengonversinya dengan kurs",
        "will use the rate applied when it settles",
    ]),
]


@dataclass
class MessageEvidence:
    message_id: str
    user_id: Optional[str]
    request_id: Optional[str]
    related_event_id: Optional[str]
    kind: str
    currency: Optional[str] = None
    amount: Optional[float] = None
    date: Optional[str] = None
    percent: Optional[float] = None
    raw_text: str = ""


def classify_message(row: dict) -> MessageEvidence:
    text = str(row.get("message_text") or "")
    low = text.lower()
    kind = "unclassified"
    for candidate_kind, phrases in _RULES:
        if _any(low, phrases):
            kind = candidate_kind
            break
    currency, amount = _first_amount(text)
    date = _first_date(text)
    pct = None
    pm = PERCENT_RE.search(text)
    if pm:
        pct = float(pm.group(1))
    return MessageEvidence(
        message_id=row.get("message_id"),
        user_id=row.get("user_id") if isinstance(row.get("user_id"), str) else None,
        request_id=row.get("request_id") if isinstance(row.get("request_id"), str) else None,
        related_event_id=row.get("related_event_id") if isinstance(row.get("related_event_id"), str) else None,
        kind=kind,
        currency=currency,
        amount=amount,
        date=date,
        percent=pct,
        raw_text=text,
    )


def parse_all_messages(messages_df) -> List[MessageEvidence]:
    out = []
    for row in messages_df.to_dict("records"):
        out.append(classify_message(row))
    return out


# Kinds that must never be treated as available/forecastable cash, no matter
# what number appears in the message text (untrusted evidence safety net).
NEVER_TRUST_AS_INCOME = {
    "bonus_pending", "gig_payout_pending", "merchant_refund_pending",
    "investment_valuation_change", "prize_pending", "scam_advance_fee",
    "min_payments_separate_cards", "card_dispute_pending",
    "internal_transfer", "fx_settlement_note",
}

# Kinds that revise the user's forward-looking recurring salary baseline.
SALARY_KINDS = {
    "salary_increase", "salary_temporary", "salary_reduced_leave",
    "salary_date_change", "salary_resumed", "salary_first",
    "salary_base_commission_pending", "household_ended_remaining",
    "employment_ended", "seasonal_contract_ended",
}
