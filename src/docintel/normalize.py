"""Parsing and normalisation helpers for amounts, dates and currencies."""

from __future__ import annotations

import re
from collections import Counter
from datetime import date, datetime

from dateutil import parser as date_parser

# Symbols that map unambiguously (or by strong convention) to an ISO 4217 code.
CURRENCY_SYMBOLS: dict[str, str] = {
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "₨": "PKR",
    "₩": "KRW",
    "₺": "TRY",
    "₦": "NGN",
    "₱": "PHP",
    "R$": "BRL",
    "A$": "AUD",
    "C$": "CAD",
    "US$": "USD",
    "$": "USD",
}
# Ambiguous local abbreviations that cannot live in CURRENCY_SYMBOLS (e.g. "Rs" is also INR).
CURRENCY_ALIASES: dict[str, tuple[str, ...]] = {"PKR": ("Rs", "Rs.")}
ISO_CURRENCIES: frozenset[str] = frozenset(
    [
        "USD",
        "EUR",
        "GBP",
        "JPY",
        "INR",
        "PKR",
        "AUD",
        "CAD",
        "CHF",
        "CNY",
        "SEK",
        "NOK",
        "DKK",
        "PLN",
        "BRL",
        "MXN",
        "ZAR",
        "AED",
        "SAR",
        "SGD",
        "HKD",
        "NZD",
        "KRW",
        "TRY",
        "NGN",
        "PHP",
        "EGP",
        "KES",
        "BDT",
    ]
)
_CODE_RE = re.compile(r"\b(" + "|".join(sorted(ISO_CURRENCIES)) + r")\b")
_AMOUNT_CLEAN_RE = re.compile(r"[^\d,.\-()]")


def parse_amount(value: object) -> float | None:
    """Parse '1,234.50', '$ 99', '(12.00)', '1.234,50' into a float. Returns None if empty."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return round(float(value), 4)
    text = str(value).strip()
    if not text:
        return None
    negative = (text.startswith("(") and text.endswith(")")) or text.startswith("-")
    cleaned = _AMOUNT_CLEAN_RE.sub("", text).strip("()-")
    if not cleaned or not any(ch.isdigit() for ch in cleaned):
        raise ValueError(f"not a monetary amount: {value!r}")
    if "," in cleaned and "." in cleaned:
        # Whichever separator comes last is the decimal separator.
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        head, _, tail = cleaned.rpartition(",")
        # '12,50' is a decimal comma; '1,250' is a thousands separator.
        cleaned = f"{head.replace(',', '')}.{tail}" if len(tail) == 2 else cleaned.replace(",", "")
    amount = float(cleaned)
    return round(-amount if negative else amount, 4)


def parse_date(value: object, dayfirst: bool = False) -> date | None:
    """Parse common date spellings into a `date`. ISO strings are always unambiguous."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return date.fromisoformat(text)
    try:
        return date_parser.parse(text, dayfirst=dayfirst, fuzzy=True).date()
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"unparseable date: {value!r}") from exc


def normalize_currency(value: object) -> str | None:
    """Map a code or symbol ('usd', '€', 'Rs') to an ISO 4217 code."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    upper = text.upper()
    if upper in ISO_CURRENCIES:
        return upper
    if text in CURRENCY_SYMBOLS:
        return CURRENCY_SYMBOLS[text]
    for code, aliases in CURRENCY_ALIASES.items():
        if upper in {a.upper() for a in aliases}:
            return code
    raise ValueError(f"unknown currency: {value!r}")


def detect_currency(text: str) -> str | None:
    """Guess the document currency from explicit ISO codes first, then symbols."""
    codes = Counter(_CODE_RE.findall(text.upper()))
    if codes:
        return codes.most_common(1)[0][0]
    symbol_hits: Counter[str] = Counter()
    for symbol in sorted(CURRENCY_SYMBOLS, key=len, reverse=True):
        count = text.count(symbol)
        if count:
            symbol_hits[CURRENCY_SYMBOLS[symbol]] += count
            text = text.replace(symbol, " ")
    if re.search(r"\bRs\.?\s?\d", text):
        symbol_hits["PKR"] += 1
    return symbol_hits.most_common(1)[0][0] if symbol_hits else None


def amounts_close(a: float, b: float, abs_tol: float, rel_tol: float) -> bool:
    """True when two amounts agree within an absolute or relative tolerance."""
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


def canonical_text(value: object) -> str:
    """Lower-case, collapse whitespace, strip punctuation edges: used for fuzzy grounding."""
    return re.sub(r"\s+", " ", str(value)).strip().strip(".,:;").lower()
