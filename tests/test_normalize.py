from datetime import date

import pytest

from docintel.normalize import (
    amounts_close,
    detect_currency,
    normalize_currency,
    parse_amount,
    parse_date,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$1,234.50", 1234.5),
        ("1.234,50 €", 1234.5),
        ("12,50", 12.5),
        ("1,250", 1250.0),
        ("(12.00)", -12.0),
        ("Rs 4,065,324.53", 4065324.53),
        (99, 99.0),
        ("", None),
        (None, None),
    ],
)
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == expected


def test_parse_amount_rejects_text():
    with pytest.raises(ValueError):
        parse_amount("twelve")


@pytest.mark.parametrize(
    ("raw", "dayfirst", "expected"),
    [
        ("2025-03-05", False, date(2025, 3, 5)),
        ("March 5, 2025", False, date(2025, 3, 5)),
        ("05/03/2025", True, date(2025, 3, 5)),
        ("5 Mar 2025", False, date(2025, 3, 5)),
    ],
)
def test_parse_date(raw, dayfirst, expected):
    assert parse_date(raw, dayfirst=dayfirst) == expected


def test_parse_date_rejects_garbage():
    with pytest.raises(ValueError):
        parse_date("not a date at all")


@pytest.mark.parametrize(
    ("raw", "code"), [("usd", "USD"), ("€", "EUR"), ("£", "GBP"), ("Rs", "PKR"), (None, None)]
)
def test_normalize_currency(raw, code):
    assert normalize_currency(raw) == code


def test_normalize_currency_unknown():
    with pytest.raises(ValueError):
        normalize_currency("dollars-ish")


def test_detect_currency_prefers_iso_codes_then_symbols():
    assert detect_currency("Total EUR 12.00 (approx $13)") == "EUR"
    assert detect_currency("Subtotal £10.00 Total £12.00") == "GBP"
    assert detect_currency("Total Rs 4,500") == "PKR"
    assert detect_currency("no money here") is None


def test_amounts_close_uses_abs_or_relative_tolerance():
    assert amounts_close(100.0, 100.01, 0.02, 0.0)
    assert not amounts_close(100.0, 100.5, 0.02, 0.001)
    assert amounts_close(100000.0, 100040.0, 0.02, 0.005)
