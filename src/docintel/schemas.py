"""Strict Pydantic schemas for the structured output the LLM must produce.

Field validators accept the loose formats LLMs (and documents) produce, such as
"$1,200.00" or "03/04/2025", and normalise them, so that a `ValidationError`
only means the output is genuinely wrong. Those errors drive the repair loop.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

from docintel.normalize import normalize_currency, parse_amount, parse_date


class DocumentType(StrEnum):
    INVOICE = "invoice"
    RECEIPT = "receipt"
    OTHER = "other"


Amount = Annotated[float | None, BeforeValidator(parse_amount)]
OptDate = Annotated[date | None, BeforeValidator(parse_date)]
Currency = Annotated[str | None, BeforeValidator(normalize_currency)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @field_validator("*", mode="before")
    @classmethod
    def _empty_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none", "n/a"}:
            return None
        return value


class LineItem(_Strict):
    description: str = Field(min_length=1)
    quantity: Amount = None
    unit_price: Amount = None
    amount: Amount = Field(default=None, description="Line total (quantity x unit price).")


class Invoice(_Strict):
    vendor_name: str | None = None
    vendor_address: str | None = None
    customer_name: str | None = None
    invoice_number: str | None = None
    invoice_date: OptDate = None
    due_date: OptDate = None
    currency: Currency = Field(default=None, description="ISO 4217 code, e.g. USD.")
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Amount = None
    tax: Amount = None
    total: Amount = None


class Receipt(_Strict):
    merchant_name: str | None = None
    merchant_address: str | None = None
    receipt_number: str | None = None
    transaction_date: OptDate = None
    currency: Currency = Field(default=None, description="ISO 4217 code, e.g. EUR.")
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Amount = None
    tax: Amount = None
    tip: Amount = None
    total: Amount = None
    payment_method: str | None = None


class KeyValue(_Strict):
    key: str = Field(min_length=1)
    value: str | None = None


class GenericDocument(_Strict):
    title: str | None = None
    summary: str | None = Field(default=None, description="One sentence describing the document.")
    fields: list[KeyValue] = Field(default_factory=list)


class Classification(_Strict):
    document_type: DocumentType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str | None = None


DocumentSchema = Invoice | Receipt | GenericDocument

SCHEMA_FOR_TYPE: dict[DocumentType, type[Invoice] | type[Receipt] | type[GenericDocument]] = {
    DocumentType.INVOICE: Invoice,
    DocumentType.RECEIPT: Receipt,
    DocumentType.OTHER: GenericDocument,
}
