"""Generate a synthetic but realistic set of invoices and receipts with ground truth.

Every document is rendered as a digital PDF (text layer) with reportlab, and a
"scanned" twin is produced by rasterising it and adding rotation, blur and noise
(saved as JPG, or as an image-only PDF so the PDF OCR fallback is exercised).
All companies and people are fictional.
"""

from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
from PIL import Image, ImageFilter
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas

VENDORS = [
    ("Northwind Analytics LLC", "410 Harbor Blvd, Seattle, WA 98101", "USD"),
    ("Bluepeak Software GmbH", "Lindenstrasse 12, 10969 Berlin", "EUR"),
    ("Kestrel Print & Design Ltd", "22 Wharf Road, Bristol BS1 4RN", "GBP"),
    ("Indus Cloud Solutions (Pvt) Ltd", "Plot 7, Gulberg III, Lahore", "PKR"),
    ("Maple Leaf Logistics Inc.", "88 King St W, Toronto, ON M5H", "CAD"),
    ("Atlas Office Supply Co.", "1200 Commerce Ave, Austin, TX 78701", "USD"),
    ("Studio Lumen SARL", "5 Rue des Arts, 69002 Lyon", "EUR"),
    ("Granite Legal Services LLP", "14 Fleet Street, London EC4Y", "GBP"),
]
CUSTOMERS = [
    "Acme Retail Group",
    "Helios Health Partners",
    "Orbit Media Ltd",
    "Juniper Foods",
    "Vertex Engineering",
    "Cobalt Fintech",
    "Silverline Hotels",
    "Nimbus Education",
]
SERVICES = [
    ("Data pipeline consulting (hours)", 85, 160),
    ("Cloud hosting - monthly plan", 49, 499),
    ("UX design workshop", 300, 1200),
    ("Annual software license", 199, 2400),
    ("A4 copy paper (box)", 18, 45),
    ("Freight shipment Toronto-Montreal", 120, 900),
    ("Legal review of contract", 250, 900),
    ("Brand identity package", 800, 3000),
    ("On-site support visit", 150, 450),
    ("API usage overage (1k calls)", 2, 15),
    ("Laptop docking station", 89, 260),
    ("Training session (per seat)", 60, 220),
]
MERCHANTS = [
    ("FRESH MART", "221 Elm Street, Springfield", "USD", 0.0725, False),
    ("CAFE MERIDIAN", "8 Canal Street, Amsterdam", "EUR", 0.09, True),
    ("THE COPPER KETTLE", "17 High St, Oxford OX1", "GBP", 0.20, True),
    ("CITY PHARMACY", "45 Mall Road, Lahore", "PKR", 0.17, False),
    ("TECHNOHUB STORE", "300 Market St, San Francisco", "USD", 0.08625, False),
]
GROCERY = [
    ("Organic milk 1L", 1.29, 3.5),
    ("Sourdough bread", 2.5, 6),
    ("Espresso", 2.2, 3.8),
    ("Cappuccino", 2.9, 4.6),
    ("Club sandwich", 6.5, 12),
    ("Paracetamol 500mg", 1.1, 4),
    ("USB-C cable 2m", 9, 19),
    ("Bananas (kg)", 0.9, 2.4),
    ("Greek salad", 7, 13),
    ("Sparkling water", 1.2, 3),
    ("Vitamin C tablets", 4, 11),
    ("Phone case", 12, 29),
]
SYMBOL = {"USD": "$", "EUR": "€", "GBP": "£", "PKR": "Rs ", "CAD": "C$"}
TAX_RATE = {"USD": 0.08, "EUR": 0.19, "GBP": 0.20, "PKR": 0.16, "CAD": 0.13}
TAX_LABEL = {"EUR": "VAT", "GBP": "VAT", "CAD": "HST", "PKR": "GST"}
SCAN_DPI = 170
REGULAR, BOLD = "Helvetica", "Helvetica-Bold"


@dataclass
class Sample:
    doc_id: str
    kind: str
    pdf: bytes
    truth: dict[str, Any]


def _money(value: float, currency: str, european: bool) -> str:
    text = f"{value:,.2f}"
    if european:
        text = text.replace(",", "_").replace(".", ",").replace("_", ".")
        return f"{text} {SYMBOL[currency].strip()}"
    return f"{SYMBOL[currency]}{text}"


def _fmt_date(value: date, style: int) -> str:
    formats = ["%B %d, %Y", "%d/%m/%Y", "%Y-%m-%d", "%d %b %Y"]
    return value.strftime(formats[style])


def _line_items(
    rng: random.Random, catalogue: list, n: int, qtys: list[int], scale: float
) -> list[dict[str, Any]]:
    items = []
    for desc, lo, hi in rng.sample(catalogue, n):
        qty = rng.choice(qtys)
        price = round(rng.uniform(lo, hi) * scale, 2)
        items.append(
            {
                "description": desc,
                "quantity": float(qty),
                "unit_price": price,
                "amount": round(qty * price, 2),
            }
        )
    return items


def _invoice(rng: random.Random, i: int) -> Sample:
    vendor, address, currency = VENDORS[i % len(VENDORS)]
    european = currency == "EUR"
    issued = date(2025, 1, 1) + timedelta(days=rng.randint(0, 270))
    due = issued + timedelta(days=rng.choice([14, 30, 45]))
    date_style = 1 if currency in {"EUR", "GBP", "PKR"} else rng.choice([0, 2, 3])
    number = f"INV-{issued.year}-{rng.randint(1000, 9999)}"
    customer = rng.choice(CUSTOMERS)
    tax_rate = TAX_RATE[currency]
    scale = 280 if currency == "PKR" else 1
    items = _line_items(rng, SERVICES, rng.randint(2, 6), [1, 1, 2, 3, 5, 10, 12], scale)
    subtotal = round(sum(it["amount"] for it in items), 2)
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)

    buf = io.BytesIO()
    c = Canvas(buf, pagesize=A4)
    width, height = A4
    c.setFont(BOLD, 18)
    c.drawString(20 * mm, height - 25 * mm, vendor)
    c.setFont(REGULAR, 9)
    c.drawString(20 * mm, height - 31 * mm, address)
    c.setFont(BOLD, 22)
    c.drawRightString(width - 20 * mm, height - 25 * mm, "INVOICE")
    c.setFont(REGULAR, 10)
    meta = [
        ("Invoice No:", number),
        ("Invoice Date:", _fmt_date(issued, date_style)),
        ("Due Date:", _fmt_date(due, date_style)),
        ("Currency:", currency),
    ]
    for row, (label, value) in enumerate(meta):
        y = height - (40 + 6 * row) * mm
        c.drawRightString(width - 60 * mm, y, label)
        c.drawRightString(width - 20 * mm, y, value)
    c.setFont(BOLD, 10)
    c.drawString(20 * mm, height - 45 * mm, "Bill To:")
    c.setFont(REGULAR, 10)
    c.drawString(20 * mm, height - 51 * mm, customer)

    y = height - 75 * mm
    right = width - 20 * mm
    c.setFillGray(0.9)
    c.rect(18 * mm, y - 2 * mm, width - 36 * mm, 8 * mm, fill=1, stroke=0)
    c.setFillGray(0)
    c.setFont(BOLD, 10)
    c.drawString(20 * mm, y, "Description")
    c.drawRightString(130 * mm, y, "Qty")
    c.drawRightString(165 * mm, y, "Unit Price")
    c.drawRightString(right, y, "Amount")
    c.setFont(REGULAR, 10)
    for it in items:
        y -= 8 * mm
        c.drawString(20 * mm, y, it["description"])
        c.drawRightString(130 * mm, y, f"{it['quantity']:g}")
        c.drawRightString(165 * mm, y, _money(it["unit_price"], currency, european))
        c.drawRightString(right, y, _money(it["amount"], currency, european))
    y -= 6 * mm
    c.line(110 * mm, y, right, y)
    tax_label = TAX_LABEL.get(currency, "Sales Tax")
    totals = [
        ("Subtotal", subtotal, False),
        (f"{tax_label} ({tax_rate:.0%})", tax, False),
        ("Total Due", total, True),
    ]
    for label, value, bold in totals:
        y -= 7 * mm
        c.setFont(BOLD if bold else REGULAR, 11 if bold else 10)
        c.drawRightString(150 * mm, y, label)
        c.drawRightString(right, y, _money(value, currency, european))
    c.setFont(REGULAR, 8)
    c.drawString(
        20 * mm,
        20 * mm,
        f"Payment terms: Net {(due - issued).days}. "
        f"Please quote {number} with your payment. Thank you!",
    )
    c.showPage()
    c.save()
    truth = {
        "document_type": "invoice",
        "vendor_name": vendor,
        "customer_name": customer,
        "invoice_number": number,
        "invoice_date": issued.isoformat(),
        "due_date": due.isoformat(),
        "currency": currency,
        "line_items": items,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
    }
    return Sample(f"invoice_{i:02d}", "invoice", buf.getvalue(), truth)


def _receipt(rng: random.Random, i: int) -> Sample:
    merchant, address, currency, tax_rate, has_tip = MERCHANTS[i % len(MERCHANTS)]
    scale = 280 if currency == "PKR" else 1
    when = date(2025, 1, 1) + timedelta(days=rng.randint(0, 270))
    items = _line_items(rng, GROCERY, rng.randint(2, 5), [1, 1, 1, 2, 3], scale)
    subtotal = round(sum(it["amount"] for it in items), 2)
    tax = round(subtotal * tax_rate, 2)
    tip = round(subtotal * 0.1, 2) if has_tip else None
    total = round(subtotal + tax + (tip or 0), 2)
    number = f"{rng.randint(100000, 999999)}"
    payment = rng.choice(["VISA **** 4821", "MASTERCARD **** 1937", "CASH"])

    width = 80 * mm
    height = (110 + 7 * len(items)) * mm
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(width, height))
    y = height - 12 * mm

    def row(text: str, size: int = 9, bold: bool = False, value: str | None = None) -> None:
        nonlocal y
        c.setFont(BOLD if bold else REGULAR, size)
        if value is None:
            c.drawCentredString(width / 2, y, text)
        else:
            c.drawString(6 * mm, y, text)
            c.drawRightString(width - 6 * mm, y, value)
        y -= size + 5

    sym = SYMBOL[currency]
    date_fmt = "%m/%d/%Y" if currency == "USD" else "%d/%m/%Y"
    row(merchant, 13, bold=True)
    row(address, 8)
    row(f"Receipt #{number}", 8)
    row(
        f"Date: {when.strftime(date_fmt)}  Time: {rng.randint(8, 21):02d}:{rng.randint(0, 59):02d}",
        8,
    )
    row("-" * 42, 8)
    for it in items:
        label = it["description"]
        if it["quantity"] != 1:
            label = f"{it['quantity']:g} x {label} @ {it['unit_price']:.2f}"
        row(label, 8, value=f"{it['amount']:.2f}")
    row("-" * 42, 8)
    row("SUBTOTAL", 9, value=f"{sym}{subtotal:,.2f}")
    row(f"TAX {tax_rate * 100:g}%", 9, value=f"{sym}{tax:,.2f}")
    if tip is not None:
        row("TIP", 9, value=f"{sym}{tip:,.2f}")
    row("TOTAL", 11, bold=True, value=f"{sym}{total:,.2f}")
    row(f"Paid by {payment}", 8)
    row("Thank you for shopping with us!", 8)
    c.showPage()
    c.save()
    truth = {
        "document_type": "receipt",
        "merchant_name": merchant,
        "receipt_number": number,
        "transaction_date": when.isoformat(),
        "currency": currency,
        "line_items": items,
        "subtotal": subtotal,
        "tax": tax,
        "tip": tip,
        "total": total,
        "payment_method": payment,
    }
    return Sample(f"receipt_{i:02d}", "receipt", buf.getvalue(), truth)


def _letter(rng: random.Random, i: int) -> Sample:
    sender = VENDORS[(i + 3) % len(VENDORS)][0]
    body = [
        sender,
        "",
        date(2025, 6, 2 + i).strftime("%B %d, %Y"),
        "",
        f"Dear {rng.choice(CUSTOMERS)} team,",
        "",
        "We are writing to confirm the kickoff meeting for the platform migration",
        "project. The agenda covers scope, milestones and the communication plan.",
        "Please bring a list of stakeholders and any compliance requirements.",
        "",
        "Meeting location: Conference Room B",
        "Duration: 90 minutes",
        "",
        "Kind regards,",
        "Project Management Office",
    ]
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=A4)
    _, height = A4
    c.setFont(REGULAR, 11)
    for n, text in enumerate(body):
        c.drawString(25 * mm, height - (30 + 7 * n) * mm, text)
    c.showPage()
    c.save()
    return Sample(f"letter_{i:02d}", "other", buf.getvalue(), {"document_type": "other"})


def scan(pdf: bytes, rng: random.Random, dpi: int = SCAN_DPI) -> Image.Image:
    """Rasterise a PDF page and simulate a scanner capture (rotation, blur, noise)."""
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        pix = doc[0].get_pixmap(dpi=dpi, alpha=False)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    angle = rng.uniform(1.0, 3.5) * rng.choice([-1, 1])
    image = image.rotate(
        angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(245, 245, 240)
    )
    image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 0.9)))
    array = np.asarray(image).astype(np.float32)
    array = array * rng.uniform(0.88, 0.97) + rng.uniform(5, 15)  # dull paper / exposure
    noise_rng = np.random.default_rng(rng.randint(0, 2**31))
    array += noise_rng.normal(0, rng.uniform(6, 12), array.shape)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def _image_pdf(image: Image.Image, dpi: int = SCAN_DPI) -> bytes:
    """Wrap an image in a PDF with no text layer (like a scanner's PDF output)."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    with pymupdf.open() as doc:
        page = doc.new_page(width=image.width * 72 / dpi, height=image.height * 72 / dpi)
        page.insert_image(page.rect, stream=buffer.getvalue())
        return doc.tobytes(deflate=True)


def generate_dataset(
    out_dir: Path, seed: int = 7, n_invoices: int = 8, n_receipts: int = 5, n_other: int = 2
) -> list[dict[str, Any]]:
    """Write documents and `ground_truth.json` to `out_dir`; return the manifest."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = [_invoice(rng, i) for i in range(n_invoices)]
    samples += [_receipt(rng, i) for i in range(n_receipts)]
    samples += [_letter(rng, i) for i in range(n_other)]
    manifest: list[dict[str, Any]] = []
    for n, sample in enumerate(samples):
        digital = f"{sample.doc_id}_digital.pdf"
        (out_dir / digital).write_bytes(sample.pdf)
        manifest.append({"file": digital, "variant": "digital", **sample.truth})
        if sample.kind == "other":
            continue
        image = scan(sample.pdf, rng)
        if n % 2 == 0:
            scanned = f"{sample.doc_id}_scanned.pdf"
            (out_dir / scanned).write_bytes(_image_pdf(image))
        else:
            scanned = f"{sample.doc_id}_scanned.jpg"
            image.save(out_dir / scanned, quality=75)
        manifest.append({"file": scanned, "variant": "scanned", **sample.truth})
    (out_dir / "ground_truth.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
