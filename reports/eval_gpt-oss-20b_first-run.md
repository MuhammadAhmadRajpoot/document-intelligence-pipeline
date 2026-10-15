# First evaluation run: openai/gpt-oss-20b

These are the tables printed by `docintel eval` for the first full run on 2026-10-15
(28 documents, concurrency 2, wall time 411.7 s, CPU-only Windows 11 laptop with an
Intel Core i7-7500U, 2 cores / 4 threads). The JSON report from this run was deleted by
mistake before a re-run, so only the printed tables below are kept. The re-run itself could
not finish because the key's Groq daily token quota for this model ran out.

The run used an earlier commit. Since then:

- HTTP 429 handling was added. In this run, `receipt_02_scanned.pdf` failed because all
  three extraction attempts were rejected by the per-minute token limit.
- Grounding was improved (row-label hints, fuzzy matching, whole-word currency aliases).
  This changes confidence scores and review flags, not the extracted values.

Remaining real error: `invoice_07_scanned.jpg` got a wrong `vendor_name` (OCR).
Correct receipts flagged for review: `receipt_03_*`, because the PKR currency was printed
as "Rs", which the grounding at that commit could not locate.

| Slice | Docs | Field accuracy | Classification | Docs fully correct | Flagged for review | Mean latency |
|---|---|---|---|---|---|---|
| All | 28 | 96.0% | 100.0% | 92.3% | 10.7% | 29.06 s |
| digital (variant) | 15 | 100.0% | 100.0% | 100.0% | 6.7% | 23.9 s |
| scanned (variant) | 13 | 92.0% | 100.0% | 84.6% | 15.4% | 35.01 s |
| invoice (type) | 16 | 99.4% | 100.0% | 93.8% | 0.0% | 33.2 s |
| other (type) | 2 | n/a | 100.0% | n/a | 0.0% | 14.39 s |
| receipt (type) | 10 | 90.0% | 100.0% | 90.0% | 30.0% | 25.36 s |

| Field | Overall | Digital | Scanned |
|---|---|---|---|
| currency | 96.2% | 100.0% | 92.3% |
| customer_name | 100.0% | 100.0% | 100.0% |
| due_date | 100.0% | 100.0% | 100.0% |
| invoice_date | 100.0% | 100.0% | 100.0% |
| invoice_number | 100.0% | 100.0% | 100.0% |
| line_items | 96.2% | 100.0% | 92.3% |
| merchant_name | 90.0% | 100.0% | 80.0% |
| receipt_number | 90.0% | 100.0% | 80.0% |
| subtotal | 96.2% | 100.0% | 92.3% |
| tax | 96.2% | 100.0% | 92.3% |
| tip | 90.0% | 100.0% | 80.0% |
| total | 96.2% | 100.0% | 92.3% |
| transaction_date | 90.0% | 100.0% | 80.0% |
| vendor_name | 93.8% | 100.0% | 87.5% |
