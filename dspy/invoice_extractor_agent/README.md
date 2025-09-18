# Invoice Extractor Agent

## What this agent does
- Extracts a fixed set of invoice fields from free‑form invoice text into a flat JSON object
- Uses a simple DSPy `Predict` module with an `InvoiceExtraction` signature (rationale + extracted dict)
- Evaluates accuracy with a per‑field partial‑credit metric and optionally optimizes prompts with GEPA
- Inputs: `text` (invoice text)
- Outputs: `extracted` (JSON with 8 keys), `rationale` (short reasoning)

---

## High-level workflow
```
[Start: Load dataset / provide text]
     |
     v
[LLM: Base extraction (Predict)]
     |
     v
[Code: Score with per‑field metric]
     |
     v
[LLM: GEPA optimization (optional)]
     |
     v
[Code: Evaluate & save optimized program]
     |
     v
[Output: JSON with 8 required fields]
```
- Inputs: invoice `text`
- Outputs: JSON with 8 keys: `company`, `billed_to`, `invoice_number`, `invoice_date`, `total_amount`, `bank_name`, `account_name`, `account_number`
- External services: OpenAI models

---

## Components and external services
- LLMs: `openai/gpt-5-mini` (base extraction), `gpt-5` (reflection LM for GEPA)
- Env: `OPENAI_API_KEY`
- Data: `dspy/invoice_extractor_agent/invoice_ner_clean.csv`
- Saved program: `dspy/invoice_extractor_agent/invoice_program/`
- Dependencies (pip): `dspy`, `pandas`, `python-dotenv`
- Notes: Do not change API keys in `.env`.

---

## Copy‑paste prompts

### 1) Generation prompt (optimized 8‑field extraction)
```
Task
From free‑form invoice text (plain text), extract a small, fixed set of top‑level invoice fields and return them as a flat JSON object.

Output format (strict)
- Return a single JSON object only (no wrapper, no comments, no rationale).
- Include exactly these 8 keys (all must be present, even if unknown), with string values:
  1) "company"
  2) "billed_to"
  3) "invoice_number"
  4) "invoice_date"
  5) "total_amount"
  6) "bank_name"
  7) "account_name"
  8) "account_number"
- If a field is not explicitly present, set it to "" (empty string).
- Do not include any extra keys or metadata. Do not rename any keys.

General extraction strategy
- Prefer clearly labeled fields; pick the value closest to the relevant label.
- Preserve formatting of values: keep currency symbols/separators, leading zeros and prefixes in IDs, and spaces/dashes in account numbers/IBANs.
- Never fabricate values. If not present, return "".
- For party fields, output only the name (no address/phone/email/URLs).

Field rules (abridged)
1) company (issuer/seller): Prefer explicit issuer/payee blocks (e.g., "From", "Vendor", "PAY TO") or the prominent brand near "INVOICE" (avoid template titles). Prefer company over person when both appear.
2) billed_to (customer/recipient): Look for "Bill To"/"Invoice To"/"Customer"/"Client"; prefer company over person. Do not derive from "PAY TO".
3) invoice_number: Use labels like "Invoice Number"/"Invoice No"/"Invoice #"; keep prefixes/hyphens/zeros.
4) invoice_date: The invoice issue date (not due/service); prefer "Invoice Date"; preserve original formatting.
5) total_amount: Use the grand total ("TOTAL"/"Amount Due"/"Balance Due"); preserve currency formatting.
6) bank_name: From "Bank Name"/"Bank:"; use alphabetic bank name, not codes.
7) account_name: From "Account Name"/"Account Holder"/"Beneficiary".
8) account_number: From "Account Number"/"Account No"/"A/C No"/IBAN; keep spaces/dashes.
```

### 2) Evaluation/system schema
```
Inputs
- text (string): Raw invoice text

Outputs
- rationale (string): Brief reasoning, list of detected fields
- extracted (object): JSON dict with the 8 required keys (string values)

Return the fields in this structured order:
[[ ## text ## ]]
{text}
[[ ## rationale ## ]]
{rationale}
[[ ## extracted ## ]]
{extracted}
[[ ## completed ## ]]
```

### Fixed list (required keys)
```
["company","billed_to","invoice_number","invoice_date","total_amount","bank_name","account_name","account_number"]
```

---

## Scoring/aggregation (code)
JS:
```javascript
// +0.5 for key presence, +0.5 for exact value match; averaged by gold key count
function fieldAccuracyMetric(gold, pred) {
  const keys = Object.keys(gold);
  let present = 0, correct = 0;
  for (const k of keys) {
    if (Object.prototype.hasOwnProperty.call(pred, k)) {
      present += 1;
      if (pred[k] === gold[k]) correct += 1;
    }
  }
  return ((0.5 * present) + (0.5 * correct)) / keys.length;
}
```

Python:
```python
def field_accuracy_metric(gold: dict, pred: dict) -> float:
    keys = list(gold.keys())
    present = sum(1 for k in keys if k in pred)
    correct = sum(1 for k in keys if k in pred and pred[k] == gold[k])
    return ((0.5 * present) + (0.5 * correct)) / max(1, len(keys))
```

---

## Implementation steps
1) Install: `pip install dspy pandas python-dotenv`
2) Configure LM: `dspy.configure(lm=dspy.LM("openai/gpt-5-mini", api_key=OPENAI_API_KEY, temperature=1, max_tokens=16000))`
3) Load CSV dataset (`invoice_ner_clean.csv`) and parse `Final_Output` JSON to dicts
4) Build tiny train/valid splits and convert to `dspy.Example`
5) Define `InvoiceExtraction` signature and base `extractor = dspy.Predict(InvoiceExtraction)`
6) Implement `field_accuracy_metric` and evaluate with `Evaluate`
7) (Optional) Optimize with `GEPA` using a reflection LM (e.g., `gpt-5`) and compile an `optimized_program`
8) Re‑evaluate, then save the program: `optimized_program.save("./invoice_program/", save_program=True)`
9) Inference: call `optimized_program(text=raw_invoice_text)` and read `extracted`

---

## Example I/O
- Input (abridged): `"INVOICE\nAcme Corp\nBill To: Globex\nInvoice No: INV-00123\nInvoice Date: 2024-01-03\nTOTAL: $1,200.00\nBank: Borcelle Bank\nAccount Name: Acme Corp\nAccount No: 0123 4567 8901"`
- Output (abridged):
  - extracted: `{"company":"Acme Corp","billed_to":"Globex","invoice_number":"INV-00123","invoice_date":"2024-01-03","total_amount":"$1,200.00","bank_name":"Borcelle Bank","account_name":"Acme Corp","account_number":"0123 4567 8901"}`

---

## Files
- `dspy/invoice_extractor_agent/invoice-extractor.ipynb`
- `dspy/invoice_extractor_agent/invoice_program/` (saved optimized program)
- `dspy/invoice_extractor_agent/invoice_ner_clean.csv`

