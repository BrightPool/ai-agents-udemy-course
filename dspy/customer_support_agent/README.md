# Customer Support Agent

## What this agent does
- Handles e‑commerce support queries (status, refunds, returns, docs) with a ReAct agent
- Calls internal tools: documentation lookup, mock order search, refund, status
- Enforces a concise, professional tone and rejects irrelevant (non‑ecommerce) prompts
- Inputs: `user_message`, optional `customer_email`, `order_id`
- Outputs: `answer` (customer‑facing), plus `action` and `tool_result`

---

## High-level workflow
```
[Start: Provide user_message (+ optional email/order_id)]
     |
     v
[Guard: ecommerce relevance check]
     |
     v
[ReAct: choose tool]
     |  
     +--> [doc_search]
     |          |
     |          v
     |     [Policy/doc text]
     |
     +--> [search_orders]
     |          |
     |          v
     |     [Order details]
     |
     +--> [get_status]
     |          |
     |          v
     |     [Status summary]
     |
     +--> [refund]
     |          |
     |          v
     |     [Refund outcome]
     |
     v
[Compose final reply]
     |
     v
[Output: action, tool_result, answer]
```
- Inputs: `user_message`, optionally `customer_email`, `order_id`
- Outputs: `action`, `tool_result`, `answer`
- External services: none (uses local mock DB and local .txt docs)

---

## Components and external services
- LLMs: `openai/gpt-5-mini` (configured via DSPy)
- Env: `OPENAI_API_KEY` (or Anthropic variant if swapped; current notebook config uses OpenAI)
- Local assets: `dspy/customer_support_agent/documentation/*.txt` (shipping/returns/products/account/payment)
- Dependencies (pip): `dspy`, `python-dotenv`
- Notes: History is maintained with `dspy.History` inside the session.

---

## Copy‑paste prompts

### 1) ReAct system schema (SupportReActSignature)
```
System/Instructions:
You are a professional customer service representative for TechStore, an e-commerce company specializing in electronics and tech accessories. Use provided customer context to personalize responses.
Guidelines: Verify order IDs before refunds; use documentation for policy info; search orders for lookups; use status for quick checks; escalate politely when needed.

Tools available:
- doc_search(query: string, category: string = "auto") -> string
- search_orders(customer_email: string = "", order_id: string = "") -> string
- refund(order_id: string, reason: string = "Customer request") -> string
- get_status(order_id: string) -> string

Outputs:
- action: primary tool used (one of: doc_search, search_orders, refund, get_status, answer_direct)
- tool_result: most relevant tool output used (may be empty for answer_direct)
- answer: final customer-facing reply (concise, professional)
```

### 2) Guard / relevance heuristic (concept)
```
Reject if the user_message is not ecommerce-related (keywords: order, purchase, delivery, tracking, return, refund, exchange, product, warranty, account, payment, shipping, etc.).
Return: {"answer": rejection_message, "action": "reject", "tool_result": ""}
```

---

## Scoring/aggregation (code)
JS:
```javascript
// Example: simple template builder for support answers
function supportReply(header, bodyLines) {
  return `${header}\n\n${bodyLines.join("\n")}`.trim();
}
```

Python:
```python
# Example: format an order status snippet from tool data

def format_status_snippet(status_block: str) -> str:
    return status_block.strip()
```

---

## Implementation steps
1) Install: `pip install dspy python-dotenv`
2) Configure LM: `dspy.configure(lm=dspy.LM("openai/gpt-5-mini", api_key=OPENAI_API_KEY, temperature=1, max_tokens=16000))`
3) Prepare local docs under `documentation/` (shipping, returns, products, account, payment)
4) Implement tools:
   - `doc_search(query, category="auto")` – loads and trims doc text; prefixes header and appends the user query
   - `search_orders(customer_email, order_id)` – queries mock `ORDERS_DATABASE`
   - `get_status(order_id)` – returns compact order status
   - `refund(order_id, reason)` – eligibility check and state update
5) Build the ReAct agent with `SupportReActSignature` and the four tools; keep `max_iters` small (e.g., 3)
6) Add an ecommerce relevance check before invoking ReAct; maintain `dspy.History`
7) Call `run_support_agent(user_message, customer_email?, order_id?)` and return `{answer, action, tool_result}`

---

## Example I/O
- Input: `"What's the status of order ORD-001?"` (+ email `john.doe@email.com`)
- Output (abridged):
  - action: `get_status`
  - tool_result: `"Order ID: ORD-001\nStatus: Delivered\n..."`
  - answer: `"Hi John — I checked order ORD-001... Delivered on 2024-01-18. If you haven’t received it..."`

---

## Files
- `dspy/customer_support_agent/customer-support-agent.ipynb`
- `dspy/customer_support_agent/documentation/*.txt`
