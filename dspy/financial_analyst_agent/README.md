# Financial Analyst Agent

## What this agent does
- Answers finance questions by combining Yahoo Finance data with code-based analysis
- Uses ReAct to call tools: price/history fetchers and a ProgramOfThought analysis step
- Produces concise numeric results and a final textual answer; maintains brief conversation history
- Inputs: `user_message` (string)
- Outputs: `answer` (string), plus `action` and `tool_result` for traceability

---

## High-level workflow
```
[Start: Provide user_message]
     |
     v
[ReAct: choose tool]
     |  
     +--> [Tool: yf_get_price]
     |          |
     |          v
     |     [Tool result]
     |
     +--> [Tool: yf_get_history]
     |          |
     |          v
     |     [Tool result]
     |
     +--> [Tool: run_finance_analysis (ProgramOfThought)]
     |          |
     |          v
     |     [Code execution → numeric answer]
     |
     v
[Compose final answer]
     |
     v
[Output: action, tool_result, answer]
```
- Inputs: `user_message`
- Outputs: `action`, `tool_result`, `answer`
- External services: OpenAI models, Yahoo Finance (yfinance), Deno for ProgramOfThought execution

---

## Components and external services
- LLMs: `openai/gpt-5-mini`
- Env: `OPENAI_API_KEY`, `DENO_CONFIG` (points to workspace `deno.json`)
- Tools/APIs: `yfinance` (Yahoo Finance)
- Dependencies (pip): `dspy`, `yfinance`, `python-dotenv` (optional)
- Notes: Ensure `deno` is installed and on PATH; notebook sets `DENO_CONFIG` to project `deno.json`.

---

## Copy‑paste prompts

### 1) ProgramOfThought generation prompt (FinanceAnalysis)
```
System/Instructions:
Write Python code to analyze provided financial data/questions and output the final numeric or textual result. Keep code minimal and safe; print only the final result.

Inputs:
- question (string)
- context (string)

Output:
- answer (string)
```

### 2) ReAct system schema (FinanceReActSignature)
```
System/Instructions:
You are a financial analyst. Use tools to fetch market data and perform deterministic analyses.

Tools available:
- yf_get_price(ticker: str) -> str  # latest price summary
- yf_get_history(ticker: str, period: str = "5d", interval: str = "1d") -> str  # recent OHLCV
- run_finance_analysis(question: str, context: str = "") -> str  # ProgramOfThought analysis

Behavior:
- For data retrieval, call yf_get_price / yf_get_history.
- For calculations, comparisons, or aggregations, call run_finance_analysis with a concise question and include retrieved context.
- Finish with a concise, well-structured answer.

Outputs:
- action: primary tool used (one of: yf_get_price, yf_get_history, run_finance_analysis, answer_direct)
- tool_result: the most relevant tool output used
- answer: final financial analysis answer
```

---

## Scoring/aggregation (code)
JS:
```javascript
// Example: compute percentage return
function pctReturn(startClose, endClose) {
  if (typeof startClose !== 'number' || typeof endClose !== 'number') return 0;
  return ((endClose / startClose) - 1) * 100;
}
```

Python:
```python
def pct_return(start_close: float, end_close: float) -> float:
    if not isinstance(start_close, (int, float)) or not isinstance(end_close, (int, float)):
        return 0.0
    return ((end_close / start_close) - 1.0) * 100.0
```

---

## Implementation steps
1) Install: `pip install dspy yfinance python-dotenv`
2) Configure LM: `dspy.configure(lm=dspy.LM("openai/gpt-5-mini", api_key=OPENAI_API_KEY, temperature=1, max_tokens=16000))`
3) Ensure `deno` is installed and set `DENO_CONFIG` to the repo `deno.json`
4) Implement tools:
   - `yf_get_price(ticker)` – compact latest price (uses `Ticker().fast_info` or `history()` fallback)
   - `yf_get_history(ticker, period, interval)` – compact last ≤10 rows OHLCV string
   - `run_finance_analysis(question, context)` – wraps `ProgramOfThought(FinanceAnalysis)`
5) Define ReAct signature `FinanceReActSignature` and `FinancialAnalystAgent` module
6) Instantiate agent and call with a user request; inspect `action`, `tool_result`, `answer`

---

## Example I/O
- Input: `"Compare AAPL and MSFT 5-day performance and provide the numeric returns."`
- Output (abridged):
  - action: `run_finance_analysis`
  - tool_result: multi-line numeric summary with computed returns
  - answer: `"Summary (5 trading days ... AAPL outperformed MSFT by 2.04pp)."`

---

## Files
- `dspy/financial_analyst_agent/financial_analyst.ipynb`
- `deno.json` (project root)

