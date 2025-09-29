### Guide: Writing Agent READMEs (course-ready, tool-agnostic)

- Audience: Non-technical readers. They should understand the flow and be able to rebuild it in any workflow tool or simple scripts.
- Style: Plain, concise, happy path only. Use ASCII diagrams, avoid platform-specific naming. Always highlight external services and include copy-paste prompts.

## Required sections

1) What this agent does
- 2–5 bullets explaining purpose, inputs, outputs.
- Call out any optimization/feedback loops at a high level.

2) High-level workflow (ASCII diagram)
- Use monospace ASCII so it renders everywhere.
- Example:
```
[Start: Provide Input]
     |
     v
[LLM: Step 1]
     |
     v
[LLM: Step 2]
     |
     v
[Code: Compute Metrics]
     |
     v
[Output: Results]
```
- If an optimization loop exists, show a clear single loop-back arrow.

3) Components and external services
- List LLM models (full identifiers), tools/APIs, and required env vars.
- Example:
  - LLMs: `openai/gpt-5-mini`, `openai/gpt-5`
  - Env: `OPENAI_API_KEY`
  - Notes: Do not change API keys in `.env`.

4) Copy-paste prompts for LLM steps
- Provide exact optimized prompts where available; otherwise include the best approximation derived from signatures.
- Include:
  - Generation prompt (system/instructions).
  - Any evaluator/critic schema prompts (system/instructions) plus required inputs structure.
  - If prompts were optimized, paste the exact optimized instructions printed from the program or notebook outputs.

5) Scoring/aggregation snippet (Code)
- Include a short, copyable code block that computes the main metric(s).
- Provide both JavaScript and Python if helpful.
- Example (JS):
```
const ratingScores = { hilarious: 5, funny: 4, meh: 3, "not funny": 2, offensive: 1 };
const avg = Math.round((responses.reduce((s, r) => s + ratingScores[r], 0) / responses.length) * 100) / 100;
```

6) Implementation steps (tool-agnostic)
- Describe a minimal, linear flow anyone can implement:
  - Trigger/input parameters
  - LLM call #1 with prompt X (and how to pass variables)
  - LLM call #2 with prompt Y (and required input arrays/fields)
  - Code step: compute metrics and optional feedback string
  - Output: what to return/save
- Optional: If there’s an optimization loop, explain it as a simple “propose → evaluate → keep best” loop without naming a specific platform.

7) Example I/O (brief)
- Provide one concise example input and a summarized output shape (not long content).
- Keep it safe and course-appropriate.

8) Files
- Point to key notebook(s), saved program directory (if any), and artifacts like CSVs.

## Extraction tips (DSPy-specific)

- Optimized prompts:
  - Print from the optimized program (in a notebook):
    - Iterate and print: `for name, pred in optimized_program.named_predictors(): print(pred.signature.instructions)`
  - Or load saved program: `loaded = dspy.load("./program_dir/")` then print instructions.
- Evaluator/system schemas:
  - Use `lm.inspect_history(n=1)` to capture system schema if needed.
  - If the evaluator uses fixed profiles/labels, paste those lists verbatim.
- If no optimized prompt exists, derive a clean copy-paste version from the signature docstring + fields.

## Style rules

- ASCII diagrams only (no Mermaid).
- Tool-agnostic language. Avoid naming specific platforms; say “workflow/orchestration tool” or “script step.”
- Single happy path. Don’t add fallbacks or alternate conventions.
- Highlight external services and any costs/keys.
- Clear section headers, brief sentences, bullet lists.
- Keep it immediately actionable for interns: minimal jargon, concrete steps, copy-paste blocks.

## Final checklist (use before publishing)

- What this agent does: concise and accurate
- ASCII diagram renders correctly in monospace
- External services listed (models, env vars)
- Prompts included:
  - Generation prompt (optimized if available)
  - Evaluator/system schema and fixed lists (e.g., personas)
- Scoring snippet present and correct
- Implementation steps are linear and tool-agnostic
- Example I/O included (short)
- Files section points to real paths
- No platform-specific language
- No fallbacks or confusing alternatives

## Starter template (copy and fill)

```
# <Agent Name>

## What this agent does
- <Bullet 1>
- <Bullet 2>
- <Bullet 3>

---

## High-level workflow
<Insert an ASCII diagram in your final README using a fenced code block. Example>
  [Start: Provide <Input>]
       |
       v
  [LLM: <Step 1>]
       |
       v
  [LLM: <Step 2>]
       |
       v
  [Code: <Metric/Aggregation>]
       |
       v
  [Output: <Result>]

- Inputs: <list>
- Outputs: <list>
- External services: <models/APIs>

---

## Components and external services
- LLMs: <provider/model-1>, <provider/model-2>
- Env: <ENV_VAR_1>, <ENV_VAR_2>
- Notes: <keys/costs/limits>

---

## Copy‑paste prompts

### 1) Generation prompt
<Insert your prompt in a fenced code block in the final README>

### 2) Evaluation/system schema
<Insert your schema in a fenced code block in the final README>

Fixed lists (if any):
<Insert list in a fenced code block in the final README>

---

## Scoring/aggregation (code)
JS: <Insert a JS fenced code block>

Python: <Insert a Python fenced code block>

---

## Implementation steps
1) Trigger: <input param(s)>
2) LLM: Generation (use the prompt above; pass variables: <var names>)
3) LLM: Evaluation (use schema above; pass <fields/lists>)
4) Code: Compute metrics and optional feedback
5) Output: return <fields>

Optional (advanced):
- Optimization loop: propose → evaluate → keep best

---

## Example I/O
- Input: <example>
- Output (abridged):
  - <field>: <value>
  - <field>: <value>

---

## Files
- <notebook>.ipynb
- <artifact>.csv
- <saved_program_dir>/
```


