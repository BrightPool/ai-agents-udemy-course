# Deep Researcher (DSPy)

## What this agent does
- Generates concise, well-structured research reports from a user-provided topic or question.
- Uses a simple ReAct loop with two tools: `openai_search` (research via model knowledge) and `think_tool` (reflection/planning).
- Maintains conversation history to keep context across turns.
- Returns structured outputs: `action`, `tool_result`, `answer`.

---

## High-level workflow
```
[Start: Provide user_message]
      |
      v
[LLM (ReAct): Plan next step]
      |
      v
[Tool: openai_search(queries, max_results, topic)]
      |
      v
[LLM: Observe results]
      |
      v
[Tool: think_tool(reflection)]
      |
      v
[LLM: Finalize]
      |
      v
[Output: {action, tool_result, answer}]
```
- Inputs: `user_message` (string)
- Outputs: `action` (string), `tool_result` (string), `answer` (string)
- External services: OpenAI LLM (see below)

---

## Components and external services
- LLMs: `openai/gpt-5-mini`
- Env: `OPENAI_API_KEY`
- Notes: Do not change API keys in `.env`.

---

## Copy‑paste prompts

### 1) Generation prompt
```
You are a deep research assistant. Use tools to gather information (`openai_search` for research, `think_tool` for reflection).
Keep searches focused, reflect after searches, and finish with a concise, well-structured research answer.

When finished, produce:
- action: the primary tool used (one of: openai_search, think_tool, answer_direct)
- tool_result: the most relevant tool output you used (may be empty for answer_direct)
- answer: the final research answer/report

Be clear, factual, and professional.
```

### 2) Evaluation/system schema
```
Inputs
- user_message (str): The user's research request
- history (History): Conversation history (optional)

Intermediate interaction (looped by the agent)
- next_thought (str)
- next_tool_name (one of: openai_search, think_tool, finish)
- next_tool_args (object)

Allowed tools and arguments
1) openai_search
   - queries: string[] (required)
   - max_results: integer (default 5)
   - topic: "general" | "news" (default "general")
2) think_tool
   - reflection: string (required)
3) finish
   - {} (no arguments)

Final outputs
- reasoning (str): brief plan and justification
- action (str): chosen action/tool used most prominently
- tool_result (str): tool output used to answer
- answer (str): final research answer
```

Fixed lists
```
next_tool_name: ["openai_search", "think_tool", "finish"]
openai_search.topic: ["general", "news"]
```

---

## Scoring/aggregation (code)
JS
```javascript
// Minimal aggregation: extract core fields for downstream use
function aggregateResearchResult(result) {
  const action = result?.action ?? "";
  const tool_result = result?.tool_result ?? "";
  const answer = result?.answer ?? "";
  return { action, tool_result, answer };
}
```

Python
```python
# Minimal aggregation: extract core fields for downstream use
def aggregate_research_result(result):
    return {
        "action": getattr(result, "action", ""),
        "tool_result": getattr(result, "tool_result", ""),
        "answer": getattr(result, "answer", ""),
    }
```

---

## Implementation steps
1) Trigger: input `user_message` (string)
2) LLM: Generation (use the prompt above; pass `user_message` and optional `history`)
3) LLM: Tool calls (agent decides):
   - `openai_search({ queries: [...], max_results, topic })`
   - `think_tool({ reflection })`
   - `finish({})`
4) Code: Aggregate the final structured output (`action`, `tool_result`, `answer`)
5) Output: return `answer` (primary), plus `action` and `tool_result` for traceability

---

## Example I/O
- Input: "Summarize the latest approaches to structured reasoning in LLMs."
- Output (abridged):
  - action: `think_tool`
  - tool_result: "Synthesis of collected findings: key approaches are Chain-of-Thought, Self-Consistency, Least-to-Most, PAL, Tree of Thoughts, ReAct/tool-augmented prompting, and verifier/refinement + retrieval pipelines."
  - answer: "Concise summary — latest approaches to structured reasoning in LLMs ..."

---

## Files
- `dspy/open_deep_research/deep-researcher.ipynb`
