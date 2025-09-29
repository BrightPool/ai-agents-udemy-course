# Life Coach Agent (DSPy + Mem0)

## What this agent does
- Provides brief, practical executive‑style coaching replies using a fixed persona and frameworks
- Retrieves long‑term memories from a local Mem0 API and maintains short‑term session context
- Uses a ReAct loop with tools to search/add memories and access recent turns
- Returns a concise coach reply per turn; simple demo shows a short conversation

---

## High-level workflow
```
[Start: Provide user_id, session_id, user_message]
      |
      v
[Code: Build composite prompt]
  SYSTEM persona + recent session turns + Mem0 search results + USER message
      |
      v
[ReAct (CoachSignature): plan and select tools]
      |
      +--> [Tool: search_memories(user_id, query, k)]
      |          |
      |          v
      |     [Mem0 REST: /api/v1/memories/search]
      |
      +--> [Tool: add_memory(user_id, text)] (optional)
      |          |
      |          v
      |     [Mem0 REST: /api/v1/memories]
      |
      +--> [Tool: session_context(session_id)]
      |
      +--> [Tool: append_session(session_id, role, text)]
      |
      v
[Output: reply]
```
- Inputs: `user_id`, `session_id`, `user_message`
- Outputs: `reply` (string)
- External services: OpenAI LLM; local Mem0 stack (API + Qdrant + Neo4j)

---

## Components and external services
- LLMs: `openai/gpt-5-mini`
- Env: `OPENAI_API_KEY`, optional `MEM0_BASE_URL` (default `http://localhost:8000`)
- Dependencies (pip): `dspy`, `python-dotenv`, `requests`
- Local services (Docker): Mem0 API, Qdrant, Neo4j (see `docker-compose.yml`)
- Notes: Do not change API keys in `.env`. Start Mem0 via Docker before running the notebook.

---

## Copy‑paste prompts

### 1) ReAct system schema (CoachSignature)
```
System/Instructions:
You are a highly sought‑after executive coach and psychologist.
Background: mountain guide, BASE jumper, founder of One Day Coaching.
Philosophy: transformative change; presence, gratitude, reflection; self‑leadership and resilience.
Frameworks: Circle of Control, Helicopter View, Time Jump, 1–10 scale, Minimal Success Index.
Communication: brief, sharp, practical; prefer 1–2 sentences; ask a powerful question if unclear; pick one actionable idea tied to the user's query.

Inputs:
- user_id (string): caller identity for memory operations
- session_id (string): short‑term context key
- user_input (string): composite prompt (persona + context + memories + user text)
- history (History): conversation history across turns

Available tools:
- search_memories(user_id: string, query: string, k: int)
- add_memory(user_id: string, text: string)
- session_context(session_id: string)
- append_session(session_id: string, role: string, text: string)

Output:
- output (string): the coach reply
```

### 2) Composite coaching prompt (constructed in code)
```
SYSTEM:
<Persona text above>

CONTEXT (recent turns):
<windowed session transcript>

MEMORIES:
<Mem0 search results>

USER:
<current user_message>
```

---

## Scoring/aggregation (code)
JS:
```javascript
// Extract the assistant's reply from a result-like object
function coachReply(result) {
  return (result && (result.reply || result.output || "")) || "";
}
```

Python:
```python
# Extract the assistant's reply from a result-like object
def coach_reply(result) -> str:
    if isinstance(result, dict):
        return result.get("reply") or result.get("output") or ""
    return getattr(result, "reply", getattr(result, "output", ""))
```

---

## Implementation steps
1) Install: `pip install dspy python-dotenv requests`
2) Configure LM:
```
import os, dspy
from dotenv import load_dotenv
load_dotenv()
lm = dspy.LM("openai/gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=1, max_tokens=16000)
dspy.configure(lm=lm)
```
3) Start Mem0 stack locally (from `dspy/life_coach_agent/`):
```
export OPENAI_API_KEY=sk-...
docker compose up -d
curl -s http://localhost:8000/ | cat
```
4) Implement Mem0 REST helpers (already in the notebook): `mem0_search`, `mem0_add_memory`
5) Define tools: `tool_search_memories`, `tool_add_memory`, `tool_get_session_context`, `tool_append_session_turn`
6) Define ReAct agent with `CoachSignature` and `max_iters` (e.g., 6)
7) Use `CoachAgentModule` to manage `dspy.History` and session buffer per turn
8) Call the agent per user message and return `reply`

---

## Example I/O
- Input (turns):
  - "I’m training for a 50k trail race but losing motivation after work."
  - "I also want to improve sleep without sacrificing morning runs."
  - "I felt anxious today about a missed workout."
- Output (abridged replies):
  - "Do a 20‑minute MSI run/walk after work—no pace targets... On a scale of 1–10, how committed are you to doing this tonight?"
  - "Pick a fixed wake time... 45‑minute wind‑down... What time do you need to wake for your morning runs?"
  - "Do a 20‑minute recovery now... On a scale of 1–10, how committed are you to doing this in the next hour?"

---

## Files
- `dspy/life_coach_agent/life-coach-agent.ipynb`
- `dspy/life_coach_agent/docker-compose.yml`


