### Porting LangGraph Agents to DSPy: A Practical, No-Nonsense Guide

This guide walks a junior developer through porting a LangGraph agent to DSPy, using the existing customer support agent as the working example. It covers where to look in the LangGraph project, how to map those concepts into DSPy, conversation history, tools, prompts, and assets like documentation files.

---

### 1) Read the source: what to inspect in LangGraph

Start by skimming these files to understand behavior and interfaces:

- `langgraph/customer_support_agent/src/agent/graph.py`: Orchestration, state shape, LLM binding, routing.
- `langgraph/customer_support_agent/src/agent/tools.py`: Tool signatures and business logic.
- `langgraph/customer_support_agent/src/agent/models.py`: Pydantic models (input/output schemas).
- `langgraph/customer_support_agent/src/agent/documentation/*.txt`: Local docs used by tools (agentic file reading).
- `langgraph/customer_support_agent/run_graph.py`: Minimal “how to call” example.
- `langgraph/customer_support_agent/README.md`: Feature overview and architecture notes.

Key LangGraph structures to mirror in DSPy:

```53:62:langgraph/customer_support_agent/src/agent/graph.py
class CustomerServiceState(TypedDict):
    # Core state
    messages: Annotated[List[AnyMessage], add_messages]

    # Required inputs
    customer_email: str
    order_id: str
```

```302:312:langgraph/customer_support_agent/src/agent/graph.py
# Define the graph
graph = (
    StateGraph(CustomerServiceState, context_schema=Context)
    .add_node("llm_call", llm_call)
    .add_node("tool_node", tool_node)
    .add_edge(START, "llm_call")
    .add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    .add_edge("tool_node", "llm_call")
    .compile(name="Customer Service Agent")
)
```

```160:171:langgraph/customer_support_agent/src/agent/graph.py
# Define tools
tools = [
    search_documentation_tool,
    search_orders_tool,
    refund_customer_tool,
    get_order_status_tool,
]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)
```

What this tells you:
- State keeps `messages` (conversation) and inputs like `customer_email`, `order_id`.
- The LLM step decides if tools are needed; a conditional edge routes to `ToolNode`.
- Tools are bound to the LLM for function calling.

---

### 2) Concept mapping: LangGraph → DSPy

| Concept | LangGraph | DSPy |
| --- | --- | --- |
| Conversation state | `messages: Annotated[..., add_messages]` | `dspy.History` (explicitly passed) |
| Orchestration | `StateGraph` nodes/edges | `dspy.ReAct` loop (`max_iters`) |
| Tool binding | `llm.bind_tools([...])` | Pass Python functions in `tools=[...]` |
| Prompt | `SystemMessage(...)` with context | Signature docstring + input fields |
| Inputs | `customer_email`, `order_id` | Signature input fields; pass through wrapper |
| Relevance filtering | `is_relevant_query()` before LLM | Same function before invoking ReAct |
| Models | Pydantic models | Optional; use Python types or keep simple |

Standard policy:
- Whenever an agent needs tools (function calls), use DSPy ReAct to handle tool-calling loops rather than manual orchestration.
- Always include a `history: dspy.History` input in your Signature and pass a maintained `History(messages=[])` each turn.

---

### 3) Copy assets: keep local files in sync

- Copy the documentation files from:
  - `langgraph/customer_support_agent/src/agent/documentation/*.txt`
- To:
  - `dspy/customer_support_agent/documentation/*.txt`

The DSPy notebook expects docs here:
- `dspy/customer_support_agent/customer-support-agent.ipynb` → Cell 2 sets `DOCS_DIR = BASE_DIR / "documentation"`.

Do not rename categories; the simple heuristics rely on the same names: `shipping`, `returns`, `products`, `account`, `payment`.

---

### 4) Build the DSPy scaffold

Open `dspy/customer_support_agent/customer-support-agent.ipynb` and track these pieces.

#### 4.1 Configure the language model
- See Cell 1 for LM configuration. Match your provider (Anthropic, OpenAI, etc.) via `dspy.LM(...)` and `dspy.configure(lm=lm)`.
- Ensure `ANTHROPIC_API_KEY` (or relevant) is set via `.env`.

#### 4.2 Conversation history

- Start a persistent history for the session:
  - Cell 1 creates: `conversation_history = History(messages=[])`
- Pass `conversation_history` directly to components; no extra sanitization needed.

How this maps:
- LangGraph’s `add_messages` automatically aggregates; in DSPy you explicitly maintain `History` and pass it each call.

Preferred pattern for chat:
- Own `History` inside your `dspy.Module` (e.g., `self.history = dspy.History(messages=[])`) and maintain it there for continuity across turns.
- Avoid global history variables; expose a simple `chat(text)` that forwards into your module.
- Avoid naming collisions: don't name your module attribute `history` if your framework/tools expect `module.history` to be a list; prefer `self.conversation_history` instead and pass it to Signatures as the `history` input field.

#### 4.3 Tools (plain functions)

Port each `@tool` into a normal Python function. See Cell 3 in the notebook for the working versions:

- `doc_search(query: str, category: str = "auto") -> str`
- `search_orders(customer_email: str = "", order_id: str = "") -> str`
- `refund(order_id: str, reason: str = "Customer request") -> str`
- `get_status(order_id: str) -> str`

They mirror the logic in `tools.py`:
- Database lookups
- Refund rules
- Documentation loading and truncation for large files

Keep function names concise; these names will be referenced by the ReAct planner.

#### 4.4 Relevance filtering

Copy the same heuristic:
- See Cell 2 `is_relevant_query(query: str) -> bool`
- Call it before invoking the model to short-circuit irrelevant questions.

#### 4.5 Signature and ReAct agent

Use a Signature with a docstring that replaces LangGraph’s system prompt. For research-style agents, prefer DSPy ReAct to handle tool calling loops instead of manual orchestration. See Cell 5:

```python
class SupportReActSignature(dspy.Signature):
    """
    You are a professional customer service representative for TechStore, an e-commerce company specializing in electronics and tech accessories.
    Use provided customer context to personalize responses.
    Guidelines: Verify order IDs before refunds; use documentation for policy info; search orders for lookups; use status for quick checks; escalate politely when needed.

    You can call tools: doc_search, search_orders, refund, get_status.
    When finished, produce:
    - `action`: the primary tool used (one of: doc_search, search_orders, refund, get_status, answer_direct)
    - `tool_result`: the most relevant tool output you used (may be empty for answer_direct)
    - `answer`: the final customer-facing reply
    Keep responses concise and professional.
    """
    user_message: str = dspy.InputField(description="The customer's message")
    customer_email: str = dspy.InputField(description="Customer email if available")
    order_id: str = dspy.InputField(description="Order ID if available")
    history: dspy.History = dspy.InputField(description="Conversation history")

    reasoning: str = dspy.OutputField(description="Brief plan and justification")
    action: str = dspy.OutputField(description="Chosen action/tool")
    tool_result: str = dspy.OutputField(description="Tool output used to answer")
    answer: str = dspy.OutputField(description="Final answer")
```

Instantiate ReAct with tools:
```python
react_agent = dspy.ReAct(
    SupportReActSignature,
    tools=[doc_search, search_orders, refund, get_status],
    max_iters=3,
)
```

Notes:
- `max_iters` handles looping over tool calls.
- The Signature docstring is your “system prompt” replacement. Put the important policies there (role, guidelines, rejection criteria).
- Input fields mirror the context you want available.

For deep research agents (e.g., `open_deep_research`), implement a single ReAct Signature (e.g., `ResearchReActSignature`) and provide tools like `openai_search` and `think_tool`. Avoid building multi-phase orchestration; let ReAct plan tool usage.

#### 4.6 Module-based wrapper with history (preferred)

Wrap your ReAct agent in a `dspy.Module` and have the module own its `History` to keep chat state consistent across turns and make reuse straightforward:

```python
class MyAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.history = dspy.History(messages=[])
        self.react = dspy.ReAct(MySignature, tools=[...], max_iters=3)

    def forward(self, user_message: str):
        # Append user message
        self.history.messages.append({"role": "user", "content": user_message})
        # Run ReAct with internal history
        result = self.react(user_message=user_message, history=self.history)
        # Append assistant answer
        answer = getattr(result, "answer", "")
        if isinstance(answer, str) and answer.strip():
            self.history.messages.append({"role": "assistant", "content": answer})
        return result
```

For simple cases, expose a convenience `chat(text)` that calls your module; the module maintains `History` internally.

See Cell 5 `run_support_agent(...)`. Example:

```python
def run_support_agent(user_message: str, customer_email: str = "", order_id: str = "") -> dict:
    if not isinstance(user_message, str) or not user_message.strip():
        return {"answer": "", "action": "reject", "tool_result": ""}

    if user_message and not is_relevant_query(user_message):
        rejection = (
            "I'm sorry, but I can only assist with e-commerce related inquiries such as order status, "
            "product information, shipping, returns, refunds, account and payment issues."
        )
        conversation_history.messages.append({"role": "user", "content": user_message})
        conversation_history.messages.append({"role": "assistant", "content": rejection})
        return {"answer": rejection, "action": "reject", "tool_result": ""}

    conversation_history.messages.append({"role": "user", "content": user_message})

    result = react_agent(
        user_message=user_message,
        customer_email=customer_email,
        order_id=order_id,
        history=conversation_history,
    )

    answer = getattr(result, "answer", "")
    if isinstance(answer, str) and answer.strip():
        conversation_history.messages.append({"role": "assistant", "content": answer})

    return {
        "answer": getattr(result, "answer", ""),
        "action": getattr(result, "action", ""),
        "tool_result": getattr(result, "tool_result", ""),
    }
```

#### 4.7 Commenting conventions

- Do not add comments that state code was "ported from LangGraph". Keep comments focused on intent and behavior in the DSPy implementation.

---

### 5) Minimal working test

Mirror the LangGraph input shape and call your wrapper. See Cell 6:

```python
input_payload = {
    "messages": [{"role": "human", "content": "What's the status of order ORD-001?"}],
    "customer_email": "john.doe@email.com",
    "order_id": "ORD-001",
}

user_text = input_payload["messages"][0]["content"]
resp = run_support_agent(
    user_message=user_text,
    customer_email=input_payload["customer_email"],
    order_id=input_payload["order_id"],
)
print(resp)
```

Expected output includes `action` (e.g., `get_status`), the `tool_result`, and a customer-facing `answer`.

---

### 6) What to preserve from LangGraph

- Tools behavior: parameters, names, and output formatting. The ReAct agent learns tool usage via names and docstring context.
- Relevance guard: preserve the same heuristics.
- Inputs: keep `customer_email` and `order_id`; they’re part of your happy path for orders/status/refunds.
- Documentation assets: same category names and content files.
- Tone and policy: recreate prompt content in the Signature docstring.

---

### 7) Environment and configuration

- `.env` remains the source of truth for API keys. Don’t change existing keys.
- DSPy LM configuration happens once (Cell 1). Keep it simple and targeted to the model you’ll use.
- If you’re using Anthropic, keep `temperature` low for consistent tool decisions, just like LangGraph did.
- For OpenAI reasoning models, set `temperature=1.0` and `max_tokens >= 16000` when creating the LM, e.g. `dspy.LM('openai/gpt-5', temperature=1.0, max_tokens=16000)`.

---

### 8) Troubleshooting and gotchas

- Empty content blocks: avoid pushing empty strings into `History`.
- Tool name mismatches: the ReAct planner uses the names you provide. If you rename a tool, update the Signature docstring and the `tools=[...]` list.
- Missing docs: if documentation files aren’t copied to `dspy/customer_support_agent/documentation`, `doc_search` will return “not found” and degrade quality.
- Over-long docs: `doc_search` trims content. This mirrors the LangGraph warning/truncation approach.
- Max iterations: if the agent loops unnecessarily, reduce `max_iters` or tighten the prompt within the Signature.
- History attribute collision: don’t name a module attribute `history`; some internals expect a list-like object there. Use `self.conversation_history = dspy.History(messages=[])` and pass it to Signatures as `history`.
- dspy.Predict signature format: when using inline predictors, use a valid mapping string, e.g., `dspy.Predict("question -> answer")`, not just `"answer"`.

---

### 9) Quick porting checklist

- Copy `documentation/*.txt` to `dspy/customer_support_agent/documentation/`.
- Recreate tools as plain functions with the same behavior.
- Implement `is_relevant_query` and the `_classify_query_to_category` heuristic.
- Create a `dspy.Signature` whose docstring mirrors the LangGraph system prompt.
- Build a `dspy.ReAct` agent with `tools=[...]` and `max_iters` (use ReAct for any tool-using agents).
- Include `history: dspy.History` in your Signature.
- Prefer a Module-based wrapper that owns `self.conversation_history = dspy.History(messages=[])` and updates it in `forward`.
- Call your agent via `agent(user_message="...")` repeatedly to maintain chat continuity.
- Avoid comments like “ported from LangGraph”.
- For OpenAI reasoning models, use `temperature=1.0` and `max_tokens >= 16000` in `dspy.LM(...)`.

---

### 10) Where to look in this repo

- LangGraph orchestration: `langgraph/customer_support_agent/src/agent/graph.py`
- LangGraph tools: `langgraph/customer_support_agent/src/agent/tools.py`
- LangGraph doc assets: `langgraph/customer_support_agent/src/agent/documentation/*.txt`
- DSPy notebook and implementation: `dspy/customer_support_agent/customer-support-agent.ipynb` (Cells 1–8)
- DSPy doc assets: `dspy/customer_support_agent/documentation/*.txt`

---

This file includes everything necessary for a junior developer to port an AI agent from LangGraph to DSPy in this repo.
