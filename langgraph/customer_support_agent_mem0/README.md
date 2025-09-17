# Customer Service Agent with LangGraph

A sophisticated customer service agent built with LangGraph that demonstrates advanced AI agent patterns including **agentic file reading**, dynamic tool selection, and stateful conversation management. The agent specializes in e-commerce customer support with intelligent query processing and documentation search capabilities.

## How It Works

### Core Architecture

The agent operates through a sophisticated multi-step process:

1. **Query Reception**: Receives customer messages through LangGraph's state management
2. **Relevance Filtering**: Uses pattern matching to reject off-topic queries (e.g., weather, sports)
3. **Tool Selection**: LLM analyzes the query and selects appropriate tools from the available set
4. **Tool Execution**: Executes selected tools with proper error handling and validation
5. **Response Generation**: Combines tool results with conversational context for natural responses

### Agentic File Reading System

The documentation search implements **agentic file reading** - a pattern where the agent dynamically loads and searches through relevant `.txt` files based on query content:

```
User Query: "What's your return policy?"

1. Agent classifies query → "returns" category
2. Agent loads `/documentation/returns.txt` file
3. Agent searches file content for relevant sections
4. Agent returns contextual information to user
```

**Technical Implementation:**

- Files are stored in `src/agent/documentation/` directory
- Each category has its own `.txt` file (shipping.txt, returns.txt, etc.)
- Content is loaded dynamically using `pathlib.Path`
- Keyword-based search finds relevant sections within loaded content

## Features

- **Order Management**: Search orders by email or order ID, check order status
- **Refund Processing**: Process refunds for eligible orders with validation
- **Agentic Documentation Search**: Dynamic loading of `.txt` files based on query classification
- **Smart Query Filtering**: Rejects queries unrelated to e-commerce customer service
- **Conversation Memory**: Maintains context across multiple interactions
- **LangGraph Server**: Runs as a local server with LangGraph Studio integration

## Project Structure

```
customer_support_agent/
├── langgraph.json              # LangGraph configuration and tool definitions
├── pyproject.toml              # Python dependencies and project metadata
├── .env.example               # Environment variables template (API keys)
├── README.md                   # This comprehensive documentation
└── src/
    └── agent/
        ├── __init__.py         # Package initialization
        ├── graph.py           # Main LangGraph workflow and state management
        ├── tools.py           # Customer service tools with agentic file reading
        ├── models.py          # Pydantic models for type safety
        └── documentation/     # Agentic file reading system
            ├── shipping.txt   # Shipping policies and information
            ├── returns.txt    # Return and refund policies
            ├── products.txt   # Product warranties and specifications
            ├── account.txt    # Account management and login
            └── payment.txt    # Payment methods and security
```

## Setup

### Mem0 local server (Docker)

This agent calls a local Mem0 server at `http://localhost:8000`. Spin up Mem0's dependencies with Docker, then run the Mem0 server.

1) Start Qdrant (and optional Neo4j for graph memory) with Docker Compose

```yaml
# docker-compose.mem0.yml
version: "3.8"
services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  neo4j:
    image: neo4j:5
    container_name: neo4j
    ports:
      - "7474:7474"   # HTTP UI
      - "7687:7687"   # Bolt
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs

volumes:
  qdrant_data:
  neo4j_data:
  neo4j_logs:
```

```bash
docker compose -f docker-compose.mem0.yml up -d
# Qdrant now on http://localhost:6333, Neo4j on http://localhost:7474 (bolt://localhost:7687)
```

2) Run the Mem0 server (Docker)

- Clone the Mem0 open-source repository and use its Docker setup (recommended):

```bash
git clone https://github.com/mem0ai/mem0
cd mem0/server
cp .env.example .env
# Edit .env to set your LLM provider (e.g., OPENAI_API_KEY) and vector/graph configs
# Ensure Qdrant host=host.docker.internal, port=6333 (from the compose above)
# For graph memory with Neo4j, set NEO4J_URI=bolt://host.docker.internal:7687 and NEO4J_AUTH

docker compose up -d
# Mem0 API should now be on http://localhost:8000
```

3) Quick health check for Mem0

```bash
curl -s -X POST \
  http://localhost:8000/api/v1/memories/search \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "demo",
    "query": "test",
    "k": 1,
    "enable_graph": true
  }'
```

4) Configure this project to talk to Mem0

Set the environment variables (either export them or add to a `.env` in this project root):

```bash
export MEM0_BASE_URL=http://localhost:8000
export OPENAI_API_KEY=sk-...
```

Notes:
- If you're not using graph memory, you can omit Neo4j entirely; Mem0 will work with just Qdrant.
- See Mem0 docs for full configuration details: `https://docs.mem0.ai`.

### 1. Install uv

```bash
brew install uv
# or:
# pipx install uv
```

### 2. Create and activate a virtual environment

```bash
cd customer_support_agent
uv venv
source .venv/bin/activate
# Tip: You can skip activation and prefix commands with `uv run` instead.
```

### 3. Install dependencies

```bash
uv sync
# Include dev tools (ruff, pytest, etc.):
# uv sync --group dev
```

### 3. Set Up Environment Variables

Create a `.env` file with your API keys:

```bash
cp .env.example .env
# Edit .env and add your API keys
LANGSMITH_API_KEY=lsv2...
ANTHROPIC_API_KEY=sk-ant-api03-...
```

### 4. Launch LangGraph Server

Start the LangGraph API server locally:

```bash
uv run langgraph dev
```

Sample output:

```
>    Ready!
>
>    - API: http://localhost:2024
>
>    - Docs: http://localhost:2024/docs
>
>    - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### 5. Test in LangGraph Studio

Open LangGraph Studio in your browser using the URL provided in the output:

```
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### 6. Test the API

Install the LangGraph Python SDK (dev-only):

```bash
uv add --group dev langgraph-sdk
```

Send a message to the agent:

```python
from langgraph_sdk import get_client
import asyncio

client = get_client(url="http://localhost:2024")

async def main():
    async for chunk in client.runs.stream(
        None,  # Threadless run
        "agent", # Name of assistant. Defined in langgraph.json.
        input={
        "messages": [{
            "role": "human",
            "content": "What's the status of order ORD-001?",
            }],
        },
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")

asyncio.run(main())
```

## Usage Examples

### Order Status Inquiries

```
Input: "What's the status of order ORD-001?"
Output: Order status information with delivery details
```

### Refund Processing

```
Input: "I need a refund for order ORD-002"
Output: Refund processing confirmation with details
```

### Policy Information

```
Input: "What's your return policy?"
Output: Detailed return policy information
```

### Query Rejection

```
Input: "Tell me about the weather today"
Output: Polite redirection to e-commerce topics
```

## Available Tools

### 1. search_documentation_tool

**Purpose**: Implements agentic file reading for documentation search
**Input**: `DocumentationSearchRequest` object with query and category
**Process**:

```python
# Tool signature
def search_documentation_tool(request: DocumentationSearchRequest) -> str:
    # 1. Resolve category (auto-classify if needed)
    # 2. Load .txt file: f"{category}.txt"
    # 3. Search content for relevant sections
    # 4. Return structured JSON response
```

**Categories Supported**: shipping, returns, products, account, payment
**File Location**: `src/agent/documentation/{category}.txt`

### 2. search_orders_tool

**Purpose**: Search customer orders by email or order ID
**Input**: customer_email (optional), order_id (optional)
**Process**:

- Validates that at least one search parameter is provided
- Searches mock database for matching orders
- Returns formatted order information
- Handles not-found cases gracefully

### 3. refund_customer_tool

**Purpose**: Process refunds with validation and business rules
**Input**: order_id (required), reason (optional)
**Business Logic**:

- Verifies order exists in database
- Checks order eligibility (not cancelled, not processing)
- Calculates refund amount from order price
- Updates order status to "refunded"
- Returns confirmation with refund details

### 4. get_order_status_tool

**Purpose**: Quick order status lookup
**Input**: order_id (required)
**Process**:

- Direct database lookup by order ID
- Returns formatted status information
- Includes delivery dates when available

## Technical Architecture

### State Management with LangGraph

The agent uses LangGraph's `StateGraph` for managing conversation state:

```python
# State structure
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_step: str  # routing decision

# Graph nodes
def agent_node(state: AgentState) -> AgentState:
    # LLM processes messages and decides on tool calls

def tool_node(state: AgentState) -> AgentState:
    # Execute selected tools and return results

def route_query(state: AgentState) -> str:
    # Route to appropriate node based on query analysis
```

### Query Processing Pipeline

1. **Input Reception**: Messages enter through LangGraph state
2. **Relevance Check**: `is_relevant_query()` filters off-topic questions
3. **LLM Analysis**: Claude analyzes query intent and context
4. **Tool Selection**: LLM chooses appropriate tools via function calling
5. **Execution**: Tools run with proper error handling
6. **Response**: Results combined with conversation history

### Auto-Classification Logic

The documentation tool uses heuristic classification:

```python
def _classify_query_to_category(query: str) -> str:
    # Keyword matching for categories
    shipping_keywords = ["shipping", "delivery", "tracking"]
    returns_keywords = ["return", "refund", "exchange"]
    # ... pattern matching logic
```

## Mock Data

The agent uses mock data including:

- **Orders Database**: 5 sample orders with different statuses (delivered, shipped, processing, cancelled)
- **Documentation Files**: Separate `.txt` files for each category:
  - `shipping.txt`: Shipping policies, rates, tracking info
  - `returns.txt`: Return policy, process, refund timeline
  - `products.txt`: Warranty, compatibility, specifications
  - `account.txt`: Password reset, order history, profile updates
  - `payment.txt`: Accepted methods, security, billing

## Key Features

### Agentic File Reading

**How it works:**

- Agent receives query: "What's your return policy?"
- Classifies query using keyword matching → "returns"
- Dynamically loads `returns.txt` file from disk
- Searches file content for relevant sections
- Returns contextual information without hardcoded data

**Benefits:**

- Documentation can be updated without code changes
- Supports unlimited documentation categories
- Keyword-based search finds relevant content
- File-based storage is easily maintainable

### Smart Query Filtering

**Implementation:**

```python
def is_relevant_query(query: str) -> bool:
    relevant_keywords = ["order", "shipping", "return", "refund", "product"]
    irrelevant_keywords = ["weather", "sports", "politics"]
    # Pattern matching logic
```

**Rejection Examples:**

- "What's the weather today?" → Rejected
- "How do I track my order?" → Accepted
- "Tell me about baseball" → Rejected

### Professional Prompting

The system prompt includes:

- **Role Definition**: "You are a professional e-commerce customer service agent"
- **Service Catalog**: Explicit list of available tools and capabilities
- **Interaction Guidelines**: Professional tone, helpful responses
- **Rejection Criteria**: Clear rules for off-topic queries
- **Context Awareness**: Instructions to maintain conversation flow

### LangChain Tool Integration

**Tool Architecture:**

```python
@tool
def search_documentation_tool(request: DocumentationSearchRequest) -> str:
    # Pydantic-validated input
    # Business logic execution
    # Structured JSON output

# Tool binding
llm_with_tools = llm.bind_tools(tools)
```

**Execution Flow:**

1. LLM receives user query + tool descriptions
2. LLM decides which tool(s) to call
3. LangGraph executes tool with proper error handling
4. Tool results returned to LLM for response generation

### LangGraph State Management

**State Structure:**

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    next_step: str  # Routing decision

# Conditional routing
def route_query(state: AgentState) -> str:
    if should_reject_query(state):
        return "reject_node"
    elif needs_tool_call(state):
        return "tool_node"
    else:
        return "agent_node"
```

**Benefits:**

- Persistent conversation context
- Conditional execution paths
- Error recovery and retry logic
- Observable state transitions

## Customization

### Adding New Tools

1. Create a new function in `src/agent/tools.py` with the `@tool` decorator
2. Add it to the `tools` list in `src/agent/graph.py`
3. Update the prompt to mention the new capability
4. The LLM will automatically learn to use the new tool

### Modifying Data

- Edit `src/agent/tools.py` to add more orders or documentation
- Update the mock database structure as needed

### Adjusting Behavior

- Modify the prompt in `src/agent/graph.py` to change agent personality
- Adjust `is_relevant_query()` function to change filtering criteria
- Update routing logic in `route_query()` function

## Working with Agentic File Reading

### Adding New Documentation Categories

1. **Create Documentation File**:

   ```bash
   # Add new category file
   echo "# New Category Documentation" > src/agent/documentation/newcategory.txt
   ```

2. **Update Classification Logic**:

   ```python
   # In tools.py, add keywords for new category
   new_category_keywords = ["keyword1", "keyword2"]
   ```

3. **Update Models** (if needed):
   ```python
   # In models.py, add to DocumentationCategory literal
   DocumentationCategory = Literal[
       "shipping", "returns", "products", "account", "payment", "newcategory"
   ]
   ```

### Modifying Existing Documentation

- Edit `.txt` files directly in `src/agent/documentation/`
- No code changes required - agent loads files dynamically
- Supports Markdown formatting for better readability
- Keyword search automatically adapts to new content

## Course Demonstration Notes

This agent demonstrates advanced AI agent patterns:

- **Agentic File Reading**: Dynamic loading of documentation from `.txt` files
- **Tool Composition**: Multiple tools working together for complex tasks
- **State Management**: LangGraph's StateGraph for conversation persistence
- **Conditional Routing**: Smart routing based on query analysis
- **Error Handling**: Robust error handling and graceful degradation
- **Pydantic Validation**: Type-safe data structures throughout
- **Professional UX**: Customer service focused design and interaction
- **Development Workflow**: Complete LangGraph CLI to production pipeline

**Key Technical Demonstrations:**

1. **Query Processing Pipeline**: Input → Relevance Check → Classification → Tool Selection → Execution
2. **Agentic Patterns**: File reading, content search, dynamic categorization
3. **Stateful Conversations**: Context preservation across multiple interactions
4. **Tool Orchestration**: LLM-driven tool selection and execution
5. **Error Recovery**: Graceful handling of edge cases and invalid inputs

Perfect for understanding how to build production-ready AI agents with sophisticated behavior patterns.

## Development Workflow

1. **Start Server**: `langgraph dev`
2. **Test in Studio**: Use the web UI for interactive testing
3. **API Testing**: Use the Python SDK for programmatic testing
4. **Iterate**: Modify code and restart server to see changes
5. **Deploy**: Use LangGraph Platform for production deployment

This demonstrates the complete LangGraph development lifecycle from local development to production deployment.
