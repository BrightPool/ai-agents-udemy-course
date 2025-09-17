# Mem0-Powered Coaching Agent with LangGraph

A sophisticated coaching agent built with LangGraph that demonstrates advanced AI agent patterns including **persistent memory management**, dynamic memory retrieval, and stateful conversation management. The agent specializes in executive coaching with intelligent memory processing and personalized coaching responses using Mem0's vector and graph memory capabilities.

## How It Works

### Core Architecture

The agent operates through a sophisticated memory-augmented coaching process:

1. **Memory Retrieval**: Searches Mem0 for relevant coaching memories and context about the user
2. **Coaching Analysis**: LLM analyzes the user's query with their coaching history and personalized context
3. **Response Generation**: Generates personalized coaching responses using proven coaching frameworks
4. **Memory Storage**: Stores the user's latest message in Mem0 for future coaching sessions

### Mem0 Memory Management System

The coaching agent implements **persistent memory management** - a pattern where the agent maintains long-term memory of coaching conversations using Mem0's hybrid vector and graph database:

```
User Query: "I'm struggling with motivation for my fitness goals"

1. Agent searches Mem0 for user's coaching history → relevant past conversations
2. Agent loads user's previous fitness discussions and progress tracking
3. Agent incorporates memory context into coaching response
4. Agent stores new message for future reference

Technical Implementation:
- Uses httpx for HTTP communication with Mem0 API
- Supports both vector similarity search and graph-based memory connections
- Configurable memory retrieval (k parameter for top-K results)
- Best-effort memory operations that don't fail the coaching session
- Environment-based configuration for Mem0 base URL and API settings

## Features

- **Persistent Memory**: Long-term memory storage using Mem0's vector and graph database
- **Personalized Coaching**: Tailored responses based on user's coaching history and progress
- **Executive Coaching Frameworks**: Implements proven coaching methodologies (Circle of Control, Helicopter View, etc.)
- **Memory-Augmented Responses**: Incorporates relevant past conversations into current coaching
- **Configurable Memory Retrieval**: Adjustable parameters for memory search depth and relevance
- **LangGraph Server**: Runs as a local server with LangGraph Studio integration

## Project Structure

```

customer_support_agent_mem0/
├── langgraph.json # LangGraph configuration
├── pyproject.toml # Python dependencies and project metadata
├── docker-compose.yml # Mem0 service stack (Qdrant + Neo4j)
├── .env.mem0.example # Mem0 environment variables template
├── README.md # This comprehensive documentation
└── src/
└── agent/
├── **init**.py # Package initialization
├── graph.py # Main LangGraph workflow with Mem0 integration
├── models.py # Pydantic models for type safety (legacy)
└── tools.py # Legacy customer service tools (not used by coaching agent)

````

## Setup

### Prerequisites

- **Docker & Docker Compose**: Required for running Mem0 services locally
- **uv**: Modern Python package manager for dependency management
- **OpenAI API Key**: Required for the coaching LLM responses

### Mem0 Memory Infrastructure

The coaching agent requires Mem0's hybrid memory system for persistent conversation memory. The repository includes a complete Docker stack with:

- **Mem0 API Server**: Main memory service at `http://localhost:8000`
- **Qdrant Vector Database**: Vector similarity search for semantic memory retrieval
- **Neo4j Graph Database**: Graph-based memory connections and relationships

#### 1. Start Mem0 Services

```bash
# Copy and configure environment file
cp .env.mem0.example .env.mem0
# Edit .env.mem0 with your OpenAI API key and other settings

# Start the complete memory stack
docker compose up -d

# Verify services are running
# Mem0 API:        http://localhost:8000
# Qdrant console:  http://localhost:6333
# Neo4j browser:   http://localhost:7474 (bolt://localhost:7687)
````

#### 2. Test Mem0 Connection

```bash
# Health check - should return search results structure
curl -s -X POST \
  http://localhost:8000/api/v1/memories/search \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "demo",
    "query": "test coaching session",
    "k": 1,
    "enable_graph": true
  }'
```

#### 3. Configure Environment Variables

Set the environment variables for the coaching agent:

```bash
export MEM0_BASE_URL=http://localhost:8000
export OPENAI_API_KEY=sk-your-openai-api-key-here
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

### Motivation and Goal Setting

```
Input: "I'm struggling to stay motivated with my fitness goals"
Output: Personalized coaching response incorporating user's past fitness discussions,
         using frameworks like Circle of Control and Process-oriented goals
```

### Career Transition Coaching

```
Input: "I'm thinking about changing careers but I'm scared"
Output: Coaching response drawing from user's previous career discussions,
         applying Helicopter View and Time Jump frameworks
```

### Leadership Development

```
Input: "My team isn't performing well and I don't know what to do"
Output: Leadership coaching incorporating user's management history,
         using frameworks like Weakness Strategy and Reflection Loop
```

### Personal Growth Coaching

```
Input: "I want to be more disciplined but keep failing"
Output: Discipline coaching with personalized advice based on user's
         previous attempts, using MSI (Minimal Success Index) framework
```

## Core Components

### Memory-Augmented Coaching Pipeline

The coaching agent operates through a streamlined three-node pipeline that doesn't use traditional LangChain tools but instead leverages Mem0's memory capabilities:

#### 1. Memory Search Node (`mem0_search`)

**Purpose**: Retrieves relevant coaching memories from Mem0's hybrid database
**Process**:

```python
# Searches for top-K relevant memories based on user query
# Supports both vector similarity and graph-based memory retrieval
# Returns formatted memory context for coaching analysis
# Best-effort operation that doesn't fail the session
```

**Configuration**:

- `k`: Number of memories to retrieve (default: 3)
- `enable_graph`: Whether to use graph-based memory connections (default: true)
- `user_id`: Unique identifier for the coaching client

#### 2. LLM Coaching Node (`llm`)

**Purpose**: Generates personalized coaching responses using OpenAI GPT-4.1-mini
**Features**:

- Comprehensive coaching system prompt with proven frameworks
- Memory-augmented responses incorporating user's coaching history
- Brief, practical coaching style focused on transformation
- Framework-based responses (Circle of Control, Helicopter View, etc.)

#### 3. Memory Storage Node (`mem0_add`)

**Purpose**: Persists user's latest message to Mem0 for future coaching sessions
**Process**:

```python
# Stores user input in vector and graph memory
# Enables future memory retrieval and relationship building
# Best-effort operation that doesn't interrupt coaching flow
```

## Technical Architecture

### State Management with LangGraph

The coaching agent uses LangGraph's `StateGraph` for managing coaching conversation state with persistent memory:

```python
# State structure for coaching sessions
class CoachingAgentState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]  # Conversation history
    user_id: str                    # Unique coaching client identifier
    k: int                         # Memory retrieval depth (optional)
    enable_graph: bool             # Graph memory toggle (optional)
    memories_text: str             # Retrieved memories for context
    assistant_text: str            # Generated coaching response

# Execution context with runtime configuration
class Context(TypedDict, total=False):
    openai_api_key: str           # OpenAI API key
    mem0_base_url: str           # Mem0 service URL
    mem0_default_k: int          # Default memory retrieval count
    mem0_enable_graph: bool      # Default graph memory setting
```

### Memory-Augmented Coaching Pipeline

1. **Memory Retrieval**: `mem0_search_node()` queries Mem0 for relevant coaching context
2. **Coaching Analysis**: `llm_node()` generates personalized responses with memory context
3. **Memory Persistence**: `mem0_add_node()` stores new interactions for future sessions
4. **State Persistence**: LangGraph's `MemorySaver` maintains conversation continuity

### Memory Processing Logic

The agent implements sophisticated memory processing:

```python
def mem0_search_node(state: CoachingAgentState, runtime: Runtime[Context]):
    # 1. Extract latest user message from conversation
    # 2. Query Mem0 API with configurable parameters
    # 3. Process and rank memory results by relevance score
    # 4. Return formatted memory context for coaching

def llm_node(state: CoachingAgentState, runtime: Runtime[Context]):
    # 1. Combine system prompt with retrieved memories
    # 2. Generate coaching response using GPT-4.1-mini
    # 3. Apply coaching frameworks based on user context
    # 4. Return structured coaching response
```

## Memory Infrastructure

The coaching agent relies on Mem0's persistent memory system rather than mock data:

### Mem0 Memory Components

- **Vector Memory (Qdrant)**: Semantic similarity search for relevant coaching memories
- **Graph Memory (Neo4j)**: Relationship-based memory connections and context linking
- **Hybrid Retrieval**: Combines vector similarity with graph-based relationship traversal
- **User-Specific Memory**: Isolated memory spaces for each coaching client (`user_id`)
- **Temporal Memory**: Chronological ordering and time-based memory relevance

### Memory Configuration Options

```python
# Configurable memory retrieval parameters
memory_config = {
    "k": 3,                    # Number of memories to retrieve
    "enable_graph": True,      # Use graph relationships
    "score_threshold": 0.7,    # Minimum relevance score
    "temporal_boost": True     # Boost recent memories
}
```

### Memory Persistence Strategy

- **Automatic Storage**: Every user message is automatically stored in Mem0
- **Relationship Building**: Graph connections between related coaching topics
- **Semantic Indexing**: Vector embeddings for natural language memory retrieval
- **Best-Effort Operations**: Memory failures don't interrupt coaching sessions

## Key Features

### Memory-Augmented Coaching

**How it works:**

- User shares: "I'm struggling with motivation for my fitness goals"
- Agent searches Mem0 for user's previous fitness and motivation discussions
- Retrieves relevant coaching history and progress tracking
- Generates personalized response incorporating past context
- Stores new interaction for future coaching continuity

**Benefits:**

- Personalized coaching based on user's unique journey
- Consistent coaching relationship across sessions
- Context-aware responses that build on previous work
- Long-term memory enables sophisticated coaching strategies

### Executive Coaching Frameworks

**Implementation:**

The agent uses proven coaching methodologies through a comprehensive system prompt:

```python
COACH_SYSTEM_PROMPT = """
You are a highly sought-after executive coach and psychologist...
Philosophy: transformations, not incremental changes
Core Frameworks:
- Circle of Control → agency vs victimhood
- Helicopter View → shift perspective, self-coaching from altitude
- Time Jump → imagine future state already achieved
- Process-oriented goals → controllable actions, most effective
"""
```

**Framework Applications:**

- **Motivation Issues**: Applies Circle of Control and Process goals
- **Career Transitions**: Uses Helicopter View and Time Jump techniques
- **Leadership Challenges**: Implements Reflection Loop and Weakness Strategy
- **Personal Growth**: Applies MSI (Minimal Success Index) and Flow Channel

### Professional Coaching Communication

The coaching system prompt emphasizes:

- **Expert Positioning**: "Highly sought-after executive coach and psychologist"
- **Transformation Focus**: "Philosophy: transformations, not incremental changes"
- **Practical Style**: "Be brief, sharp, and practical. Prefer 1–2 sentences over paragraphs"
- **Framework Integration**: Extensive coaching frameworks (Circle of Control, Helicopter View, etc.)
- **Memory Awareness**: Incorporates user's coaching history for personalized responses

### Direct API Integration

**Architecture Approach:**

The coaching agent uses direct HTTP API calls rather than LangChain tools for maximum flexibility:

```python
# Direct Mem0 API integration
def mem0_search_node(state: CoachingAgentState, runtime: Runtime[Context]):
    with httpx.Client(timeout=20) as client:
        resp = client.post(f"{base_url}/api/v1/memories/search", json=payload)
        # Process memory results directly

# Direct OpenAI API integration
def llm_node(state: CoachingAgentState, runtime: Runtime[Context]):
    llm = ChatOpenAI(model="gpt-4.1-mini", api_key=api_key)
    response = llm.invoke([system_message] + messages)
    # Process coaching response directly
```

**Benefits:**

1. **Full Control**: Direct API calls provide complete control over requests and responses
2. **Performance**: Eliminates LangChain abstraction overhead
3. **Flexibility**: Easy to customize API parameters and error handling
4. **Transparency**: Clear understanding of all API interactions

### LangGraph State Management

**Coaching State Structure:**

```python
class CoachingAgentState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]  # Conversation history
    user_id: str                    # Unique coaching client identifier
    k: int                         # Memory retrieval depth
    enable_graph: bool             # Graph memory toggle
    memories_text: str             # Retrieved coaching memories
    assistant_text: str            # Generated coaching response

# Linear pipeline (no conditional routing needed)
graph = StateGraph(CoachingAgentState, context_schema=Context)
graph.add_edge(START, "mem0_search")
graph.add_edge("mem0_search", "llm")
graph.add_edge("llm", "mem0_add")
graph.add_edge("mem0_add", END)
```

**Benefits:**

- **Memory Persistence**: LangGraph's MemorySaver maintains coaching context across sessions
- **Linear Pipeline**: Streamlined three-node architecture optimized for coaching workflow
- **Context Schema**: Runtime configuration through Context TypedDict
- **Best-Effort Operations**: Memory failures don't break the coaching conversation

## Customization

### Modifying Coaching Frameworks

1. **Update System Prompt**: Edit `COACH_SYSTEM_PROMPT` in `src/agent/graph.py`
2. **Add New Frameworks**: Include additional coaching methodologies and techniques
3. **Adjust Communication Style**: Modify the brief, practical coaching approach
4. **Customize Philosophy**: Update core principles (transformation vs incremental changes)

### Memory Configuration

- **Adjust Retrieval Depth**: Modify default `k` parameter for memory retrieval
- **Toggle Graph Memory**: Enable/disable Neo4j graph-based memory connections
- **Configure Timeouts**: Adjust httpx timeout values for API calls
- **Customize Scoring**: Modify memory relevance ranking logic

### LLM Configuration

- **Change Model**: Switch between different OpenAI models (GPT-4, GPT-3.5-turbo)
- **Adjust Temperature**: Modify response creativity (currently set to 0.3 for coaching)
- **Update Context Window**: Configure maximum token limits for conversations
- **Customize System Prompt**: Tailor the coaching personality and expertise

## Working with Mem0 Memory System

### Understanding Memory Retrieval

The coaching agent uses Mem0's hybrid memory system for context-aware coaching:

1. **Vector Memory (Qdrant)**: Semantic search for relevant coaching conversations
2. **Graph Memory (Neo4j)**: Relationship-based connections between coaching topics
3. **Hybrid Retrieval**: Combines both approaches for optimal memory retrieval
4. **User Isolation**: Each `user_id` has their own memory space for privacy

### Memory Configuration Options

```python
# Memory retrieval parameters in graph.py
def mem0_search_node(state: CoachingAgentState, runtime: Runtime[Context]):
    # Configure search parameters
    k = state.get("k") or 3  # Number of memories to retrieve
    enable_graph = state.get("enable_graph", True)  # Use graph connections

    payload = {
        "user_id": state.get("user_id", "default"),
        "query": user_text,
        "k": k,
        "enable_graph": enable_graph
    }
```

### Optimizing Memory Performance

- **Adjust K Value**: Increase for more context, decrease for faster responses
- **Graph Toggle**: Enable for complex relationship mapping, disable for simple similarity
- **User Segmentation**: Use consistent `user_id` for personalized coaching journeys
- **Memory Quality**: Regular review of stored memories for coaching effectiveness

## Course Demonstration Notes

This coaching agent demonstrates advanced AI agent patterns for memory-augmented applications:

- **Persistent Memory Management**: Hybrid vector and graph memory with Mem0
- **Memory-Augmented Generation**: Incorporating conversation history into responses
- **State Management**: LangGraph's StateGraph for coaching session persistence
- **Direct API Integration**: Bypassing LangChain tools for maximum control
- **Error Handling**: Best-effort memory operations that don't break conversations
- **Professional Coaching UX**: Executive coaching focused design and interaction
- **Development Workflow**: Complete LangGraph CLI to production pipeline

**Key Technical Demonstrations:**

1. **Memory Pipeline**: Memory Search → LLM Coaching → Memory Storage
2. **Hybrid Memory Patterns**: Vector similarity + graph relationships for context
3. **Stateful Coaching Sessions**: Context preservation across coaching interactions
4. **API Orchestration**: Direct Mem0 and OpenAI API integration
5. **Coaching Frameworks**: Implementation of proven executive coaching methodologies
6. **Personalization**: User-specific memory spaces and tailored coaching responses

Perfect for understanding how to build production-ready AI agents with persistent memory and sophisticated coaching capabilities.

## Development Workflow

1. **Start Mem0 Services**: `docker compose up -d` (Qdrant + Neo4j)
2. **Configure Environment**: Set `MEM0_BASE_URL` and `OPENAI_API_KEY`
3. **Start Server**: `uv run langgraph dev`
4. **Test in Studio**: Use LangGraph Studio web UI for coaching interactions
5. **API Testing**: Use Python SDK for programmatic coaching sessions
6. **Iterate**: Modify coaching frameworks and restart server to see changes
7. **Deploy**: Use LangGraph Platform for production coaching deployment

This demonstrates the complete memory-augmented agent development lifecycle from local development to production deployment.

### Example Coaching Session

```python
from langgraph_sdk import get_client

client = get_client(url="http://localhost:2024")

# Start a coaching conversation
async def coaching_session():
    async for chunk in client.runs.stream(
        None,  # Threadless run
        "agent",  # Graph name
        input={
            "messages": [{
                "role": "human",
                "content": "I'm struggling with work-life balance"
            }],
            "user_id": "coaching_client_001",  # Unique client identifier
            "k": 3,  # Retrieve 3 relevant memories
            "enable_graph": True  # Use graph memory
        }
    ):
        print(f"Coach: {chunk.data}")
```
