# Marketing Blog Agent with LangGraph

A sophisticated marketing blog writer built with LangGraph that demonstrates advanced AI agent patterns including **vector search**, dynamic tool selection, and stateful content management. The agent creates comprehensive marketing blog posts by retrieving relevant context, managing outlines, and assembling final content.

## How It Works

### Core Architecture

The agent operates through a structured multi-step process:

1. **Topic Reception**: Receives blog topic through LangGraph's state management
2. **Relevance Filtering**: Uses pattern matching to reject off-topic requests (e.g., weather, sports)
3. **Context Retrieval**: Searches marketing corpus via vector search for relevant information
4. **Outline Creation**: Generates structured outline for the blog post
5. **Section Writing**: Writes individual sections with proper research and context
6. **Content Assembly**: Combines all sections into final blog post

### Vector Search System

The agent implements **semantic search** over a marketing corpus using vector embeddings:

```
User Request: "Write a blog about Nimbus pricing strategy"

1. Agent searches marketing corpus for pricing-related content
2. Retrieves relevant snippets about tiers, features, and competitive positioning
3. Uses context to inform blog content and claims
4. Grounds marketing copy in actual product information
```

**Technical Implementation:**

- Uses OpenAI embeddings with FAISS for fast vector search
- Includes 50+ marketing snippets about Nimbus (revenue automation platform)
- Fallback to numpy-based search when FAISS unavailable
- Semantic search finds relevant content by meaning, not keywords

## Features

- **Context Retrieval**: Vector search over marketing corpus to ground claims
- **Outline Management**: Dynamic creation and modification of blog structure
- **Section Writing**: Individual section drafting with research integration
- **Content Editing**: Section-level editing and refinement capabilities
- **Blog Assembly**: Final compilation of all sections into complete post
- **State Persistence**: Maintains outline and sections across interactions
- **LangGraph Server**: Runs as a local server with LangGraph Studio integration

## Project Structure

```
marketing_blog_agent/
├── langgraph.json              # LangGraph configuration and tool definitions
├── pyproject.toml              # Python dependencies and project metadata
├── .env.example               # Environment variables template (API keys)
├── README.md                   # This comprehensive documentation
└── src/
    └── agent/
        ├── __init__.py         # Package initialization
        ├── graph.py           # Main LangGraph workflow and state management
        ├── tools.py           # Marketing blog tools with vector search and content management
        ├── models.py          # Pydantic models for type safety and structured data
        └── documentation/     # Legacy documentation files (can be repurposed or removed)
            ├── shipping.txt   # Legacy files - not used by blog agent
            ├── returns.txt
            ├── products.txt
            ├── account.txt
            └── payment.txt
```

## Pydantic Models

The agent uses comprehensive Pydantic models for type safety and structured data validation:

### Core Models

- **`BlogMetadata`**: Blog post metadata including title, topic, author, target audience, tone, and word count goals
- **`BlogSection`**: Individual blog sections with title, content, word count, and completion status
- **`SearchResult`**: Vector search results with content, similarity scores, and metadata
- **`BlogOutline`**: Blog structure with sections list, estimated word count, and finalization status
- **`ToolResponse`**: Standardized tool execution responses with success status, messages, and data
- **`BlogConfiguration`**: Blog generation settings including section limits, word count constraints, and style guides

These models ensure consistent data handling throughout the blog writing pipeline and provide excellent IDE support with type checking and autocomplete.

## Setup

### 1. Install uv

```bash
brew install uv
# or:
# pipx install uv
```

### 2. Create and activate a virtual environment

```bash
cd marketing_blog_agent
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
OPENAI_API_KEY=sk-proj-...
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

Send a blog writing request to the agent:

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
                "content": "Write a blog post about Nimbus pricing strategy",
            }],
            "topic": "Nimbus Pricing Strategy"
        },
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")

asyncio.run(main())
```

## Usage Examples

### Blog Post Creation

```
Input: "Write a blog about our product launch strategy"
Output: Complete blog post with outline, sections, and assembled final content
```

### Context-Aware Writing

```
Input: "Create content about Nimbus enterprise features"
Output: Blog post grounded in actual product information from marketing corpus
```

### Outline Management

```
Input: "Help me outline a post about competitive advantages"
Output: Structured outline with sections ready for content development
```

### Off-Topic Request Handling

```
Input: "Tell me about the weather today"
Output: Polite redirection to marketing/blog writing topics
```

## Available Tools

### 1. search_context

**Purpose**: Vector search over marketing corpus for relevant context
**Input**: query (string), k (number of results, default 4)
**Process**:

```python
# Tool signature
def search_context(query: str, k: int = 4) -> str:
    # 1. Embed query using OpenAI embeddings
    # 2. Search FAISS index for similar content
    # 3. Return top-k results with scores
    # 4. Return structured JSON response
```

**Features**:

- Semantic search using vector embeddings
- 50+ marketing snippets about Nimbus platform
- FAISS acceleration with numpy fallback
- Returns id, text, and similarity score

### 2. change_outline

**Purpose**: Replace the current blog outline with a new structure
**Input**: new_outline (List[str] of section titles)
**Process**:

- Updates the persistent outline state
- Returns JSON confirmation with new outline
- Maintains outline across agent interactions

### 3. write_section

**Purpose**: Write and persist a blog section draft
**Input**: section_title (string), draft (string content)
**Process**:

- Saves section content to persistent state
- Returns JSON confirmation of successful save
- Content is authored by the LLM based on retrieved context

### 4. edit_section

**Purpose**: Replace an existing section with edited content
**Input**: section_title (string), new_draft (string content)
**Process**:

- Updates existing section with new content
- Returns JSON confirmation of successful edit
- Enables iterative content refinement

### 5. assemble_blog

**Purpose**: Compile final blog post from outline and sections
**Input**: None (uses current state)
**Process**:

- Combines all sections in outline order
- Formats with proper Markdown headers
- Returns complete blog post as JSON
- Ready for publication or further editing

## Technical Architecture

### State Management with LangGraph

The agent uses LangGraph's `StateGraph` for managing blog writing state:

```python
# State structure
class BlogAgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    topic: str  # Blog topic for context

# Graph nodes
def llm_call(state: BlogAgentState) -> Dict[str, Any]:
    # GPT-4o processes messages and decides on tool calls for blog writing

def tool_node(state: BlogAgentState) -> Dict[str, Any]:
    # Execute selected blog writing tools and return results

def should_continue(state: BlogAgentState) -> Literal["tool_node", END]:
    # Route to tool execution or end based on GPT-4o tool calls
```

### Blog Writing Pipeline

1. **Topic Reception**: Blog topic enters through LangGraph state
2. **Relevance Check**: `is_relevant_query()` filters non-marketing requests
3. **LLM Analysis**: GPT-4o analyzes request and plans blog creation
4. **Tool Orchestration**: LLM calls tools to search context, create outline, write sections
5. **Content Assembly**: Tools combine sections into final blog post
6. **Response**: Complete blog post returned to user

### Vector Search Implementation

The agent uses semantic search over marketing corpus:

```python
def _search(query: str, k: int = 3) -> list[dict]:
    # Embed query using OpenAI or fallback
    q = _embed_texts([query])[0]
    q = q / (np.linalg.norm(q) + 1e-12)

    # Search FAISS index or numpy fallback
    if _INDEX_BACKEND == "faiss":
        distances, indices = _INDEX.search(q.reshape(1, -1), k)
        # Return results with scores
```

## Marketing Corpus

The agent uses a comprehensive marketing corpus including:

- **Product Information**: Nimbus revenue automation platform details
- **Pricing Tiers**: Starter ($99/mo), Pro ($499/mo), Scale ($1,999/mo) with features
- **Compliance & Security**: SOC 2, GDPR, HIPAA compliance information
- **Technical Specs**: API limits, data residency, integration capabilities
- **Company Voice**: Brand guidelines and messaging consistency
- **Case Studies**: Customer success stories and ROI metrics
- **Support Details**: SLA commitments, response times, support channels

**Sample Corpus Entries**:

- "Nimbus is the revenue automation platform for RevOps and Data teams"
- "Pricing: Starter $99/mo up to 5 seats; Pro $499/mo up to 25 seats"
- "Compliance: SOC 2 Type II and ISO 27001 certified"
- "Features: Rules Engine, Playbooks, and Workflows with audit logs"

## Key Features

### Vector Search Context Retrieval

**How it works:**

- Agent receives request: "Write about Nimbus pricing strategy"
- GPT-4o searches marketing corpus for pricing-related content
- Retrieves relevant snippets about tiers, features, and competitive positioning
- Uses context to inform blog content and ground claims in reality

**Benefits:**

- Marketing copy grounded in actual product information
- Consistent messaging across all content
- Reduces hallucinations and ensures accuracy
- Semantic search finds relevant content by meaning, not keywords

### State-Persistent Content Management

**Implementation:**

```python
@dataclass
class BlogStateMemory:
    topic: Optional[str] = None
    outline: List[str] = None
    sections: dict[str, str] = None
```

**Features:**

- Outline persists across tool calls
- Section drafts maintained in memory
- Iterative content refinement enabled
- Final assembly combines all components

### Smart Request Filtering

**Implementation:**

```python
def is_relevant_query(query: str) -> bool:
    q = query.lower()
    keywords = ["blog", "outline", "section", "marketing", "write", "post"]
    return any(k in q for k in keywords)
```

**Rejection Examples:**

- "What's the weather today?" → Rejected
- "Write a blog about our product" → Accepted
- "Tell me about baseball" → Rejected

### Professional Marketing Writing

The system prompt includes:

- **Role Definition**: "You are a senior marketing blog writer using GPT-4o"
- **Writing Style**: Practical voice, crisp verbs, short sentences, no hype
- **Content Goals**: Clear multi-level outlines, research-grounded sections
- **Quality Standards**: US English, Oxford comma, sentence case headings
- **Workflow**: Plan before writing, use tools to manage state

### LangGraph Tool Orchestration

**Tool Architecture:**

```python
@tool
def search_context(query: str, k: int = 4) -> str:
    # Vector search over marketing corpus
    # Returns JSON with relevant snippets and scores

@tool
def change_outline(new_outline: List[str]) -> str:
    # Update blog structure
    # Returns JSON confirmation

# Tool binding
llm_with_tools = llm.bind_tools(tools)  # GPT-4o with tool calling
```

**Execution Flow:**

1. GPT-4o receives blog request + tool descriptions
2. GPT-4o plans content creation and tool usage
3. LangGraph executes tools in sequence (search → outline → write → assemble)
4. Tool results inform subsequent GPT-4o calls
5. Final blog post assembled and returned

### State Management with LangGraph

**State Structure:**

```python
class BlogAgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    topic: str  # Blog topic for context

# Conditional routing
def should_continue(state: BlogAgentState) -> Literal["tool_node", END]:
    # Continue if GPT-4o made tool calls, otherwise end
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    return END
```

**Benefits:**

- Persistent blog creation context
- Tool call orchestration and sequencing
- Error handling and recovery
- Observable content creation workflow

## Customization

### Adding New Tools

1. Create a new function in `src/agent/tools.py` with the `@tool` decorator
2. Add it to the `tools` list in `src/agent/__init__.py` and `src/agent/graph.py`
3. Update the system prompt in `graph.py` to describe the new capability
4. The LLM will automatically learn to use the new tool

### Modifying Marketing Corpus

- Edit the `EXAMPLE_DOCS` list in `src/agent/tools.py` to add new marketing content
- Update the corpus with your company's products, features, and messaging
- Rebuild the vector index by restarting the application
- Content changes are immediately searchable without code redeployment

### Adjusting Writing Style

- Modify the system prompt in `src/agent/graph.py` to change writing personality
- Adjust `is_relevant_query()` function in `graph.py` to change topic filtering
- Update writing guidelines (US vs UK English, tone, formatting preferences)
- Customize the blog structure and section naming conventions

## Working with Vector Search

### Expanding the Marketing Corpus

1. **Add New Content Snippets**:

   ```python
   # In tools.py, add to EXAMPLE_DOCS list
   EXAMPLE_DOCS.append({
       "id": "new_feature_1",
       "text": "New feature description with key benefits and use cases"
   })
   ```

2. **Content Guidelines**:

   - Keep snippets concise (1-2 sentences)
   - Include specific product details, pricing, features
   - Add customer success metrics and ROI data
   - Include compliance and security information
   - Maintain consistent brand voice and messaging

3. **Rebuild Index**:

   ```python
   # Restart the application to rebuild vector index
   # Index automatically rebuilds on startup with new content
   ```

### Optimizing Search Results

- **Query Formulation**: Use specific product terms and feature names
- **Content Quality**: More detailed snippets improve search relevance
- **Deduplication**: Avoid duplicate information across snippets
- **Metadata**: Include specific metrics, dates, and version numbers
- **Context Preservation**: Keep related information together in snippets

## Course Demonstration Notes

This agent demonstrates advanced AI agent patterns for content creation:

- **Vector Search Integration**: Semantic search over marketing corpus using embeddings
- **Stateful Content Management**: Persistent outline and section management across tool calls
- **Tool Orchestration**: Sequential execution of search, outline, writing, and assembly tools
- **Context-Grounded Writing**: Marketing copy informed by actual product information
- **Iterative Content Refinement**: Section-level editing and improvement capabilities
- **Professional Writing Standards**: Consistent voice, formatting, and quality guidelines
- **LangGraph Workflow**: Complete content creation pipeline from concept to publication

**Key Technical Demonstrations:**

1. **Content Creation Pipeline**: Topic → Context Search → Outline → Section Writing → Assembly
2. **Vector Search Patterns**: Embedding-based retrieval with FAISS and numpy fallbacks
3. **State Persistence**: Memory management for complex multi-step content creation
4. **Tool Sequencing**: LLM-driven orchestration of specialized content tools
5. **Quality Assurance**: Grounding claims in marketing corpus to ensure accuracy

Perfect for understanding how to build production-ready AI content creation agents with research capabilities and professional writing standards.

## Development Workflow

1. **Start Server**: `langgraph dev`
2. **Test in Studio**: Use the web UI for interactive testing
3. **API Testing**: Use the Python SDK for programmatic testing
4. **Iterate**: Modify code and restart server to see changes
5. **Deploy**: Use LangGraph Platform for production deployment

This demonstrates the complete LangGraph development lifecycle from local development to production deployment.
