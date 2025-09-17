# 🔬 Open Deep Research Agent

A lightweight AI research assistant that turns a question into a comprehensive report using a simple DAG and concurrent query execution. Built with LangGraph.

## 🤖 What This Agent Does

**Simply put:** You give it a research question, and it breaks it down, researches it thoroughly using AI, and delivers a comprehensive report.

**Example:**

- **Input:** "Research the impact of AI on healthcare delivery systems"
- **Output:** A detailed report covering current applications, benefits, challenges, regulations, and future predictions

## 🏗️ How It Works (Simple Version)

The system runs a **5-stage DAG**:

1. **Clarifier** (`clarify_with_user`) – Analyzes user queries and requests clarification if needed
2. **Planner** (`write_research_brief`) – Transforms messages into structured research briefs
3. **Query Generator** (`generate_research_queries`) – Produces concrete search queries from the brief
4. **Query Runner** (`run_queries`) – Executes queries concurrently using OpenAI search
5. **Report Writer** (`final_report_generation`) – Synthesizes findings into final reports

## 🎯 Key Features

- **Smart Question Analysis**: Clarifies ambiguous inputs
- **Focused Planning**: Generates a structured research brief
- **Parallel Queries**: Runs multiple queries concurrently for speed
- **AI-Powered Search**: Uses OpenAI for comprehensive coverage
- **Structured Reports**: Clear, organized final reports in Markdown
- **Robust Handling**: Graceful handling of token/context limits
- **Local Server**: Runs locally with a web UI via LangGraph Studio

## Project Structure

```
open_deep_research/
├── langgraph.json              # LangGraph configuration
├── pyproject.toml              # Dependencies and metadata
├── .env.example                # Environment variables template (API keys)
├── README.md                   # This documentation
└── src/
    └── agent/
        ├── __init__.py
        ├── configuration.py    # Minimal config (models, num_queries, etc.)
        ├── deep_researcher.py  # Simple DAG: clarify → brief → queries → run → report
        ├── prompts.py          # Clarification, brief, query generation, report
        ├── state.py            # DAG state definitions
        └── utils.py            # OpenAI search tool and helpers
```

## 🚀 Quick Start (5 minutes)

### Step 1: Install uv package manager

```bash
brew install uv
# or
pipx install uv
```

### Step 2: Set up the project

```bash
cd langgraph/open_deep_research
uv venv
source .venv/bin/activate  # On Mac/Linux
# or on Windows: .venv\Scripts\activate
```

### Step 3: Install everything

```bash
uv sync
uv pip install -e .
```

### Step 4: Add your OpenAI API key

```bash
cp .env.example .env
# Edit .env file and add:
OPENAI_API_KEY=sk-proj-your-key-here
```

### Step 5: Start the research server

```bash
uv run langgraph dev
```

**Success!** You'll see:

```
Ready!
- API: http://localhost:2024
- Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### Step 6: Try it out

1. Open the Studio URL in your browser
2. Type a research question like: _"Research the benefits of renewable energy"_
3. Watch the agents work together to create a comprehensive report!

## 💡 What You Can Research

The agent works great for:

**📊 Business & Technology**

- "Research the impact of AI on software development productivity"
- "Analyze the growth of electric vehicle market share"
- "Explore the benefits of microservices architecture"

**🏥 Healthcare & Science**

- "Research telemedicine adoption trends"
- "Analyze climate change mitigation strategies"
- "Explore CRISPR gene editing applications"

**💼 Industry Analysis**

- "Research the future of remote work"
- "Analyze the impact of 5G on IoT development"
- "Explore sustainable energy transition strategies"

## 🔧 API Usage (Advanced)

For programmatic access:

```bash
uv add --group dev langgraph-sdk
```

```python
from langgraph_sdk import get_client
import asyncio

client = get_client(url="http://localhost:2024")

async def research():
    async for chunk in client.runs.stream(
        None,  # Threadless run
        "open_deep_research",
        input={
            "messages": [{
                "role": "human",
                "content": "Research the impact of AI on healthcare"
            }]
        }
    ):
        print(f"Event: {chunk.event}")
        if chunk.data:
            print(chunk.data)

asyncio.run(research())
```

## 🛠️ How Research Actually Works

### The AI Research Tools

**🔍 OpenAI Search Tool** (`openai_search`)

- Uses OpenAI's knowledge to produce comprehensive summaries
- Executes multiple queries concurrently for faster coverage
- Single tool invocation handles all queries in parallel
- Returns consolidated research findings for report generation

```python
# How queries are executed in the code
results = await openai_search.ainvoke({"queries": queries}, config)
```

## 🏛️ Technical Architecture (For Developers)

### Simple DAG Structure

```
Main Workflow (AgentState)
├── clarify_with_user       # Analyzes and clarifies user queries
├── write_research_brief    # Generates structured research briefs
├── generate_research_queries # Creates concrete search queries
├── run_queries            # Executes queries concurrently
└── final_report_generation # Synthesizes final report
```

### 5-Phase Pipeline with Routing Logic

1. **Clarification** (`clarify_with_user`)

   - Uses structured output (`ClarifyWithUser`) to analyze queries
   - Prevents infinite loops with `maximum_clarification_attempts` limit
   - Routes to END if clarification needed, or to planning phase

2. **Planning** (`write_research_brief`)

   - Uses structured output (`ResearchQuestion`) for consistent briefs
   - Transforms user messages into focused research objectives

3. **Query Generation** (`generate_research_queries`)

   - Generates configurable number of queries (default: 6)
   - Parses and trims query list to exact count
   - Routes to query execution phase

4. **Parallel Query Execution** (`run_queries`)

   - Single `openai_search` tool invocation handles all queries
   - Internally parallelized for performance
   - Returns consolidated findings

5. **Final Report** (`final_report_generation`)
   - Robust token limit handling with progressive truncation
   - Retry logic with 3 attempts maximum
   - Graceful error recovery and state cleanup

### Why Parallel Processing Matters

**Without Parallel Queries:**

- Query 1 → Query 2 → Query 3 (sequential)
- Total time: 3x longer

**With Parallel Queries:**

- All queries run at the same time
- Total time: similar to a single query

**Real Example (within the tool):**

```python
results = await openai_search.ainvoke({"queries": queries}, config)
```

## ⚙️ Customization & Development

### Configuration Options (`src/agent/configuration.py`)

The agent behavior can be customized through the `Configuration` class:

- **Model Configuration**:

  - `research_model`: Model for clarification and planning phases
  - `final_report_model`: Model for report generation
  - `research_model_max_tokens`: Token limits for research phases
  - `final_report_model_max_tokens`: Token limits for report generation

- **Query Settings**:

  - `num_queries`: Number of search queries to generate (default: 6)

- **Safety Limits**:
  - `maximum_clarification_attempts`: Prevents infinite clarification loops
  - `max_structured_output_retries`: Retry limit for structured output failures

### Adding New Research Capabilities

- **Extend Query Execution**: Modify `run_queries` in `src/agent/deep_researcher.py` to add additional tools alongside `openai_search`
- **Add New Phases**: Insert new nodes in the DAG workflow
- **Custom Tools**: Add new research tools in `src/agent/utils.py`

### Prompt Customization (`src/agent/prompts.py`)

Available prompts for customization:

- `clarify_with_user_instructions`: Query analysis and clarification logic
- `transform_messages_into_research_topic_prompt`: Research brief generation
- `generate_research_queries_prompt`: Query generation from briefs
- `final_report_generation_prompt`: Report synthesis and formatting

### Error Handling & Robustness

The implementation includes several robustness features:

- **Token Limit Management**: Progressive truncation with retry logic in report generation
- **Infinite Loop Prevention**: Maximum clarification attempts limit
- **Structured Output Reliability**: Retry mechanisms for structured AI responses
- **Graceful Degradation**: Fallback error messages when operations fail
- **State Cleanup**: Automatic cleanup of intermediate research data

### Development Workflow

1. **Make Changes** - Edit code in `src/agent/`
2. **Restart Server** - `uv run langgraph dev`
3. **Test in Studio** - Use the web interface for testing
4. **Iterate** - Make changes and restart as needed

## 🎓 Learning Outcomes

This project demonstrates:

- **LangGraph StateGraph Architecture** - Building complex workflows with routing logic
- **Structured Output Patterns** - Using Pydantic models for reliable AI responses
- **Async Parallel Processing** - Concurrent query execution for performance
- **Robust Error Handling** - Token limits, retries, and graceful degradation
- **Configuration Management** - Runtime model and behavior customization
- **Tool Development** - Creating and integrating custom AI research tools
- **State Management** - Complex state graphs with cleanup and routing
- **Production-Ready Patterns** - Retry logic, logging, and error recovery

## 📚 Resources

- **LangGraph Documentation**: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph)
- **LangChain Documentation**: [python.langchain.com](https://python.langchain.com)
- **OpenAI API Reference**: [platform.openai.com/docs](https://platform.openai.com/docs)

---

**Happy Researching! 🚀**

Got questions? The code is well-commented and the architecture is designed to be easy to understand and extend.
