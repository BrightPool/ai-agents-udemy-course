# 🔬 Open Deep Research Agent

A lightweight AI research assistant that turns a question into a comprehensive report using a simple DAG and concurrent query execution. Built with LangGraph.

## 🤖 What This Agent Does

**Simply put:** You give it a research question, and it breaks it down, researches it thoroughly using AI, and delivers a comprehensive report.

**Example:**

- **Input:** "Research the impact of AI on healthcare delivery systems"
- **Output:** A detailed report covering current applications, benefits, challenges, regulations, and future predictions

## 🏗️ How It Works (Simple Version)

The system runs a **5-stage DAG**:

1. **Clarifier** – Optionally asks for missing info
2. **Planner** – Writes a focused research brief
3. **Query Generator** – Produces N concrete search queries
4. **Query Runner** – Executes queries concurrently using OpenAI
5. **Report Writer** – Synthesizes a comprehensive final report

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

**🔍 OpenAI Search Tool**

- Uses OpenAI's knowledge to produce comprehensive summaries
- Executes multiple queries concurrently for faster coverage
- Supports both general research and news-focused queries

**🤔 Think Tool (optional)**

- Available for experiments, but not required in the simplified DAG

## 🏛️ Technical Architecture (For Developers)

### Simple DAG

```
Main Workflow (AgentState)
├── clarify_with_user
├── write_research_brief
├── generate_research_queries
├── run_queries
└── final_report_generation
```

### 5-Phase Pipeline

1. **Clarification** – Ensure the request is unambiguous
2. **Planning** – Create a focused research brief
3. **Query Generation** – Produce N concrete, diverse queries
4. **Parallel Query Execution** – Run queries concurrently via OpenAI
5. **Final Report** – Synthesize a comprehensive, cited report

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

### Adding New Research Capabilities

- **Add/Swap Tools**: Edit `run_queries` in `src/agent/deep_researcher.py` to combine additional tools alongside `openai_search`.
- **Modify Agent Behavior**: Edit `src/agent/configuration.py` to adjust:

- Model assignments for different phases
- Token limits and iteration counts
- Number of generated queries (`NUM_QUERIES`)

**Customize Prompts:**
Edit `src/agent/prompts.py` to tweak clarification, brief creation, query generation, and reporting.

### Development Workflow

1. **Make Changes** - Edit code in `src/agent/`
2. **Restart Server** - `uv run langgraph dev`
3. **Test in Studio** - Use the web interface for testing
4. **Iterate** - Make changes and restart as needed

## 🎓 Learning Outcomes

This project demonstrates:

- **DAG-based AI Systems** - A simple, reliable research pipeline
- **Clear State Management** - Minimal, focused state definitions
- **Parallel I/O** - Running multiple web queries concurrently
- **Tool Integration** - Building and using custom AI tools
- **Practical Architecture** - Maintainable and easy to customize

## 📚 Resources

- **LangGraph Documentation**: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph)
- **LangChain Documentation**: [python.langchain.com](https://python.langchain.com)
- **OpenAI API Reference**: [platform.openai.com/docs](https://platform.openai.com/docs)

---

**Happy Researching! 🚀**

Got questions? The code is well-commented and the architecture is designed to be easy to understand and extend.
