# 🔬 Open Deep Research Agent

A sophisticated AI research assistant that orchestrates multiple specialized agents to conduct comprehensive, high-quality research. Built with LangGraph for reliable multi-agent coordination.

## 🤖 What This Agent Does

**Simply put:** You give it a research question, and it breaks it down, researches it thoroughly using AI, and delivers a comprehensive report.

**Example:**

- **Input:** "Research the impact of AI on healthcare delivery systems"
- **Output:** A detailed report covering current applications, benefits, challenges, regulations, and future predictions

## 🏗️ How It Works (Simple Version)

The system uses **6 specialized AI agents** working together:

1. **Clarifier Agent** - Makes sure your question is clear
2. **Planner Agent** - Breaks your question into research tasks
3. **Supervisor Agent** - Coordinates the research team
4. **Researcher Agents** - Do the actual research (work in parallel)
5. **Compression Agent** - Organizes findings
6. **Report Writer** - Creates the final report

## 🎯 Key Features

- **Smart Question Analysis** - Asks for clarification when needed
- **Intelligent Task Breaking** - Splits complex topics into manageable pieces
- **Parallel Research** - Multiple researchers work simultaneously for speed
- **AI-Powered Search** - Uses OpenAI's knowledge for comprehensive research
- **Structured Reports** - Clear, organized final reports
- **Error Handling** - Gracefully handles research challenges
- **Local Server** - Runs on your machine with web interface

## Project Structure

```
open_deep_research/
├── langgraph.json              # LangGraph configuration and tool definitions
├── pyproject.toml              # Python dependencies and project metadata
├── .env.example               # Environment variables template (API keys)
├── README.md                   # This comprehensive documentation
└── src/
    └── open_deep_research/
        ├── __init__.py         # Package initialization
        ├── deep_researcher.py  # Main LangGraph workflow and multi-agent coordination
        ├── configuration.py    # OpenAI-only configuration with GPT-5-mini defaults
        ├── prompts.py          # Specialized prompts for each agent role
        ├── state.py            # Type definitions and state management
        └── utils.py            # Research tools and utility functions
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
cd open_deep_research
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

- Uses OpenAI's knowledge base for comprehensive research
- Provides structured information based on the model's training data
- Supports both general research and news-focused queries

**🤔 Think Tool**

- Allows agents to pause and reflect on their research progress
- Helps make strategic decisions about next research steps
- Creates "thinking breaks" for better research quality

**✅ Research Complete Tool**

- Signals when individual research tasks are finished
- Tells the supervisor that a researcher agent is done
- Triggers the transition to report generation

## 🏛️ Technical Architecture (For Developers)

### How Multiple Agents Work Together

The system uses **nested StateGraphs** - think of them as specialized AI "departments":

```
Main Workflow (AgentState)
├── Clarification Agent
├── Planning Agent
├── Supervisor Subgraph (SupervisorState)
│   ├── Supervisor Agent (coordinates research)
│   └── Tool Execution (delegates tasks)
└── Researcher Subgraph (ResearcherState)
    ├── Individual Researcher Agents
    ├── Tool Execution (search, think, complete)
    └── Compression Agent
```

### The 6-Phase Research Pipeline

1. **Clarification** - "Do I understand the question correctly?"
2. **Planning** - "What specific aspects should I research?"
3. **Supervision** - "Which researchers should work on which tasks?"
4. **Parallel Research** - Multiple researchers work simultaneously
5. **Synthesis** - "How do all these findings fit together?"
6. **Final Report** - Structured, comprehensive output

### Why Parallel Processing Matters

**Without Parallel Processing:**

- Agent 1 researches → Agent 2 researches → Agent 3 researches
- Total time: 3x longer

**With Parallel Processing:**

- All 3 agents research at the same time
- Total time: Same as single agent!
- Better: Each agent specializes in their sub-topic

**Real Example:**

```python
# Supervisor delegates 3 research tasks simultaneously
research_tasks = [
    researcher_subgraph.ainvoke({"topic": "Current AI applications"}, config),
    researcher_subgraph.ainvoke({"topic": "Technical challenges"}, config),
    researcher_subgraph.ainvoke({"topic": "Future predictions"}, config)
]
all_results = await asyncio.gather(*research_tasks)  # Runs in parallel!
```

## ⚙️ Customization & Development

### Adding New Research Capabilities

**1. Add a New Research Tool:**

```python
# In utils.py
@tool_decorator(description="My new research tool")
async def my_new_tool(query: str) -> str:
    # Your tool logic here
    return f"Research results for: {query}"

# Add to get_all_tools() function
async def get_all_tools(config: RunnableConfig):
    tools = [tool_decorator(ResearchComplete), think_tool, my_new_tool]
    return tools
```

**2. Modify Agent Behavior:**
Edit `configuration.py` to adjust:

- Model assignments for different phases
- Token limits and iteration counts
- Concurrent research limits

**3. Customize Prompts:**
Edit `prompts.py` to modify agent personalities and instructions for different research domains.

### Development Workflow

1. **Make Changes** - Edit code in `src/open_deep_research/`
2. **Restart Server** - `uv run langgraph dev`
3. **Test in Studio** - Use the web interface for testing
4. **Iterate** - Make changes and restart as needed

## 🎓 Learning Outcomes

This project demonstrates:

- **Multi-Agent AI Systems** - Coordinating multiple specialized agents
- **Complex State Management** - Handling sophisticated agent interactions
- **Parallel Processing** - Running multiple research tasks simultaneously
- **Tool Integration** - Building and using custom AI tools
- **Production-Ready Architecture** - Scalable, maintainable AI agent systems

## 📚 Resources

- **LangGraph Documentation**: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph)
- **LangChain Documentation**: [python.langchain.com](https://python.langchain.com)
- **OpenAI API Reference**: [platform.openai.com/docs](https://platform.openai.com/docs)

---

**Happy Researching! 🚀**

Got questions? The code is well-commented and the architecture is designed to be easy to understand and extend.
