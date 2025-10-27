# LangGraph Cloud Template (Python + uv)

This repository is a stripped-down starter kit for the LangGraph Cloud live examples. It keeps only the essentials so you can clone the layout, wire up your own graph, and deploy quickly with `uv`.

## What's Included

```
langgraph-cloud-template/
├── langgraph.json   # Placeholder wiring for LangGraph Cloud
├── pyproject.toml   # Core dependencies: LangGraph, LangChain, OpenAI, Pydantic, dotenv
├── .env.example     # Minimal environment variable sketch
└── src/agent/
    ├── __init__.py  # Empty export surface ready for your graph/tools
    ├── graph.py     # Start your graph definition here
    ├── models.py    # Place Pydantic schemas here
    ├── tools.py     # Register tool callables here
    └── utils.py     # Helper functions or shared code
```

The Python modules are intentionally empty—use them as blank canvases while you work through the course or replicate the live demos.

## Getting Started with uv

1. **Install uv**
   ```bash
   brew install uv  # or: pipx install uv
   ```

2. **Create a virtual environment**
   ```bash
   cd langgraph-cloud-template
   uv venv
   source .venv/bin/activate  # optional; you can also use `uv run`
   ```

3. **Install dependencies**
   ```bash
   uv sync
   # Add developer tooling if needed:
   # uv sync --group dev
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Fill in your LangGraph/OpenAI credentials and any course-specific settings
   ```

5. **Run the local LangGraph dev server**
   ```bash
   uv run langgraph dev
   ```

   Then open the printed URLs to explore the docs or LangGraph Studio.

## Customising the Template

- Define your graph in `src/agent/graph.py` and export it from `src/agent/__init__.py`.
- Add tools to `src/agent/tools.py` as you follow along with the live sessions.
- Describe request/response payloads in `src/agent/models.py`.
- Use `src/agent/utils.py` for shared helpers to keep your graph lean.

Feel free to duplicate this folder whenever you need a clean starting point for a new LangGraph Cloud agent.
