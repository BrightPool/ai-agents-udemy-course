# AI Agents Udemy Course

## Prerequisites

- Python 3.11+ installed
- [uv](https://docs.astral.sh/uv/) installed
- Relevant API keys (e.g., Anthropic, LangSmith)

## Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality and security. The hooks include:

- **gitleaks**: Scans for secrets and sensitive information in your code

### Installation

1. Install pre-commit:

   ```bash
   brew install pre-commit
   ```

2. Install the pre-commit hooks:

   ```bash
   pre-commit install
   ```

3. (Optional) Run hooks on all files:
   ```bash
   pre-commit run --all-files
   ```

The hooks will automatically run before each commit to check for secrets and other issues.
