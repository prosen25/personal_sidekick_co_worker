# Personal Sidekick Co-Worker

An AI-powered personal co-worker app built with LangGraph + Gradio.
It runs a worker/evaluator loop with tool use (web search, Wikipedia, Python REPL, Playwright browser automation, file tools, and push notifications) to complete user tasks against success criteria.

## Project Structure

- `src/app.py`: Gradio UI entrypoint.
- `src/sidekick.py`: Orchestrates worker, tools, evaluator, and graph execution.
- `src/worker.py`: Task-solving agent with tool calling.
- `src/evaluator.py`: Evaluates whether success criteria are met.
- `src/sidekick_tools.py`: Tool factory/utilities (Playwright, file tools, search, etc.).
- `tests/`: Unit tests.

## Prerequisites

- Python `3.12+`
- Bash shell
- Internet access for model/tool APIs

## Setup and Run (Bash, from scratch)

Run the following commands in order.

### 1) Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

### 2) Go to project root

```bash
cd /path/to/personal_sidekick_co_worker
```

### 3) Create/update environment and install dependencies

```bash
uv sync --all-groups
```

### 4) Install Playwright browser binaries (required)

```bash
uv run playwright install chromium
```

### 5) Configure environment variables

Create `.env` with required keys:

```bash
cat > .env << 'EOF'
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key
WORKER_MODEL=gpt-4o-preview
EVALUATOR_MODEL=gpt-4o-mini

# Optional (only if you want push notifications)
PUSHOVER_USER=your_pushover_user
PUSHOVER_TOKEN=your_pushover_token
EOF
```

Create sandbox directory used by file tools:

```bash
mkdir -p sandbox
```

## Run the App

```bash
uv run python -m src.app
```

This launches the Gradio UI in your browser.

## Run Tests

```bash
uv run pytest
```

## Useful Commands

```bash
# Run a single test file
uv run pytest tests/test_sidekick.py

# Re-sync dependencies after dependency changes
uv sync --all-groups
```

## Troubleshooting

- If `uv` is not found after install, re-open terminal or re-run:
  - `export PATH="$HOME/.local/bin:$PATH"`
- If browser/tool errors mention Playwright executable missing, run:
  - `uv run playwright install chromium`
- If OpenAI/Serper calls fail, verify keys in `.env` and restart the app.
