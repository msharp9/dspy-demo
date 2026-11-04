# dspy-demo

A student-friendly introduction to DSPy: signatures, modules, optimizers, and saving programs.


## Setup

This project uses `uv` for dependency management.

1.  **Install `uv`**:
    Follow the instructions at [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/).
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Install Python**:
    `uv` manages Python versions for you.
    ```bash
    uv python install
    ```

3.  **Install Dependencies**:
    ```bash
    uv sync
    ```


## Running the demos

The demos use a local [Ollama](https://ollama.com) model (`llama3.2`). Start Ollama and pull the model first:

```bash
ollama pull llama3.2
ollama pull llama3.1:8b   # reflection model for the GEPA demo
```

| Script | What it shows |
| --- | --- |
| `src/main.py` | Signatures, `Predict`, `ChainOfThought`, a `ReAct` agent with Wikipedia tools, and a multi-step `Module` |
| `src/class_demo.py` | Optimizing with `BootstrapFewShot` on GSM8K math problems, then saving and loading the program |
| `src/saving.py` | Saving a program as state-only JSON vs. the whole program |
| `src/optimizers.py` | Optimizing a 77-way intent classifier with `MIPROv2`, measuring accuracy before and after (~10 min locally) |
| `src/gepa_demo.py` | **New in DSPy 3:** `GEPA`, a reflective optimizer. The metric returns written feedback, and a stronger model (`REFLECTION_MODEL`, default `llama3.1:8b`) reads it to rewrite the prompt (~20 min locally; lower `max_metric_calls` for a quicker run) |

Shared pieces: `src/config.py` picks the model (change `MODEL` to try another one) and `src/data.py` loads the datasets.

Run one from the repo root, e.g. `uv run python src/main.py`. Saved programs are written to `dspy_*/` folders, which are git-ignored.


## Contributing

### Setup Pre-Commit (Optional)

All linting and testing is done in CI, however, if you want to run them locally, you can use prek (preferred) or pre-commit as documented below.

```bash
uvx prek install
```

### Running Tests
Run the test suite using `pytest`. This will also check that code coverage is at least 80%.
```bash
uv run pytest
```

### Linting and Formatting
Check for linting errors using `ruff`:
```bash
uv run ruff check
```

Format code:
```bash
uv run ruff format
```
