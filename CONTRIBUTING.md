# Contributing

Thanks for your interest in improving SQL Agent API.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

uvicorn app.main:app --reload
```

The server reloads on file changes. You can exercise it with `curl` or any HTTP client against `POST /chat` and `POST /db/tables` (see [README.md](README.md) for request/response shapes). With no `db_url`, `/chat` runs against the bundled `app/chinook.db` sample database, so you can try changes without wiring up a real database.

## Project layout

- `app/api/routes.py` — HTTP endpoints and request/response models.
- `app/services/graph.py` — the LangGraph state machine definition and routing logic.
- `app/services/nodes.py` — the individual graph nodes (intent classification, query generation, validation, execution, answer generation).
- `app/services/tools.py` — LangChain SQL tools (list tables, get schema, execute query) bound to a given database.
- `app/services/agent.py` — wires an LLM + tools into a runnable agent.
- `app/core/database.py` — connection/engine/schema caching and the read-only execution guard.

## Testing

There is currently no automated test suite — if you're fixing a bug or adding a feature, contributions that add test coverage (e.g. `pytest` tests for `app/services/nodes.py`'s routing logic, or `app/core/database.py`'s caching behavior) are especially welcome.

## Submitting changes

1. Fork the repo and create a branch for your change.
2. Keep changes focused — smaller, single-purpose PRs are easier to review.
3. Describe the "why" in your PR description, not just the "what".
4. Open a pull request using the provided template.

## Reporting issues

Please use the bug report / feature request templates when opening an issue — they help us get the context we need (repro steps, expected vs. actual behavior, environment) without back-and-forth.
