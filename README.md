# SQL Agent API

A natural-language-to-SQL agent, served as a FastAPI HTTP API and orchestrated with [LangGraph](https://github.com/langchain-ai/langgraph). Ask a question in plain English, get back an answer grounded in your database — with schema-aware query generation, hallucination checks, and a read-only execution guard.

```
POST /chat
{ "input": "What were the top 5 best-selling artists last year?" }

→ { "response": "The top 5 best-selling artists were ...", "sql_query": "SELECT ..." }
```

## Features

- **Model-agnostic** — powered by [LiteLLM](https://github.com/BerriAI/litellm), so you can point it at OpenAI, Anthropic, Groq, a local model server, or anything LiteLLM supports, per request.
- **Prompt-injection detection** — regex-based screening rejects "ignore previous instructions"-style attempts before any LLM call is made.
- **Intent classification** — cheap heuristics short-circuit casual greetings/thanks; only ambiguous input costs an LLM call.
- **Schema-aware query generation** — the agent lists tables, fetches relevant schema, then generates SQL scoped to only the tables it was shown.
- **Hallucination + safety checks** — every generated query is parsed with [sqlglot](https://github.com/tobymao/sqlglot) and rejected if it references a table that doesn't exist or contains a DML/DDL statement (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, ...).
- **Read-only at the connection level, too** — a SQLAlchemy `before_execute` hook blocks mutating statements even if one slips past the query checker.
- **Automatic retries** — on a bad query or a DB error, the agent regenerates the query (up to `MAX_RETRIES`) before giving up gracefully.
- **Connection & schema caching** — engines, `SQLDatabase` wrappers, and table lists are cached per connection string to avoid repeated introspection.
- **Usage tracking** — every response includes prompt/completion/total token counts, API call count, and duration.

## Architecture

The agent is a LangGraph state machine (`app/services/graph.py`, nodes in `app/services/nodes.py`):

```
classify_intent
   ├── injection  → handle_injection → END
   ├── casual     → handle_casual    → END
   └── query
        └── list_tables → get_schema → generate_query → check_query
                                              ▲               │
                                              └── retry ──────┤
                                                               ▼
                                                        execute_query
                                              ▲                │
                                              └── retry ───────┤
                                                               ▼
                                                     generate_answer → END

  (any node can fall through to handle_error → END once MAX_RETRIES is hit)
```

## Quickstart

### Local (Python)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

uvicorn app.main:app --reload
```

The API starts on `http://localhost:8000`. With no `db_url` supplied, `/chat` falls back to the bundled Chinook sample database (`app/chinook.db`) — a classic music-store schema, useful for trying things out immediately.

### Docker

```bash
docker build -t sql-agent-api .
docker run -p 8000:8000 --env-file .env sql-agent-api
```

## API Reference

### `POST /chat`

Ask a natural-language question.

```jsonc
// Request
{
  "input": "How many customers are there?",
  "db_url": "postgresql://user:pass@host:5432/mydb",  // optional — omit to use the bundled demo DB
  "api_key": "sk-...",                                  // your LLM provider's API key
  "model_name": "gpt-4o-mini",                           // any LiteLLM-supported model string
  "api_base": null,                                      // optional, for local/self-hosted models
  "schema": null                                          // optional pre-fetched schema (see /db/tables)
}
```

```jsonc
// Response
{
  "response": "There are 59 customers.",
  "sql_query": "SELECT COUNT(*) FROM customers",
  "prompt_tokens": 412,
  "completion_tokens": 18,
  "total_tokens": 430,
  "api_calls": 3,
  "duration_ms": 842
}
```

### `POST /db/tables`

Validate a database URL and list its usable tables.

```jsonc
// Request
{ "db_url": "postgresql://user:pass@host:5432/mydb" }

// Response
{ "status": "success", "tables": ["customers", "orders", "products"] }
```

## Configuration

| Variable       | Default                    | Description                                  |
|----------------|-----------------------------|-----------------------------------------------|
| `CORS_ORIGINS` | `http://localhost:8000`     | Comma-separated list of allowed CORS origins. |

LLM provider credentials and model selection are passed **per request** in the `/chat` body, not via environment variables — this API is designed to be multi-tenant/multi-model out of the box.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Licensed under the [Apache License 2.0](LICENSE).
