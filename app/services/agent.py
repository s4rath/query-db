import time
from langchain_litellm import ChatLiteLLM
from langchain_community.utilities import SQLDatabase
from .tools import get_sql_tools
from .graph import build_sql_agent
from app.core.config import settings
from app.core.logger import log_info, log_error


def build_agent_from_db(db: SQLDatabase, llm):
    """Factory function to create an agent for a specific database."""
    tools = get_sql_tools(db, llm)
    return build_sql_agent(llm, tools, dialect=db.dialect)
    

def run_agent(
    user_input: str,
    db: SQLDatabase,
    api_key: str = None,
    model_name: str = None,
    api_base: str = None,
    schema: dict = None,
    start_time: float = None
):
    log_info(f"Running agent with input: {user_input}")

    start_time = start_time or time.perf_counter()

    try:
        if not model_name:
            raise ValueError(
                "Model name is required. Please provide a model name in the request."
            )

        if not api_key and not api_base:
            raise ValueError(
                "API key (or api_base for local models) is required. "
                "Please provide them in the request."
            )

        _api_key = api_key or "dummy-local-key"

        llm = ChatLiteLLM(
            model=model_name,
            temperature=0,
            api_key=_api_key,
            api_base=api_base,
        )

        agent = build_agent_from_db(db, llm)

        result = agent.invoke({
            "question": user_input,
            "tables": "",
            "schema": "",
            "schema_dict": schema or {},
            "query": "",
            "query_checked": False,
            "query_result": "",
            "final_answer": "",
            "error": "",
            "retry_count": 0,
            "messages": [],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
        })

        # Log usage metrics to console
        log_info(
            "LLM Usage Metrics",
            prompt_tokens=result.get("prompt_tokens", 0),
            completion_tokens=result.get("completion_tokens", 0),
            total_tokens=result.get("total_tokens", 0),
            api_calls=result.get("api_calls", 0)
        )

        return {
            "response": result["final_answer"],
            "sql_query": result.get("query"),
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
            "api_calls": result.get("api_calls", 0),
            "duration_ms": int((time.perf_counter() - start_time) * 1000)
        }

    except Exception as e:
        log_error("Error in agent execution", error=e)
        raise e