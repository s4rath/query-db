from functools import partial
from langgraph.graph import StateGraph, END
from app.core.logger import log_info
from .nodes import (
    AgentState,
    classify_intent_node,
    handle_casual_node,
    handle_injection_node,
    list_tables_node,
    get_schema_node,
    generate_query_node,
    check_query_node,
    execute_query_node,
    generate_answer_node,
    handle_error_node,
)

MAX_RETRIES = 3

# def route_intent(state: AgentState) -> str:
#     """After intent classification — route to DB pipeline or casual handler."""
#     return "list_tables" if state.get("intent") == "query" else "handle_casual"

def route_intent(state: AgentState) -> str:
    intent = state.get("intent", "query")
    if intent == "injection":
        return "handle_injection"
    if intent == "casual":
        return "handle_casual"
    return "list_tables"

def should_retry_after_check(state: AgentState) -> str:
    """
    After execute_query:
    - If error AND retry count within limit → go back to generate_query
    - If error AND retries exhausted         → go to handle_error
    - If success                             → go to execute_query
    """
    if state.get("error"):
        if state.get("retry_count", 0) >= MAX_RETRIES:
            print(f"🔁 Retry limit ({MAX_RETRIES}) reached → handle_error")
            return "handle_error"
        print(f"🔁 Retrying query (attempt {state['retry_count']})...")
        return "generate_query"     # Loop back and rewrite the query
    return "execute_query"

def should_retry_after_execute(state: AgentState) -> str:
    """
    After execute_query:
    - DB error + retries left → retry generate_query
    - DB error + retries exhausted → handle_error
    - Success → generate_answer
    """
    if state.get("error"):
        if state.get("retry_count", 0) >= MAX_RETRIES:
            log_info(f"🔁 Retry limit ({MAX_RETRIES}) reached → handle_error")
            return "handle_error"
        log_info(f"🔁 Retrying query (attempt {state['retry_count']})...")
        return "generate_query"
    return "generate_answer"


def route_after_list_tables(state: AgentState) -> str:
    if state.get("error"):
        return "handle_error"
    return "get_schema"


def build_sql_agent(llm, tools, dialect: str = "sqlite"):

    # Bind dependencies into each node using partial
    # so the graph only passes `state` during execution
    nodes = {
        "classify_intent": partial(classify_intent_node, llm=llm),
        "handle_injection": handle_injection_node,
        "handle_casual":    partial(handle_casual_node, llm=llm),
        "list_tables":     partial(list_tables_node,   tools=tools),
        "get_schema":      partial(get_schema_node,    tools=tools, llm=llm),
        "generate_query":  partial(generate_query_node, llm=llm, dialect=dialect),
        "check_query":     check_query_node,
        "execute_query":   partial(execute_query_node, tools=tools),
        "generate_answer": partial(generate_answer_node, llm=llm),
        "handle_error":    handle_error_node,
    }

    # ── Build the graph ──────────────────────────────
    workflow = StateGraph(AgentState)

    # Register all nodes
    for name, fn in nodes.items():
        workflow.add_node(name, fn)

    # ── Entry point ──────────────────────────────────
    workflow.set_entry_point("classify_intent")

    # ── Intent routing ───────────────────────────────
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "handle_injection": "handle_injection",
            "list_tables":  "list_tables",    # DB question → full pipeline
            "handle_casual": "handle_casual", # Casual → direct response
        }
    )

    # ── Casual path ──────────────────────────────────
    workflow.add_edge("handle_injection", END)
    workflow.add_edge("handle_casual", END)

    # ── DB query pipeline ────────────────────────────
    workflow.add_conditional_edges(
        "list_tables",
        route_after_list_tables,
        {
            "get_schema": "get_schema",
            "handle_error": "handle_error",
        }
    )
    workflow.add_edge("get_schema",     "generate_query")
    workflow.add_edge("generate_query", "check_query")
    
    # Conditional: after check, retry or execute?
    workflow.add_conditional_edges(
        "check_query",
        should_retry_after_check,
        {
            "generate_query":  "generate_query",   # retry loop if hallucination
            "execute_query": "execute_query",    # continue to execution if OK
            "handle_error":    "handle_error",
        }
    )

    # Conditional: after execution, retry or answer?
    workflow.add_conditional_edges(
        "execute_query",
        should_retry_after_execute,
        {
            "generate_query":  "generate_query",   # retry loop if DB error
            "generate_answer": "generate_answer",
            "handle_error":    "handle_error",
        }
    )

    workflow.add_edge("generate_answer", END)
    workflow.add_edge("handle_error",    END)

    return workflow.compile()
