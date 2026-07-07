from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
)
from langchain_community.tools import QuerySQLDatabaseTool

def get_sql_tools(db, llm):
    list_tables_tool   = ListSQLDatabaseTool(db=db)
    get_schema_tool    = InfoSQLDatabaseTool(db=db)
    query_tool         = QuerySQLDatabaseTool(db=db)

    return {
        "list_tables":   list_tables_tool,
        "get_schema":    get_schema_tool,
        "execute_query": query_tool,
    }