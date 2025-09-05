# Imports
import json
from typing import Annotated, Any, Generator, Optional, Sequence, TypedDict, Union
from uuid import uuid4

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
    VectorSearchRetrieverTool,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    convert_to_openai_messages,
)
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)



LLM_ENDPOINT_NAME = "databricks-gpt-oss-120b"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)


system_prompt = """You are the Foodly Escalation Agent.
Your job is to escalate a conversation to a human support specialist.

You MUST always know the user’s ID (user_id) or email_id before creating a ticket.

If user_id is not provided in the conversation context, do not guess or proceed.
Instead, ask the user politely for their registered email or phone number, or any ID you can use to look up their account.

Once you have user_id, use the escalation tool to create a support ticket.
Confirm to the user that their case has been escalated, and provide the ticket ID and ETA from the tool output.

"""

###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
###############################################################################

uc_tool_names  = ("agents.escalation.*",)


tools  = UCFunctionToolkit(function_names=list(uc_tool_names)).tools




from helpers import LangGraphResponsesAgent, create_tool_calling_agent
AGENT = create_tool_calling_agent(llm, tools, system_prompt)




