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


system_prompt = """You are the Foodly Order Agent.  
Your job is to help with user requests about their orders: list recent orders, check order status, show order details, and request cancellations.  
Use only the provided order-related tools (e.g., get_all_orders, get_order_status, get_order_details, cancel_order).  
Always confirm the order ID with the user before taking any action.  
Never process payments, refunds, or handle policy questions — those belong to other agents.  
If a request is outside your scope, reply: “I can’t handle that, but I can escalate it for you.”
Keep replies friendly, accurate, and safe.
"""

###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
###############################################################################

uc_tool_names  = ("agents.orders_data.*",)


tools  = UCFunctionToolkit(function_names=list(uc_tool_names)).tools


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]



from helpers import LangGraphResponsesAgent, create_tool_calling_agent
AGENT = create_tool_calling_agent(llm, tools, system_prompt)
