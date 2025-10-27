from typing import Annotated, List, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class CoachingAgentState(TypedDict, total=False):
    """Mutable state for the coaching agent across DAG nodes.

    This TypedDict defines the structure of the agent's state as it flows
    through the LangGraph execution graph. It contains both input parameters
    and derived fields that are populated during execution.

    Attributes:
        messages: Conversation history with automatic message addition support.
        user_id: Unique identifier for the user/conversation (required).
        k: Number of memories to retrieve (optional, uses default if not set).
        enable_graph: Whether to use graph-based memory relationships.
        memories_text: Retrieved memories joined as a string.
        assistant_text: Generated assistant response content.
    """

    # Conversation messages
    messages: Annotated[List[AnyMessage], add_messages]

    # Required inputs
    user_id: str

    # Optional inputs controlling Mem0 search
    k: int
    enable_graph: bool

    # Derived/ephemeral fields
    memories_text: str


class Context(TypedDict, total=False):
    """Execution context for the Mem0 coaching agent.

    Defines configuration options that can be provided via RunnableConfig.context
    when executing the graph, or sourced from environment variables. These
    settings control various aspects of the agent's behavior and external
    service connections.

    Attributes:
        openai_api_key: OpenAI API key for LLM access.
        mem0_default_k: Default number of memories to retrieve in searches.
        mem0_enable_graph: Whether to enable graph-based memory relationships.
        qdrant_host: Qdrant vector database hostname.
        qdrant_port: Qdrant vector database port number.
        neo4j_uri: Neo4j graph database connection URI.
        neo4j_username: Neo4j database username.
        neo4j_password: Neo4j database password.
    """

    openai_api_key: str
    mem0_default_k: int
    mem0_enable_graph: bool
    qdrant_host: str
    qdrant_port: int
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
