"""Mem0-powered Coaching LangGraph Agent.

This module implements a conversational coaching agent that leverages Mem0
for long-term memory and personalized responses. The agent follows a
three-step process:

1) Search Mem0 for relevant memories (top K, with optional graph enabled)
2) Generate a reply using an OpenAI chat model with a coaching system prompt
3) Write the user's message back to Mem0 for long-term memory
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

from agent.utils import (
    extract_latest_user_text,
    get_default_k,
    get_mem0_client,
    get_openai_llm,
    load_env_if_exists,
)

load_env_if_exists()


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
    assistant_text: str


def _extract_latest_user_text(state: CoachingAgentState) -> str:
    """Extract the most recent user message from the conversation state.

    Internal helper function that wraps the utility function for state-specific
    message extraction.

    Args:
        state: Current agent state containing conversation messages.

    Returns:
        str: Content of the latest human message, or empty string if none found.
    """
    return extract_latest_user_text(state.get("messages", []))


def mem0_search_node(
    state: CoachingAgentState, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """Query Mem0 for top-K relevant memories and join them into a string.

    This node searches the user's memory store for relevant past conversations
    and context that can inform the coaching response. It handles various
    response formats from the Mem0 SDK and ensures robust error handling.

    Args:
        state: Current agent state containing user input and search parameters.
        runtime: LangGraph runtime providing access to configuration context.

    Returns:
        Dict[str, Any]: State update containing 'memories_text' field with
                        concatenated relevant memories, or empty dict if no
                        user text or search fails.

    Note:
        Uses best-effort error handling - search failures are logged but don't
        interrupt the conversation flow.
    """
    user_text = _extract_latest_user_text(state)
    if not user_text:
        return {}

    # Determine K and graph flag from context/env defaults
    k = int(state.get("k") or get_default_k(runtime, fallback=3))

    m = get_mem0_client(runtime)

    memories_text = ""
    try:
        # Use limit=k to align with Mem0 Python SDK
        results: Any = m.search(
            query=user_text, user_id=state.get("user_id", "default"), limit=int(k)
        )

        # Normalize to a list regardless of SDK return shape
        items: List[Any] = []
        if isinstance(results, dict) and isinstance(results.get("results"), list):
            items = results.get("results", [])
        elif isinstance(results, list):
            items = results

        # Sort by score if present, else keep order
        def score_of(item: Any) -> float:
            """Extract score from memory item for sorting."""
            try:
                return float(item.get("score", 0.0)) if isinstance(item, dict) else 0.0
            except Exception:
                return 0.0

        items_sorted = sorted(items, key=score_of, reverse=True) if items else []
        topn = items_sorted[: max(1, int(k))] if items_sorted else []
        memory_texts: List[str] = []
        for r in topn:
            if isinstance(r, dict):
                mem_val = r.get("memory")
                if isinstance(mem_val, str):
                    memory_texts.append(mem_val)
            elif isinstance(r, str):
                memory_texts.append(r)
        memories_text = "\n".join(memory_texts)
    except Exception as exc:  # Best-effort memory; don't fail the run
        print(f"⚠️  Mem0 search failed: {exc}")  # noqa: T201

    return {"memories_text": memories_text}


COACH_SYSTEM_PROMPT = (
    "You are a a highly sought-after executive coach and psychologist. Your diverse background "
    "includes experience as a mountain guide, BASE jumper, and founder of One Day Coaching. "
    "Your coaching philosophy centers on transformative change, emphasizing mindset shifts over "
    "incremental improvements. You are known for coaching elite athletes, to multiple elite "
    "competition victories. Your approach incorporates frameworks like the Circle of Control and "
    "Helicopter View to foster self-leadership, presence, and resilience. He values reflection, "
    "gratitude, and a focus on process-oriented goals. You exist to help people achieve sustainable success.\n\n"
    "Philosophy: transformations, not incremental changes.\n\n"
    "Core Philosophy\n"
    "- Change vs Transformation:\n"
    "- Change = skill acquisition, reversible, fades with neglect (pull-ups, languages).\n"
    "- Transformation = permanent mindset shift, irreversible (butterfly, popcorn, fatherhood).\n"
    "- Coaching goal → transformation.\n"
    "- Presence & Awareness: anchor in the moment through gratitude, sensory focus, reflection.\n"
    "- Self-leadership: regulate inner states (focus, gratitude, energy) like dashboard instruments.\n"
    "- Coaching style: questions, not prescriptions; trigger self-discovery, not obedience.\n"
    "- Passion vs Profession: preserve passion → prevents burnout, sustains fire.\n\n"
    "Frameworks & Tools\n"
    "- Circle of Control → agency vs victimhood; discard uncontrollables.\n"
    "- Helicopter View → shift perspective, step outside tunnel vision, self-coaching from altitude.\n"
    "- Time Jump → imagine future state already achieved, describe backward.\n"
    "- 1–10 Scale → precision self-assessment, 0.1-step improvements.\n"
    "- Gratitude Practice → daily anchor, builds resilience.\n"
    "- Checklists & Rules → externalize memory, reduce errors under stress.\n"
    "- Reflection Loop → act, reflect, adjust; expert hallmark.\n"
    "- Success Recipe Analysis → extract repeatable patterns from past wins.\n"
    "- Fun Injection → joy fuels discipline, sustains long goals.\n"
    "- Grandchildren Perspective → big-picture values orientation.\n"
    "- Anger Diagnostic → anger = unmet needs indicator.\n\n"
    "Motivation & Goal-Setting\n"
    "Goal Types:\n"
    "- Ranking → external, unstable, unhelpful.\n"
    "- Performance → measurable, objective metrics.\n"
    "- Process → controllable actions, most effective.\n"
    "- Mastery → long-term growth, most enduring.\n"
    "- Minimal Success Index (MSI) → define “good enough” baseline, balance ambition with realism.\n"
    "- Wave Model of Training → intensity cycles; build base, peak later.\n"
    "- Small Goals → Momentum → micro-wins sustain progression.\n"
    "- Discipline vs Motivation → discipline bridges low-motivation periods.\n\n"
    "Mindset Principles\n"
    "- Satisfaction Formula → Reality ÷ Expectations; adjust expectations to increase satisfaction.\n"
    "- Weakness Strategy → accept/manage weaknesses, amplify strengths.\n"
    "- Flow Channel → balance challenge + skill for optimal state.\n"
    "- Post-Peak Valleys → dips are natural, part of growth cycle.\n"
    "- Risk & Resilience → safety first; resilience from reframing setbacks.\n"
    "- Comparison Trap → avoid performance drain by focusing inward.\n\n"
    "Support & Team Dynamics\n"
    "- Micro-check-ins: “How are you?” → continuous self-awareness.\n"
    "- Non-directive support: athlete defines needs; coach listens, reflects.\n"
    "- Trust & Honesty: foundation of sustainable performance.\n"
    "- Bubble Focus: stay inward-focused despite competition/noise.\n"
    "- Key Themes for AI Trainer Design\n"
    "- Transformation-oriented training, not just skill drills.\n"
    "- Structured questioning as primary coaching tool.\n\n"
    "Explicit frameworks (Circle of Control, Helicopter View, Time Jump, MSI).\n"
    "- Goal-setting architecture: Process → Performance → Mastery (ranking minimized).\n"
    "- Fun & joy as motivational multipliers.\n"
    "- Reflection loops baked into training cycles.\n"
    "- Emphasis on presence, gratitude, emotional regulation.\n"
    "- Focus on self-leadership and autonomy.\n\n"
    "Communication style:\n"
    "- Be brief, sharp, and practical.\n"
    "- Prefer 1–2 sentences over paragraphs\n"
    "- ask a powerful question if its unclear how you can help the user.\n"
    "- Only expand if asked.\n"
    "- Pick one idea (the most relevant to the user query) linked to your key coaching concepts, and find a way to make it actionable\n"
)


def llm_node(state: CoachingAgentState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Generate an assistant reply using the coaching system prompt and memories.

    This node is the core of the coaching agent, combining the system prompt
    with retrieved memories to generate personalized coaching responses.
    It handles message formatting and ensures proper response structure.

    Args:
        state: Current agent state containing messages and retrieved memories.
        runtime: LangGraph runtime providing access to configuration context.

    Returns:
        Dict[str, Any]: State update containing the AI response message and
                        assistant text content. Returns empty dict if no user
                        text is available.

    Note:
        Includes a fallback response mechanism in case the LLM doesn't return
        a properly formatted AIMessage.
    """
    llm = get_openai_llm(runtime)
    user_text = _extract_latest_user_text(state)
    if not user_text:
        return {}

    memories = state.get("memories_text", "")

    system_content = COACH_SYSTEM_PROMPT
    if memories:
        system_content += (
            f"\n\nHere is some additional information on your pupil:\n{memories}"
        )

    system_message = SystemMessage(content=system_content)

    response = llm.invoke([system_message] + state.get("messages", []))
    if isinstance(response, AIMessage):
        assistant_text = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        return {"messages": [response], "assistant_text": assistant_text}
    # Fallback (should not generally happen)
    return {
        "messages": [AIMessage(content="I'm here to help.")],
        "assistant_text": "I'm here to help.",
    }


def mem0_add_node(
    state: CoachingAgentState, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """Write the user's latest message to Mem0 for long-term memory.

    This final node stores the user's input in the memory system to build
    a persistent understanding of the user's context, preferences, and
    conversation history for future interactions.

    Args:
        state: Current agent state containing the user's message to store.
        runtime: LangGraph runtime providing access to configuration context.

    Returns:
        Dict[str, Any]: Empty dictionary (no state changes needed).

    Note:
        Uses best-effort error handling - memory storage failures are logged
        but don't interrupt the conversation flow.
    """
    user_text = _extract_latest_user_text(state)
    if not user_text:
        return {}

    m = get_mem0_client(runtime)
    try:
        m.add(user_text, user_id=state.get("user_id", "default"))
    except Exception as exc:  # Best-effort write; don't fail the run
        print(f"⚠️  Mem0 add failed: {exc}")  # noqa: T201
    return {}


# Compiled LangGraph for the Mem0-powered coaching agent
# Execution flow: START → mem0_search → llm → mem0_add → END
graph = (
    StateGraph(CoachingAgentState, context_schema=Context)
    .add_node("mem0_search", mem0_search_node)
    .add_node("llm", llm_node)
    .add_node("mem0_add", mem0_add_node)
    .add_edge(START, "mem0_search")
    .add_edge("mem0_search", "llm")
    .add_edge("llm", "mem0_add")
    .add_edge("mem0_add", END)
    .compile(name="Mem0 Coaching Agent")
)
