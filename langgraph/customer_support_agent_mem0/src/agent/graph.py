"""Mem0-powered Coaching LangGraph Agent.

This graph mirrors the provided n8n workflow:
1) Search Mem0 for relevant memories (top K, with optional graph enabled)
2) Generate a reply using an OpenAI chat model with a coaching system prompt
3) Write the user's message back to Mem0 for long-term memory
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict, cast

import httpx
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.utils import convert_to_secret_str
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

# Load environment variables from .env file if it exists
_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent
_env_file = _project_root / ".env"

if _env_file.exists():
    load_dotenv(_env_file)
    print(f"✅ Loaded environment variables from {_env_file}")  # noqa: T201
else:
    print(f"ℹ️  No .env file found at {_env_file}")  # noqa: T201
    print("   Environment variables will be loaded from system environment or runtime context")  # noqa: T201


class Context(TypedDict, total=False):
    """Execution context for the Mem0 coaching agent.

    These can be provided via RunnableConfig.context when executing the graph
    or sourced from environment variables.
    """

    openai_api_key: str
    mem0_base_url: str
    mem0_default_k: int
    mem0_enable_graph: bool


class CoachingAgentState(TypedDict, total=False):
    """Mutable state for the coaching agent across DAG nodes."""

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


def _get_openai_llm(runtime: Runtime[Context]) -> ChatOpenAI:
    ctx = getattr(runtime, "context", None)
    api_key_value: Optional[str] = None
    if isinstance(ctx, dict):
        api_key_value = cast(Optional[str], ctx.get("openai_api_key"))
    if not api_key_value:
        api_key_value = os.getenv("OPENAI_API_KEY")

    return ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=convert_to_secret_str(api_key_value or ""),
        temperature=0.3,
        timeout=30,
    )


def _get_mem0_base_url(runtime: Runtime[Context]) -> str:
    ctx = getattr(runtime, "context", None)
    if isinstance(ctx, dict):
        base = cast(Optional[str], ctx.get("mem0_base_url"))
        if base:
            return base.rstrip("/")
    return (os.getenv("MEM0_BASE_URL") or "http://localhost:8000").rstrip("/")


def _extract_latest_user_text(state: CoachingAgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            return msg.content
    return ""


def mem0_search_node(state: CoachingAgentState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Query Mem0 for top-K relevant memories and join them into a string."""
    user_text = _extract_latest_user_text(state)
    if not user_text:
        return {}

    base_url = _get_mem0_base_url(runtime)

    # Determine K and graph flag from state or context/env defaults
    default_k = 3
    ctx = getattr(runtime, "context", None)
    if isinstance(ctx, dict) and isinstance(ctx.get("mem0_default_k"), int):
        default_k = cast(int, ctx.get("mem0_default_k"))

    k = int(state.get("k") or default_k)

    default_enable_graph = True
    if isinstance(ctx, dict) and isinstance(ctx.get("mem0_enable_graph"), bool):
        default_enable_graph = cast(bool, ctx.get("mem0_enable_graph"))

    enable_graph_flag = bool(state.get("enable_graph") if state.get("enable_graph") is not None else default_enable_graph)

    payload = {
        "user_id": state.get("user_id", "default"),
        "query": user_text,
        "k": k,
        "enable_graph": enable_graph_flag,
    }

    memories_text = ""
    try:
        with httpx.Client(timeout=20) as client:
            resp = client.post(f"{base_url}/api/v1/memories/search", json=payload)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", []) if isinstance(data, dict) else []
            # Normalize and sort by score if present
            def score_of(item: Any) -> float:
                try:
                    return float(item.get("score", 0.0))
                except Exception:
                    return 0.0

            if isinstance(results, list):
                results_sorted = sorted(results, key=score_of, reverse=True)
                top10 = results_sorted[:10]
                memory_texts: List[str] = []
                for r in top10:
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
    """Generate an assistant reply using the coaching system prompt and memories."""
    llm = _get_openai_llm(runtime)
    user_text = _extract_latest_user_text(state)
    memories = state.get("memories_text", "")

    system_content = COACH_SYSTEM_PROMPT
    if memories:
        system_content += f"\n\nHere is some additional information on your pupil:\n{memories}"

    system_message = SystemMessage(content=system_content)

    response = llm.invoke([system_message] + state.get("messages", []))
    if isinstance(response, AIMessage):
        assistant_text = response.content if isinstance(response.content, str) else str(response.content)
        return {"messages": [response], "assistant_text": assistant_text}
    # Fallback (should not generally happen)
    return {"messages": [AIMessage(content="I'm here to help.")], "assistant_text": "I'm here to help."}


def mem0_add_node(state: CoachingAgentState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Write the user's latest message to Mem0 for long-term memory."""
    user_text = _extract_latest_user_text(state)
    if not user_text:
        return {}

    base_url = _get_mem0_base_url(runtime)
    enable_graph_flag = bool(
        state.get("enable_graph")
        if state.get("enable_graph") is not None
        else True
    )

    payload = {
        "user_id": state.get("user_id", "default"),
        "memory": user_text,
        "enable_graph": enable_graph_flag,
    }

    try:
        with httpx.Client(timeout=20) as client:
            resp = client.post(f"{base_url}/api/v1/memories", json=payload)
            resp.raise_for_status()
    except Exception as exc:  # Best-effort write; don't fail the run
        print(f"⚠️  Mem0 add failed: {exc}")  # noqa: T201
    return {}


# Build the graph mirroring the n8n flow
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
