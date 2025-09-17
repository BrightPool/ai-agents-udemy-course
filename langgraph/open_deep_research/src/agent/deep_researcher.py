"""Main LangGraph implementation for the Deep Research Agent (OpenAI-only).

Simplified DAG version without supervisor/researcher loops.

Workflow:
1. Clarify with user (optional)
2. Write research brief
3. Generate concrete queries
4. Run queries asynchronously
5. Generate final report
"""

from typing import cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent.configuration import (
    Configuration,
)
from agent.prompts import (
    clarify_with_user_instructions,
    final_report_generation_prompt,
    generate_research_queries_prompt,
    transform_messages_into_research_topic_prompt,
)
from agent.state import (
    AgentState,
    ClarifyWithUser,
    ResearchQuestion,
)
from agent.utils import (
    get_api_key_for_model,
    get_model_token_limit,
    get_today_str,
    is_token_limit_exceeded,
    openai_search,
)

# Initialize a configurable model that we will use throughout the agent
# This allows dynamic configuration of model parameters at runtime
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


# =============================================================================
# PHASE 1: CLARIFICATION AGENT
# =============================================================================

"""
The Clarification Agent analyzes user queries to determine if they need clarification
before proceeding with research. This ensures the research scope is well-defined
and prevents wasted effort on ambiguous queries.
"""


async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[str]:
    """PHASE 1: Clarification Agent.

    Analyzes user messages to determine if clarification is needed before research begins.
    This prevents wasted effort on ambiguous or poorly scoped queries.

    Decision Logic:
    1. Check if clarification is enabled in configuration
    2. If enabled, analyze query using structured AI model
    3. Route to either clarification request or direct research planning

    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences

    Returns:
        Command routing to either END (with clarification question) or research planning
    """
    # Step 1: Configure (we always ask for clarification in this simplified DAG)
    configurable = Configuration.from_runnable_config(config)

    # Step 2: Configure AI model for clarification analysis
    messages = state["messages"]
    model_config: RunnableConfig = {
        "configurable": {
            "model": configurable.research_model,
            "max_tokens": configurable.research_model_max_tokens,
            "api_key": configurable.get_openai_api_key(),
        },
        "tags": ["langsmith:nostream"],  # Disable streaming for structured output
    }

    # Create model with structured output (ClarifyWithUser) and retry logic
    clarification_model = (
        configurable_model.with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )

    # Step 3: Analyze the user's query for clarity requirements
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), date=get_today_str()
    )
    response = cast(
        ClarifyWithUser,
        await clarification_model.ainvoke([HumanMessage(content=prompt_content)]),
    )

    # This is to stop infinite loops in case the LLM thinks we incrasingly need clarification
    if (
        state.get("clarification_attempts", 0)
        >= configurable.maximum_clarification_attempts
    ):
        return Command(
            goto="write_research_brief",
            update={
                "messages": [AIMessage(content=response.verification)],
                "clarification_attempts": {
                    "type": "override",
                    "value": state.get("clarification_attempts", 0) + 1,
                },
            },
        )

    if response.need_clarification:
        # Step 4: Always ask for clarification and end the run
        return Command(
            goto=END,
            update={
                "messages": [AIMessage(content=response.question)],
                "clarification_attempts": state.get("clarification_attempts", 0) + 1,
            },
        )
    else:
        return Command(
            goto="write_research_brief",
            update={
                "messages": [AIMessage(content=response.verification)],
                "clarification_attempts": {state.get("clarification_attempts", 0) + 1},
            },
        )


# =============================================================================
# PHASE 2: RESEARCH PLANNING AGENT
# =============================================================================

"""
The Research Planning Agent transforms user queries into structured research briefs
and initializes the supervisor agent. This is the bridge between user input and
the multi-agent research execution.
"""


async def write_research_brief(
    state: AgentState, config: RunnableConfig
) -> Command[str]:
    """PHASE 2: Research Planning Agent.

    Transforms user messages into a structured research brief that will guide
    the entire research process. This agent analyzes the user's request and
    creates a focused, actionable research plan.

    Process:
    1. Extract and analyze user messages using AI model
    2. Generate structured research question/brief
    3. Initialize supervisor agent with appropriate context and prompts

    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings

    Returns:
        Command routing to research supervisor with initialized research brief
    """
    # Step 1: Configure AI model for research brief generation
    configurable = Configuration.from_runnable_config(config)
    research_model_config: RunnableConfig = {
        "configurable": {
            "model": configurable.research_model,
            "max_tokens": configurable.research_model_max_tokens,
            "api_key": configurable.get_openai_api_key(),
        },
        "tags": ["langsmith:nostream"],  # Structured output needs non-streaming
    }

    # Create model with structured output (ResearchQuestion) for consistent formatting
    research_model = (
        configurable_model.with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # Step 2: Generate structured research brief from user input
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])), date=get_today_str()
    )
    response = cast(
        ResearchQuestion,
        await research_model.ainvoke([HumanMessage(content=prompt_content)]),
    )

    # Step 3: Proceed to generate concrete research queries
    return Command(
        goto="generate_research_queries",
        update={"research_brief": response.research_brief},
    )


# =============================================================================
# PHASE 3: GENERATE QUERIES AGENT
# =============================================================================


async def generate_research_queries(
    state: AgentState, config: RunnableConfig
) -> Command[str]:
    """Generate a set of concrete queries to research the brief."""
    configurable = Configuration.from_runnable_config(config)
    model_config: RunnableConfig = {
        "configurable": {
            "model": configurable.research_model,
            "max_tokens": configurable.research_model_max_tokens,
            "api_key": configurable.get_openai_api_key(),
        },
        "tags": ["langsmith:nostream"],
    }

    query_model = configurable_model.with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    ).with_config(model_config)

    prompt_content = generate_research_queries_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        research_brief=state.get("research_brief", ""),
        date=get_today_str(),
        num_queries=getattr(configurable, "num_queries", 6),
    )

    response = await query_model.ainvoke([HumanMessage(content=prompt_content)])
    # Parse lines into queries
    text = str(response.content)
    queries = [q.strip("- ") for q in text.splitlines() if q.strip()]
    # Trim to requested count
    queries = queries[: int(getattr(configurable, "num_queries", 6))]

    return Command(
        goto="run_queries",
        update={"queries": {"type": "override", "value": queries}},
    )


# =============================================================================
# PHASE 4: RUN QUERIES AGENT
# =============================================================================


async def run_queries(state: AgentState, config: RunnableConfig) -> Command[str]:
    """Execute all generated queries asynchronously using OpenAI search."""
    queries = state.get("queries", []) or []
    if not queries:
        return Command(goto="final_report_generation")

    # Invoke the OpenAI search tool (internally parallelized)
    results = await openai_search.ainvoke({"queries": queries}, config)

    return Command(
        goto="final_report_generation",
        update={"notes": {"type": "override", "value": [results]}},
    )


# =============================================================================
# PHASE 5: FINAL REPORT GENERATION
# =============================================================================

"""
Final Report Generation: The synthesis and reporting phase

This agent takes all the research findings from the multi-agent research process
and synthesizes them into a concise, well-structured final report.
Handles token limits gracefully with progressive truncation and retry logic.
"""


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """PHASE 6: Final Report Generation Agent.

    Synthesizes all collected research findings into a concise final report.
    This is the culmination of the entire research process, transforming raw research
    data into a structured, readable report for the user.

    Key Features:
    - Concise synthesis of key research findings
    - Structured report generation with clear sections
    - Token limit handling with progressive truncation
    - Retry logic for robust report generation
    - Error recovery and graceful degradation

    Report Structure:
    - Executive summary of key findings
    - Detailed analysis sections
    - Supporting evidence and data
    - Conclusions and implications
    - References and sources

    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys

    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)

    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    writer_model_config: RunnableConfig = {
        "configurable": {
            "model": configurable.final_report_model,
            "max_tokens": configurable.final_report_model_max_tokens,
            "api_key": get_api_key_for_model(configurable.final_report_model, config),
        },
        "tags": ["langsmith:nostream"],
    }

    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None

    while current_retry <= max_retries:
        try:
            # Create concise prompt with all research context
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str(),
            )

            # Generate the final report
            final_report = await configurable_model.with_config(
                writer_model_config
            ).ainvoke([HumanMessage(content=final_report_prompt)])

            # Return successful report generation
            return {
                "final_report": final_report.content,
                "messages": [final_report],
                **cleared_state,
            }

        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1

                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(
                        configurable.final_report_model
                    )
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [
                                AIMessage(
                                    content="Report generation failed due to token limits"
                                )
                            ],
                            **cleared_state,
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = (model_token_limit or 0) * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int((findings_token_limit or 0) * 0.9)

                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [
                        AIMessage(content="Report generation failed due to an error")
                    ],
                    **cleared_state,
                }

    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [
            AIMessage(content="Report generation failed after maximum retries")
        ],
        **cleared_state,
    }


# =============================================================================
# MAIN RESEARCH WORKFLOW CONSTRUCTION
# =============================================================================

"""
Main Deep Researcher Workflow: The complete end-to-end research orchestration

This is the master workflow that coordinates all phases of the research process.
It implements a sophisticated multi-agent research pipeline that transforms
user queries into concise research reports.

Workflow Architecture:
1. **Clarification Phase**: Analyze and clarify user intent
2. **Planning Phase**: Generate structured research briefs
3. **Research Phase**: Multi-agent parallel research execution
4. **Synthesis Phase**: Compress and organize findings
5. **Reporting Phase**: Generate final concise reports

Key Design Principles:
- **Modular Architecture**: Each phase is a specialized agent
- **State Management**: Complex state graphs manage agent interactions
- **Error Resilience**: Robust error handling throughout the pipeline
- **Scalability**: Easy to add new agents and research capabilities
"""

# Create the main research workflow with efficient state management
deep_researcher_builder = StateGraph(AgentState)

# Add core workflow nodes - simplified DAG phases
deep_researcher_builder.add_node(
    "clarify_with_user", clarify_with_user
)  # PHASE 1: Query analysis and clarification
deep_researcher_builder.add_node(
    "write_research_brief", write_research_brief
)  # PHASE 2: Research planning and brief generation
deep_researcher_builder.add_node(
    "generate_research_queries", generate_research_queries
)  # PHASE 3: Generate concrete queries
deep_researcher_builder.add_node("run_queries", run_queries)  # PHASE 4: Execute queries
deep_researcher_builder.add_node(
    "final_report_generation", final_report_generation
)  # PHASE 5: Final report synthesis and formatting

# Define the main workflow sequence - linear progression through research phases
deep_researcher_builder.add_edge(
    START, "clarify_with_user"
)  # Begin with user query analysis
deep_researcher_builder.add_edge("clarify_with_user", END)
deep_researcher_builder.add_edge("write_research_brief", "generate_research_queries")
deep_researcher_builder.add_edge("generate_research_queries", "run_queries")
deep_researcher_builder.add_edge(
    "run_queries", "final_report_generation"
)  # Query execution triggers report generation
deep_researcher_builder.add_edge(
    "final_report_generation", END
)  # Final report ends the workflow

# Compile the complete research orchestration system
deep_researcher = deep_researcher_builder.compile()
