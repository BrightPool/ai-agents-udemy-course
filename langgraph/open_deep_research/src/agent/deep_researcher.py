"""Main LangGraph implementation for the Deep Research Agent (OpenAI-only).

This module implements a sophisticated multi-agent research system that orchestrates
complex research tasks through coordinated AI agents. The system follows a
hierarchical architecture:

1. **Clarification Agent**: Analyzes user queries for clarity and scope
2. **Supervisor Agent**: Breaks down complex research into manageable sub-tasks
3. **Researcher Agents**: Execute parallel research on individual topics
4. **Compression Agent**: Synthesizes findings into coherent summaries
5. **Report Generator**: Creates comprehensive final reports

Key Features:
- Multi-agent coordination with nested StateGraphs
- Parallel processing for speed and efficiency
- Intelligent task decomposition and planning
- Robust error handling and token limit management
- Structured research methodology with clear phases

Architecture:
- Main workflow: Clarification → Planning → Parallel Research → Synthesis → Reporting
- Supervisor subgraph: Coordinates research delegation and progress tracking
- Researcher subgraph: Individual research execution with tool integration
"""

import asyncio
from typing import cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
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
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from agent.state import (
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from agent.utils import (
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    remove_up_to_last_ai_message,
    think_tool,
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
    # Step 1: Check if clarification feature is enabled
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification and proceed directly to research planning
        return Command(goto="write_research_brief")

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

    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # Query needs clarification - end with question for user
        return Command(
            goto=END, update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Query is clear - proceed to research planning with verification
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]},
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

    # Step 3: Initialize supervisor agent with research brief and system instructions
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations,
    )

    # Return command that initializes supervisor with research context
    return Command(
        goto="research_supervisor",
        update={
            "research_brief": response.research_brief,  # The structured research plan
            "supervisor_messages": {
                "type": "override",  # Replace any existing supervisor messages
                "value": [
                    SystemMessage(
                        content=supervisor_system_prompt
                    ),  # Supervisor instructions
                    HumanMessage(content=response.research_brief),  # Research task
                ],
            },
        },
    )


# =============================================================================
# PHASE 3: SUPERVISOR AGENT
# =============================================================================

"""
The Supervisor Agent is the central coordinator of the research process. It analyzes
the research brief, breaks down complex tasks into manageable sub-tasks, and delegates
work to specialized researcher agents. This agent makes strategic decisions about
research direction and manages the overall research workflow.
"""


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[str]:
    """PHASE 3: Supervisor Agent (Main Coordinator).

    The lead research supervisor that orchestrates the entire research process.
    This agent analyzes the research brief and makes strategic decisions about
    how to break down and execute complex research tasks.

    Core Responsibilities:
    1. Analyze research brief and current progress
    2. Break down complex research into sub-tasks
    3. Delegate research work to specialized researcher agents
    4. Monitor progress and make strategic decisions
    5. Signal completion when research objectives are met

    Available Actions (via tools):
    - ConductResearch: Delegate specific research tasks to worker agents
    - ResearchComplete: Signal that research is finished
    - think_tool: Strategic reflection and planning

    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings

    Returns:
        Command routing to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    research_model_config: RunnableConfig = {
        "configurable": {
            "model": configurable.research_model,
            "max_tokens": configurable.research_model_max_tokens,
            "api_key": configurable.get_openai_api_key(),
        },
        "tags": ["langsmith:nostream"],
    }

    # Define supervisor's available tools for research coordination
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]

    # Configure model with tools, retry logic, and model settings
    research_model = (
        configurable_model.bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # Step 2: Generate supervisor's strategic response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)

    # Step 3: Update state and proceed to tool execution phase
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],  # Add supervisor's response
            "research_iterations": state.get("research_iterations", 0)
            + 1,  # Track iterations
        },
    )


async def supervisor_tools(
    state: SupervisorState, config: RunnableConfig
) -> Command[str]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.

    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase

    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings

    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = cast(AIMessage, supervisor_messages[-1])

    # Define exit criteria for research phase
    exceeded_allowed_iterations = (
        research_iterations > configurable.max_researcher_iterations
    )
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", ""),
            },
        )

    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}

    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call
        for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "think_tool"
    ]

    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(
            ToolMessage(
                content=f"Reflection recorded: {reflection_content}",
                name="think_tool",
                tool_call_id=tool_call["id"],
            )
        )

    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call
        for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductResearch"
    ]

    if conduct_research_calls:
        try:
            # Limit concurrent research units to prevent resource exhaustion
            allowed_conduct_research_calls = conduct_research_calls[
                : configurable.max_concurrent_research_units
            ]
            overflow_conduct_research_calls = conduct_research_calls[
                configurable.max_concurrent_research_units :
            ]

            # Execute research tasks in parallel
            research_tasks = [
                researcher_subgraph.ainvoke(
                    {
                        "researcher_messages": [
                            HumanMessage(content=tool_call["args"]["research_topic"])
                        ],
                        "research_topic": tool_call["args"]["research_topic"],
                    },
                    cast(RunnableConfig, config),
                )
                for tool_call in allowed_conduct_research_calls
            ]

            tool_results = await asyncio.gather(*research_tasks)

            # Create tool messages with research results
            for observation, tool_call in zip(
                tool_results, allowed_conduct_research_calls
            ):
                all_tool_messages.append(
                    ToolMessage(
                        content=observation.get(
                            "compressed_research",
                            "Error synthesizing research report: Maximum retries exceeded",
                        ),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )

            # Handle overflow research calls with error messages
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(
                    ToolMessage(
                        content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                        name="ConductResearch",  # String name for ToolMessage
                        tool_call_id=overflow_call["id"],
                    )
                )

            # Aggregate raw notes from all research results
            raw_notes_concat = "\n".join(
                [
                    "\n".join(observation.get("raw_notes", []))
                    for observation in tool_results
                ]
            )

            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]

        except Exception:
            # End research phase on error for simplicity in OpenAI-only setup
            return Command(
                goto=END,
                update={
                    "notes": get_notes_from_tool_calls(supervisor_messages),
                    "research_brief": state.get("research_brief", ""),
                },
            )

    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(goto="supervisor", update=update_payload)


# =============================================================================
# SUPERVISOR SUBGRAPH CONSTRUCTION
# =============================================================================

"""
Supervisor Subgraph: The core research coordination engine

This subgraph implements the supervisor's decision-making loop:
1. supervisor: Analyzes current state and makes strategic decisions
2. supervisor_tools: Executes the decisions (delegates research, reflects, completes)

The supervisor runs in a loop until research is complete, managing:
- Task decomposition and delegation
- Progress monitoring and iteration limits
- Strategic decision-making via think_tool
- Parallel research execution coordination
"""

# Create supervisor subgraph with its specialized state management
supervisor_builder = StateGraph(SupervisorState)

# Add core supervisor nodes
supervisor_builder.add_node("supervisor", supervisor)  # Strategic decision-making
supervisor_builder.add_node(
    "supervisor_tools",
    supervisor_tools,  # Decision execution and research delegation
)

# Define supervisor workflow: continuous loop until completion
supervisor_builder.add_edge(START, "supervisor")  # Start with strategic analysis

# Compile into executable subgraph for the main research workflow
supervisor_subgraph = supervisor_builder.compile()


# =============================================================================
# PHASE 4: RESEARCHER AGENT
# =============================================================================

"""
Researcher Agent: The specialized research execution engine

Each researcher agent is assigned a specific research sub-task by the supervisor.
They conduct focused, in-depth research using available tools and then compress
their findings into coherent summaries.

Key responsibilities:
- Execute focused research on assigned topics
- Use search tools to gather information
- Apply strategic thinking via think_tool
- Compress findings into structured summaries
- Handle tool failures gracefully
"""


async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[str]:
    """PHASE 4: Individual Researcher Agent.

    Specialized research execution for individual sub-tasks. Each researcher
    is assigned a specific research topic by the supervisor and conducts
    focused investigation using available tools.

    Research Process:
    1. Analyze assigned research topic from supervisor
    2. Use available tools (search, think_tool) to gather information
    3. Make strategic decisions about research direction
    4. Generate comprehensive findings

    Tool Integration:
    - OpenAI-powered search for information gathering
    - think_tool for strategic research planning
    - ResearchComplete to signal task completion

    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability

    Returns:
        Command routing to researcher_tools for tool execution
    """
    # Step 1: Load configuration and validate research environment
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])

    # Validate that research tools are available
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please ensure OpenAI API key is set."
        )

    # Step 2: Configure the researcher AI model with available tools
    research_model_config: RunnableConfig = {
        "configurable": {
            "model": configurable.research_model,
            "max_tokens": configurable.research_model_max_tokens,
            "api_key": configurable.get_openai_api_key(),
        },
        "tags": ["langsmith:nostream"],
    }

    # Prepare system prompt for focused research behavior
    researcher_prompt = research_system_prompt.format(
        mcp_prompt="",  # Removed MCP integration
        date=get_today_str(),
    )

    # Configure model with tools, retry logic, and research-specific settings
    research_model = (
        configurable_model.bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # Step 3: Generate researcher's strategic response to current findings
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)

    # Step 4: Update state and proceed to tool execution phase
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],  # Add researcher's analysis
            "tool_call_iterations": state.get("tool_call_iterations", 0)
            + 1,  # Track iterations
        },
    )


# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(
    state: ResearcherState, config: RunnableConfig
) -> Command[str]:
    """Execute tools called by the researcher, including search tools and strategic thinking.

    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search) - Information gathering
    3. ResearchComplete - Signals completion of individual research task

    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings

    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = cast(AIMessage, researcher_messages[-1])

    # Early exit if no tool calls were made
    has_tool_calls = bool(most_recent_message.tool_calls)
    if not has_tool_calls:
        return Command(goto="compress_research")

    # Step 2: Handle tool calls (search + think_tool)
    tools = await get_all_tools(config)
    tools_by_name: dict[str, object] = {}
    for tool in tools:
        from langchain_core.tools import BaseTool as _BaseTool

        if isinstance(tool, _BaseTool):
            name = getattr(tool, "name", "web_search")
            tools_by_name[name] = tool
        else:
            # Fallback for dict-style tool definitions
            name = getattr(tool, "get", lambda *_: "web_search")("name", "web_search")  # type: ignore[misc]
            tools_by_name[name] = tool

    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config)
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)

    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=observation, name=tool_call["name"], tool_call_id=tool_call["id"]
        )
        for observation, tool_call in zip(observations, tool_calls)
    ]

    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = (
        state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    )
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_iterations or research_complete_called:
        # End research and proceed to compression
        return Command(
            goto="compress_research", update={"researcher_messages": tool_outputs}
        )

    # Continue research loop with tool results
    return Command(goto="researcher", update={"researcher_messages": tool_outputs})


async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.

    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.

    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings

    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config(
        cast(
            RunnableConfig,
            {
                "configurable": {
                    "model": configurable.compression_model,
                    "max_tokens": configurable.compression_model_max_tokens,
                    "api_key": configurable.get_openai_api_key(),
                },
                "tags": ["langsmith:nostream"],
            },
        )
    )

    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])

    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(
        HumanMessage(content=compress_research_simple_human_message)
    )

    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3

    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(
                date=get_today_str()
            )
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages

            # Execute compression
            response = await synthesizer_model.ainvoke(messages)

            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join(
                [
                    str(message.content)
                    for message in filter_messages(
                        researcher_messages, include_types=["tool", "ai"]
                    )
                ]
            )

            # Return successful compression result
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content],
            }

        except Exception as e:
            synthesis_attempts += 1

            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue

            # For other errors, continue retrying
            continue

    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join(
        [
            str(message.content)
            for message in filter_messages(
                researcher_messages, include_types=["tool", "ai"]
            )
        ]
    )

    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content],
    }


# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState,
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)  # Main researcher logic
researcher_builder.add_node(
    "researcher_tools", researcher_tools
)  # Tool execution handler
researcher_builder.add_node(
    "compress_research", compress_research
)  # Research compression

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")  # Entry point to researcher
researcher_builder.add_edge("compress_research", END)  # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()


# =============================================================================
# PHASE 6: FINAL REPORT GENERATION
# =============================================================================

"""
Final Report Generation: The synthesis and reporting phase

This agent takes all the research findings from the multi-agent research process
and synthesizes them into a comprehensive, well-structured final report.
Handles token limits gracefully with progressive truncation and retry logic.
"""


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """PHASE 6: Final Report Generation Agent.

    Synthesizes all collected research findings into a comprehensive final report.
    This is the culmination of the entire research process, transforming raw research
    data into a structured, readable report for the user.

    Key Features:
    - Comprehensive synthesis of all research findings
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
            # Create comprehensive prompt with all research context
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
user queries into comprehensive research reports.

Workflow Architecture:
1. **Clarification Phase**: Analyze and clarify user intent
2. **Planning Phase**: Generate structured research briefs
3. **Research Phase**: Multi-agent parallel research execution
4. **Synthesis Phase**: Compress and organize findings
5. **Reporting Phase**: Generate final comprehensive reports

Key Design Principles:
- **Modular Architecture**: Each phase is a specialized agent
- **State Management**: Complex state graphs manage agent interactions
- **Error Resilience**: Robust error handling throughout the pipeline
- **Scalability**: Easy to add new agents and research capabilities
"""

# Create the main research workflow with comprehensive state management
deep_researcher_builder = StateGraph(AgentState)

# Add core workflow nodes - each represents a major research phase
deep_researcher_builder.add_node(
    "clarify_with_user", clarify_with_user
)  # PHASE 1: Query analysis and clarification
deep_researcher_builder.add_node(
    "write_research_brief", write_research_brief
)  # PHASE 2: Research planning and brief generation
deep_researcher_builder.add_node(
    "research_supervisor", supervisor_subgraph
)  # PHASE 3-5: Multi-agent research execution (supervisor + researchers + compression)
deep_researcher_builder.add_node(
    "final_report_generation", final_report_generation
)  # PHASE 6: Final report synthesis and formatting

# Define the main workflow sequence - linear progression through research phases
deep_researcher_builder.add_edge(
    START, "clarify_with_user"
)  # Begin with user query analysis
deep_researcher_builder.add_edge(
    "research_supervisor", "final_report_generation"
)  # Research completion triggers report generation
deep_researcher_builder.add_edge(
    "final_report_generation", END
)  # Final report ends the workflow

# Compile the complete research orchestration system
deep_researcher = deep_researcher_builder.compile()
