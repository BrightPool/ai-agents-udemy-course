# Video Response Generator DAG Agent with LangGraph

A sophisticated AI agent built with LangGraph that demonstrates **advanced Directed Acyclic Graph (DAG) patterns** for generating brand video responses using fictitious Veo3 personas. This agent showcases LangGraph's powerful directional flow capabilities for creating diverse video content across multiple specialized nodes.

## How It Works

### Core DAG Architecture

The agent operates through a sophisticated **9-node DAG workflow** with intelligent routing and specialized video content generation:

1. **Content Classification** → Analyzes incoming requests and determines content type
2. **Specialist Routing** → Routes to appropriate specialist based on content requirements
3. **Specialized Processing** → Each specialist handles specific video content domains
4. **Tool Execution** → Specialists use domain-specific tools for content generation
5. **Quality Evaluation** → Monitors content quality throughout the generation process
6. **Decision Making** → Determines next steps based on quality and complexity
7. **Human Review** → Escalates complex content for human review
8. **Content Continuation** → Routes to different specialists for enhancement

### Advanced DAG Flow Patterns

The agent demonstrates **LangGraph's directional flow capabilities** through multiple specialized content generation nodes:

```
Brand Request: "Create a promotional video for our new product"

1. classify_content_request → Determines: content_type="script", video_type="promotional", veo3_persona="professional"
2. route_after_classification → Routes to: "script_specialist"
3. script_specialist → Generates video script with professional Veo3 persona
4. should_use_tools → Decides: "script_tool_node" (needs script generation)
5. script_tool_node → Executes script generation tools
6. script_specialist → Refines script with brand voice
7. quality_evaluator → Evaluates: quality_score=0.8
8. final_decision → Determines: END (high-quality content)
```

**Technical Implementation:**

- **9 Specialized Nodes**: Each with distinct content generation responsibilities
- **Conditional Routing**: Smart decision-making between content specialists
- **Parallel Processing**: Multiple specialists can work simultaneously
- **State Tracking**: Monitors quality and processed nodes
- **Dynamic Escalation**: Routes complex content to human review

## Features

- **Multi-Specialist Architecture**: Script, Prompt, Response, Analysis, and Generation specialists
- **Intelligent Content Classification**: Automatic categorization and Veo3 persona assignment
- **Dynamic Routing**: Smart routing based on content type and priority level
- **Quality Monitoring**: Real-time content quality tracking
- **Human Review Integration**: Automatic escalation for complex content generation
- **Tool Specialization**: Each specialist has access to relevant content generation tools
- **State Persistence**: Maintains context across multiple content generation interactions
- **LangGraph Server**: Runs as a local server with LangGraph Studio integration

## Project Structure

```
customer_support_dag_agent/
├── langgraph.json              # LangGraph configuration and tool definitions
├── pyproject.toml              # Python dependencies and project metadata
├── .env.example               # Environment variables template (API keys)
├── README.md                   # This comprehensive documentation
├── run_graph.py               # Main runner script with examples
└── src/
    └── agent/
        ├── __init__.py         # Package initialization
        ├── graph.py           # Advanced DAG workflow with 9 specialized nodes
        ├── models.py          # Pydantic models for type safety
        ├── tools.py           # Video content generation tools with Veo3 personas
        └── documentation/     # Brand documentation system
            ├── shipping.txt   # Shipping policies and information
            ├── returns.txt    # Return and refund policies
            ├── products.txt   # Product warranties and specifications
            ├── account.txt    # Account management and login
            └── payment.txt    # Payment methods and security
```

## Setup

### 1. Install uv

```bash
brew install uv
# or:
# pipx install uv
```

### 2. Create and activate a virtual environment

```bash
cd customer_support_dag_agent
uv venv
source .venv/bin/activate
# Tip: You can skip activation and prefix commands with `uv run` instead.
```

### 3. Install dependencies

```bash
uv sync
# Include dev tools (ruff, pytest, etc.):
# uv sync --group dev
```

### 4. Set Up Environment Variables

Create a `.env` file with your API keys:

```bash
cp .env.example .env
# Edit .env and add your API keys
LANGSMITH_API_KEY=lsv2...
ANTHROPIC_API_KEY=sk-ant-api03-...
```

### 5. Launch LangGraph Server

Start the LangGraph API server locally:

```bash
uv run langgraph dev
```

Sample output:

```
>    Ready!
>
>    - API: http://localhost:2024
>
>    - Docs: http://localhost:2024/docs
>
>    - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### 6. Test in LangGraph Studio

Open LangGraph Studio in your browser using the URL provided in the output:

```
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### 7. Test the API

Install the LangGraph Python SDK (dev-only):

```bash
uv add --group dev langgraph-sdk
```

Send a message to the agent:

```python
from langgraph_sdk import get_client
import asyncio

client = get_client(url="http://localhost:2024")

async def main():
    async for chunk in client.runs.stream(
        None,  # Threadless run
        "agent", # Name of assistant. Defined in langgraph.json.
        input={
        "messages": [{
            "role": "human",
            "content": "Create a promotional video script for our new tech product",
            }],
        "brand_name": "TechCorp",
        "brand_tone": "innovative and professional",
        "brand_values": ["innovation", "quality", "customer focus"],
        },
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")

asyncio.run(main())
```

## Usage Examples

### Script Generation

```
Input: "Create a promotional video script for our new tech product"
DAG Flow: classify_content_request → script_specialist → script_tool_node → quality_evaluator → END
Output: Complete video script with dialogue, narration, and call-to-action
```

### Visual Prompt Creation

```
Input: "Generate visual prompts for our brand's social media video"
DAG Flow: classify_content_request → prompt_specialist → prompt_tool_node → quality_evaluator → END
Output: Detailed visual prompts for video generation
```

### Brand Response Generation

```
Input: "Create a response to customer feedback using our friendly persona"
DAG Flow: classify_content_request → response_specialist → response_tool_node → quality_evaluator → END
Output: Authentic brand response using Veo3 persona
```

### Comprehensive Content Generation

```
Input: "URGENT: Create complete video content package for product launch"
DAG Flow: classify_content_request → generation_specialist → generation_tool_node → quality_evaluator → human_review
Output: Complete content package requiring human review
```

## DAG Node Architecture

### 1. Content Classification Node

**Purpose**: Analyzes incoming content requests and determines generation strategy
**Input**: Brand content request
**Output**: 
- `content_type`: "script", "prompt", "response", "analysis", "generation"
- `video_type`: "promotional", "educational", "testimonial", "social", "advertisement"
- `veo3_persona`: "professional", "creative", "casual", "authoritative", "friendly"
- `priority_level`: "low", "medium", "high", "urgent"

**Classification Logic**:
```python
def classify_content_request(state: VideoResponseState) -> Dict[str, Any]:
    # Keyword analysis for content type
    # Video type assessment based on request
    # Veo3 persona selection based on brand needs
    # Priority assessment based on urgency indicators
```

### 2. Specialist Nodes (5 Specialized Content Generators)

#### Script Specialist
- **Expertise**: Video script writing, dialogue creation, storytelling
- **Tools**: `generate_video_script_tool`, `analyze_brand_tone_tool`
- **Routing**: Handles all script and dialogue generation requests

#### Prompt Specialist
- **Expertise**: Visual prompt creation, scene composition, visual storytelling
- **Tools**: `create_video_prompt_tool`, `analyze_brand_tone_tool`
- **Routing**: Processes all visual prompt and scene description requests

#### Response Specialist
- **Expertise**: Brand response generation, Veo3 persona development, audience engagement
- **Tools**: `generate_veo3_response_tool`, `analyze_brand_tone_tool`
- **Routing**: Handles brand response and persona-based content requests

#### Analysis Specialist
- **Expertise**: Content quality analysis, brand alignment assessment, performance evaluation
- **Tools**: `analyze_brand_tone_tool`
- **Routing**: Manages content analysis and quality assessment requests

#### Generation Specialist
- **Expertise**: Comprehensive content creation, multi-format generation, complete content packages
- **Tools**: All tools available (`generate_video_script_tool`, `create_video_prompt_tool`, `generate_veo3_response_tool`)
- **Routing**: Handles complex, multi-format content generation requests

### 3. Tool Execution Nodes (5 Specialized Tool Sets)

Each specialist has access to domain-specific content generation tools:

- **Script Tools**: Video script generation and brand tone analysis
- **Prompt Tools**: Visual prompt creation and brand tone analysis
- **Response Tools**: Veo3 persona response generation and brand tone analysis
- **Analysis Tools**: Brand tone analysis and content evaluation
- **Generation Tools**: Full access to all tools for comprehensive content creation

### 4. Quality Evaluation Node

**Purpose**: Monitors content quality throughout the generation process
**Input**: Generated content and specialist responses
**Output**:
- `quality_score`: Numerical score (0.0-1.0)
- `requires_human`: Boolean flag for human review

**Evaluation Logic**:
```python
def quality_evaluator(state: VideoResponseState) -> Dict[str, Any]:
    # Analyze generated content for quality indicators
    # Calculate quality score based on content assessment
    # Determine if human review is needed
```

### 5. Decision Making Node

**Purpose**: Makes final routing decisions based on quality and complexity
**Input**: Quality score and review requirements
**Output**: Routes to "human_review", "continue_generation", or END

**Decision Logic**:
```python
def final_decision(state: VideoResponseState) -> Literal["human_review", "continue_generation", END]:
    # High quality → END
    # Low quality → continue_generation
    # Requires human → human_review
```

### 6. Human Review Node

**Purpose**: Handles escalation to human content reviewers for complex generation
**Input**: Generated content summary and quality metrics
**Output**: Professional review request with content summary

### 7. Content Continuation Node

**Purpose**: Routes content to different specialists when initial generation needs enhancement
**Input**: Current content generation state
**Output**: Routing message to generation specialist for additional improvements

## Veo3 Persona System

### Persona Characteristics

The agent uses five distinct Veo3 personas for brand content generation:

#### Professional Persona
- **Characteristics**: Formal, authoritative, trustworthy, business-focused
- **Use Cases**: Corporate videos, B2B content, formal announcements
- **Voice**: Clear, confident, solution-oriented
- **Visual Style**: Clean, minimalist, corporate, sophisticated lighting

#### Creative Persona
- **Characteristics**: Innovative, artistic, imaginative, visually-driven
- **Use Cases**: Product launches, creative campaigns, artistic content
- **Voice**: Inspiring, innovative, visually-focused
- **Visual Style**: Dynamic, artistic, vibrant colors, creative angles

#### Casual Persona
- **Characteristics**: Relaxed, friendly, approachable, conversational
- **Use Cases**: Social media content, behind-the-scenes, informal updates
- **Voice**: Natural, conversational, relatable
- **Visual Style**: Natural lighting, relaxed composition, everyday settings

#### Authoritative Persona
- **Characteristics**: Expert, knowledgeable, confident, credible
- **Use Cases**: Educational content, expert testimonials, thought leadership
- **Voice**: Expert, knowledgeable, confident
- **Visual Style**: Strong composition, confident framing, expert positioning

#### Friendly Persona
- **Characteristics**: Warm, personable, helpful, engaging
- **Use Cases**: Customer service content, community building, support videos
- **Voice**: Warm, helpful, engaging
- **Visual Style**: Warm lighting, approachable angles, welcoming environments

## Technical Architecture

### DAG State Management

The agent uses LangGraph's `StateGraph` with enhanced state tracking:

```python
class VideoResponseState(TypedDict):
    # Core conversation state
    messages: Annotated[List[AnyMessage], add_messages]
    brand_name: str
    brand_tone: str
    brand_values: List[str]
    
    # Video generation context
    video_type: str  # "promotional", "educational", "testimonial", "social", "advertisement"
    target_audience: str
    video_length: str  # "short", "medium", "long"
    
    # DAG flow state
    content_type: str  # "script", "prompt", "response", "analysis", "generation"
    priority_level: str  # "low", "medium", "high", "urgent"
    requires_human: bool
    processed_nodes: List[str]  # Track which nodes have been processed
    quality_score: float  # Track quality throughout flow
    
    # Veo3 persona context
    veo3_persona: str  # "professional", "creative", "casual", "authoritative", "friendly"
```

### Advanced Routing Logic

The DAG implements sophisticated conditional routing:

```python
# Route from classification to appropriate specialist
.add_conditional_edges(
    "classify_content_request",
    route_after_classification,
    {
        "script_specialist": "script_specialist",
        "prompt_specialist": "prompt_specialist", 
        "response_specialist": "response_specialist",
        "analysis_specialist": "analysis_specialist",
        "generation_specialist": "generation_specialist"
    }
)

# Route from specialists to tools or quality evaluation
.add_conditional_edges(
    "script_specialist",
    should_use_tools,
    {
        "script_tool_node": "script_tool_node",
        "quality_evaluator": "quality_evaluator"
    }
)
```

### Tool Specialization

Each specialist has access to domain-specific tools:

```python
# Script specialist tools
script_tool_node = ToolNode(tools=[generate_video_script_tool, analyze_brand_tone_tool])

# Prompt specialist tools  
prompt_tool_node = ToolNode(tools=[create_video_prompt_tool, analyze_brand_tone_tool])

# Response specialist tools
response_tool_node = ToolNode(tools=[generate_veo3_response_tool, analyze_brand_tone_tool])

# Generation specialist tools (full access)
generation_tool_node = ToolNode(tools=[generate_video_script_tool, create_video_prompt_tool, generate_veo3_response_tool])
```

## Key DAG Features

### Multi-Specialist Content Architecture

**Implementation**:

```python
def script_specialist(state: VideoResponseState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Specialized script generation with domain expertise."""
    # Script-specific LLM configuration
    # Script-specific tools and prompts
    # Script-specific response generation

def prompt_specialist(state: VideoResponseState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Specialized visual prompt generation with creative expertise."""
    # Prompt-specific visual creativity
    # Prompt-specific tools and prompts
    # Prompt-specific response generation
```

**Benefits**:

- **Domain Expertise**: Each specialist is optimized for specific content generation areas
- **Tool Specialization**: Specialists only have access to relevant content generation tools
- **Focused Output**: Specialists provide more accurate and relevant content
- **Scalability**: Easy to add new specialists for new content domains

### Intelligent Content Classification

**Implementation**:

```python
def classify_content_request(state: VideoResponseState, runtime: Runtime[Context]) -> Dict[str, Any]:
    query_lower = query.lower()
    
    # Determine content type
    if any(word in query_lower for word in ["script", "dialogue", "narration"]):
        content_type = "script"
    elif any(word in query_lower for word in ["prompt", "description", "visual"]):
        content_type = "prompt"
    # ... additional classification logic
    
    # Determine Veo3 persona
    if any(word in query_lower for word in ["professional", "corporate", "business"]):
        veo3_persona = "professional"
    elif any(word in query_lower for word in ["creative", "artistic", "innovative"]):
        veo3_persona = "creative"
    # ... persona selection logic
```

**Benefits**:

- **Automatic Routing**: Content requests are automatically routed to appropriate specialists
- **Persona Selection**: Veo3 personas are automatically selected based on brand needs
- **Context Awareness**: Classification considers brand context and content requirements
- **Flexible Classification**: Easy to modify classification rules for new content types

### Dynamic Quality Monitoring

**Implementation**:

```python
def quality_evaluator(state: VideoResponseState, runtime: Runtime[Context]) -> Dict[str, Any]:
    quality_score = state.get("quality_score", 0.5)
    
    # Analyze generated content for quality indicators
    if isinstance(last_message, AIMessage):
        content_lower = content.lower()
        
        # Positive indicators
        if any(word in content_lower for word in ["excellent", "high-quality", "engaging"]):
            quality_score += 0.2
        # Negative indicators
        elif any(word in content_lower for word in ["unable", "cannot", "limited"]):
            quality_score -= 0.1
    
    # Determine if human review is needed
    requires_human = (
        quality_score < 0.3 or 
        state.get("priority_level") == "urgent" or
        len(processed_nodes) >= 3  # After 3 nodes, consider human review
    )
```

**Benefits**:

- **Real-time Monitoring**: Quality is tracked throughout the content generation process
- **Proactive Review**: Low quality triggers human review
- **Adaptive Routing**: Quality influences routing decisions
- **Quality Assurance**: Ensures brands receive appropriate level of content quality

### Human Review Integration

**Implementation**:

```python
def human_review(state: VideoResponseState, runtime: Runtime[Context]) -> Dict[str, Any]:
    review_message = AIMessage(
        content=f"""
I've generated comprehensive video content that requires human review and refinement.

CONTENT SUMMARY:
- Content Type: {state.get('content_type', 'generation')}
- Video Type: {state.get('video_type', 'advertisement')}
- Veo3 Persona: {state.get('veo3_persona', 'friendly')}
- Nodes Processed: {', '.join(processed_nodes)}
- Quality Score: {quality_score:.2f}

The content has been created using the specified Veo3 persona and brand guidelines.
Please review the generated content and provide feedback for any necessary adjustments.
"""
    )
```

**Benefits**:

- **Seamless Transition**: Professional handoff with content context preservation
- **Context Transfer**: Human reviewers receive complete content generation summary
- **Quality Handling**: Complex content is flagged appropriately
- **Brand Experience**: Clear communication about review process

## DAG Flow Visualization

The LangGraph Studio provides visual representation of the complex DAG:

```
START
  ↓
classify_content_request
  ↓
route_after_classification
  ├── script_specialist
  ├── prompt_specialist  
  ├── response_specialist
  ├── analysis_specialist
  └── generation_specialist
  ↓
should_use_tools
  ├── script_tool_node → script_specialist
  ├── prompt_tool_node → prompt_specialist
  ├── response_tool_node → response_specialist
  ├── analysis_tool_node → analysis_specialist
  ├── generation_tool_node → generation_specialist
  └── quality_evaluator
  ↓
final_decision
  ├── human_review → END
  ├── continue_generation → generation_specialist
  └── END
```

## Customization

### Adding New Content Specialists

1. **Create Specialist Node**:

   ```python
   def new_specialist(state: VideoResponseState, runtime: Runtime[Context]) -> Dict[str, Any]:
       """New specialist for specific content domain."""
       # Specialist-specific LLM configuration
       # Specialist-specific tools and prompts
       # Specialist-specific response generation
   ```

2. **Add Routing Logic**:

   ```python
   def route_after_classification(state: VideoResponseState) -> Literal["new_specialist", ...]:
       content_type = state.get("content_type", "generation")
       if content_type == "new_content_type":
           return "new_specialist"
   ```

3. **Update Graph Structure**:

   ```python
   .add_node("new_specialist", new_specialist)
   .add_conditional_edges(
       "classify_content_request",
       route_after_classification,
       {"new_specialist": "new_specialist", ...}
   )
   ```

### Modifying Veo3 Personas

- **Persona Characteristics**: Update persona descriptions in tools
- **Persona Selection**: Modify persona selection logic in classification
- **Persona Tools**: Adjust tools to better support specific personas

### Adjusting Quality Evaluation

- **Quality Indicators**: Modify positive/negative keyword lists
- **Review Thresholds**: Adjust quality score requirements
- **Human Review Criteria**: Modify escalation conditions

## Working with the DAG

### Development Workflow

1. **Start Server**: `uv run langgraph dev`
2. **Test in Studio**: Use the web UI to visualize DAG flow
3. **API Testing**: Use the Python SDK for programmatic testing
4. **Iterate**: Modify nodes and routing logic as needed
5. **Deploy**: Use LangGraph Platform for production deployment

### DAG Visualization

The LangGraph Studio provides visual representation of:

- **Node Execution Flow**: See which specialists are processing content requests
- **State Transitions**: Monitor quality scores and routing decisions
- **Conditional Routing**: Understand decision-making logic
- **Error Handling**: Track error recovery and escalation paths

### State Inspection

Monitor state changes in real-time:

- **Content Classification**: See how requests are categorized and prioritized
- **Specialist Routing**: Track which specialists handle different content types
- **Quality Tracking**: Monitor content quality throughout the generation process
- **Review Decisions**: Understand when and why human review occurs

## Course Demonstration Notes

This agent demonstrates advanced AI agent patterns:

- **Complex DAG Architecture**: Multi-node workflows with sophisticated content generation routing
- **Specialist Pattern**: Domain-specific agents with specialized content generation tools and knowledge
- **Dynamic Routing**: Intelligent decision-making between multiple content generation paths
- **Quality Monitoring**: Real-time content quality assessment and escalation
- **Human-AI Collaboration**: Seamless handoff between AI and human content reviewers
- **State Management**: Complex state tracking across multiple content generation interactions
- **Tool Specialization**: Domain-specific tool access and usage for content generation
- **Production-Ready Architecture**: Scalable, maintainable multi-agent content generation systems

**Key Technical Demonstrations:**

1. **DAG Flow Patterns**: Classification → Routing → Specialization → Evaluation → Decision
2. **Multi-Specialist Architecture**: Domain-specific agents with specialized content generation capabilities
3. **Intelligent Routing**: Conditional edges based on content analysis and quality assessment
4. **Quality-Driven Escalation**: Dynamic escalation based on real-time content quality assessment
5. **Complex State Management**: Multi-dimensional state tracking across content generation nodes

Perfect for understanding how to build production-ready AI agents with sophisticated workflow orchestration and multi-agent content generation coordination.

## Development Workflow

1. **Start Server**: `uv run langgraph dev`
2. **Test in Studio**: Use the web UI for interactive DAG testing
3. **API Testing**: Use the Python SDK for programmatic testing
4. **Iterate**: Modify DAG structure and restart server to see changes
5. **Deploy**: Use LangGraph Platform for production deployment

This demonstrates the complete LangGraph development lifecycle from local development to production deployment with advanced DAG patterns for brand video content generation.