"""Testimonial video generator DAG agent that mirrors n8n workflow.

Phases:
- Phase 1: Define personas, pick persona, generate 3 buying situations
- Phase 2: For each buying situation, generate a Veo3 prompt
- Phase 3: Create FAL queue requests, poll until complete, download videos,
           concatenate via ffmpeg, and return a local URL reference
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, TypedDict, cast

import fal_client  # type: ignore[import-untyped]
import httpx
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.utils import convert_to_secret_str
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent.models import (
    BuyingSituation,
    BuyingSituationsOutput,
    PromptOutput,
)

# Load environment variables from .env file if it exists
_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent
_env_file = _project_root / ".env"
if _env_file.exists():
    load_dotenv(_env_file)


class VideoAgentState(TypedDict, total=False):
    """Mutable state for the video agent across DAG nodes.

    Marked total=False because nodes incrementally add keys as they progress.
    """

    # Inputs
    persona_selection: str
    image_path: str | None

    # Derived
    execution_id: str
    work_dir: str
    personas: Dict[str, Dict[str, str]]
    persona: Dict[str, str]

    # Phase 1
    buying_situations: Dict[str, BuyingSituation]
    situations_list: List[BuyingSituation]
    situation_index: int

    # Phase 2
    prompts_json: List[PromptOutput]
    prompt_strings: List[str]

    # Phase 3
    run_results: List[Dict[str, Any]]
    video_urls: List[str]
    downloaded_files: List[str]
    final_output_path: str

    # Final
    result: Dict[str, str]
    # Diagnostics
    warnings: List[str]


def _get_llm(openai_api_key: str | None = None) -> ChatOpenAI:
    """Construct the OpenAI chat model using provided key or environment variable."""
    api_key_value = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key_value:
        raise ValueError("OPENAI_API_KEY must be provided or set in environment")
    return ChatOpenAI(
        model="gpt-5-mini",
        api_key=convert_to_secret_str(api_key_value),
        temperature=0.1,
        timeout=60,
    )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _ensure_fal_key(fal_key: str | None = None) -> str:
    """Ensure the fal SDK sees a valid API key and return it."""
    key = fal_key or os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing FAL_KEY or FAL_API_KEY environment variable for fal client, or provide fal_key in context."
        )
    # fal_client looks for FAL_KEY specifically
    os.environ.setdefault("FAL_KEY", key)
    return key


FAL_QUEUE_SLUG = "fal-ai/veo3"


def _extract_video_url(payload: Dict[str, Any]) -> str | None:
    video_section = payload.get("video")
    if isinstance(video_section, dict):
        for key in ("url", "video_url"):
            value = video_section.get(key)
            if isinstance(value, str):
                return value
    for key in ("video_url", "url"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def _as_dict(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        return data
    for attr in ("model_dump", "dict", "to_dict"):
        method = getattr(data, attr, None)
        if callable(method):
            try:
                result = method()
            except Exception:  # pragma: no cover - best effort
                continue
            if isinstance(result, dict):
                return result
    if hasattr(data, "__dict__"):
        return dict(getattr(data, "__dict__"))
    return {}


def init_and_create_directory(state: VideoAgentState) -> VideoAgentState:
    """Initialize execution and create a working directory under /tmp."""
    execution_id = uuid.uuid4().hex
    work_dir = Path(f"/tmp/n8n/{execution_id}")
    _ensure_dir(work_dir)
    # Clean any leftovers if re-used
    for f in ["videos.txt", "final_output.mp4"]:
        try:
            (work_dir / f).unlink(missing_ok=True)
        except Exception:
            pass
    return {
        "execution_id": execution_id,
        "work_dir": str(work_dir),
    }


def define_personas(state: VideoAgentState) -> VideoAgentState:
    """Define a catalog of personas to mirror the n8n node."""
    personas: Dict[str, Dict[str, str]] = {
        "developer": {
            "name": "Omar Ali",
            "age": "32",
            "gender": "Male",
            "location": "San Francisco",
            "occupation": "Software Engineer",
            "income": "$200,000",
            "background": (
                "This individual's life revolves around exploration and understanding. They are deeply curious about the world, people, and how things work, often delving into complex topics such as philosophy, history, and the human psyche. Their online presence is a reflection of their inner world - an active, thoughtful space where they seek to make connections, exchange ideas, and learn from others. They love the act of learning and self-improvement. They have a strong desire to make things better for themselves and the world around them, expressed through their writing and interactions. This person is committed to their craft, valuing beauty and truth, and finds joy in the small, everyday aspects of life. They are fascinated by the power of human connection and the potential for growth within communities. They are known for their inquisitive nature, often posing questions and seeking different perspectives to broaden their understanding of the world. They have an open mind and are constantly seeking a deeper understanding of complex concepts and human behavior. Their curiosity is a constant source of inspiration, driving them to explore new ideas, challenge their beliefs, and make meaningful connections with others."
            ),
        },
        "nurse": {
            "name": "Sarah Thomspon",
            "age": "55",
            "gender": "Woman",
            "location": "Manchester, UK",
            "occupation": "Nurse",
            "income": "GBP 55k",
            "background": (
                "Sarah is a dedicated Registered Nurse with over thirty years of experience in the NHS. Raised in a working-class family, she was inspired to pursue nursing by her strong desire to help others and provide compassionate care. She is married, owns her home, and is a devout Christian, finding strength in her faith and family. Sarah is a strong advocate for her patients and believes healthcare is a fundamental right. She enjoys reading, gardening, and spending time outdoors, which helps her manage the stress that comes with her demanding job. Throughout her career, she has consistently strived to improve her skills and provide the best possible care to her patients."
            ),
        },
        "foodie": {
            "name": "Emily Carter",
            "age": "21",
            "gender": "Female",
            "location": "Los Angeles, CA",
            "occupation": "Student",
            "income": "NA",
            "background": (
                "Emily Carter is a 21-year-old college student at UCLA, majoring in Digital Marketing. Originally from a small town in Oregon, she moved to Los Angeles to pursue her passion for marketing and content creation. She's an avid foodie, constantly exploring LA's diverse culinary landscape and documenting her experiences on TikTok and Instagram. Her content focuses on restaurant reviews, food trends, and lifestyle content, and she hopes to work in social media marketing after graduation."
            ),
        },
        "health-coach": {
            "name": "Clara Johnson",
            "age": "35",
            "gender": "Female",
            "location": "New York City, NY",
            "occupation": "Health Coach",
            "income": "$70k",
            "background": (
                "Clara, a certified holistic health coach, was raised in a culturally diverse household where natural wellness was a way of life. Inspired by her family's practices, she pursued a certification in holistic health and launched her coaching business in NYC. Through her work at NaturalHealth Inc., Clara focuses on promoting holistic beauty methods, integrating wellness with skincare, and educating others on the benefits of natural products. She is passionate about community workshops and uses her platform to share DIY beauty techniques, encouraging others to embrace natural wellness. She is a Buddhist who believes beauty comes from within and enjoys meditation, nature walks, and creating healthy meals."
            ),
        },
        "tattoo-artist": {
            "name": "Jordan McCulloch",
            "age": "Age 55",
            "gender": "Male",
            "location": "Portland, OR, USA",
            "occupation": "Tattoo Artist",
            "income": "$48k",
            "background": (
                "Jordan grew up in a small coastal town in Oregon before moving to Portland to embrace the city’s alternative scene and vibrant art community. With a talent for visual storytelling and a welcoming demeanor, Jordan found a calling in tattoo artistry, channeling creativity into meaningful body art for friends and clients. Passionate about authentic self-expression and forging real connections, Jordan has built a loyal client base and a network of close friends who share a love of indie music, art shows, and late-night philosophical discussions. Jordan values openness in relationships and tends to express themselves candidly, both in art and conversation. When not at the tattoo studio, Jordan spends time volunteering for LGBTQ+ causes, making zines, and exploring the city’s eclectic food scene."
            ),
        },
    }
    return {"personas": personas}


def set_persona(state: VideoAgentState) -> VideoAgentState:
    """Select the active persona based on input selection."""
    selection = state.get("persona_selection")
    personas = state.get("personas", {})
    if not selection or selection not in personas:
        raise ValueError("persona_selection is missing or invalid")
    return {"persona": personas[selection]}


def generate_buying_situations(
    state: VideoAgentState, config: RunnableConfig
) -> VideoAgentState:
    """Ask the LLM to propose three buying situations for the persona."""
    persona = state.get("persona")
    if persona is None:
        raise KeyError("persona missing; ensure set_persona ran before this node")

    # Extract API key from context
    ctx = getattr(config, "context", {})
    openai_api_key = ctx.get("openai_api_key") if isinstance(ctx, dict) else None
    llm = _get_llm(openai_api_key)
    system = SystemMessage(
        content=(
            "Role play as {name}, a {age} {gender} from {location}, works as a {occupation} "
            "earning {income}. {background}. You have been invited to review a new product "
            "concept and reveal some deep, personal, and meaningful insights into when you would buy this offer. "
            "Your responses should be great examples of initial reactions to the concept, but also reveal what moments "
            "triggered you to search for this product, how you considered competing alternatives, and why ultimately you "
            "chose to buy this specific offer.\n\n"
            "Respond only with a valid JSON object. No markdown, code blocks, explanation, or escape characters — just raw JSON.\n\n"
            "Rules:\n"
            "- Always choose exactly 3 buying_situations.\n"
            "- Return only a single root-level JSON object (no arrays).\n"
            "- Do not wrap the JSON in quotes.\n"
            '- Do not insert escape characters like \\n or \\".\n'
            '- Ensure all property names and string values use standard double quotes (").\n'
            "- Ensure each buying_situation includes: situation, location, trigger, evaluation, conclusion.\n"
            "- Do not stop mid-way; close all braces and brackets.\n"
        ).format(**persona)
    )

    example = '{"persona":"Rhys UK Entrepreneur","buying_situations":{"celebrating_milestone":{"situation":"Celebrating a business milestone","location":"Private members\' club in London","trigger":"Closed a major deal with a new international client","evaluation":"Considered champagne and high-end cocktails, but wanted something more personal and rooted in tradition","conclusion":"Chose a premium single malt whisky to mark the achievement and share with colleagues, as it symbolised craftsmanship and success"},"hosting_partners":{"situation":"Hosting international partners","location":"Home office in Manchester","trigger":"Inviting overseas investors to discuss the next round of funding","evaluation":"Compared various spirits that would reflect UK culture; whisky stood out as a distinctly British offering","conclusion":"Bought a respected Scottish whisky to showcase local heritage and create a more authentic, memorable experience for guests"},"relaxing_weekend":{"situation":"Relaxing after a long week","location":"Apartment balcony overlooking the city","trigger":"Needed a way to unwind after late nights preparing pitches and handling staff issues","evaluation":"Looked at beer, gin, and wine; whisky appealed because it felt slower, more intentional, and suited to reflection","conclusion":"Chose whisky as a ritual drink — poured neat into a favourite glass, symbolising a pause and moment of clarity"}}}'
    prompt = SystemMessage(
        content=(
            "Return JSON with fields persona (string) and buying_situations (object with exactly 3 keys).\n"
            f"Example format: {example}"
        )
    )
    raw_content = llm.invoke([system, prompt]).content  # type: ignore[arg-type]
    text = raw_content if isinstance(raw_content, str) else json.dumps(raw_content)
    # Parse and validate
    data = None
    for _ in range(2):
        try:
            data = json.loads(text)
            break
        except Exception:
            # Ask model to repair
            repair_raw = llm.invoke(
                [
                    SystemMessage(
                        content="Repair and return valid JSON only, same schema."
                    ),
                    SystemMessage(content=text),
                ]
            ).content  # type: ignore[arg-type]
            text = repair_raw if isinstance(repair_raw, str) else json.dumps(repair_raw)
    if data is None:
        raise ValueError("Failed to parse buying situations JSON")
    validated = BuyingSituationsOutput.model_validate(data)
    situations_list = list(validated.buying_situations.values())
    return {
        "buying_situations": validated.buying_situations,
        "situations_list": situations_list,
        "situation_index": 0,
        "prompts_json": [],
        "prompt_strings": [],
    }


def generate_prompt_for_current(
    state: VideoAgentState, config: RunnableConfig
) -> VideoAgentState:
    """Generate a Veo3 prompt for the current buying situation."""
    persona = state.get("persona")
    if persona is None:
        raise KeyError("persona missing; ensure set_persona ran before this node")
    idx = state.get("situation_index", 0)
    situations_list = state.get("situations_list")
    if situations_list is None:
        raise KeyError("situations_list missing; ensure generate_buying_situations ran")
    if not (0 <= idx < len(situations_list)):
        raise IndexError("situation_index out of range")
    situation = situations_list[idx]

    # Extract API key from context
    ctx = getattr(config, "context", {})
    openai_api_key = ctx.get("openai_api_key") if isinstance(ctx, dict) else None
    llm = _get_llm(openai_api_key)

    template = (
        "You are a director of a qualitative research agency that specialises in creating realistic simulated testimonials that capture buying situations for new product concepts.\n\n"
        "Your tasks are to:\n"
        "1. Take the persona and their buying situation.\n"
        "2. Craft:\n"
        "   - A single vivid persona description\n"
        "   - A scene appearance that is realistic and convincing considering the persona background\n"
        "   - A short 6–8 second quote that captures the emotion/reason for choosing the product\n"
        "3. Assemble a final prompt string that combines these elements into a realistic video reaction setup.\n\n"
        "Return a structured object strictly matching this Pydantic schema: PromptOutput(prompt: PromptModel(scene_description: SceneDescription(persona, appearance), quote, prompt_string)). No markdown."
        "\n\nPersona\n"
        f"Name: {persona['name']}\n"
        f"Age: {persona['age']}\n"
        f"Gender: {persona['gender']}\n"
        f"Location: {persona['location']}\n"
        f"Occupation: {persona['occupation']}\n"
        f"Income: {persona['income']}\n"
        f"Background: {persona['background']}\n\n"
        "Buying Situation\n"
        f"Situation: {situation.situation}\n"
        f"Location: {situation.location}\n"
        f"Trigger: {situation.trigger}\n"
        f"Evaluation: {situation.evaluation}\n"
        f"Conclusion: {situation.conclusion}"
    )

    structured_llm = llm.with_structured_output(PromptOutput)
    response = cast(
        PromptOutput, structured_llm.invoke([SystemMessage(content=template)])
    )

    existing_prompts = cast(List[PromptOutput], state.get("prompts_json", []))
    prompts_json: List[PromptOutput] = [*existing_prompts, response]
    existing_strings = cast(List[str], state.get("prompt_strings", []))
    prompt_strings: List[str] = [*existing_strings, response.prompt.prompt_string]
    return {
        "prompts_json": prompts_json,
        "prompt_strings": prompt_strings,
    }


def next_prompt_or_submit(
    state: VideoAgentState,
) -> Literal["continue_prompt_loop", "submit_fal_requests"]:
    """Route to continue generating prompts or submit the FAL requests."""
    idx = state.get("situation_index", 0)
    total = len(state.get("situations_list", []))
    if idx + 1 < total:
        return "continue_prompt_loop"
    return "submit_fal_requests"


def increment_prompt_index(state: VideoAgentState) -> VideoAgentState:
    """Increment the situation index for the next prompt generation."""
    return {"situation_index": state.get("situation_index", 0) + 1}


def submit_fal_requests(
    state: VideoAgentState, config: RunnableConfig
) -> VideoAgentState:
    """Run Veo3 requests synchronously and capture their outputs (KISS)."""
    # Ensure API key is available to fal client
    ctx = getattr(config, "context", {})
    fal_key = ctx.get("fal_key") if isinstance(ctx, dict) else None
    _ensure_fal_key(fal_key)

    prompt_strings = state.get("prompt_strings", [])
    run_results: List[Dict[str, Any]] = []
    video_urls: List[str] = []
    warnings: List[str] = (
        state.get("warnings", []).copy() if state.get("warnings") else []
    )

    for prompt in prompt_strings:
        try:
            # Use subscribe (queue) per fal docs; returns final result
            result = fal_client.subscribe(
                "fal-ai/veo3/fast",
                arguments={
                    "prompt": prompt,
                    "duration": "8s",
                    "auto_fix": True,
                    "resolution": "720p",
                },
                with_logs=False,
            )
            result_dict = _as_dict(result)
            run_results.append(result_dict)
            url = _extract_video_url(result_dict)
            if isinstance(url, str) and url:
                video_urls.append(url)
            else:
                # Try generic keys
                gen = result_dict.get("url") or result_dict.get("video_url")
                if isinstance(gen, str) and gen:
                    video_urls.append(gen)
                else:
                    warnings.append("No video URL in fal result")
        except Exception as exc:  # pragma: no cover - network failure path
            warnings.append(f"FAL run failed: {exc}")

    # Provide video_urls so downstream can skip collection
    return {"run_results": run_results, "video_urls": video_urls, "warnings": warnings}


def collect_video_urls(state: VideoAgentState) -> VideoAgentState:
    """Collect video URLs directly from fal_client results (KISS)."""
    run_results = state.get("run_results", [])
    video_urls: List[str] = []
    for payload in run_results:
        url = _extract_video_url(payload)
        if not url:
            # If the endpoint returns the URL at top-level, try generic keys
            url = payload.get("url") or payload.get("video_url")  # type: ignore[assignment]
        if not isinstance(url, str) or not url:
            raise ValueError("No video URL returned from fal run")
        video_urls.append(url)
    return {"video_urls": video_urls}


def download_videos(state: VideoAgentState) -> VideoAgentState:
    """Download video files to the working directory."""
    work_dir_str = state.get("work_dir")
    if not work_dir_str:
        raise KeyError("work_dir missing; ensure init_and_create_directory ran")
    work_dir = Path(work_dir_str)
    _ensure_dir(work_dir)
    files: List[str] = []
    with httpx.Client(timeout=None, follow_redirects=True) as client:
        for i, url in enumerate(state.get("video_urls", [])):
            file_name = f"{i:02d}.mp4"
            file_path = work_dir / file_name
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(file_path, "wb") as f:
                    for chunk in resp.iter_bytes():
                        if chunk:
                            f.write(chunk)
            files.append(str(file_path))
    return {"downloaded_files": files}


def prepare_concat_file(state: VideoAgentState) -> VideoAgentState:
    """Prepare ffmpeg concat file listing downloaded videos."""
    work_dir_str = state.get("work_dir")
    if not work_dir_str:
        raise KeyError("work_dir missing; ensure init_and_create_directory ran")
    work_dir = Path(work_dir_str)
    list_path = work_dir / "videos.txt"
    lines = []
    for p in sorted(work_dir.glob("*.mp4")):
        lines.append(f"file '{p.name}'\n")
    list_path.write_text("".join(lines), encoding="utf-8")
    return {}


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def merge_videos_ffmpeg(state: VideoAgentState) -> VideoAgentState:
    """Merge downloaded videos using ffmpeg concat demuxer."""
    work_dir_str = state.get("work_dir")
    if not work_dir_str:
        raise KeyError("work_dir missing; ensure init_and_create_directory ran")
    work_dir = Path(work_dir_str)
    if not _which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg.")
    videos_txt = work_dir / "videos.txt"
    if not videos_txt.exists():
        raise FileNotFoundError(f"Missing {videos_txt}")
    final_path = work_dir / "final_output.mp4"
    cmd = [
        "ffmpeg",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(videos_txt),
        "-c",
        "copy",
        "-y",
        str(final_path),
    ]
    subprocess.run(cmd, cwd=str(work_dir), check=True, capture_output=True)
    return {"final_output_path": str(final_path)}


def complete(state: VideoAgentState) -> VideoAgentState:
    """Produce the final result payload with a local URL reference."""
    execution_id = state.get("execution_id")
    if not execution_id:
        raise KeyError("execution_id missing; expected from init")
    video_url = f"http://localhost:3001/video/{execution_id}/final_output.mp4"
    return {
        "result": {
            "videoUrl": video_url,
            "executionId": execution_id,
            "message": "Video processing complete",
        }
    }


def prompt_loop_router(
    state: VideoAgentState,
) -> Literal["generate_prompt_for_current", "increment_index_and_loop"]:
    """Generate first, then decide whether to continue the loop."""
    return "generate_prompt_for_current"


# Build the graph mirroring the n8n flow
graph = (
    StateGraph(VideoAgentState)
    # Phase 0: Init + input
    .add_node("init_and_create_directory", init_and_create_directory)
    .add_node("define_personas", define_personas)
    .add_node("set_persona", set_persona)
    # Phase 1: Generate buying situations
    .add_node("generate_buying_situations", generate_buying_situations)
    # Phase 2: Loop to generate prompts per situation
    .add_node("generate_prompt_for_current", generate_prompt_for_current)
    .add_node("increment_prompt_index", increment_prompt_index)
    # Phase 3: Video generation and merging
    .add_node("submit_fal_requests", submit_fal_requests)
    .add_node("collect_video_urls", collect_video_urls)
    .add_node("download_videos", download_videos)
    .add_node("prepare_concat_file", prepare_concat_file)
    .add_node("merge_videos_ffmpeg", merge_videos_ffmpeg)
    .add_node("complete", complete)
    # Flow
    .add_edge(START, "init_and_create_directory")
    .add_edge("init_and_create_directory", "define_personas")
    .add_edge("define_personas", "set_persona")
    .add_edge("set_persona", "generate_buying_situations")
    # Prompt generation loop: generate -> decide -> either continue or move on
    .add_edge("generate_buying_situations", "generate_prompt_for_current")
    .add_conditional_edges(
        "generate_prompt_for_current",
        next_prompt_or_submit,
        {
            "continue_prompt_loop": "increment_prompt_index",
            "submit_fal_requests": "submit_fal_requests",
        },
    )
    .add_edge("increment_prompt_index", "generate_prompt_for_current")
    # Submit requests -> wait -> poll -> branch until all complete
    .add_edge("submit_fal_requests", "collect_video_urls")
    # After completion: fetch URLs -> download -> prepare -> merge -> complete
    .add_edge("collect_video_urls", "download_videos")
    .add_edge("download_videos", "prepare_concat_file")
    .add_edge("prepare_concat_file", "merge_videos_ffmpeg")
    .add_edge("merge_videos_ffmpeg", "complete")
    .add_edge("complete", END)
    .compile(name="Testimonial Video Generator DAG Agent")
)
