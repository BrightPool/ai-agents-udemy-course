"""Video response generation tools for brand content with Veo3 personas.

This module defines the tools used throughout the video response generator agent
for creating brand video content using fictitious Veo3 personas.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.tools import tool


@tool
def generate_video_script_tool(
    brand_name: str,
    video_type: str,
    veo3_persona: str,
    target_audience: str,
    video_length: str,
    brand_tone: str,
    content_description: str
) -> str:
    """Generate a video script for brand content using Veo3 persona.
    
    Args:
        brand_name: Name of the brand
        video_type: Type of video (promotional, educational, testimonial, social, advertisement)
        veo3_persona: Veo3 persona to use (professional, creative, casual, authoritative, friendly)
        target_audience: Target audience for the video
        video_length: Length of video (short, medium, long)
        brand_tone: Brand tone and voice
        content_description: Description of the content to create
    
    Returns:
        Generated video script with dialogue, narration, and call-to-action
    """
    # Veo3 persona characteristics
    persona_characteristics = {
        "professional": "formal, authoritative, trustworthy, business-focused",
        "creative": "innovative, artistic, imaginative, visually-driven",
        "casual": "relaxed, friendly, approachable, conversational",
        "authoritative": "expert, knowledgeable, confident, credible",
        "friendly": "warm, personable, helpful, engaging"
    }
    
    persona_desc = persona_characteristics.get(veo3_persona, "friendly, approachable")
    
    # Video length guidelines
    length_guidelines = {
        "short": "15-30 seconds, concise messaging",
        "medium": "30-60 seconds, balanced content",
        "long": "60+ seconds, detailed storytelling"
    }
    
    length_desc = length_guidelines.get(video_length, "30-60 seconds")
    
    script_template = f"""
VIDEO SCRIPT FOR {brand_name.upper()}

Video Type: {video_type.title()}
Veo3 Persona: {veo3_persona.title()} ({persona_desc})
Target Audience: {target_audience}
Length: {length_desc}
Brand Tone: {brand_tone}

SCRIPT:

[SCENE 1: Opening Hook]
Veo3 Persona: "{content_description[:50]}... Let me show you how {brand_name} can help you."

[SCENE 2: Main Content]
Veo3 Persona: "As someone who understands {target_audience}, I know that {brand_name} delivers exactly what you need. Our {brand_tone} approach ensures you get results that matter."

[SCENE 3: Call to Action]
Veo3 Persona: "Ready to experience the difference? Visit {brand_name} today and see why we're the trusted choice for {target_audience}."

VISUAL NOTES:
- Maintain {veo3_persona} persona throughout
- Use {brand_tone} visual style
- Include {brand_name} branding elements
- Optimize for {video_length} format

TONE GUIDELINES:
- Voice: {persona_desc}
- Pace: Appropriate for {video_length}
- Energy: Engaging but {brand_tone}
"""
    
    return script_template


@tool
def analyze_brand_tone_tool(
    brand_name: str,
    brand_values: List[str],
    existing_content: Optional[str] = None
) -> str:
    """Analyze and define brand tone for video content.
    
    Args:
        brand_name: Name of the brand
        brand_values: List of brand values and principles
        existing_content: Optional existing content to analyze
    
    Returns:
        Brand tone analysis and recommendations
    """
    values_text = ", ".join(brand_values) if brand_values else "quality, innovation, customer focus"
    
    analysis = f"""
BRAND TONE ANALYSIS FOR {brand_name.upper()}

Brand Values: {values_text}

TONE RECOMMENDATIONS:

1. VOICE CHARACTERISTICS:
   - Professional yet approachable
   - Confident but not arrogant
   - Innovative while being reliable
   - Customer-centric messaging

2. LANGUAGE STYLE:
   - Clear and concise communication
   - Industry-appropriate terminology
   - Positive and solution-oriented
   - Inclusive and accessible

3. EMOTIONAL TONE:
   - Trustworthy and credible
   - Inspiring and motivating
   - Supportive and helpful
   - Forward-thinking and progressive

4. VEo3 PERSONA ALIGNMENT:
   - Professional: Formal, authoritative, business-focused
   - Creative: Innovative, artistic, visually-driven
   - Casual: Relaxed, friendly, conversational
   - Authoritative: Expert, knowledgeable, confident
   - Friendly: Warm, personable, engaging

5. CONTENT GUIDELINES:
   - Always lead with value proposition
   - Use storytelling to connect emotionally
   - Include clear calls-to-action
   - Maintain consistency across all touchpoints

BRAND VOICE MATRIX:
- What {brand_name} says: {values_text}
- How {brand_name} says it: {brand_name.lower()}-appropriate tone
- Why {brand_name} says it: To build trust and drive action
"""
    
    return analysis


@tool
def create_video_prompt_tool(
    brand_name: str,
    video_type: str,
    veo3_persona: str,
    visual_style: str,
    scene_description: str,
    brand_colors: Optional[List[str]] = None
) -> str:
    """Create detailed visual prompts for video generation.
    
    Args:
        brand_name: Name of the brand
        video_type: Type of video content
        veo3_persona: Veo3 persona characteristics
        visual_style: Desired visual style
        scene_description: Description of the scene
        brand_colors: Optional brand color palette
    
    Returns:
        Detailed visual prompt for video generation
    """
    colors_text = ", ".join(brand_colors) if brand_colors else "brand-appropriate colors"
    
    # Veo3 persona visual characteristics
    persona_visuals = {
        "professional": "clean, minimalist, corporate, sophisticated lighting, formal composition",
        "creative": "dynamic, artistic, vibrant colors, creative angles, innovative framing",
        "casual": "natural lighting, relaxed composition, everyday settings, authentic moments",
        "authoritative": "strong composition, confident framing, expert positioning, credible visuals",
        "friendly": "warm lighting, approachable angles, welcoming environments, engaging visuals"
    }
    
    persona_visual = persona_visuals.get(veo3_persona, "warm, approachable, engaging")
    
    prompt = f"""
VISUAL PROMPT FOR {brand_name.upper()} VIDEO

Video Type: {video_type.title()}
Veo3 Persona: {veo3_persona.title()}
Visual Style: {visual_style}
Brand Colors: {colors_text}

DETAILED VISUAL PROMPT:

Scene: {scene_description}

Visual Characteristics:
- Persona Style: {persona_visual}
- Lighting: Professional, well-lit, {veo3_persona}-appropriate
- Composition: {visual_style} framing, {brand_name} branding visible
- Colors: {colors_text} palette, brand-consistent
- Mood: {veo3_persona} energy, {video_type} appropriate

Camera Work:
- Movement: Smooth, professional camera work
- Angles: {veo3_persona}-appropriate perspectives
- Focus: Clear, sharp imagery with {brand_name} elements
- Transitions: Seamless, brand-consistent cuts

Brand Elements:
- Logo placement: Strategic, visible but not overwhelming
- Color scheme: {colors_text}
- Typography: {brand_name} brand fonts
- Visual identity: Consistent with {brand_name} guidelines

Technical Specifications:
- Resolution: High definition, professional quality
- Aspect ratio: Optimized for {video_type} format
- Frame rate: Smooth, professional standard
- Audio: Clear, {veo3_persona} voice quality

Style Guidelines:
- Maintain {brand_name} visual identity
- Ensure {veo3_persona} persona consistency
- Create engaging, professional content
- Optimize for {video_type} distribution
"""
    
    return prompt


@tool
def generate_veo3_response_tool(
    brand_name: str,
    veo3_persona: str,
    context: str,
    response_type: str,
    brand_tone: str,
    target_audience: str
) -> str:
    """Generate brand responses using Veo3 personas.
    
    Args:
        brand_name: Name of the brand
        veo3_persona: Veo3 persona to use
        context: Context for the response
        response_type: Type of response (social, email, video, etc.)
        brand_tone: Brand tone and voice
        target_audience: Target audience
    
    Returns:
        Generated response using Veo3 persona
    """
    # Veo3 persona response styles
    persona_responses = {
        "professional": "formal, business-focused, authoritative, solution-oriented",
        "creative": "innovative, artistic, imaginative, visually-inspired",
        "casual": "relaxed, friendly, conversational, approachable",
        "authoritative": "expert, knowledgeable, confident, credible",
        "friendly": "warm, personable, helpful, engaging"
    }
    
    persona_style = persona_responses.get(veo3_persona, "warm, personable, helpful")
    
    response_template = f"""
VEo3 PERSONA RESPONSE FOR {brand_name.upper()}

Persona: {veo3_persona.title()} ({persona_style})
Response Type: {response_type.title()}
Context: {context}
Brand Tone: {brand_tone}
Target Audience: {target_audience}

GENERATED RESPONSE:

[Opening]
"Hi there! As someone who represents {brand_name}, I want to address {context}..."

[Main Response]
"At {brand_name}, we understand that {target_audience} face unique challenges. That's why we've developed solutions that {brand_tone} approach to help you succeed."

[Veo3 Persona Voice]
"Our {veo3_persona} approach means we're {persona_style} in everything we do. We're not just another company - we're your partner in achieving your goals."

[Call to Action]
"Ready to experience the {brand_name} difference? Let's connect and see how we can help you with {context}."

[Closing]
"Thanks for considering {brand_name}. We're here to support {target_audience} every step of the way."

PERSONA NOTES:
- Voice: {persona_style}
- Tone: {brand_tone}
- Approach: {veo3_persona}-appropriate
- Audience: {target_audience}-focused
- Brand: {brand_name} values maintained

RESPONSE GUIDELINES:
- Maintain {brand_name} brand voice
- Use {veo3_persona} persona consistently
- Address {target_audience} needs
- Include clear next steps
- End with {brand_name} value proposition
"""
    
    return response_template