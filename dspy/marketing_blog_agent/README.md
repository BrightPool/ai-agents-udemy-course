# Marketing Blog Writer Agent

## What this agent does
- Generates a full marketing blog post from a single topic in a practical, no‑hype voice
- Retrieves relevant company snippets via vector search (FAISS) and uses them as writing context
- Produces an outline, drafts each section with retrieved_context, and assembles a final markdown draft
- Inputs: `topic` (string)
- Outputs: `final_blog` (markdown string), plus optional `outline` (list of section titles) and section drafts

---

## High-level workflow
```
[Start: Provide topic]
     |
     v
[Code: Embed corpus]
     |
     v
[Code: Vector search (FAISS)]
     |
     v
[LLM: Generate outline]
     |
     v
[Loop over outline]
     |
     +--> [LLM: Write section with retrieved_context]
     |          |
     |          v
     |     [Store section draft]
     |
     v
[Code: Assemble blog]
     |
     v
[Output: Final blog markdown]
```
- Inputs: `topic`
- Outputs: `final_blog` (markdown)
- External services: OpenAI models, FAISS embeddings/search

---

## Components and external services
- LLMs: `openai/gpt-5-mini` (generation), `openai/text-embedding-3-large` (embeddings)
- Env: `OPENAI_API_KEY`
- Dependencies (pip): `dspy`, `faiss-cpu`, `python-dotenv`, `pandas`
- Notes: Do not change API keys in `.env`. The notebook uses temperature `1` and `max_tokens=16000` for the generator.

---

## Copy‑paste prompts

### 1) Outline generation prompt
```
System/Instructions:
Create a clear, multi‑level outline for a marketing blog post.

Inputs:
- topic (string): The blog topic.

Output:
- outline (array of strings): An ordered list of section titles.
```

### 2) Section drafting prompt
```
System/Instructions:
Write a focused section with context from prior company writing. Use only the provided retrieved_context for facts. Tone: practical, no‑hype. Length: 3–6 paragraphs.

Inputs:
- topic (string): The blog topic.
- section_title (string): Which section to write.
- retrieved_context (array of strings): Relevant snippets from prior posts.

Output:
- draft (string): The section content.
```

### 3) Agent/system schema (for orchestration)
```
System/Instructions:
You are a marketing blog writer. Given `topic`, create an outline, write each section using prior company writing as context (via vector search), optionally edit sections for continuity, and finish with a coherent draft. Use only the available tools. When finished, return the complete blog draft in `process_result`.

Inputs:
- topic (string)

Outputs:
- reasoning (string): High‑level plan and justification of actions
- process_result (string): Final blog draft text

Available tools:
- search_context(query: string, k: int)
- change_outline(new_outline: array of strings)
- write_section(topic: string, section_title: string)
- edit_section(topic: string, section_title: string, instruction: string)
- assemble_blog()
```

---

## Scoring/aggregation (code)
JS:
```javascript
function assembleBlog(outline, sections) {
  return outline
    .map((title) => `# ${title}\n\n${sections[title] || ""}`)
    .join("\n\n").trim();
}
```

Python:
```python
def assemble_blog(outline, sections):
    parts = []
    for title in outline:
        body = sections.get(title, "")
        parts.append(f"# {title}\n\n{body}")
    return "\n\n".join(parts).strip()
```

---

## Implementation steps
1) Trigger: provide `topic` (string)
2) Code: embed prior company snippets and build a FAISS index
3) LLM: generate `outline` using the outline prompt (pass `topic`)
4) For each section title in `outline`:
   - Code: run vector search with `topic + section_title` to get `retrieved_context`
   - LLM: draft the section using the section prompt (pass `topic`, `section_title`, `retrieved_context`)
5) Code: assemble the final blog with the aggregator snippet above
6) Output: return `final_blog` (markdown). Optionally also return the `outline` and per‑section drafts.

---

## Example I/O
- Input: `"Launch blog for Nimbus Scale EU customers covering SOC 2, ISO 27001, Frankfurt data residency, private Slack support hours, Salesforce and Snowflake integrations, Concierge Migration, Growth Summit details, Acme Logistics case study, and promo code BUILD25 in a practical, no‑hype tone."`
- Output (abridged):
  - outline: `["Introduction…", "Security & Compliance…", "Data residency in Frankfurt…", …]`
  - final_blog: `"# Introduction…\n\n<draft>\n\n# Security & Compliance…\n\n<draft>\n\n…"`

---

## Files
- `dspy/marketing_blog_agent/blog-writer.ipynb`

