# Joke Generator Agent (DSPy)

## What this agent does
- Generates a short, original joke for a given topic.
- Simulates five expert audience personas to rate the joke and provide reactions.
- Optimizes the joke-writing prompt so the generated jokes score higher with the personas.

---

## High-level workflow
```
[Start: Provide Topic]
     |
     v
[LLM: Generate Joke]
     |
     v
[LLM: Audience Evaluation]
     |
     v
[Code: Compute Average Score]
     |
     v
[Optimization Run?] --No--> [Output: Joke + Score + Reactions]
     |
    Yes
     v
[LLM: Iteratively Improve Prompt]
     |
     v
[Evaluate Candidate Prompt]
     |
     `------> loops back to [Optimization Run?]
```

- **Inputs**: topic (e.g., "Coffee", "Remote Work").
- **Outputs**: generated joke, 5 persona reactions, 5 persona ratings, average score (out of 5).
- **External services**: OpenAI models (`openai/gpt-5-mini`, `openai/gpt-5`).

---

## Components and external services
- **Language Models (LLM)**
  - `openai/gpt-5-mini` (cheaper) and `openai/gpt-5` (smarter) via DSPy `dspy.LM`.
- **Audience Evaluator (LLM Signature)**
  - A DSPy signature that takes a `joke` and fixed `profiles` and returns `reactions` and `responses` (ratings from a fixed set).
- **Prompt Optimizer (DSPy GEPA)**
  - Uses feedback from the audience evaluator to iteratively refine the joke-generation prompt.

Environment note: The notebook loads `OPENAI_API_KEY` from `.env`. Do not change your keys.

---

## Copy‑paste prompts for LLM steps

### 1) Joke Generation (Optimized Prompt)
Use this as the system/instructions for your LLM node that generates the joke from a `topic` input.

```
Task
- Write one funny, original joke based on a provided topic that could impress seasoned comedy professionals.

Input format
- You will receive:
  - topic: a single word or short phrase (e.g., Entertainment, Culture, Misunderstanding).

Output format
- Return only the joke text as plain text (no preamble, labels, quotes, hashtags, emojis, or explanations).
- Length: 1–2 sentences, ideally 15–35 words.
- End on the strongest, most surprising concrete word or phrase.

Audience bar (what the joke must satisfy)
- Club owner who’s seen everything: wants a fresh angle with a crisp, earned twist.
- Pro comedian: allergic to hack premises and groaner puns.
- Festival curator: seeks a distinctive voice and memorable perspective.
- Critic: values clear structure and a touch of smart social/cultural observation.
- Comedy-writing professor: expects economy, tight setup→turn, and timing.

Hard avoids
- No single, obvious pun or one-note wordplay.
- No stock setups: “X walks into a bar,” “objects go to therapy,” “Wi‑Fi vs commitment,” petri‑dish “culture grows on people,” franchise/sequel therapy, or any prefab template.
- Don’t lean on familiar premises unless the angle is unmistakably novel:
  - “Disaster turned into a hustle/startup spin.”
  - “Cats own the humans” dynamic.
  - “Vengeance isn’t cinematic; it’s paperwork/shipping.”
- Avoid listy rule-of-three unless you truly subvert it; no filler tags (“these days,” “amirite,” “that’s just Monday,” “now more than ever”).
- No punching down, slurs, or needlessly cruel targets.
- Specifically avoid stale tech overlays:
  - Religion-as-tech/app/loyalty/gamified salvation (points, tiers, QR codes at the pearly gates, CAPTCHAs for heaven).
  - Identity-as-software (updates, release notes, patches, rollbacks, versioning).
  - Any premise that reduces to “X but like an app/algorithm.”

Recommended approach
1) Narrow the topic to one vivid, contemporary angle or image. Prefer behind-the-scenes behavior, quiet rules/etiquette, ritual mechanics, trend performance, or how meaning/status is performed (e.g., awards-season rituals, museum etiquette, audience behavior).
2) Brainstorm 3–5 distinct premises. Immediately discard anything that feels familiar, tweet-y, or pun-led. Quick test: would a jaded comic say “I’ve heard that”? If yes, toss it.
3) Choose the freshest premise and write one tight setup that clearly frames an expectation.
4) Use one strong device only—misdirection, semantic flip, framing shift, or compressed analogy—and make the turn feel earned, not clever-for-its-own-sake.
5) Prefer specific nouns and surprising, concrete imagery over abstractions; keep verbs active; trim every extra word.
6) Land the twist on the final word or phrase; make it a charged, concrete image rather than an adverb or vague tag.
7) Optional: add one very short, smart tag only if it elevates the premise and isn’t stock.

Self-check before returning
- Is the angle genuinely unexpected (not “religion-as-loyalty-program,” “identity-as-update,” or “X as an app/algorithm”)?
- Is the structure clear (setup → turn → concise payoff) with tight timing?
- Is there a subtle social/cultural observation or perspective that feels current?
- Would a club owner not roll their eyes, and a pro comic not groan?
- Are there zero filler tags, hacky tropes, or listy padding?
- Does it end on the strongest, most surprising concrete word or phrase?
- Does it fit 1–2 sentences and roughly 15–35 words?
- Is it unmistakably original (not echoing common “cats-as-landlords,” “hustle-from-ruin,” or “revenge-as-bureaucracy” beats)?

Deliverable
- One joke that follows all the above, returned as plain text only.
```

If your tool supports a user input, pass `topic` as the single input variable.


### 2) Audience Evaluation (System Schema Prompt)
Use this for the evaluation step to get structured reactions and ratings from five personas. Provide `joke` as the input and pass the `profiles` array below.

```
Your input fields are:
1. `joke` (str): A joke to evaluate
2. `profiles` (list[str]): Profiles of the audience members
Your output fields are:
1. `reactions` (list[str]): Short reaction from each audience member explaining their inner thought process when hearing the joke (one per profile, same order)
2. `responses` (list["hilarious"|"funny"|"meh"|"not funny"|"offensive"]): Rating from each audience member (one per profile, same order)
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## joke ## ]]
{joke}

[[ ## profiles ## ]]
{profiles}

[[ ## reactions ## ]]
{reactions}

[[ ## responses ## ]]
{responses}

[[ ## completed ## ]]
In adhering to this structure, your objective is:
        Decide if the joke is funny.
```

Fixed `profiles` array to send as input:
```
[
  "35-year-old comedy club owner who has seen every major standup special and demands originality",
  "42-year-old comedy critic who writes for The New Yorker and analyzes joke structure and social commentary",
  "38-year-old professional comedian who performs nightly and is tired of hacky material",
  "45-year-old comedy festival curator who looks for unique voices and fresh perspectives",
  "40-year-old comedy writing professor who teaches advanced joke construction and timing"
]
```

Expected outputs:
- `reactions`: list of 5 strings
- `responses`: list of 5 items from {hilarious, funny, meh, not funny, offensive}


### 3) Scoring Logic (Code)
After you get `responses`, compute the average score out of 5:

```
const ratingScores = {
  "hilarious": 5,
  "funny": 4,
  "meh": 3,
  "not funny": 2,
  "offensive": 1,
};

const total = responses.reduce((sum, r) => sum + ratingScores[r], 0);
const avgScore = Math.round((total / responses.length) * 100) / 100; // 2 decimals
```

You can also build a feedback string by pairing each persona with its reaction.

---

## Implementation steps
Below is a minimal, happy‑path mapping that works in any workflow/orchestration tool or plain scripts. Use your LLM provider's chat/completions API.

1) **Trigger**
   - Manual Trigger or Webhook with input `topic` (string).

2) **LLM: Generate Joke**
   - Call type: Chat/completions
   - System/Instructions: use the "Joke Generation (Optimized Prompt)" above.
   - Input: provide a variable `topic` (e.g., JSON `{ "topic": "Coffee" }`).
   - Output: `joke` as plain text. If your API returns structured JSON, extract the text field.

3) **LLM: Audience Evaluation**
   - Call type: Chat/completions
   - System/Instructions: use the "Audience Evaluation (System Schema Prompt)" above.
   - Input fills placeholders:
     - `joke`: the text from the previous node
     - `profiles`: the fixed array provided above
   - Parse output to get `reactions[]` and `responses[]`.

4) **Code: Compute Score + Feedback**
   - Implementation: Code step (JavaScript or Python)
   - Inputs: `responses[]`, `reactions[]`, and `profiles[]` from previous node; `joke` from step 2.
   - Compute `avgScore` using the scoring snippet.
   - Optionally build a feedback string that pairs each profile to its reaction.

5) **Output**
   - Return a result (e.g., HTTP response, stdout, or saved file)
   - Include JSON:
     - `topic`, `joke`, `avgScore`, `responses`, `reactions`, `profiles`.

Optional (advanced):
- To emulate DSPy’s optimization (GEPA), set up a loop that:
  - Proposes prompt variations to the Joke Generation step,
  - Runs Audience Evaluation and scoring,
  - Keeps the best‑scoring prompt. Most workflow tools support this with conditionals and loop/batch constructs.

---

## Notes on optimization results
- In the notebook, optimizing the prompt improved average metric from ~62.6% to ~77.2% against a held‑out set.
- The exact optimized instructions printed from the saved program are included verbatim above under "Joke Generation (Optimized Prompt)".

---

## Example I/O
- Input topic: `Coffee`
- Output (abridged):
  - `joke`: a 1–2 sentence original joke per the prompt
  - `responses`: ["funny", "funny", "meh", ...]
  - `reactions`: ["Short inner thought 1", ...]
  - `avgScore`: e.g., 3.8

---

## Files
- `comedian-agent.ipynb`: End‑to‑end DSPy notebook
- `evaluation_results.csv`: Example evaluation export
- `comedian_program/`: Saved optimized DSPy program (if present)
