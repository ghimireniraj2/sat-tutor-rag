# backend/prompts.py
import json
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# System prompts — define tutor behaviour per mode
# ---------------------------------------------------------------------------

EXPLAIN_SYSTEM = """You are an expert SAT tutor helping a student understand
a concept or problem. Your goal is to guide the student to understanding,
not just give them the answer.

Rules:
- Use the provided context to ground your explanation
- Break down concepts into clear steps
- Use simple language — avoid jargon unless you explain it
- If the student seems to be asking for a direct answer to a practice problem,
  guide them with hints rather than solving it for them
- Keep responses concise — 150-300 words unless the concept requires more
- If the context does not contain enough information to answer well, say so
  rather than making things up

You are talking to a student preparing for the SAT exam."""


PRACTICE_SYSTEM = """You are an SAT question generator. Generate a single
SAT-style practice question based on the topic and context provided.

You MUST respond with valid JSON only. No preamble, no explanation, no
markdown code blocks. Just the raw JSON object.

Required format:
{
  "question": "The full question text",
  "choices": {
    "A": "First choice",
    "B": "Second choice",
    "C": "Third choice",
    "D": "Fourth choice"
  },
  "correct": "A",
  "explanation": "Why this answer is correct, and why the others are wrong"
}

Rules:
- Question must be clearly worded and unambiguous
- All four choices must be plausible — avoid obviously wrong distractors
- Explanation must reference the correct mathematical or grammatical reasoning
- Difficulty should match the topic complexity from the context
- Never repeat a question the student has already seen in this conversation"""


# ---------------------------------------------------------------------------
# User prompt builders — assemble the full prompt for each request
# ---------------------------------------------------------------------------

def build_explain_prompt(
    query: str,
    context_chunks: list[dict],
    history: list[dict],
) -> list[dict]:
    """
    Build the messages list for Explain mode.
    Returns a list of message dicts for the Groq API.
    """
    # Assemble context string from retrieved chunks
    context = "\n\n---\n\n".join(
        f"[Source: {c['metadata'].get('source', 'unknown')} | "
        f"Topic: {c['metadata'].get('topic', 'unknown')}]\n{c['text']}"
        for c in context_chunks
    )

    messages = [{"role": "system", "content": EXPLAIN_SYSTEM}]

    # Inject conversation history (last 4 turns)
    messages.extend(history[-8:])  # 4 turns = 8 messages (user + assistant)

    # Add retrieved context + current query
    messages.append({
        "role": "user",
        "content": f"""Here is relevant content from SAT study materials:

{context}

---

Student question: {query}"""
    })

    return messages


def build_practice_prompt(
    topic: str,
    context_chunks: list[dict],
    history: list[dict],
) -> list[dict]:
    """
    Build the messages list for Practice mode.
    Returns a list of message dicts for the Groq API.
    """
    context = "\n\n---\n\n".join(
        f"[Topic: {c['metadata'].get('topic', 'unknown')} | "
        f"Difficulty: {c['metadata'].get('difficulty', 'unknown')}]\n{c['text']}"
        for c in context_chunks
    )

    # Extract previously seen questions from history to avoid repeats
    seen_questions = []
    for msg in history:
        if msg["role"] == "assistant":
            try:
                import json
                parsed = json.loads(msg["content"])
                if "question" in parsed:
                    seen_questions.append(parsed["question"][:100])
            except Exception:
                pass

    seen_note = ""
    if seen_questions:
        seen_note = f"\n\nAvoid generating questions similar to these already seen:\n" + \
                    "\n".join(f"- {q}" for q in seen_questions[-3:])

    messages = [{"role": "system", "content": PRACTICE_SYSTEM}]
    messages.extend(history[-4:])  # shorter history for practice mode
    messages.append({
        "role": "user",
        "content": f"""Generate a practice question on this topic: {topic}

Reference material:
{context}{seen_note}"""
    })

    return messages

# ---------------------------------------------------------------------------
# Structured output — Practice mode
# ---------------------------------------------------------------------------

class PracticeQuestion(BaseModel):
    question: str
    choices: dict[str, str]   # {"A": "...", "B": "...", "C": "...", "D": "..."}
    correct: str              # "A", "B", "C", or "D"
    explanation: str


def parse_practice_response(raw: str) -> PracticeQuestion:
    """
    Parse and validate LLM response for Practice mode.
    Raises ValueError if response is not valid JSON or missing fields.
    """
    try:
        data = json.loads(raw)
        return PracticeQuestion(**data)
    except Exception as e:
        raise ValueError(f"Invalid practice question format: {e}\nRaw: {raw}")