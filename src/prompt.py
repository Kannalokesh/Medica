"""
Prompt engineering helpers.
This module builds system prompt and user prompt. The user prompt requests JSON output
whenever possible to make frontend rendering reliable.
"""

SYSTEM_INSTRUCTIONS = """You are Medico — a careful, concise, evidence-minded medical assistant.
- Always be explicit about uncertainty. If the answer is not known, say so.
- Provide a short answer (1-3 sentences), then a short 'Supporting details' list.
- If the user asks for diagnosis, treatment, or dosing, include a safety disclaimer and recommend consulting a licensed clinician.
- Avoid hallucinations: base answers only on the provided context. If context doesn't contain the answer, say you don't have enough information.
"""

FEW_SHOT_EXAMPLES = [
    {
        "q": "What are typical symptoms of appendicitis?",
        "a": "Typical symptoms include right lower quadrant abdominal pain, nausea, and fever. \n\nSupporting details:\n- Pain often begins periumbilically before localizing.\n- Fever and elevated WBC may be present."
    }
]

# Instruct model to output JSON with specific keys.
JSON_INSTRUCTIONS = """
IMPORTANT: Return output **as a JSON object only** (no extra commentary). Use the exact keys below:

{
  "summary": "<1-3 sentence plain-text summary>",
  "key_facts": ["fact 1", "fact 2", "..."],
  "supporting_details": ["detail 1", "detail 2", "..."],
  "disclaimer": "<short disclaimer string>",
  "sources": [
    {"source": "filename.pdf"},
    ...
  ]
}

- The "summary" should be concise.
- "key_facts" should contain up to 4 bullet-style facts (strings).
- "supporting_details" should contain 1-4 short supporting points.
- "disclaimer" should be present for any clinical content (short).
- "sources" should reflect the retrieved docs used (if none, an empty list).
If you cannot produce JSON for any reason, produce a plain answer but clearly label it as non-JSON.
"""

def build_system_prompt():
    return SYSTEM_INSTRUCTIONS

def render_user_prompt(question: str, context: str, max_context_chars: int = 3000) -> str:
    """
    Renders the user prompt containing the retrieval context and the question, then requests JSON output.
    """
    ctx = (context or "").strip()
    if len(ctx) > max_context_chars:
        ctx = ctx[:max_context_chars] + "\n\n[TRUNCATED]"

    example = ""
    if len(FEW_SHOT_EXAMPLES) > 0:
        ex = FEW_SHOT_EXAMPLES[0]
        example = f"Example Q: {ex['q']}\nExample A: {ex['a']}\n\n"

    prompt = f"""{example}You are provided with the following retrieved documents (if any). Use them to answer the user's question; if the documents don't contain the answer, say you don't have enough information.

RETRIEVED DOCUMENTS:
{ctx}

USER QUESTION:
{question}

INSTRUCTIONS:
- Answer using only the retrieved documents if they contain the answer.
- First, produce a short summary.
- Then produce key facts and supporting details.
- Include a short disclaimer for clinical content.
- MOST IMPORTANT: Try to return a JSON object with keys: summary, key_facts, supporting_details, disclaimer, sources. See explicit JSON format instructions below.
{JSON_INSTRUCTIONS}
"""
    return prompt
