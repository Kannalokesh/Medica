# src/llm_client.py
from dotenv import load_dotenv
load_dotenv()

import os
from typing import List
from google import genai

# Generation client (Gemini)
API_KEY_ENV = os.environ.get("GEMINI_API_KEY") 
_client = genai.Client(api_key=API_KEY_ENV) if API_KEY_ENV else genai.Client()

def generate_text(system_prompt: str, user_prompt: str, model: str = "gemini-2.5-flash", max_output_tokens: int = 512) -> str:
    """
    Use Gemini to generate a response. We supply a system prompt and a user prompt.
    """
    try:
        # Try models.generate_content (common in examples)
        resp = _client.models.generate_content(model=model, contents=f"{system_prompt}\n\n{user_prompt}")
        if hasattr(resp, "text") and resp.text:
            return resp.text
        # fallback nested path
        try:
            return resp.candidates[0].content.parts[0].text
        except Exception:
            return str(resp)
    except Exception:
        # Fallback to responses.generate if available
        try:
            resp = _client.responses.generate(
                model=model,
                input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_output_tokens=max_output_tokens
            )
            # Try parsing common shapes
            if hasattr(resp, "output") and getattr(resp.output, "content", None):
                out = ""
                for item in resp.output.content:
                    if hasattr(item, "text"):
                        out += item.text
                    else:
                        out += str(item)
                if out:
                    return out
            if hasattr(resp, "text") and resp.text:
                return resp.text
            return str(resp)
        except Exception as e:
            raise RuntimeError(f"Failed to call Gemini generation: {e}")

# -------------------------
# Embeddings: sentence-transformers (local)
# -------------------------
# We use a lightweight, fast model for local embeddings:
#   - all-MiniLM-L6-v2  (384 dims)
# It avoids cloud billing and works offline.

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception as e:
    raise RuntimeError("Missing dependency: install sentence-transformers (and its dependencies). "
                       "Run: pip install sentence-transformers") from e

# instantiate model once
# default model: all-MiniLM-L6-v2 (good for RAG prototyping)
_S2T_MODEL_NAME = os.environ.get("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
try:
    _S2T_MODEL = SentenceTransformer(_S2T_MODEL_NAME)
except Exception as e:
    # Give a clear error that model download may be required
    raise RuntimeError(f"Failed to load SentenceTransformer model '{_S2T_MODEL_NAME}': {e}")

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Return embeddings for a list of texts using sentence-transformers.
    - Accepts a single string or list[str].
    - Returns list of lists (vectors).
    """
    if texts is None:
        return []
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list):
        texts = list(texts)

    # encode -> numpy array (num_texts x dim)
    try:
        vectors = _S2T_MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    except TypeError:
        # older versions may not have convert_to_numpy flag
        vectors = _S2T_MODEL.encode(texts, show_progress_bar=False)
        vectors = np.array(vectors)

    # convert to python lists for storage/Chroma consumption
    out = []
    for vec in vectors:
        # ensure floats are native python floats
        out.append([float(x) for x in vec.tolist()])

    return out
