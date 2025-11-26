#src/llm_client.py

import os
from typing import List
from threading import Lock
from dotenv import load_dotenv
load_dotenv()

# Support both names for convenience (GENAI_API_KEY or GEMINI_API_KEY)
_API_KEY = os.environ.get("GEMINI_API_KEY")
_USE_VERTEX = os.environ.get("USE_VERTEXAI") in ("1", "true", "True")
_VERTEX_PROJECT = os.environ.get("VERTEXAI_PROJECT")
_VERTEX_LOCATION = os.environ.get("VERTEXAI_LOCATION")

_client = None
_client_lock = Lock()

def _init_genai_client():
    """
    Lazy initialize google.genai.Client only when first needed.
    This avoids heavy import/initialization during module import.
    """
    global _client
    if _client is not None:
        return

    with _client_lock:
        if _client is not None:
            return

        # Import inside function to avoid import-time overhead
        try:
            import google.genai as genai
        except Exception as e:
            raise RuntimeError(f"Missing dependency google-genai or failed to import: {e}")

        if _API_KEY:
            _client = genai.Client(api_key=_API_KEY)
            return

        if _USE_VERTEX:
            if not (_VERTEX_PROJECT and _VERTEX_LOCATION):
                raise ValueError(
                    "Vertex AI mode requires VERTEXAI_PROJECT and VERTEXAI_LOCATION environment variables."
                )
            # Ensure GOOGLE_APPLICATION_CREDENTIALS is set to point to service account JSON
            # (Render: /etc/secrets/service-account.json if you uploaded it as Secret File)
            _client = genai.Client(vertexai=True, project=_VERTEX_PROJECT, location=_VERTEX_LOCATION)
            return

        raise ValueError(
            "No genai credentials provided. Set GENAI_API_KEY or GEMINI_API_KEY for Google AI API, "
            "or set USE_VERTEXAI=1 and VERTEXAI_PROJECT & VERTEXAI_LOCATION for Vertex AI mode."
        )

def _get_client():
    if _client is None:
        _init_genai_client()
    return _client

def generate_text(system_prompt: str, user_prompt: str, model: str = "gemini-2.5-flash", max_output_tokens: int = 512) -> str:
    """
    Use Gemini to generate a response. We supply a system prompt and a user prompt.
    """
    client = _get_client()
    # Keep try/except shapes from your original code but use the lazy client
    try:
        # Try models.generate_content (common in examples)
        resp = client.models.generate_content(model=model, contents=f"{system_prompt}\n\n{user_prompt}")
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
            resp = client.responses.generate(
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
# Lazy-load SentenceTransformer to avoid heavy memory at import time.
_S2T_MODEL_NAME = os.environ.get("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_s2t_model = None
_s2t_lock = Lock()

def _ensure_s2t_model():
    global _s2t_model
    if _s2t_model is not None:
        return
    with _s2t_lock:
        if _s2t_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np  # keep available for dtype conversions
        except Exception as e:
            raise RuntimeError(
                "Missing dependency: install sentence-transformers (and its dependencies). "
                "Run: pip install sentence-transformers"
            ) from e

        try:
            _s2t_model = SentenceTransformer(_S2T_MODEL_NAME)
        except Exception as e:
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

    _ensure_s2t_model()

    # encode -> numpy array (num_texts x dim)
    try:
        vectors = _s2t_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    except TypeError:
        # older versions may not have convert_to_numpy flag
        vectors = _s2t_model.encode(texts, show_progress_bar=False)
        import numpy as np
        vectors = np.array(vectors)

    # convert to python lists for storage/Chroma consumption
    out = []
    for vec in vectors:
        out.append([float(x) for x in vec.tolist()])

    return out
