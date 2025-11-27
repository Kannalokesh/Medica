from dotenv import load_dotenv
load_dotenv()

import os
from typing import List
from threading import Lock

# --------
# Configuration (Gemini API key)
# --------
# Accept either GENAI_API_KEY or GEMINI_API_KEY (backwards compatibility)
_API_KEY = os.environ.get("GENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")

# Client container and lock for lazy init
_client = None
_client_lock = Lock()

def _init_genai_client():
    """
    Lazy-initialize google.genai client using API key (Gemini / Google AI API path).
    Raises a clear error if API key is not provided.
    """
    global _client
    if _client is not None:
        return

    with _client_lock:
        if _client is not None:
            return

        if not _API_KEY:
            raise ValueError(
                "Missing Gemini/GENAI API key. Set environment variable GENAI_API_KEY or GEMINI_API_KEY."
            )

        try:
            import google.genai as genai
        except Exception as e:
            # Provide helpful message if dependency missing
            raise RuntimeError(
                "Failed to import google.genai. Is google-genai installed in your environment?"
            ) from e

        # Create the client using API key ONLY (no Vertex / GCP path)
        try:
            _client = genai.Client(api_key=_API_KEY)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize genai.Client with provided API key: {e}") from e

def _get_client():
    if _client is None:
        _init_genai_client()
    return _client

# --------
# Text generation
# --------
def generate_text(system_prompt: str, user_prompt: str, model: str = "gemini-2.5-flash", max_output_tokens: int = 512) -> str:
    """
    Generate text from Gemini (Google genai). Uses lazy client initialization.
    Returns the string output (best-effort).
    """
    client = _get_client()

    # Try common modern API shapes: models.generate_content, then client.responses.generate
    try:
        # Some genai versions expose models.generate_content
        resp = client.models.generate_content(model=model, contents=f"{system_prompt}\n\n{user_prompt}")
        # Common shapes:
        if hasattr(resp, "text") and resp.text:
            return resp.text
        # older/nested shapes
        try:
            return resp.candidates[0].content.parts[0].text
        except Exception:
            return str(resp)
    except Exception:
        # Fallback to responses.generate shape
        try:
            resp = client.responses.generate(
                model=model,
                input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_output_tokens=max_output_tokens
            )
            # Try to assemble text from common response shapes
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
            raise RuntimeError(f"Failed to call Gemini generation: {e}") from e

# -------------------------
# Embeddings: local sentence-transformers (lazy)
# -------------------------
_S2T_MODEL_NAME = os.environ.get("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_s2t_model = None
_s2t_lock = Lock()

def _ensure_s2t_model():
    """
    Lazy-load the sentence-transformers model to avoid heavy startup memory usage.
    Raises clear errors if dependency or download fails.
    """
    global _s2t_model
    if _s2t_model is not None:
        return

    with _s2t_lock:
        if _s2t_model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np  # for handling older API branches
        except Exception as e:
            raise RuntimeError(
                "Missing dependency: sentence-transformers (and its dependencies). "
                "Install with: pip install sentence-transformers"
            ) from e

        try:
            _s2t_model = SentenceTransformer(_S2T_MODEL_NAME)
        except Exception as e:
            raise RuntimeError(f"Failed to load SentenceTransformer model '{_S2T_MODEL_NAME}': {e}") from e

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

    try:
        vectors = _s2t_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    except TypeError:
        # older versions may not support convert_to_numpy
        vectors = _s2t_model.encode(texts, show_progress_bar=False)
        import numpy as np
        vectors = np.array(vectors)

    out = []
    for vec in vectors:
        out.append([float(x) for x in vec.tolist()])

    return out
