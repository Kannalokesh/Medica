from dotenv import load_dotenv
load_dotenv()

import os
from typing import List
from threading import Lock

# --------
# Configuration (Gemini API key + embedding model)
# --------
_API_KEY =  os.environ.get("GEMINI_API_KEY")
# Set this in Render env. Example names vary by provider/version. Required.

_EMBED_MODEL = os.environ.get("GENAI_EMBEDDING_MODEL", "text-embedding-004")

_client = None
_client_lock = Lock()

def _init_genai_client():
    """
    Lazy-init google.genai client using API key only (Gemini / Google AI API path).
    Raises clear error if API key is missing.
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
            raise RuntimeError(
                "Failed to import google.genai. Ensure google-genai is installed in your environment."
            ) from e

        try:
            # Always initialize the client with api_key (force non-Vertex path)
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
    Generate text from Gemini (google.genai). Returns a text string.
    """
    client = _get_client()

    try:
        # prefer models.generate_content if present
        resp = client.models.generate_content(model=model, contents=f"{system_prompt}\n\n{user_prompt}")
        if hasattr(resp, "text") and resp.text:
            return resp.text
        try:
            return resp.candidates[0].content.parts[0].text
        except Exception:
            return str(resp)
    except Exception:
        # fallback to responses.generate
        try:
            resp = client.responses.generate(
                model=model,
                input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_output_tokens=max_output_tokens
            )
            # assemble textual output from common shapes
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
# Embeddings: remote via genai (Gemini)
# -------------------------
def _parse_embedding_response(resp):
    """
    Parse common genai embedding response shapes into a list of vectors.
    Returns list[list[float]] or raises ValueError if unable to parse.
    """
    # Common recent shape: resp.data -> list of {embedding: [...]}
    try:
        # try attribute-style
        data = getattr(resp, "data", None)
        if data:
            out = []
            for item in data:
                # item might be a dict-like or object with .embedding
                if isinstance(item, dict) and "embedding" in item:
                    out.append([float(x) for x in item["embedding"]])
                elif hasattr(item, "embedding"):
                    emb = getattr(item, "embedding")
                    out.append([float(x) for x in list(emb)])
                elif isinstance(item, (list, tuple)):
                    # fallback: item might be vector itself
                    out.append([float(x) for x in item])
                else:
                    # try item["embedding"] if dict-like
                    try:
                        emb = item["embedding"]  # may raise
                        out.append([float(x) for x in emb])
                    except Exception:
                        raise
            if out:
                return out
    except Exception:
        pass

    # Older / alternative shapes: resp["data"] or resp[0]["embedding"]
    try:
        # dict-like access
        if isinstance(resp, dict) and "data" in resp:
            out = []
            for item in resp["data"]:
                if isinstance(item, dict) and "embedding" in item:
                    out.append([float(x) for x in item["embedding"]])
            if out:
                return out
    except Exception:
        pass

    # Last-resort: resp itself is a single vector or list of vectors
    try:
        if isinstance(resp, (list, tuple)):
            # list of vectors
            first = resp[0]
            if isinstance(first, (list, tuple)):
                return [[float(x) for x in vec] for vec in resp]
    except Exception:
        pass

    raise ValueError("Unable to parse embedding response shape from genai client: " + str(type(resp)))

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Return embeddings for a list of texts by calling Gemini embeddings via google.genai.
    - Requires GENAI_EMBEDDING_MODEL env var to be set (example: 'textembedding-gecko-001' or provider equivalent).
    - Returns list of lists (vectors).
    """
    if texts is None:
        return []
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list):
        texts = list(texts)

    if not _EMBED_MODEL:
        raise RuntimeError(
            "GENAI_EMBEDDING_MODEL is not set. Set environment variable GENAI_EMBEDDING_MODEL "
            "to the embedding model name you want to use (e.g. 'textembedding-gecko-001' or provider-specific name)."
        )

    client = _get_client()

    # call the embeddings endpoint (try common method names)
    try:
        # Preferred modern shape
        if hasattr(client, "embeddings") and hasattr(client.embeddings, "create"):
            resp = client.embeddings.create(model=_EMBED_MODEL, input=texts)
            return _parse_embedding_response(resp)
        # fallback older shape
        if hasattr(client, "models") and hasattr(client.models, "embed"):
            resp = client.models.embed(model=_EMBED_MODEL, input=texts)
            return _parse_embedding_response(resp)
        # last resort: try client.responses.embed or client.embed
        if hasattr(client, "embed"):
            resp = client.embed(model=_EMBED_MODEL, input=texts)
            return _parse_embedding_response(resp)
    except Exception as e:
        # bubble a helpful error
        raise RuntimeError(f"Failed to obtain embeddings from genai client: {e}") from e

    raise RuntimeError("No supported embeddings API found on genai client. Please check the client version.")
