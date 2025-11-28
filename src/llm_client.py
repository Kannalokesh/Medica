# src/llm_client.py
from dotenv import load_dotenv
load_dotenv()

import os
from typing import List
from threading import Lock

# --------
# Configuration (Gemini API key + embedding model)
# --------
_API_KEY = os.environ.get("GEMINI_API_KEY")
# recommended default; set in Render env to override if needed
_DEFAULT_EMBED_MODEL = "text-embedding-004"
_EMBED_MODEL = os.environ.get("GENAI_EMBEDDING_MODEL", _DEFAULT_EMBED_MODEL)

_client = None
_client_lock = Lock()

# -------------------------
# lazy init genai client
# -------------------------
def _init_genai_client():
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
                "Failed to import google.genai. Ensure google-genai package is installed in your environment."
            ) from e

        try:
            # Force API-key initialization path (non-Vertex)
            _client = genai.Client(api_key=_API_KEY)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize genai.Client with provided API key: {e}") from e

def _get_client():
    if _client is None:
        _init_genai_client()
    return _client

# -------------------------
# generation (text)
# -------------------------
def generate_text(system_prompt: str, user_prompt: str, model: str = "gemini-2.5-flash", max_output_tokens: int = 512) -> str:
    """
    Generate text from Gemini (google.genai). Returns best-effort text string.
    """
    client = _get_client()

    # Try modern models.generate_content shape first
    try:
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            resp = client.models.generate_content(model=model, contents=f"{system_prompt}\n\n{user_prompt}")
            if hasattr(resp, "text") and resp.text:
                return resp.text
            # try nested shapes
            try:
                return resp.candidates[0].content.parts[0].text
            except Exception:
                return str(resp)
        # fallback to client.responses.generate
        if hasattr(client, "responses") and hasattr(client.responses, "generate"):
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
        # bubble an informative error
        raise RuntimeError(f"Failed to call Gemini generation: {e}") from e

    # if none of the above patterns are available, show helpful error
    raise RuntimeError("No supported generation API found on genai client. Please check client version.")

# -------------------------
# embedding response parser (robust)
# -------------------------
def _parse_embedding_response(resp) -> List[List[float]]:
    """
    Convert different genai embedding response shapes into List[List[float]].
    Common shapes handled:
      - resp.embeddings -> list of objects with .values or .embedding
      - resp.data -> list of dicts {'embedding': [...]}
      - dict resp with 'data' key
      - resp is list-of-vectors
    Raises ValueError if unable to parse.
    """
    # 1) attribute-style 'embeddings'
    try:
        data = getattr(resp, "embeddings", None)
        if data:
            out = []
            for item in data:
                # item may have .values or .embedding or be a list
                if hasattr(item, "values"):
                    out.append([float(x) for x in item.values])
                elif hasattr(item, "embedding"):
                    emb = getattr(item, "embedding")
                    # emb might be an object or list
                    if hasattr(emb, "values"):
                        out.append([float(x) for x in emb.values])
                    else:
                        out.append([float(x) for x in list(emb)])
                elif isinstance(item, (list, tuple)):
                    out.append([float(x) for x in item])
                else:
                    # try dict-like access
                    try:
                        emb = item["embedding"]
                        out.append([float(x) for x in emb])
                    except Exception:
                        raise
            if out:
                return out
    except Exception:
        pass

    # 2) attribute-style 'data' with nested embedding objects
    try:
        data = getattr(resp, "data", None)
        if data:
            out = []
            for item in data:
                # item might be dict-like or object
                if isinstance(item, dict) and "embedding" in item:
                    out.append([float(x) for x in item["embedding"]])
                elif hasattr(item, "embedding"):
                    emb = getattr(item, "embedding")
                    if hasattr(emb, "values"):
                        out.append([float(x) for x in emb.values])
                    else:
                        out.append([float(x) for x in list(emb)])
            if out:
                return out
    except Exception:
        pass

    # 3) dict-like response: resp['data']
    try:
        if isinstance(resp, dict) and "data" in resp:
            out = []
            for item in resp["data"]:
                if isinstance(item, dict) and "embedding" in item:
                    out.append([float(x) for x in item["embedding"]])
            if out:
                return out
    except Exception:
        pass

    # 4) resp itself is list of vectors
    try:
        if isinstance(resp, (list, tuple)):
            first = resp[0]
            if isinstance(first, (list, tuple)):
                return [[float(x) for x in vec] for vec in resp]
            # if only single vector returned as flat list
            if all(isinstance(x, (float, int)) for x in resp):
                return [[float(x) for x in resp]]
    except Exception:
        pass

    raise ValueError(f"Unable to parse embedding response shape from genai client (type={type(resp)})")

# -------------------------
# embeddings: try multiple api shapes safely
# -------------------------
def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Return embeddings for a list of texts by calling Gemini embeddings via google.genai.
    Tries multiple client methods for compatibility with different genai versions.
    """
    if texts is None:
        return []
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list):
        texts = list(texts)

    model = os.environ.get("GENAI_EMBEDDING_MODEL", _EMBED_MODEL)
    client = _get_client()

    errors = []

    # 1) Modern: client.embeddings.create(model=..., input=[...])
    try:
        embeddings_api = getattr(client, "embeddings", None)
        create_fn = getattr(embeddings_api, "create", None) if embeddings_api is not None else None
        if callable(create_fn):
            resp = create_fn(model=model, input=texts)
            return _parse_embedding_response(resp)
    except Exception as e:
        errors.append(("client.embeddings.create", repr(e)))

    # 2) Some clients expose client.models.embed(...)
    try:
        models_obj = getattr(client, "models", None)
        embed_fn = getattr(models_obj, "embed", None) if models_obj is not None else None
        if callable(embed_fn):
            resp = embed_fn(model=model, input=texts)
            return _parse_embedding_response(resp)
    except Exception as e:
        errors.append(("client.models.embed", repr(e)))

    # 3) Some versions expose client.embed_content(model=..., contents=[...])
    try:
        embed_content_fn = getattr(client, "embed_content", None)
        if callable(embed_content_fn):
            resp = embed_content_fn(model=model, contents=texts)
            return _parse_embedding_response(resp)
    except Exception as e:
        errors.append(("client.embed_content", repr(e)))

    # 4) Some older clients provide client.embed(...)
    try:
        embed_fn = getattr(client, "embed", None)
        if callable(embed_fn):
            resp = embed_fn(model=model, input=texts)
            return _parse_embedding_response(resp)
    except Exception as e:
        errors.append(("client.embed", repr(e)))

    # 5) legacy: client.responses.create(...) with embeddings (rare)
    try:
        responses_obj = getattr(client, "responses", None)
        create_resp_fn = getattr(responses_obj, "create", None) if responses_obj is not None else None
        if callable(create_resp_fn):
            # try to call with embedding intent (some SDKs may support)
            try:
                resp = create_resp_fn(model=model, input=texts, embeddings=True)
                return _parse_embedding_response(resp)
            except Exception:
                # try alternative signature
                resp = create_resp_fn(model=model, input=[{"role": "user", "content": t} for t in texts])
                # parse if possible
                try:
                    return _parse_embedding_response(resp)
                except Exception:
                    pass
    except Exception as e:
        errors.append(("client.responses.create", repr(e)))

    # If we reached here, nothing worked. Build informative error message
    err_msgs = "; ".join([f"{m}: {ex}" for (m, ex) in errors]) if errors else "unknown"
    raise RuntimeError(
        "No supported embeddings API found on genai client. Please check the client version and available methods. "
        f"Attempted: {err_msgs}"
    )
