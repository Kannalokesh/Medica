# src/llm_client.py
from dotenv import load_dotenv
load_dotenv()

import os
from typing import List
from threading import Lock

# Config: API key + default embedding model
_API_KEY = os.environ.get("GENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
_DEFAULT_EMBED_MODEL = os.environ.get("GENAI_EMBEDDING_MODEL", "gemini-embedding-001")  # docs example
_client = None
_client_lock = Lock()

def _init_genai_client():
    global _client
    if _client is not None:
        return
    with _client_lock:
        if _client is not None:
            return
        if not _API_KEY:
            raise ValueError("Missing GENAI_API_KEY or GEMINI_API_KEY environment variable.")
        try:
            import google.genai as genai
        except Exception as e:
            raise RuntimeError("Unable to import google.genai. Is google-genai installed?") from e
        try:
            _client = genai.Client(api_key=_API_KEY)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize genai.Client: {e}") from e

def _get_client():
    if _client is None:
        _init_genai_client()
    return _client

def generate_text(system_prompt: str, user_prompt: str, model: str = "gemini-2.5-flash", max_output_tokens: int = 512) -> str:
    client = _get_client()

    # Try common generation shapes
    # 1) client.models.generate_content(...)
    try:
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            resp = client.models.generate_content(model=model, contents=f"{system_prompt}\n\n{user_prompt}")
            if hasattr(resp, "text") and resp.text:
                return resp.text
            try:
                return resp.candidates[0].content.parts[0].text
            except Exception:
                return str(resp)
    except Exception:
        pass

    # 2) client.responses.generate(...)
    try:
        if hasattr(client, "responses") and hasattr(client.responses, "generate"):
            resp = client.responses.generate(
                model=model,
                input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_output_tokens=max_output_tokens
            )
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
        raise RuntimeError(f"Generation failed: {e}") from e

    raise RuntimeError("No supported generation API found on genai client.")


# -------------------------
# Embedding response parser
# -------------------------
def _parse_embedding_response(resp) -> List[List[float]]:
    # 1) attribute-style .embeddings
    try:
        embeddings_attr = getattr(resp, "embeddings", None)
        if embeddings_attr:
            out = []
            for item in embeddings_attr:
                if hasattr(item, "values"):
                    out.append([float(x) for x in item.values])
                elif hasattr(item, "embedding"):
                    emb = getattr(item, "embedding")
                    if hasattr(emb, "values"):
                        out.append([float(x) for x in emb.values])
                    else:
                        out.append([float(x) for x in list(emb)])
                elif isinstance(item, (list, tuple)):
                    out.append([float(x) for x in item])
                else:
                    try:
                        emb = item["embedding"]
                        out.append([float(x) for x in emb])
                    except Exception:
                        raise
            if out:
                return out
    except Exception:
        pass

    # 2) attribute-style .data
    try:
        data_attr = getattr(resp, "data", None)
        if data_attr:
            out = []
            for item in data_attr:
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

    # 3) dict-like resp['data']
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

    # 4) resp is list-of-vectors
    try:
        if isinstance(resp, (list, tuple)):
            first = resp[0]
            if isinstance(first, (list, tuple)):
                return [[float(x) for x in vec] for vec in resp]
            if all(isinstance(x, (int, float)) for x in resp):
                return [[float(x) for x in resp]]
    except Exception:
        pass

    raise ValueError(f"Unable to parse embedding response (type={type(resp)})")


# -------------------------
# Robust get_embeddings using models.embed_content primary
# -------------------------
def get_embeddings(texts: List[str]) -> List[List[float]]:
    if texts is None:
        return []
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list):
        texts = list(texts)

    model = os.environ.get("GENAI_EMBEDDING_MODEL", _DEFAULT_EMBED_MODEL)
    client = _get_client()
    attempts = []

    # 1) client.models.embed_content(model=..., contents=...)
    try:
        models_obj = getattr(client, "models", None)
        if models_obj is not None and hasattr(models_obj, "embed_content"):
            resp = models_obj.embed_content(model=model, contents=texts)
            return _parse_embedding_response(resp)
    except Exception as e:
        attempts.append(("client.models.embed_content", repr(e)))

    # 2) client.embeddings.create(model=..., input=...)
    try:
        embeddings_obj = getattr(client, "embeddings", None)
        if embeddings_obj is not None and hasattr(embeddings_obj, "create"):
            resp = embeddings_obj.create(model=model, input=texts)
            return _parse_embedding_response(resp)
    except Exception as e:
        attempts.append(("client.embeddings.create", repr(e)))

    # 3) client.models.embed(model=..., input=...)
    try:
        if models_obj is not None and hasattr(models_obj, "embed"):
            resp = models_obj.embed(model=model, input=texts)
            return _parse_embedding_response(resp)
    except Exception as e:
        attempts.append(("client.models.embed", repr(e)))

    # 4) client.embed_content(model=..., contents=...)
    try:
        if hasattr(client, "embed_content"):
            resp = client.embed_content(model=model, contents=texts)
            return _parse_embedding_response(resp)
    except Exception as e:
        attempts.append(("client.embed_content", repr(e)))

    # 5) client.embed(model=..., input=...)
    try:
        if hasattr(client, "embed"):
            resp = client.embed(model=model, input=texts)
            return _parse_embedding_response(resp)
    except Exception as e:
        attempts.append(("client.embed", repr(e)))

    # 6) nested defensive check for other "models" style attrs
    try:
        for name in dir(client):
            if "models" in name.lower():
                maybe = getattr(client, name)
                if hasattr(maybe, "embed_content"):
                    try:
                        resp = maybe.embed_content(model=model, contents=texts)
                        return _parse_embedding_response(resp)
                    except Exception as e:
                        attempts.append((f"{name}.embed_content", repr(e)))
    except Exception:
        pass

    attempts_str = "; ".join([f"{m} => {ex}" for m, ex in attempts]) if attempts else "no attempts recorded"
    raise RuntimeError(
        "No supported embeddings API found on genai client. Please check the client version and available methods. "
        f"Attempted: {attempts_str}"
    )
