from dotenv import load_dotenv
load_dotenv()

import os
import json
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from threading import Lock

# Keep imports for our local modules (these modules themselves lazy-init heavy resources)
from src.llm_client import generate_text, get_embeddings
from src.cache import Cache
from src.prompt import build_system_prompt, render_user_prompt

# NOTE: chromadb imports/clients can be heavy. We'll lazy-init the client below.
_CHROMA_LOCK = Lock()
_chroma_collection = None

def get_chroma_collection():
    """
    Lazily initialize and return a Chroma collection.
    This avoids heavy startup memory usage at import time.
    """
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    with _CHROMA_LOCK:
        if _chroma_collection is not None:
            return _chroma_collection

        # import here to avoid import-time overhead
        try:
            from chromadb import Client, PersistentClient
        except Exception as e:
            raise RuntimeError(f"Failed importing chromadb: {e}")

        CHROMA_DIR = os.environ.get("CHROMA_PERSIST_DIR")  # optional
        CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "medico_docs")

        if CHROMA_DIR:
            client = PersistentClient(path=CHROMA_DIR)
        else:
            client = Client()
        _chroma_collection = client.get_or_create_collection(name=CHROMA_COLLECTION)
        return _chroma_collection

# Config
RATE_LIMIT = os.environ.get("RATE_LIMIT", "20 per minute")  # default rate
EMBED_CACHE_TTL = int(os.environ.get("EMBED_CACHE_TTL", 60 * 60 * 24))

app = Flask(__name__, template_folder="templates")

# Rate limiter (newer flask-limiter: create then init_app)
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])
limiter.init_app(app)

# Cache (likely lightweight)
cache = Cache(db_path=os.environ.get("CACHE_DB", "./medico_cache.sqlite3"))

@app.route("/", methods=["GET"])
def index():
    if os.path.exists("templates/index.html"):
        return render_template("index.html")
    return "Medico Chatbot - Gemini + Chroma (local embeddings)"

@app.route("/chat", methods=["POST"])
@limiter.limit(RATE_LIMIT)
def chat():
    payload = request.get_json(force=True, silent=True) or {}
    query = payload.get("query") or payload.get("question") or ""
    top_k = int(payload.get("top_k", 5))
    if not query or not query.strip():
        return jsonify({"error": "empty query"}), 400

    # 1) Embed with cache
    cache_key = f"embed:{query}"
    emb = cache.get(cache_key)
    if emb is None:
        try:
            emb_list = get_embeddings([query])
            emb = emb_list[0]
            cache.set(cache_key, emb, ttl=EMBED_CACHE_TTL)
        except Exception as e:
            return jsonify({"error": "embedding_error", "details": str(e)}), 500

    # 2) Query Chroma (lazy-init)
    try:
        collection = get_chroma_collection()
        results = collection.query(query_embeddings=[emb], n_results=top_k, include=["documents","metadatas","distances"])
    except Exception as e:
        return jsonify({"error": "chroma_query_error", "details": str(e)}), 500

    # aggregate docs/metas/distances (Chroma returns nested lists)
    documents = []
    metadatas = []
    distances = []
    for docs_list in results.get("documents", []):
        documents.extend(docs_list)
    for meta_list in results.get("metadatas", []):
        metadatas.extend(meta_list)
    for dist_list in results.get("distances", []):
        distances.extend(dist_list)

    # Build context for prompt
    context_blocks = []
    for i, doc_text in enumerate(documents[:top_k]):
        meta = metadatas[i] if i < len(metadatas) else {}
        source = meta.get("source", "unknown")
        chunk = meta.get("chunk", i)
        context_blocks.append(f"[source: {source} | chunk: {chunk}]\n{doc_text}")

    retrieval_context = "\n\n---\n\n".join(context_blocks)
    system_prompt = build_system_prompt()
    user_prompt = render_user_prompt(question=query, context=retrieval_context)

    # 3) Attempt to generate model output (we ask model to return JSON if possible)
    try:
        raw = generate_text(system_prompt=system_prompt, user_prompt=user_prompt, model="gemini-2.5-flash")
    except Exception as e:
        return jsonify({"error": "generation_error", "details": str(e)}), 500

    # 4) Try to parse the model output as JSON (preferred)
    structured = None
    try:
        # model may return text that is JSON or wrapped; try to find JSON substring
        text = raw.strip()
        # case: model returned only JSON
        if text.startswith("{") and text.endswith("}"):
            structured = json.loads(text)
        else:
            # try to locate first "{" to last "}"
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = text[start:end+1]
                structured = json.loads(candidate)
    except Exception:
        structured = None

    # 5) If parsing failed, create structured JSON heuristically from raw text
    if structured is None:
        # Heuristic: summary = first paragraph or first sentence
        raw_text = raw.strip()
        paras = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
        summary = ""
        if paras:
            import re
            sentences = re.split(r'(?<=[.!?])\s+', paras[0])
            summary = " ".join(sentences[:2]).strip()
        else:
            summary = raw_text[:200] + ("..." if len(raw_text) > 200 else "")

        lines = raw_text.splitlines()
        key_facts = []
        supporting = []
        for ln in lines:
            l = ln.strip()
            if not l: continue
            if l.startswith("-") or l.startswith("*") or l and l[0].isdigit():
                if len(key_facts) < 3:
                    key_facts.append(l.lstrip("-*0123456789. ").strip())
                else:
                    supporting.append(l.lstrip("-*0123456789. ").strip())
        if not key_facts:
            for p in paras[1:3]:
                s = p.split(". ")
                if s:
                    key_facts.append(s[0].strip() + ("" if s[0].endswith(".") else "."))
        if not supporting:
            for p in paras[3:5]:
                supporting.append(p if len(p) < 200 else p[:200] + "...")

        structured = {
            "summary": summary,
            "key_facts": key_facts,
            "supporting_details": supporting,
            "disclaimer": "This is informational only — consult a licensed clinician for personalized advice.",
            "sources": []
        }

    # 6) attach sources (include distance/score if present)
    sources_out = []
    for i, meta in enumerate(metadatas[:top_k]):
        entry = {
            "source": meta.get("source", "unknown"),
            "chunk": meta.get("chunk", i),
        }
        if i < len(distances):
            entry["score"] = distances[i]
        sources_out.append(entry)
    structured["sources"] = sources_out

    # 7) cache the generation (optional)
    gen_cache_key = "gen:" + str(hash(structured.get("summary","") + json.dumps(structured.get("key_facts",[]))))
    cache.set(gen_cache_key, structured, ttl=int(os.environ.get("GEN_CACHE_TTL", 60 * 30)))

    return jsonify(structured)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    # For local dev only; in production Gunicorn will be used
    app.run(host="0.0.0.0", port=port, debug=debug)
