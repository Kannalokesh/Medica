#app.py

import os, json, traceback
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from src.llm_client import generate_text, get_embeddings
from src.cache import Cache
from src.prompt import build_system_prompt, render_user_prompt
from chromadb import Client, PersistentClient

# Config
CHROMA_DIR = os.environ.get("CHROMA_PERSIST_DIR")  # optional
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "medico_docs")
RATE_LIMIT = os.environ.get("RATE_LIMIT", "20 per minute")  # default rate
EMBED_CACHE_TTL = int(os.environ.get("EMBED_CACHE_TTL", 60 * 60 * 24))

app = Flask(__name__, template_folder="templates")

# Rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])
limiter.init_app(app)

# --------------------------------------------------
# TEMP DEBUG ENDPOINT (REMOVE AFTER DEBUGGING)
# --------------------------------------------------
@app.route("/_debug_check", methods=["GET"])
def debug_check():
    out = {"ok": True, "checks": {}}

    # 1) Environment variables (non-secret)
    try:
        out["checks"]["GENAI_API_KEY_present"] = bool(os.environ.get("GENAI_API_KEY"))
        out["checks"]["GEMINI_API_KEY_present"] = bool(os.environ.get("GEMINI_API_KEY"))
        out["checks"]["USE_VERTEXAI"] = os.environ.get("USE_VERTEXAI")
        out["checks"]["VERTEXAI_PROJECT_present"] = bool(os.environ.get("VERTEXAI_PROJECT"))
        out["checks"]["VERTEXAI_LOCATION_present"] = bool(os.environ.get("VERTEXAI_LOCATION"))
        out["checks"]["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

        # list secret files
        try:
            if os.path.exists("/etc/secrets"):
                out["checks"]["secrets_list"] = os.listdir("/etc/secrets")
            else:
                out["checks"]["secrets_list"] = []
        except Exception as e:
            out["checks"]["secrets_list_error"] = str(e)

    except Exception:
        out["checks"]["env_error"] = traceback.format_exc()

    # 2) sentence-transformers model check
    try:
        from src.llm_client import _ensure_s2t_model
        try:
            _ensure_s2t_model()
            out["checks"]["s2t_model"] = "ok"
        except Exception as e:
            out["checks"]["s2t_model_error"] = str(e)
            out["checks"]["s2t_model_trace"] = traceback.format_exc()
    except Exception as e:
        out["checks"]["s2t_import_error"] = str(e)
        out["checks"]["s2t_import_trace"] = traceback.format_exc()

    # 3) genai client init check
    try:
        from src import llm_client
        try:
            llm_client._init_genai_client()
            out["checks"]["genai_init"] = "ok"
        except Exception as e:
            out["checks"]["genai_init_error"] = str(e)
            out["checks"]["genai_init_trace"] = traceback.format_exc()
    except Exception as e:
        out["checks"]["genai_module_error"] = str(e)
        out["checks"]["genai_module_trace"] = traceback.format_exc()

    return jsonify(out)
# --------------------------------------------------



# Chroma client
if CHROMA_DIR:
    client = PersistentClient(path=CHROMA_DIR)
else:
    client = Client()
collection = client.get_or_create_collection(name=CHROMA_COLLECTION)

# Cache
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
        return jsonify({"error": "empty_query"}), 400

    # 1) embed with cache
    cache_key = f"embed:{query}"
    emb = cache.get(cache_key)
    if emb is None:
        try:
            emb_list = get_embeddings([query])
            emb = emb_list[0]
            cache.set(cache_key, emb, ttl=EMBED_CACHE_TTL)
        except Exception as e:
            return jsonify({"error": "embedding_error", "details": str(e)}), 500

    # 2) query chroma
    try:
        results = collection.query(
            query_embeddings=[emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        return jsonify({"error": "chroma_query_error", "details": str(e)}), 500

    documents = []
    metadatas = []
    distances = []

    for docs_list in results.get("documents", []):
        documents.extend(docs_list)
    for meta_list in results.get("metadatas", []):
        metadatas.extend(meta_list)
    for dist_list in results.get("distances", []):
        distances.extend(dist_list)

    # build retrieval context
    context_blocks = []
    for i, doc_text in enumerate(documents[:top_k]):
        meta = metadatas[i] if i < len(metadatas) else {}
        source = meta.get("source", "unknown")
        chunk = meta.get("chunk", i)
        context_blocks.append(f"[source: {source} | chunk: {chunk}]\n{doc_text}")

    retrieval_context = "\n\n---\n\n".join(context_blocks)
    system_prompt = build_system_prompt()
    user_prompt = render_user_prompt(question=query, context=retrieval_context)

    # 3) generate response
    try:
        raw = generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="gemini-2.5-flash"
        )
    except Exception as e:
        return jsonify({"error": "generation_error", "details": str(e)}), 500

    # 4) try parse JSON output
    structured = None
    try:
        text = raw.strip()
        if text.startswith("{") and text.endswith("}"):
            structured = json.loads(text)
        else:
            s = text.find("{")
            e = text.rfind("}")
            if s != -1 and e != -1:
                structured = json.loads(text[s:e + 1])
    except Exception:
        structured = None

    # 5) fallback heuristic JSON
    if structured is None:
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
            if not l:
                continue
            if l.startswith("-") or l.startswith("*") or l[0].isdigit():
                if len(key_facts) < 3:
                    key_facts.append(l.lstrip("-*0123456789. ").strip())
                else:
                    supporting.append(l.lstrip("-*0123456789. ").strip())

        if not key_facts:
            for p in paras[1:3]:
                s = p.split(". ")
                if s:
                    key_facts.append(s[0].strip())

        if not supporting:
            for p in paras[3:5]:
                supporting.append(p[:200] + "..." if len(p) > 200 else p)

        structured = {
            "summary": summary,
            "key_facts": key_facts,
            "supporting_details": supporting,
            "disclaimer": "This is informational only — consult a licensed clinician.",
            "sources": []
        }

    # 6) attach sources
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

    # 7) save to cache
    gen_cache_key = "gen:" + str(hash(structured.get("summary", "")))
    cache.set(gen_cache_key, structured, ttl=int(os.environ.get("GEN_CACHE_TTL", 60 * 30)))

    return jsonify(structured)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
