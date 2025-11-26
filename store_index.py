# store_index.py
from dotenv import load_dotenv
load_dotenv()

import os
from src.helper import load_pdf_file, docs_to_chunks
from src.llm_client import get_embeddings
from chromadb import Client, PersistentClient
import math
import time

def create_chroma_collection(collection_name="medico_docs", persist_directory=None):
    if persist_directory:
        client = PersistentClient(path=persist_directory)
    else:
        client = Client()
    collection = client.get_or_create_collection(name=collection_name)
    return client, collection

def chunked(iterable, size):
    """Yield successive chunks of up to `size` from iterable (list)."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]

def index_pdfs(pdf_dir, collection_name="medico_docs", persist_dir=None, embed_batch_size=64, add_batch_size=512):
    """
    Index PDFs:
      - embed texts in batches of embed_batch_size (default 64)
      - add to Chroma in add_batch_size groups (default 512)
    """
    print(f"[index_pdfs] Loading PDFs from {pdf_dir}")
    docs = load_pdf_file(pdf_dir)
    chunks = docs_to_chunks(docs)
    if not chunks:
        print("[index_pdfs] No chunks found. Exiting.")
        return None, None

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [c["id"] for c in chunks]

    print(f"[index_pdfs] Total chunks: {len(texts)}")
    # 1) compute embeddings in batches
    vectors = []
    B = max(1, int(embed_batch_size))
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        print(f"[index_pdfs] Embedding batch {i}..{i+len(batch)}")
        emb = get_embeddings(batch)
        vectors.extend(emb)
        time.sleep(0.01)  # small sleep to avoid resource spikes

    # 2) create/attach collection
    client, collection = create_chroma_collection(collection_name, persist_directory=persist_dir)

    # 3) add to collection in safe add_batch_size chunks
    ADD = max(1, int(add_batch_size))
    total = len(texts)
    print(f"[index_pdfs] Adding to Chroma in chunks of {ADD} (total {total})")
    try:
        for start in range(0, total, ADD):
            end = min(total, start + ADD)
            sub_texts = texts[start:end]
            sub_meta = metadatas[start:end]
            sub_ids = ids[start:end]
            sub_vectors = vectors[start:end]
            print(f"[index_pdfs] Adding items {start}..{end} ({len(sub_ids)}) to collection")
            collection.add(documents=sub_texts, metadatas=sub_meta, ids=sub_ids, embeddings=sub_vectors)
    except Exception as e:
        print("[store_index] collection.add failed:", e)
        # Try fallback: add each small slice individually
        print("[store_index] Attempting per-slice fallback (very slow)...")
        for start in range(0, total, ADD):
            end = min(total, start + ADD)
            try:
                collection.add(
                    documents=texts[start:end],
                    metadatas=metadatas[start:end],
                    ids=ids[start:end],
                    embeddings=vectors[start:end]
                )
            except Exception as e2:
                print(f"[store_index] Fallback add for {start}..{end} failed:", e2)
                raise

    # optional persist: PersistentClient automatically persists, but if using in-memory client nothing to do
    try:
        # If client has persist method (older clients), call; safe to ignore otherwise
        if hasattr(client, "persist"):
            client.persist()
    except Exception:
        pass

    print(f"[index_pdfs] Indexed {len(texts)} chunks into Chroma collection '{collection_name}'")
    return client, collection

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pdf_dir", required=True)
    p.add_argument("--persist_dir", default=None)
    p.add_argument("--collection", default="medico_docs")
    p.add_argument("--embed_batch_size", type=int, default=int(os.environ.get("EMBED_BATCH_SIZE", 64)))
    p.add_argument("--add_batch_size", type=int, default=int(os.environ.get("CHROMA_ADD_BATCH_SIZE", 512)))
    args = p.parse_args()
    index_pdfs(args.pdf_dir, collection_name=args.collection, persist_dir=args.persist_dir,
               embed_batch_size=args.embed_batch_size, add_batch_size=args.add_batch_size)
