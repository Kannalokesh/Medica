# gemini_index.py
import os
import glob
import json
from pathlib import Path
from typing import List
from chromadb import Client, PersistentClient

# Ensure your repo root is the current working directory when running
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "Medica" / "data"   # put your docs here
CHROMA_DIR = Path(os.environ.get("CHROMA_PERSIST_DIR", REPO_ROOT / "Medica" / "chroma_db"))
NEW_COLLECTION = os.environ.get("CHROMA_COLLECTION", "medico_docs_gemini768")
BATCH_SIZE = 16   # number of text chunks per embed call
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 800))         # characters per chunk
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))   # overlap between chunks

# Import your embedding function from your project
from src.llm_client import get_embeddings

# PDF loader helper (uses PyPDF2)
def load_pdf_text(path: Path) -> str:
    try:
        from PyPDF2 import PdfReader
    except Exception as e:
        raise RuntimeError("PyPDF2 is required to load PDFs. Install: pip install PyPDF2") from e
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            # continue - some pages may fail text extraction
            pages.append("")
    return "\n".join(pages)

def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = max(start + chunk_size - overlap, end)
    return [c for c in chunks if c]

def discover_files(data_dir: Path):
    patterns = ["**/*.txt", "**/*.md", "**/*.pdf"]
    files = []
    for pat in patterns:
        files.extend(sorted(data_dir.glob(pat)))
    return files

def main():
    print("Repo root:", REPO_ROOT)
    print("Loading files from:", DATA_DIR)
    files = discover_files(DATA_DIR)
    print(f"Found {len(files)} files to index.")

    if not files:
        print("No files found. Put .txt/.md/.pdf files under data/docs/ and retry.")
        return

    # prepare chroma client (persistent)
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=NEW_COLLECTION)
    print(f"Using Chroma DB at {CHROMA_DIR}, collection '{NEW_COLLECTION}'")

    all_docs = []
    all_metadatas = []
    all_ids = []
    idx = 0

    for file_path in files:
        print("Processing:", file_path)
        try:
            if file_path.suffix.lower() == ".pdf":
                text = load_pdf_text(file_path)
            else:
                text = load_text_file(file_path)
        except Exception as e:
            print(f" WARNING: Failed to load {file_path}: {e}")
            continue

        # Pre-sanitize: collapse long whitespace
        text = " ".join(text.split())

        chunks = split_text(text)
        for ci, chunk in enumerate(chunks):
            doc_id = f"{file_path.name}__chunk_{ci}"
            all_docs.append(chunk)
            meta = {"source": file_path.name, "chunk": ci}
            all_metadatas.append(meta)
            all_ids.append(doc_id)
            idx += 1

    print(f"Total chunks to embed: {len(all_docs)}")

    # compute embeddings in batches
    embeddings = []
    for i in range(0, len(all_docs), BATCH_SIZE):
        batch = all_docs[i:i+BATCH_SIZE]
        print(f"Embedding batch {i}..{i+len(batch)-1}")
        embs = get_embeddings(batch)
        if not isinstance(embs, list) or not embs:
            raise RuntimeError("get_embeddings returned invalid response")
        embeddings.extend(embs)

    print("Adding documents into collection...")
    # delete existing collection with same name? Here we just use get_or_create and add
    try:
        collection.add(documents=all_docs, metadatas=all_metadatas, ids=all_ids, embeddings=embeddings)
    except Exception as e:
        # Some chroma versions expect parameters named differently
        try:
            collection.add(documents=all_docs, metadata=all_metadatas, ids=all_ids, embeddings=embeddings)
        except Exception as e2:
            raise RuntimeError(f"Failed to add to collection: {e} / {e2}") from e2

    # persist
    try:
        client.persist()
    except Exception:
        pass

    print("Reindexing complete. First embedding vector length:", len(embeddings[0]) if embeddings else "none")

if __name__ == "__main__":
    main()
