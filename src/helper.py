from pathlib import Path
from typing import List
import PyPDF2

def load_pdf_file(directory: str) -> List[dict]:
    """
    Read all PDFs in `directory` and return list of dicts:
    [{'id': <id>, 'text': <full_text>, 'metadata': {'source': path}}]
    """
    docs = []
    p = Path(directory)
    for i, f in enumerate(sorted(p.glob("*.pdf"))):
        try:
            reader = PyPDF2.PdfReader(str(f))
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
            txt = "\n".join(text_parts).strip()
            docs.append({"id": f"{f.stem}_{i}", "text": txt, "metadata": {"source": str(f), "filename": f.name}})
        except Exception as e:
            print(f"[helper.load_pdf_file] Failed to read {f}: {e}")
    return docs

def text_split(doc_text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split a large text into chunks approximating chunk_size (characters),
    preserving words and keeping overlap characters between chunks.
    """
    if not doc_text:
        return []
    words = doc_text.split()
    chunks = []
    cur = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + 1 > chunk_size:
            chunks.append(" ".join(cur))
            # prepare overlap (characters)
            if overlap > 0:
                # keep last N characters of current chunk as new start
                tail = " ".join(cur)[-overlap:]
                cur = tail.split()
            else:
                cur = []
            cur_len = sum(len(x) + 1 for x in cur)
        cur.append(w)
        cur_len += len(w) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def docs_to_chunks(docs:list, chunk_size:int=500, overlap:int=50):
    """
    Convert list of docs (from load_pdf_file) to chunked documents:
    returns list of {"id":..., "text":..., "metadata": {...}}
    """
    out = []
    for doc in docs:
        chunks = text_split(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, chunk in enumerate(chunks):
            out.append({
                "id": f"{doc['id']}_chunk_{i}",
                "text": chunk,
                "metadata": {"source": doc["metadata"].get("source"), "filename": doc["metadata"].get("filename"), "chunk": i}
            })
    return out
