
# Medica - an AI powered Medical Assistant

## Overview
Medica is a medical QA assistant:
- Generation: Google Gemini (`gemini-2.5-flash`)
- Embeddings: `sentence-transformers`
- Vector DB: ChromaDB
- Web: Flask
- Caching: SQLite (simple TTL)
- Rate limiting: Flask-Limiter
- Deployment: Render (render.yaml included)

## Setup (local)
1. Create a virtualenv and install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
