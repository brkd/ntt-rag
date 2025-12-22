# ntt-rag

RAG-based PDF Question Answering system built on **FastAPI**, **ChromaDB**, and an **OpenAI-compatible** inference endpoint (Docker Compose uses **Ollama**).

The API ingests PDFs on startup, stores embeddings in Chroma, and answers questions via retrieval + LLM generation.

## What’s included

- **PDF ingestion**: load PDFs from a directory, clean text, chunk it, embed it, and store in Chroma
- **Versioned indexing**: keeps a JSON file of document/chunk hashes and only updates changed chunks
- **RAG API**:
  - `GET /health`
  - `POST /ask` (question → answer + sources)

## API usage

- **Health**

```bash
curl -s http://localhost:9632/health
```

- **Ask**

```bash
curl -s http://localhost:9632/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What information is in the documents related to 2014?"}'
```

## Configuration (env vars)

The app reads configuration from a `.env` file in the repo root and/or environment variables (prefix: `NTT_RAG_`).

Required:
- **`NTT_RAG_PDF_LOCATION`**: directory containing PDFs (e.g. `./data/raw`)
- **`NTT_RAG_INFERENCE_SERVER_URL`**: OpenAI-compatible base URL (e.g. `http://localhost:11434/v1`)

Common optional settings (with defaults):
- **`NTT_RAG_LLM_MODEL`**: `llama3.2:3b`
- **`NTT_RAG_CHROMA_HOST`**: `localhost`
- **`NTT_RAG_CHROMA_PORT`**: `8000`
- **`NTT_RAG_CHROMA_COLLECTION`**: `ntt-rag`
- **`NTT_RAG_EMBEDDING_MODEL`**: `Qwen/Qwen3-Embedding-0.6B`
- **`NTT_RAG_CHUNK_SIZE`**: `880`
- **`NTT_RAG_CHUNK_OVERLAP`**: `100`
- **`NTT_RAG_N_SOURCE_RETRIEVAL`**: `20`
- **`NTT_RAG_LLM_MAX_TOKENS`**: `512`
- **`NTT_RAG_LLM_TEMPERATURE`**: `0.0`
- **`NTT_RAG_DATA_VERSION_FILE`**: `.document_versions.json`
- **`NTT_RAG_API_HOST`**: `0.0.0.0`
- **`NTT_RAG_API_PORT`**: `9632`

### Example `.env`

```bash
# Where your PDFs live
NTT_RAG_PDF_LOCATION=./data/raw

# OpenAI-compatible endpoint (Ollama is the default choice)
NTT_RAG_INFERENCE_SERVER_URL=http://localhost:11434/v1

# Model to use (also used by docker-compose’s Ollama bootstrap)
NTT_RAG_LLM_MODEL=llama3.2:3b

# If you run Chroma on a non-default port (e.g. mapped to 8001), update these:
# NTT_RAG_CHROMA_HOST=localhost
# NTT_RAG_CHROMA_PORT=8000
```

## Run with Docker (recommended)

This brings up:
- `chromadb` (persistent data in `./vectordb_mount`)
- `ollama` (persistent models in `./ollama_mount`)
- `fastapi` on `http://localhost:9632`

### 1) (Optional) Create `.env`

If you don’t create it, Docker Compose will still run, but `.env` is the easiest way to customize the model and other settings.

At minimum, to auto-pull an Ollama model on startup:

```bash
echo 'NTT_RAG_LLM_MODEL=llama3.2:3b' > .env
```

### 2) Start everything

```bash
docker compose up --build
```

### 3) Verify

```bash
curl -s http://localhost:9632/health
```

### Notes

- **First startup can take time**: PDFs are ingested and embeddings are computed on app startup, and the embedding model may download from Hugging Face.
- **Chroma port mapping**: Chroma is mapped to `localhost:8001` (container `8000`). The FastAPI container talks to it internally on `chromadb:8000`.

## Run locally (Python)

Local mode runs the FastAPI app on your machine; you still need:
- a running **ChromaDB** server, and
- an OpenAI-compatible **inference endpoint** (e.g. Ollama).

### 1) Create a virtualenv + install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 2) Start dependencies

#### Option A: Run ChromaDB via Docker

```bash
docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/vectordb_mount:/chroma/chroma" \
  -e IS_PERSISTENT=TRUE \
  -e ANONYMIZED_TELEMETRY=FALSE \
  chromadb/chroma
```

#### Option B: Run Ollama

- If you have Ollama installed natively, start it and pull a model:

```bash
ollama serve
```

In another terminal:

```bash
ollama pull llama3.2:3b
```

Or run Ollama in Docker:

```bash
docker run --rm -p 11434:11434 -v "$(pwd)/ollama_mount:/root/.ollama" ollama/ollama
```

### 3) Set required env vars

```bash
export NTT_RAG_PDF_LOCATION=./data/raw
export NTT_RAG_INFERENCE_SERVER_URL=http://localhost:11434/v1
```

If your Chroma runs somewhere else:

```bash
export NTT_RAG_CHROMA_HOST=localhost
export NTT_RAG_CHROMA_PORT=8000
```

### 4) Run the API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 9632
```

Then open `http://localhost:9632/docs` for the interactive Swagger UI.

## Smoke test (optional)

There’s a simple script that loads PDFs, ingests them, then asks one question:

```bash
PYTHONPATH=src python src/scripts/run_rag_smoke_test.py
```

## Troubleshooting

- **The app fails at startup with config errors**: ensure `NTT_RAG_PDF_LOCATION` and `NTT_RAG_INFERENCE_SERVER_URL` are set (or present in `.env`).
- **Slow first run**: embedding model download + PDF ingestion can be lengthy; subsequent runs should be faster due to persisted Chroma + model cache.
- **Want to reindex from scratch**:
  - stop the stack
  - remove `./vectordb_mount` contents and the version file (by default `.document_versions.json`, or whatever you set via `NTT_RAG_DATA_VERSION_FILE`)


