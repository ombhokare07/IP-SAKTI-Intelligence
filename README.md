# IP-SAKTI Intelligence

Phase 1 builds a trustworthy, source-cited retrieval-augmented generation
pipeline. The current implementation covers FastAPI configuration, PDF text
extraction, deterministic chunking, BGE embeddings, persistent Chroma storage,
and a PDF ingestion command. It also includes a deterministic Trust and
Evidence Engine that evaluates retrieval relevance, citation integrity, claim
support, contradiction signals, hallucination risk, and overall grounding
quality around an injected retriever and answer generator.

## Setup

Use Python 3.11 and install the Phase-1 dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and adjust local settings as needed. Never commit
the `.env` file.

## Run

```powershell
python -m uvicorn backend.main:app --reload
python -m scripts.test_pdf_processing
python -m scripts.ingest_documents
python -m scripts.verify_vector_store
python -m scripts.test_retrieval
python -m pytest tests -q
```

The ingestion command can also be run directly as
`python scripts\ingest_documents.py`. By default it scans only
`data/raw/india/patents/`; explicit PDF files or directories can still be
passed as positional arguments.

The ingestion command recursively discovers PDFs, preserves source filename,
one-based page number, resolved file path, and deterministic chunk ID, embeds
the chunks with `BAAI/bge-base-en-v1.5`, and upserts them into persistent
Chroma storage.

## Trust and Evidence Engine

`rag.pipeline.RAGPipeline` accepts retriever and grounded-answer-generator
implementations. It refuses to generate a definitive answer when retrieved
evidence is insufficient and otherwise returns citations plus a compact trust
summary. The scores describe grounding quality within the current knowledge
base; they are not probabilities of legal correctness.

The `/api/chat` route uses the pipeline assigned to `app.state.rag_pipeline`.
Until a concrete retriever and LLM adapter are configured there, the endpoint
returns HTTP 503. With the API running, inspect a configured query with:

```powershell
python -m scripts.test_rag_query "Is a manufacturing licence required?"
```
