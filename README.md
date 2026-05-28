# Intelldoc

Document RAG pipeline for parsing internal documents, enriching images/tables, chunking content, embedding chunks, indexing them in Qdrant, and asking questions with a chatbot.

## Pipeline

```text
┌────────────────────┐
│ data/raw/*.pdf     │
└─────────┬──────────┘
          │
          ▼
┌────────────────────────────────────────────┐
│ src/indexing/ingestion/parse_pdf.py        │
│ Extract text, tables, images, metadata     │
└─────────┬──────────────────────────────────┘
          │ elements.json
          ▼
┌────────────────────────────────────────────┐
│ src/indexing/enrichment/summarize_elements.py │
│ Generate image descriptions with Bedrock   │
└─────────┬──────────────────────────────────┘
          │ enriched_elements.json
          ▼
┌────────────────────────────────────────────┐
│ src/indexing/chunking/chunker.py           │
│ Create RAG chunks                          │
└─────────┬──────────────────────────────────┘
          │ chunks.json
          ▼
┌────────────────────────────────────────────┐
│ src/indexing/embedding/embed_and_index.py  │
│ Embed chunks and save vectors + payloads   │
└─────────┬──────────────────────────────────┘
          │ Qdrant collection
          ▼
┌────────────────────────────────────────────┐
│ src/chatbot/chat.py                        │
│ Retrieve chunks and answer questions       │
└────────────────────────────────────────────┘
```

## Project Structure

```text
src/
  indexing/
    ingestion/
      parse_pdf.py
      parse_docx.py
    enrichment/
      summarize_elements.py
    chunking/
      chunker.py
    embedding/
      embed_and_index.py

  chatbot/
    retrieval.py
    chat.py
```

## Setup

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

Create your local Bedrock env file:

```powershell
Copy-Item .env.example .env.bedrock
```

Fill `.env.bedrock` with your real AWS values. This file is ignored by Git.

```env
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=eu-central-1
MODEL_ID=eu.amazon.nova-lite-v1:0
EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
EMBEDDING_DIMENSIONS=1024
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=intelldoc_chunks
```

Start Qdrant:

```powershell
docker compose up -d qdrant
```

Qdrant dashboard:

```text
http://localhost:6333/dashboard
```

## Run The Pipeline

Put PDFs in:

```text
data/raw/
```

Parse PDFs:

```powershell
python src\indexing\ingestion\parse_pdf.py
```

Generate image descriptions:

```powershell
python src\indexing\enrichment\summarize_elements.py
```

Create chunks:

```powershell
python src\indexing\chunking\chunker.py
```

Embed and index into Qdrant:

```powershell
python src\indexing\embedding\embed_and_index.py --recreate-collection
```

Ask a question:

```powershell
python src\chatbot\chat.py "What is this document about?"
```

Retrieve chunks without generating an answer:

```powershell
python src\chatbot\retrieval.py "What is this document about?" --top-k 5
```

## Outputs

Each parsed document gets its own directory:

```text
data/parsed/pdf/<document_id>/
  elements.json
  enriched_elements.json
  chunks.json
  text.txt
  images/
  tables/
```

`elements.json` is the parser output.

`enriched_elements.json` keeps the document order but replaces useful image elements with LLM-generated `image_description` text.

`chunks.json` is the local source of truth before embedding.

Qdrant stores each chunk vector plus the retrieval payload used by the chatbot,
including chunk text, document/page identifiers, source element IDs, and
image/table paths.

## Notes

Tesseract OCR is optional but recommended for richer PDF table/image extraction. Without Tesseract, the parser falls back to `pypdf`, which extracts text and embedded images but cannot create fully structured HTML tables.

Local data, Qdrant storage, and secrets are ignored by Git:

```text
data/
qdrant_data/
.env.bedrock
```
