# Offline Multimodal Legal RAG Chatbot

A fully offline, privacy-first RAG (Retrieval-Augmented Generation) chatbot built for Indian legal documents. It understands PDFs, Word documents, images, and audio — all running on your own machine with no data leaving your device.

Built for Smart India Hackathon 2024.

---

## What it does

You upload Indian legal documents — court judgments, FIRs, bail applications, statutes, diagrams, audio recordings — and then ask questions in plain English. The system finds the relevant parts of those documents and uses a local language model to generate an answer grounded in what it actually found, not in what the model memorized during training.

It works across four document types at once. Ask about a FIR and it pulls text from the DOCX. Ask about a court hierarchy diagram and it reads the image. Ask about NCRB crime statistics and it cross-references the PDF and the chart image together. The answer always cites which file it came from.

---

## Key features

**Multimodal ingestion**
Processes PDF (via pdfplumber + unstructured), DOCX (python-docx), images (CLIP + Tesseract OCR), and audio (faster-whisper) in parallel. Everything gets embedded into a unified FAISS vector index.

**Hybrid retrieval**
Combines two search strategies — CLIP semantic embeddings and BM25 keyword search — then fuses their ranked results using Reciprocal Rank Fusion. A cross-encoder (ms-marco-MiniLM) reranks the top 50 candidates to get the final 15. This beats either approach alone, especially on legal text where exact terminology matters.

**Adaptive FAISS index**
Automatically picks the right index type based on corpus size:
- Under 5,000 chunks → `IndexFlatIP` (exact search, maximum accuracy)
- 5,000–100,000 chunks → `IndexHNSWFlat` (approximate, ~99% recall, much faster)
- Over 100,000 chunks → `IndexIVFFlat` (scalable ANN for very large corpora)

**Confidence gate**
Before calling the LLM, the system scores every retrieved chunk against the query using the cross-encoder. If nothing scores above the confidence threshold, it declines to answer rather than hallucinate. This is what lets it correctly refuse questions that aren't in the indexed documents.

**Hallucination evaluation**
Every answer gets scored on three metrics automatically:
- *Faithfulness* — are the answer's sentences grounded in the retrieved chunks?
- *Answer relevance* — does the answer actually address the question?
- *Context recall* — did the answer use the retrieved context?

**Hardware-aware model selection**
Detects your hardware at startup and picks the best model that fits in memory. On Apple M3 8GB it defaults to `qwen2.5vl:3b` (3.2GB), which ran at **0.912 average reliability score** across 10 legal queries. Full configuration table below.

**3-layer stack**
Flask (RAG backend) → Node.js (API gateway + auth) → React (chat UI). Can also be run standalone via the Flask API or command line.

**Redis caching**
Three-layer cache: embedding cache (1h TTL), retrieval cache (30min TTL), and per-session conversation memory (1h TTL). Cuts latency significantly on repeated or similar queries.

---

## Model comparison results

Tested on 10 Indian legal queries across PDF, DOCX, image, and audio sources. All models ran locally via Ollama on Apple M3 8GB with `OLLAMA_FLASH_ATTENTION=1`.

| Model | Avg score | Avg latency | Queries declined | Size |
|---|---|---|---|---|
| **qwen2.5vl:3b** | **0.912** | **24.7s** | 0 / 10 | 3.2 GB |
| llava:7b (Q4_0, ctx=1024) | 0.900 | 35.7s | 0 / 10 | 4.7 GB |
| llava-phi3 | 0.772 | 37.7s | 3 / 10 | 2.9 GB |

**qwen2.5vl:3b** is the default for machines with 8GB or less. It's faster, more accurate, and answers every in-corpus query. **llava:7b** runs fine on 8GB with flash attention and ctx=1024 — it scored within 1.2% of qwen and occasionally outperformed it on statute text, but needs more memory and is 1.4× slower. **llava-phi3** underperformed both and incorrectly declined 3 queries that are clearly in the corpus.

Score is a composite of faithfulness, answer relevance, and context recall. Threshold for "Reliable" is 0.6.

---

## Stack

This is a MERN stack project with an additional Python/Flask service for AI:

```
React + Vite  (port 3000)   — chat UI, file upload, analytics dashboard
      ↓
Node / Express  (port 5000) — API gateway, MongoDB persistence, Socket.io
      ↓
Flask  (port 5001)          — RAG engine (FAISS, reranking, Ollama LLM)
      ↓
MongoDB                     — conversation history, analytics
Ollama                      — local LLM inference (qwen2.5vl:3b by default)
```

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB — `brew install mongodb-community` on Mac, or run via Docker
- [Ollama](https://ollama.com) — for running the LLM locally
- Tesseract OCR — `brew install tesseract` on Mac, `apt install tesseract-ocr` on Linux
- Redis (optional, for caching) — `brew install redis`

For Apple Silicon Macs, start Ollama with flash attention to reduce memory usage:
```bash
OLLAMA_FLASH_ATTENTION=1 ollama serve
```

---

## Installation

```bash
git clone https://github.com/BayesianDecoder/offline-multimodal-rag
cd offline-multimodal-rag

# Python RAG backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Pull the LLM
ollama pull qwen2.5vl:3b     # Apple Silicon / 8GB (recommended)
ollama pull llava:7b          # 12GB+ or with flash attention

# Node API gateway
cd server
cp .env.example .env          # edit if your ports differ
npm install
cd ..

# React frontend
cd client && npm install && cd ..
```

---

## Running

**Full MERN stack (recommended):**
```bash
# Terminal 1 — start MongoDB
brew services start mongodb-community   # Mac
# or: sudo systemctl start mongod       # Linux

# Terminal 2 — start Ollama
OLLAMA_FLASH_ATTENTION=1 ollama serve

# Terminal 3 — Flask RAG backend (port 5001)
source venv/bin/activate
python app.py

# Terminal 4 — Node gateway (port 5000)
cd server && npm start

# Terminal 5 — React frontend (port 3000)
cd client && npm run dev
# Open http://localhost:3000
```

**Docker (everything in one command):**
```bash
docker-compose up --build
```

**Quickstart — Flask only (no Node/MongoDB):**
```bash
source venv/bin/activate
python app.py
# Open http://localhost:5001
```

**Command line:**
```bash
python ragchatbot.py text "What are the grounds for bail in Indian law?"
python ragchatbot.py image path/to/diagram.png "What does this show?"
python ragchatbot.py audio path/to/recording.wav
```

---

## Indexing your documents

Put your files in the `dataset/` folder, then run the ingestion pipeline:

```bash
python multimodal_unified_parallel.py
```

Supported formats: `.pdf`, `.docx`, `.doc`, `.png`, `.jpg`, `.jpeg`, `.mp3`, `.wav`, `.m4a`

The pipeline processes files in parallel (configurable via `--workers`), builds the FAISS index, and saves everything to `processed_output/`. This only needs to run once unless you add new documents.

---

## Model comparison

To benchmark two models head-to-head on your own corpus:

```bash
# Default: qwen2.5vl:3b vs llava:7b
python compare_models.py

# Custom models
python compare_models.py --model-a qwen2.5vl:3b --model-b llava-phi3 --ctx-a 2048 --ctx-b 2048

# For llava:7b on 8GB — start Ollama with flash attention first
OLLAMA_FLASH_ATTENTION=1 ollama serve
python compare_models.py --model-b llava:7b --ctx-b 1024
```

Outputs `comparison_results.json` and `comparison_report.html` with per-query scores, latencies, answer previews, and hallucination analysis.

---

## Hardware configuration

The system auto-detects your hardware and picks the right tier. You can override with environment variables:

```bash
RAG_MODEL=llava:7b RAG_NUM_CTX=1024 python app.py
```

| Hardware | Default model | Reranker | Whisper |
|---|---|---|---|
| Apple Silicon ≤ 10 GB | qwen2.5vl:3b | MiniLM-L2 | tiny |
| x86, 6–12 GB RAM | llava-phi3 | MiniLM-L2 | base |
| x86, ≥ 12 GB RAM | llava:7b | MiniLM-L6 | base |

---

## Project structure

```
├── app.py                          # Flask API — upload, index, query endpoints
├── ragchatbot.py                   # Core RAG chatbot class
├── vectorstore.py                  # FAISS + BM25 hybrid search, adaptive index
├── multimodal_unified_parallel.py  # Document ingestion pipeline (parallel)
├── image_handler.py                # CLIP image embedder + OCR
├── audio_handler.py                # faster-whisper transcription
├── compare_models.py               # Head-to-head model benchmarking
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── src/
│   ├── optimization/
│   │   └── model_loader.py         # Hardware detection, model config
│   ├── cache/
│   │   └── redis_cache.py          # 3-layer Redis cache
│   └── evaluation/
│       └── hallucination_eval.py   # Faithfulness / relevance / recall scoring
│
├── server/                         # Node.js API gateway
│   └── src/routes/query.js
│
└── client/                         # React chat frontend
    └── src/
        ├── components/
        │   └── MessageBubble.jsx   # Chat UI with loading hints + eval scores
        └── lib/api.js
```

---

## How the retrieval works

1. Your query is embedded into a 512-dim CLIP vector
2. FAISS finds the top 50 semantically similar chunks (or uses HNSW/IVF for large corpora)
3. BM25 independently finds the top 50 keyword-matching chunks
4. Reciprocal Rank Fusion merges both lists, rewarding chunks that rank well in both
5. The cross-encoder reranks the top 50 fused results — it scores each (query, chunk) pair directly rather than comparing embeddings, giving much better precision
6. The confidence gate checks if the best-scoring chunk clears the minimum threshold. If not, the system declines rather than guess.
7. The top 15 chunks are formatted into a prompt and sent to the local LLM via Ollama
8. The answer and its evaluation scores come back to the UI

---

## Limitations

- **Context window**: At `ctx=2048` (default), very long legal documents get truncated in the prompt. The top 15 chunks each capped at 500 characters is ~7,500 characters of context.
- **Audio**: Only one audio chunk is indexed per file (full transcript as one document). Long recordings may lose detail.
- **OCR quality**: Image-heavy documents depend on Tesseract OCR quality. Handwritten text is not supported.
- **OOD detection**: The confidence gate can miss truly out-of-domain queries when the LLM's Metal GPU and the cross-encoder compete for memory on Apple Silicon. Setting a slightly higher threshold (`--confidence 0.08`) improves this.
- **No graph retrieval**: A Neo4J knowledge graph layer was part of the original design but not yet implemented.

---

## License

MIT
