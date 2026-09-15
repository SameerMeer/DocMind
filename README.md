# 📄 DocMind

### AI-Powered PDF Question Answering using Retrieval-Augmented Generation (RAG)

DocMind is an AI-powered PDF question-answering application that allows users to upload a PDF and ask questions about its content.

Instead of sending the entire document directly to an LLM, DocMind extracts the document text, divides it into smaller chunks, converts those chunks into embeddings, retrieves relevant information using hybrid search, reranks the retrieved passages, and generates a grounded answer using a local LLM.

---

## 🚀 Features

- Upload any text-based PDF
- Automatic PDF text extraction
- Section-aware and generic document chunking
- Sentence Transformer embeddings
- FAISS vector search
- Keyword-based retrieval
- Hybrid semantic + keyword retrieval
- Query expansion
- Exact-match retrieval signals
- Cross-encoder reranking
- Grounded LLM responses
- Source/page information
- Multi-part question support
- Refusal when information is not supported by the document
- Minimal Streamlit interface
- Local LLM inference using Ollama

---

## 🧠 Architecture

```text
                    ┌─────────────────┐
                    │   User Uploads  │
                    │       PDF       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Streamlit    │
                    │   Web Interface │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  PDF Extraction │
                    │   + Chunking    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Embeddings    │
                    │ all-MiniLM-L6-v2│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │      FAISS      │
                    │  Vector Search  │
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
       ┌─────────────────┐       ┌─────────────────┐
       │ Semantic Search │       │ Keyword Search  │
       └────────┬────────┘       └────────┬────────┘
                │                         │
                └────────────┬────────────┘
                             ▼
                    ┌─────────────────┐
                    │ Hybrid Retrieval│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Cross-Encoder   │
                    │   Reranking     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Relevant PDF    │
                    │    Context      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Ollama +        │
                    │ Llama 3.2 3B    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Grounded Answer │
                    │    + Sources    │
                    └─────────────────┘
