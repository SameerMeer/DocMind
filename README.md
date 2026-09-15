# 📄 DocMind

**AI-powered PDF Question Answering using Retrieval-Augmented Generation (RAG)**

DocMind is a document-grounded PDF chatbot. A user uploads a PDF through the web interface, DocMind extracts and chunks the document, creates embeddings, retrieves relevant evidence, reranks the results, and generates an answer using a local LLM.

The application is designed to work with **user-uploaded text-based PDFs**, not with a PDF hardcoded into the project.

## Features

- Upload any text-based PDF through the Streamlit UI
- Temporary PDF handling; no project `data/` folder is required
- Structured section-aware chunking when numbered sections are detected
- Generic chunking for ordinary PDFs
- Sentence Transformer embeddings
- FAISS vector similarity search
- Keyword retrieval for exact terms, identifiers, and numbers
- Hybrid retrieval
- Cross-encoder reranking
- Multi-part question support
- Grounded LLM answers
- Page/section/evidence display
- Refusal when the retrieved PDF context does not support an answer
- Minimal Streamlit UI

## Architecture

```text
User
  │
  │ uploads any PDF
  ▼
Streamlit
  │
  ▼
Temporary PDF
  │
  ▼
PDF Extraction + Chunking
  │
  ▼
Sentence Transformer Embeddings
  │
  ▼
FAISS
  │
  ├───────────────┐
  ▼               ▼
Semantic       Keyword
Search         Search
  │               │
  └───────┬───────┘
          ▼
   Hybrid Retrieval
          │
          ▼
 Cross-Encoder Reranking
          │
          ▼
   Relevant PDF Context
          │
          ▼
   Ollama + Llama 3.2 3B
          │
          ▼
   Grounded Answer
          │
          ▼
 Answer + Sources
```

## Project Structure

```text
DocMind/
├── app.py
├── pdf_processor.py
├── rag.py
├── requirements.txt
├── README.md
├── .gitignore
└── DocMind_AI_PDF_Chatbot.ipynb
```

The notebook is a development/learning artifact. The Streamlit application uses `app.py`, `pdf_processor.py`, and `rag.py`.

## How It Works

### 1. Upload

The user selects any PDF from the Streamlit interface.

The application temporarily writes the uploaded bytes to the operating system's temporary directory. It does not require a PDF to exist inside the project repository.

### 2. PDF processing

`pdf_processor.py` extracts text using PyPDF.

If numbered sections are detected, section information is preserved.

For ordinary PDFs without numbered sections, generic page-based chunks are created.

Long sections/pages are split into smaller overlapping chunks.

Each chunk has this structure:

```python
{
    "page": 1,
    "section": 1,
    "section_title": "Page 1",
    "text": "..."
}
```

### 3. Embeddings

Each chunk is converted into an embedding using:

```text
all-MiniLM-L6-v2
```

### 4. FAISS retrieval

Embeddings are normalized and stored in a FAISS inner-product index, which provides cosine-similarity-style semantic retrieval.

### 5. Hybrid retrieval

DocMind combines:

- semantic retrieval
- keyword retrieval
- query variants
- exact-match signals

This helps both meaning-based questions and exact identifiers/numbers.

### 6. Reranking

A cross-encoder reranks the candidate passages:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The strongest evidence is passed to the generation step.

### 7. Grounded generation

The retrieved passages are supplied to:

```text
Llama 3.2 3B
```

through Ollama.

The model is instructed to answer only from the supplied PDF context.

If the answer is not supported:

```text
I could not find the answer in the PDF.
```

### 8. Sources

The UI displays the retrieved page, section, score, and evidence text separately from the answer.

## Installation

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd DocMind
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Windows:

```powershell
venv\Scriptsctivate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Install Ollama

Make sure Ollama is installed and download the model:

```powershell
ollama pull llama3.2:3b
```

Then make sure Ollama is running.

### 5. Start DocMind

```powershell
streamlit run app.py
```

## Example Questions

The application can be used with many kinds of documents.

For a resume:

```text
What programming languages are mentioned?
What projects are listed?
What databases are mentioned?
```

For a research paper:

```text
What is the main objective?
What methodology was used?
What were the results?
```

For technical documentation:

```text
What API endpoints are available?
What database is used?
How does authentication work?
```

## Limitations

- PyPDF works best with text-based PDFs.
- Scanned/image-only PDFs require OCR support.
- Tables and complex PDF layouts may not extract perfectly.
- The current LLM is local and requires Ollama.
- Large documents may require additional retrieval and memory optimization.
- The current application processes one uploaded PDF at a time.
- Conversation memory across documents is not implemented.

## Future Improvements

- OCR for scanned PDFs
- Better table extraction
- Multiple-document collections
- Persistent vector databases
- Streaming answers
- Conversation memory
- Authentication
- Cloud LLM deployment
- Production backend/frontend separation

## Project Goal

DocMind demonstrates a practical end-to-end RAG pipeline:

```text
PDF
 ↓
Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
Vector Search
 ↓
Keyword Search
 ↓
Hybrid Retrieval
 ↓
Reranking
 ↓
Context
 ↓
LLM
 ↓
Grounded Answer
 ↓
Sources
```

The goal is to demonstrate a real working RAG application rather than a theory-only implementation.

## License

Add your preferred license before publishing the repository.
