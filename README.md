# 📄 DocMind

**AI-powered PDF Question Answering using Retrieval-Augmented Generation (RAG)**

DocMind is a local AI PDF chatbot that allows users to upload a PDF and ask questions about its content. Instead of sending the entire document to the language model, DocMind extracts the document, creates searchable vector representations, retrieves the most relevant sections, reranks them, and generates a grounded answer using only the retrieved PDF context.

## ✨ Features

- Upload a PDF and chat with its contents
- Section-aware PDF processing
- Semantic search using Sentence Transformers
- Keyword-based retrieval for exact terms, IDs, numbers, and technologies
- Hybrid retrieval
- FAISS vector similarity search
- Cross-encoder reranking
- Multi-section retrieval for multi-part questions
- Grounded answers using local Llama 3.2 3B through Ollama
- Source pages, sections, relevance scores, and evidence
- Refusal when requested information is not present in the PDF
- Minimal Streamlit interface

## 🧠 RAG Architecture

```text
                 PDF
                  │
                  ▼
          PDF Text Extraction
                  │
                  ▼
        Section-aware Chunking
                  │
                  ▼
       Sentence Transformer
            Embeddings
                  │
                  ▼
              FAISS
          Vector Database
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 Semantic Search      Keyword Search
        │                   │
        └─────────┬─────────┘
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
          Ollama + Llama 3.2
                  │
                  ▼
          Grounded Answer
                  │
                  ▼
        Answer + PDF Sources
```

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application and RAG pipeline |
| Streamlit | Web interface |
| PyPDF | PDF text extraction |
| Sentence Transformers | Text embeddings |
| FAISS | Vector similarity search |
| Cross-Encoder | Retrieval reranking |
| Ollama | Local LLM serving |
| Llama 3.2 3B | Answer generation |
| NumPy | Numerical processing |

## 📁 Project Structure

```text
DocMind/
├── data/
│   └── uploaded.pdf
├── app.py
├── pdf_processor.py
├── rag.py
├── requirements.txt
├── .gitignore
├── DocMind_AI_PDF_Chatbot.ipynb
└── DocMind_RAG_Test_Document.pdf
```

## ⚙️ How It Works

### 1. Upload

The user uploads a PDF through the Streamlit interface.

### 2. Process

`pdf_processor.py` extracts the PDF text and identifies numbered sections. Each chunk stores:

```python
{
    "page": ...,
    "section": ...,
    "section_title": ...,
    "text": ...
}
```

### 3. Embed

The chunks are converted into vector embeddings using:

```text
all-MiniLM-L6-v2
```

### 4. Retrieve

DocMind combines semantic and keyword retrieval. FAISS performs vector similarity search, while keyword retrieval helps with exact identifiers, numbers, names, and technical terms.

### 5. Rerank

A cross-encoder evaluates the retrieved question-document pairs and improves the ordering of relevant evidence.

### 6. Generate

The selected PDF context is passed to the local:

```text
Llama 3.2 3B
```

through Ollama.

The generation prompt instructs the model to answer only from the supplied PDF context.

### 7. Cite Evidence

DocMind displays the relevant page, section, relevance score, and evidence below the answer.

## 🚀 Installation

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
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and start Ollama

Install Ollama and make sure the model is available:

```bash
ollama pull llama3.2:3b
```

Start the model:

```bash
ollama run llama3.2:3b
```

Keep Ollama running while using DocMind.

### 5. Run DocMind

```bash
streamlit run app.py
```

The application will open in your browser.

## 💬 Example Questions

Using the included Smart Library test document:

```text
What database does the platform use?

What can a student do, and how long can they keep a book?

How many books can a faculty member borrow?

What happens when a reserved book becomes available?

What is Priya Nair's member ID?

What frontend and backend technologies are used?

What are the main security rules?

What was the development cost of the project?
```

For information deliberately absent from the PDF, DocMind is designed to respond:

```text
I could not find the answer in the PDF.
```

## 🔐 Grounded Answering

A major goal of DocMind is to reduce hallucination.

The LLM receives retrieved PDF context and is instructed to:

- use only the supplied PDF context
- answer every part of the question
- preserve exact identifiers and numbers
- combine relevant information from multiple sections
- avoid inventing unsupported facts
- refuse when the requested information is not available

## ⚠️ Current Limitations

- Designed primarily for text-based PDFs
- Scanned/image-only PDFs require OCR support
- Local answer generation currently depends on Ollama
- Large documents may require additional chunking and retrieval optimization
- Deployment requires an environment capable of running the selected LLM

## 🔮 Future Improvements

- OCR for scanned PDFs
- Conversation memory
- Multiple-document search
- Persistent vector databases
- Streaming responses
- Better table extraction
- Authentication and user accounts
- Cloud LLM deployment options
- Production deployment with a separate backend and frontend

## 🎯 Project Goal

DocMind was built to demonstrate a practical end-to-end RAG system rather than a theory-only implementation.

It combines:

```text
Document Processing
        +
Embeddings
        +
Vector Search
        +
Keyword Search
        +
Reranking
        +
LLM Generation
        +
Source Evidence
```

into a complete PDF question-answering application.

## 📌 License

Add your preferred license before publishing the repository.
