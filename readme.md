Here's a **complete implementation design + user guide** in one file (`README.md` style), explaining how your **CPU-only RAG PDF Chatbot with LangChain and Ollama** works.

You can save this as a `README.md` or just a reference text doc.

---

## 🤖 PDF Chatbot (CPU-Only, RAG-based) with LangChain & Ollama

This project is a **compact**, fully **CPU-compatible** AI chatbot that allows users to chat with **PDF files**. It uses **Ollama's local LLMs** (like `mistral`) and **LangChain's RAG (Retrieval-Augmented Generation)** pipeline.

---

## 📌 Features

- ✅ Works fully offline (no internet needed)
- ✅ Uses CPU-only local LLMs (via [Ollama](https://ollama.com/))
- ✅ Streams responses with a spinner
- ✅ Text and vector **caching** for fast reuse
- ✅ Supports multiple PDFs
- ✅ **Advanced multi-PDF search** capability
- ✅ Document source attribution in responses
- ✅ Well-documented and under 200 lines of code
- ✅ Frontend ready (backend clean and modular)

---

## 🧰 Tech Stack

| Component      | Tool/Library            |
|----------------|-------------------------|
| Text Extraction | `PyMuPDF` (`fitz`)      |
| Embeddings      | `OllamaEmbeddings` (CPU)|
| Vector Store    | `FAISS` (in-memory + pickled cache) |
| Text Splitter   | `RecursiveCharacterTextSplitter` |
| LLM             | `mistral` via `OllamaLLM` |
| RAG Chain       | LangChain `RunnableMap` |
| Caching         | File-based (txt + faiss `.pkl`) |
| Interaction     | CLI with streaming & spinner |

---

## 📁 Project Structure

```
chat_pdf.py             # Main chatbot script
pdfs/                   # Put your PDFs here
 └── your_file.pdf
cache/                  # Auto-generated text + vector cache
 ├── your_file.pdf.txt
 └── your_file.pdf.faiss
```

---

## 🚀 How It Works (Design Flow)

### 1. 📄 PDF Selection
- User can choose between **single PDF mode** or **multi-PDF search mode**
- For single mode: select one PDF to chat with
- For multi mode: system loads and indexes all PDFs in the folder
  
### 2. 📚 Text Extraction + Caching
- If cached `.txt` exists → load text.
- Else → extract using `fitz` (PyMuPDF) and cache to `.txt`.

### 3. 🧠 Chunking + Embedding + FAISS Index
- If `.faiss` exists → load cached index.
- Else → split text into chunks → embed with `OllamaEmbeddings` → store in FAISS → save to `.pkl`.

### 4. 🔄 RAG Chain Setup
- A **RAG chain** is created using:
  - Retriever (from FAISS)
  - PromptTemplate
  - `OllamaLLM` with streaming enabled
- Combines relevant context chunks + user query → feeds to LLM.

### 5. 💬 Chat Loop
- User asks a question.
- Background spinner starts.
- LLM streams response using `mistral` via Ollama.
- Response is printed word-by-word for smooth UX.

---

## ⚙️ Setup Instructions

### 1. 🐍 Create and activate virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows CMD:
venv\Scripts\activate

# On Windows Git Bash:
source venv/Scripts/activate

# On Unix or MacOS:
source venv/bin/activate
```

### 2. 📦 Install dependencies

```bash
pip install langchain langchain-core langchain-community langchain-ollama langchain-text-splitters pymupdf faiss-cpu
```

### 3. 📥 Install and run Ollama

Install Ollama: [https://ollama.com/download](https://ollama.com/download)

Pull the model:

```bash
ollama pull mistral
```

Make sure Ollama is running (`ollama run mistral` test works).

---

## 🧑‍💻 Run the Chatbot

```bash
python chat_pdf.py
```

Follow prompts:
- Choose mode: single PDF or multi-PDF search
- For single mode: select a specific PDF
- For multi mode: all PDFs will be loaded
- Ask questions like:
  - "What is this PDF about?"
  - "Summarize this document."
  - "What are the key takeaways?"
  - "Compare the information in all documents about X topic."

## 🔍 Multi-PDF Search

The new multi-PDF search functionality allows you to:

1. **Search across all PDFs** in your collection simultaneously
2. **Compare information** between different documents
3. **Identify the source document** for each piece of information
4. **Synthesize knowledge** from multiple documents

When using multi-PDF search, the system:
- Loads and indexes all PDFs from the `pdfs/` directory
- Retrieves relevant information from each document based on your query
- Specifies which document each piece of information came from
- Provides a coherent answer that combines knowledge from multiple sources

Example questions for multi-PDF search:
- "Which document has the most information about X?"
- "Compare how document A and document B discuss topic Y."
- "Find all mentions of Z across all documents."
- "What are the differences in how these documents approach problem W?"

---

## 🧠 Sample Prompt Template

```text
Answer the question based only on the context below:

{context}

Question: {question}
```

This helps ensure responses are grounded in your document.

---

## 🔁 Caching Explained

- Saves **text extraction** as `.txt`
- Saves **vector index** as `.faiss` (pickle)
- On next load, skips expensive operations if files exist

---

## 🛠️ Future Enhancements (Planned)

- ✅ Add Streamlit/Gradio frontend
- ✅ Upload and chat with multiple PDFs at once
- 🔒 Per-user session memory
- 🌐 Export Q&A as JSON/Markdown
- 📥 CLI/GUI drag-and-drop for PDFs

---

## 🤝 Credits

- [LangChain](https://www.langchain.com/)
- [Ollama](https://www.ollama.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [PyMuPDF](https://pymupdf.readthedocs.io/)

---

Let me know if you want this saved as a `.md` or `.txt` file or bundled with your code!