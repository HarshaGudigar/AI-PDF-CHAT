Here's a **complete implementation design + user guide** in one file (`README.md` style), explaining how your **CPU-only RAG PDF Chatbot with LangChain and Ollama** works.

You can save this as a `README.md` or just a reference text doc.

---

## 🤖 AI Doc Assist: PDF Chatbot (CPU-Only, RAG-based) with LangChain & Ollama

This project is a **compact**, fully **CPU-compatible** AI chatbot that allows users to chat with **PDF files**. It uses **Ollama's local LLMs** (like `llama3`) and **LangChain's RAG (Retrieval-Augmented Generation)** pipeline.

---

## 📌 Features

- ✅ Works fully offline (no internet needed)
- ✅ Uses CPU-only local LLMs (via [Ollama](https://ollama.com/))
- ✅ Streams responses with a spinner
- ✅ Text and vector **caching** for fast reuse
- ✅ **Multi-document search** works across all PDFs simultaneously
- ✅ Document source attribution in responses 
- ✅ **Built-in conversation memory** for follow-up questions
- ✅ **Dual interface**: console and web UI
- ✅ Simple, streamlined interface
- ✅ Well-documented and under 200 lines of code

---

## 🧰 Tech Stack

| Component      | Tool/Library            |
|----------------|-------------------------|
| Text Extraction | `PyMuPDF` (`fitz`)      |
| Embeddings      | `OllamaEmbeddings` (CPU)|
| Vector Store    | `FAISS` (in-memory + pickled cache) |
| Text Splitter   | `RecursiveCharacterTextSplitter` |
| LLM             | `llama3` via `OllamaLLM` |
| RAG Chain       | LangChain `RunnableMap` |
| Caching         | File-based (txt + faiss `.pkl`) |
| Web Interface   | `Gradio` with responsive UI |
| Interaction     | CLI with streaming & spinner or Web UI |

---

## 📁 Project Structure

```
chat_pdf.py             # Console chatbot script
chat_pdf_web.py         # Web interface script
setup.py                # Package setup for pip install
pdfs/                   # Put your PDFs here
 └── your_file.pdf
cache/                  # Auto-generated text + vector cache
 ├── your_file.pdf.txt
 └── your_file.pdf.faiss
```

---

## 🚀 How It Works (Design Flow)

### 1. 📄 Automatic PDF Loading
- All PDFs in the `pdfs/` folder are loaded automatically
- Metadata (pages, file size) is extracted and displayed
  
### 2. 📚 Text Extraction + Caching
- If cached `.txt` exists → load text
- Else → extract using `fitz` (PyMuPDF) and cache to `.txt`

### 3. 🧠 Chunking + Embedding + FAISS Index
- If `.faiss` exists → load cached index
- Else → split text into chunks → embed with `OllamaEmbeddings` → store in FAISS → save to cache

### 4. 🔄 RAG Chain with Memory
- A **RAG chain** is created using:
  - Retrievers from all PDF vector stores
  - Conversation memory
  - PromptTemplate
  - `OllamaLLM` with streaming enabled
- Combines relevant context chunks + conversation history + user query → feeds to LLM

### 5. 💬 Chat Loop
- User asks a question
- Background spinner starts
- LLM streams response with document attribution
- Response is printed word-by-word for smooth UX
- Conversation is saved to memory for context

---

## ⚙️ Setup Instructions

### Manual Setup

#### 1. 🐍 Create and activate virtual environment

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

#### 2. 📦 Install dependencies

```bash
pip install -e .
```
Or manually:
```bash
pip install langchain langchain-core langchain-community langchain-ollama langchain-text-splitters pymupdf faiss-cpu gradio
```

#### 3. 📥 Install and run Ollama

Install Ollama: [https://ollama.com/download](https://ollama.com/download)

Pull the llama3 model:

```bash
ollama pull llama3
```

Make sure Ollama is running before starting the application.

---

## 🧑‍💻 Run the Chatbot

### Console Interface

```bash
python chat_pdf.py
```

The console chatbot:
1. Automatically loads all PDFs in the `pdfs/` folder
2. Remembers your conversation history
3. Provides document sources for information

### Web Interface

```bash
python chat_pdf_web.py --web
```

The web interface offers:
1. **Modern Chat UI** with conversation history visualization
2. **Easy PDF upload** through drag-and-drop or file selection
3. **Document management** with real-time updates of loaded PDFs
4. **Clear conversation** button to reset memory
5. **Error handling** for missing PDFs or Ollama connection issues

The web interface has two main sections:
- **Left side**: PDF management area where you can upload documents and see what's loaded
- **Right side**: Chat area where you can ask questions about your PDFs

### Example Questions

Ask questions like:
- "What is this PDF about?"
- "Summarize this document."
- "What are the key takeaways?"
- "Compare the information in all documents about X topic."
- "Remember that I'm interested in X"
- "Based on what I asked earlier..."

### Special Commands

- "exit" or "quit": Exit the program (console only)
- "clear memory", "forget", or "clear context": Reset conversation memory

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

## 🛠️ Future Enhancements

- ✅ Add Gradio web interface
- ✅ Upload and chat with multiple PDFs at once
- ✅ Drag-and-drop PDF upload
- 🔄 Improved PDF text extraction with table support
- 🔄 Support for more document types (DOCX, TXT, etc.)
- 🔒 Per-user session memory
- 🌐 Export Q&A as JSON/Markdown
- 📊 Visualization of document relationships
- 🔍 Advanced search filters

---

## 🤝 Credits

- [LangChain](https://www.langchain.com/)
- [Ollama](https://www.ollama.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [PyMuPDF](https://pymupdf.readthedocs.io/)

---

## 🔄 Conversation Memory

The chatbot includes built-in conversation memory that:

1. **Remembers previous questions and answers**
2. **Maintains context** throughout your conversation
3. **Handles follow-up questions** without repeating information
4. **Stores user preferences** mentioned during the conversation

This feature is useful for:
- Asking follow-up questions
- Building on previous responses
- Instructing the bot to remember specific details
- Maintaining a coherent conversation flow

To clear the memory at any time, type "clear memory", "forget", or "clear context".

---

Let me know if you want this saved as a `.md` or `.txt` file or bundled with your code!