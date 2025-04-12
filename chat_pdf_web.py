# Standard library imports
import os
import time
import sys
import threading
import pickle
import signal
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

# Third-party imports
import fitz
import gradio as gr
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage

# Constants
PDF_DIR = "pdfs"
CACHE_DIR = "cache"
DEFAULT_MODEL = "llama3"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
ENABLE_MEMORY = False

@dataclass
class VectorStore:
    vector_store: Any
    metadata: Dict[str, Any]

# Global state
vector_stores: List[VectorStore] = []
memory: Optional[ConversationBufferMemory] = None
rag_chain: Optional[Any] = None
chat_history: List[tuple] = []

# Ensure directories exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Logging functions
def log_info(message: str) -> None:
    """Log an informational message."""
    print(f"[INFO] {message}")

def log_error(message: str) -> None:
    """Log an error message."""
    print(f"[ERROR] {message}")

def log_debug(message: str) -> None:
    """Log a debug message."""
    print(f"[DEBUG] {message}")

def extract_text(pdf_path: str, cache_path: str) -> str:
    """Extract text from PDF and cache it for future use."""
    try:
        if os.path.exists(cache_path):
            log_info(f"Using cached text for: {os.path.basename(pdf_path)}")
            return open(cache_path, "r", encoding="utf-8").read()
        
        log_info(f"Extracting text from {os.path.basename(pdf_path)}...")
        text = "\n".join([page.get_text() for page in fitz.open(pdf_path)])
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
        return text
    except Exception as e:
        log_error(f"Error extracting text from {pdf_path}: {str(e)}")
        raise

def build_vector_store(text: str, faiss_cache: str, metadata: Dict[str, Any]) -> Any:
    """Build or load a FAISS vector store for the given text."""
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        if os.path.exists(faiss_cache):
            log_info("Loading cached FAISS index...")
            vector_store = FAISS.load_local(faiss_cache, embeddings, allow_dangerous_deserialization=True)
            log_info(f"Metadata for {metadata.get('title', 'Unknown')}: {metadata}")
            return vector_store

        log_info("Embedding text...")
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        ).split_text(text)
        
        # Add document title to each chunk's metadata
        docs_with_metadata = []
        for chunk in chunks:
            docs_with_metadata.append((chunk, {"source": metadata.get("title", "Unknown")}))
        
        # Add metadata as a special chunk
        metadata_chunk = "\n".join(f"{k}: {v}" for k, v in metadata.items())
        docs_with_metadata.insert(0, (
            f"[METADATA]\n{metadata_chunk}",
            {"source": metadata.get("title", "Unknown"), "is_metadata": True}
        ))
        
        log_info(f"Added metadata for {metadata.get('title', 'Unknown')}: {metadata}")
        
        vector_store = FAISS.from_texts(
            [doc[0] for doc in docs_with_metadata],
            embeddings,
            metadatas=[doc[1] for doc in docs_with_metadata]
        )
        vector_store.save_local(faiss_cache)
        return vector_store
    except Exception as e:
        log_error(f"Error building vector store: {str(e)}")
        raise

def combine_retrievers(vector_stores: List[VectorStore]) -> callable:
    """Combine results from multiple retrievers."""
    try:
        llm = OllamaLLM(model=DEFAULT_MODEL, temperature=0)
        
        retrievers = []
        for vs in vector_stores:
            retrievers.append(vs.vector_store.as_retriever(search_kwargs={"k": 3}))
        
        def retrieve_from_all(query: str) -> List[Document]:
            all_docs = []
            for i, retriever in enumerate(retrievers):
                docs = retriever.invoke(query)
                for doc in docs:
                    if not hasattr(doc, "metadata") or doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata["source"] = vector_stores[i].metadata["title"]
                    
                    if "[METADATA]" in doc.page_content:
                        doc.metadata["is_metadata"] = True
                        
                all_docs.extend(docs)
                
                # Add metadata document if none was found
                found_metadata = any(getattr(doc.metadata, "is_metadata", False) for doc in docs)
                if not found_metadata:
                    metadata_string = "\n".join(f"{k}: {v}" for k, v in vector_stores[i].metadata.items())
                    metadata_doc = Document(
                        page_content=f"[METADATA]\n{metadata_string}",
                        metadata={"source": vector_stores[i].metadata["title"], "is_metadata": True}
                    )
                    all_docs.append(metadata_doc)
            
            return all_docs
        
        return retrieve_from_all
    except Exception as e:
        log_error(f"Error combining retrievers: {str(e)}")
        raise

def setup_conversation_memory() -> Optional[ConversationBufferMemory]:
    """Initialize conversation memory to store chat history."""
    if not ENABLE_MEMORY:
        return None
    try:
        return ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"
        )
    except Exception as e:
        log_error(f"Error setting up conversation memory: {str(e)}")
        return None

def setup_rag_chain(vector_stores: List[VectorStore], memory: Optional[ConversationBufferMemory]) -> Any:
    """Set up the RAG chain for document processing."""
    try:
        retriever = combine_retrievers(vector_stores)

        # Improved prompt template that handles personal information and PDF content
        prompt_template = (
            "You are a helpful assistant answering questions about PDF files"
            + (" and maintaining a conversation with the user.\n" if ENABLE_MEMORY else ".\n")
            + "If the question is about metadata such as number of pages or file size, "
            "answer with the information for all relevant documents.\n"
            "For questions about the documents, answer concisely based on the document content.\n"
            + ("For personal questions or information the user has shared with you previously, use the chat history to respond appropriately.\n" if ENABLE_MEMORY else "")
            + "Always include the document source name (filename) for any PDF information you provide.\n\n"
            "Document Context:\n{context}\n\n"
            + ("Chat History:\n{chat_history}\n\n" if ENABLE_MEMORY else "")
            + "Question: {question}\n\n"
            "Answer:"
        )

        prompt = PromptTemplate.from_template(prompt_template)
        llm = OllamaLLM(model=DEFAULT_MODEL, temperature=0, stream=True)

        def format_docs(docs: List[Document]) -> str:
            """Format documents for display, handling metadata specially."""
            metadata_sections = []
            content_sections = []
            
            for doc in docs:
                if doc.page_content.startswith("[METADATA]"):
                    metadata_txt = doc.page_content.replace("[METADATA]\n", "")
                    metadata_sections.append(f"Document: {doc.metadata.get('source', 'Unknown')}\nMetadata: {metadata_txt}")
                else:
                    content_sections.append(f"Document: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}")
            
            all_sections = metadata_sections + content_sections
            return "\n\n".join(all_sections)
        
        def get_chat_history(inputs: Dict[str, Any]) -> str:
            """Format chat history for the prompt."""
            if not ENABLE_MEMORY or not memory:
                return ""
            try:
                memory_data = memory.load_memory_variables({})
                history = memory_data.get("chat_history", [])
                formatted_history = []
                for message in history:
                    if isinstance(message, HumanMessage):
                        formatted_history.append(f"Human: {message.content}")
                    elif isinstance(message, AIMessage):
                        formatted_history.append(f"AI: {message.content}")
                return "\n".join(formatted_history)
            except Exception as e:
                log_error(f"Error getting chat history: {str(e)}")
                return ""

        return (
            {
                "context": lambda x: format_docs(retriever(x["question"])), 
                "question": lambda x: x["question"],
                "chat_history": get_chat_history,
            }
            | prompt
            | llm
        )
    except Exception as e:
        log_error(f"Error setting up RAG chain: {str(e)}")
        raise

def handle_sigint(sig: int, frame: Any) -> None:
    """Handle SIGINT (Ctrl+C) gracefully."""
    print("\n\n👋 Exiting gracefully. Goodbye!")
    sys.exit(0)

def load_documents() -> str:
    """Load all PDF documents and prepare the RAG chain."""
    global vector_stores, memory, rag_chain
    
    try:
        # Check for PDFs in the main PDF directory
        pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf") and os.path.isfile(os.path.join(PDF_DIR, f))]
        
        if not pdfs:
            log_info("⚠️  No PDFs found in 'pdfs/' folder.")
            return "No PDFs found. Please upload some PDFs to get started."
        
        # Initialize conversation memory
        memory = setup_conversation_memory()
        
        # Process all PDFs
        log_info("📚 Found PDFs:")
        for name in pdfs:
            log_info(f" - {name}")
        
        log_info("\n⏳ Loading all documents and setting up chat...")
        vector_stores = []
        
        for pdf in pdfs:
            try:
                pdf_path = os.path.join(PDF_DIR, pdf)
                txt_cache = os.path.join(CACHE_DIR, pdf.replace("/", "_").replace("\\", "_") + ".txt")
                faiss_cache = os.path.join(CACHE_DIR, pdf.replace("/", "_").replace("\\", "_").replace(".pdf", "_faiss"))
                
                # Get document metadata
                doc = fitz.open(pdf_path)
                num_pages = len(doc)
                file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
                metadata = {
                    "title": pdf,
                    "num_pages": num_pages,
                    "file_size_mb": f"{file_size_mb:.2f} MB"
                }
                
                # Process document
                text = extract_text(pdf_path, txt_cache)
                vector_store = build_vector_store(text, faiss_cache, metadata)
                
                vector_stores.append(VectorStore(
                    vector_store=vector_store,
                    metadata=metadata
                ))
            except Exception as doc_err:
                log_error(f"Error processing document {pdf}: {str(doc_err)}")
        
        # Create document summary
        summary = "\n📊 Document Summary:\n"
        for store in vector_stores:
            meta = store.metadata
            summary += f" - {meta['title']}: {meta['num_pages']} pages, {meta['file_size_mb']} size\n"
            
        # Set up RAG chain
        rag_chain = setup_rag_chain(vector_stores, memory)
        
        return summary + "\n✅ Ready! Chat with your documents."
    
    except Exception as e:
        log_error(f"Error loading documents: {str(e)}")
        if "Failed to connect to Ollama" in str(e):
            return f"⚠️ Error: Could not connect to Ollama. Please make sure Ollama is running.\n\nDetails: {str(e)}"
        return f"⚠️ Error loading documents: {str(e)}"

def process_question(question: str) -> str:
    """Process user questions for both CLI and web interfaces."""
    global memory, rag_chain, chat_history
    
    try:
        if not vector_stores:
            return "Please load documents first."
            
        if ENABLE_MEMORY and question.lower() in ["clear memory", "forget", "clear context"]:
            if memory:
                memory.clear()
            chat_history = []
            return "🧹 Memory cleared! Previous context has been forgotten."
        
        # Pre-process to extract any special directives
        remember_info = ""
        if ENABLE_MEMORY and question.lower().startswith(("remember ", "note that ")):
            remember_info = "I'll remember that. "
        
        # Process question through RAG chain
        response = rag_chain.invoke({"question": question})
        
        # Save the interaction to memory if enabled
        if ENABLE_MEMORY and memory:
            memory.save_context(
                {"question": question}, 
                {"answer": response.content if hasattr(response, "content") else str(response)}
            )
        
        # Format response
        answer = response.content if hasattr(response, "content") else str(response)
        if remember_info:
            answer = remember_info + answer
        
        return answer
    
    except Exception as e:
        error_msg = f"⚠️ Error processing question: {str(e)}"
        log_error(error_msg)
        return error_msg

def console_chat() -> None:
    """Run the console-based chat interface."""
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, handle_sigint)
    
    summary = load_documents()
    print(summary)
    print("\n📝 Special commands:")
    print(" - 'exit' or 'quit': End the session")
    print(" - 'clear memory' or 'forget': Reset conversation history")
    print("\n")
    
    while True:
        try:
            question = input("🧑 You: ")
            if question.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
                
            stop_event = threading.Event()
            
            def spinner_func() -> None:
                """Display a spinning indicator while processing."""
                spin_chars = ['|', '/', '-', '\\']
                i = 0
                while not stop_event.is_set():
                    sys.stdout.write(f"\r⏳ Thinking... {spin_chars[i % len(spin_chars)]}")
                    sys.stdout.flush()
                    time.sleep(0.1)
                    i += 1
                    
            # Start spinner thread
            thread = threading.Thread(target=spinner_func)
            thread.daemon = True
            thread.start()

            try:
                answer = process_question(question)
            finally:
                stop_event.set()
                thread.join()

            print("\r🤖 Bot:", flush=True)
            
            for word in answer.split():
                print(word, end=" ", flush=True)
                time.sleep(0.03)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Exiting gracefully. Goodbye!")
            break

def simple_upload_handler(files: Union[List[Any], Any, None]) -> str:
    """Safely handle uploaded files from both UploadButton and File components."""
    try:
        if files is None:
            return "\n".join([store.metadata["title"] for store in vector_stores]) if vector_stores else ""
        
        # Convert to list if it's not already
        if not isinstance(files, list):
            files = [files]
            
        log_info(f"Received {len(files)} files of type: {type(files[0]) if files else 'None'}")
        
        results = []
        for file_obj in files:
            if file_obj is None:
                continue
                
            try:
                # Get the file name
                if hasattr(file_obj, "name"):
                    file_name = os.path.basename(file_obj.name)
                elif hasattr(file_obj, "orig_name"):
                    file_name = file_obj.orig_name
                else:
                    file_name = f"uploaded_{int(time.time())}.pdf"
                
                # Create output path
                output_path = os.path.join(PDF_DIR, file_name)
                abs_output_path = os.path.abspath(output_path)
                log_info(f"Saving to: {abs_output_path}")
                
                # Handle file content based on type
                success = False
                
                if isinstance(file_obj, (bytes, bytearray)):
                    with open(output_path, "wb") as f:
                        f.write(file_obj)
                    log_info(f"Wrote bytes content for {file_name} ({len(file_obj)} bytes)")
                    success = True
                    
                elif hasattr(file_obj, "read"):
                    try:
                        content = file_obj.read()
                        with open(output_path, "wb") as f:
                            f.write(content)
                        
                        if hasattr(file_obj, "seek"):
                            try:
                                file_obj.seek(0)
                            except:
                                pass
                            
                        log_info(f"Wrote file content for {file_name} ({len(content)} bytes)")
                        success = True
                    except Exception as read_err:
                        log_error(f"Error reading file: {str(read_err)}")
                    finally:
                        if hasattr(file_obj, "close"):
                            try:
                                file_obj.close()
                            except:
                                pass
                
                elif isinstance(file_obj, str):
                    try:
                        import shutil
                        shutil.copy2(file_obj, output_path)
                        log_info(f"Copied from {file_obj} to {output_path}")
                        success = True
                    except Exception as copy_err:
                        log_error(f"Error copying file: {str(copy_err)}")
                
                elif hasattr(file_obj, "path") and file_obj.path:
                    try:
                        import shutil
                        shutil.copy2(file_obj.path, output_path)
                        log_info(f"Copied from {file_obj.path} to {output_path}")
                        success = True
                    except Exception as path_err:
                        log_error(f"Error copying from path: {str(path_err)}")
                
                elif not success:
                    try:
                        with open(output_path, "wb") as f:
                            f.write(file_obj)
                        log_info(f"Wrote raw content for {file_name}")
                        success = True
                    except Exception as raw_err:
                        log_error(f"Error writing raw content: {str(raw_err)}")
                
                # Verify file exists after save
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    log_info(f"Verified file saved: {output_path} ({file_size} bytes)")
                    results.append(f"{file_name}")
                else:
                    error = f"File not found after saving: {output_path}"
                    log_error(error)
                    results.append(error)
                
            except Exception as file_err:
                error = f"Error processing file {getattr(file_obj, 'name', 'unknown')}: {str(file_err)}"
                log_error(error)
                results.append(error)
        
        if not results:
            log_info("No valid files uploaded")
            return "\n".join([store.metadata["title"] for store in vector_stores]) if vector_stores else ""
        
        # List all files in PDF_DIR after upload
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
        log_info(f"Files in {PDF_DIR} after upload: {pdf_files}")
        
        # Force flush any file writes to disk
        time.sleep(0.5)
        
        # Reload documents after successful upload
        try:
            load_result = load_documents()
            return "\n".join([store.metadata["title"] for store in vector_stores]) if vector_stores else "No documents loaded"
        except Exception as e:
            error_msg = f"Error reloading documents: {str(e)}"
            log_error(error_msg)
            return "Error loading documents"
            
    except Exception as e:
        error_msg = f"Error in upload handler: {str(e)}"
        log_error(error_msg)
        return "\n".join([store.metadata["title"] for store in vector_stores]) if vector_stores else ""

def web_ui() -> gr.Blocks:
    """Run the Gradio web interface."""
    global chat_history
    
    # Initial load of documents 
    load_result = load_documents()
    
    # List all files in PDF_DIR at startup
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    log_info(f"Files in {PDF_DIR} at startup: {pdf_files}")
    
    # Use a basic theme that works across Gradio versions
    with gr.Blocks(title="AI Doc Assist", css="""
        .gradio-container {height: 100vh !important;}
        .chatbot-container {height: 70vh !important;}
        .file-upload {height: 20vh !important;}
        .document-list {height: 45vh !important;}
        .input-box {height: 5vh !important;}
        .send-button {height: 5vh !important; display: flex !important; align-items: center !important;}
    """) as demo:
        gr.Markdown("# 🤖 AI Doc Assist")

        if "Error" in load_result:
            gr.Markdown(f"### ⚠️ Warning\n{load_result}")
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=300):
                # PDF management - on the left
                gr.Markdown("### 📄 Document Management")
                
                # Simple file upload component
                file_component = gr.File(
                    label="Drag & Drop PDFs here (or click to upload)",
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="binary",
                    elem_classes="file-upload"
                )
                
                reload_btn = gr.Button("Reload PDFs")
                
                doc_list = gr.Textbox(
                    label="Available Documents", 
                    value="\n".join([store.metadata["title"] for store in vector_stores]) if vector_stores else "No documents loaded", 
                    lines=10,
                    elem_classes="document-list"
                )
            
            with gr.Column(scale=2, min_width=500):
                # Chat interface - on the right
                gr.Markdown("### 💬 Chat with Documents")
                
                chatbot = gr.Chatbot(
                    elem_classes="chatbot-container"
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask a question about your PDFs...",
                        container=False,
                        scale=7,
                        show_label=False,
                        elem_classes="input-box"
                    )
                    submit = gr.Button("Send", scale=1, elem_classes="send-button")
        
        # Connect file upload component
        file_component.upload(
            simple_upload_handler,
            inputs=[file_component],
            outputs=[doc_list]
        )
        
        # Handle reload button
        def reload_docs() -> str:
            """Reload documents and update the document list."""
            log_info("Manual reload requested")
            load_documents()
            pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
            log_info(f"Files in {PDF_DIR} after reload: {pdf_files}")
            return "\n".join([store.metadata["title"] for store in vector_stores]) if vector_stores else "No documents loaded"
            
        reload_btn.click(reload_docs, outputs=doc_list)

        # Add automatic refresh on page load
        demo.load(reload_docs, outputs=doc_list)
        
        # Handle chat interactions
        def respond(message: str, chat_history: List[tuple]) -> tuple:
            """Process a chat message and update the chat history."""
            if not message.strip():
                return "", chat_history
                
            answer = process_question(message)
            chat_history.append((message, answer))
            return "", chat_history
            
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        submit.click(respond, [msg, chatbot], [msg, chatbot])
        
    # Configure Gradio for stateful operation
    demo.queue()
    return demo

def main() -> None:
    """Main entry point for the application."""
    try:
        # Ensure the PDF and cache directories exist with proper permissions
        log_info(f"Ensuring directories exist: {PDF_DIR} and {CACHE_DIR}")
        os.makedirs(PDF_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Check if directories are writable
        test_file_path = os.path.join(PDF_DIR, "test_write.tmp")
        try:
            with open(test_file_path, 'w') as f:
                f.write("test")
            os.remove(test_file_path)
            log_info(f"{PDF_DIR} is writable")
        except Exception as e:
            log_error(f"WARNING: {PDF_DIR} is not writable: {str(e)}")
            
        # List existing files in PDF directory
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
        log_info(f"Existing PDF files: {pdf_files}")
    except Exception as e:
        log_error(f"Error setting up directories: {str(e)}")
    
    # Choose whether to run console or web interface
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        app = web_ui()
        app.launch(share=True)
    else:
        console_chat()

if __name__ == "__main__":
    main() 