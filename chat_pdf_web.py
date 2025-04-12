import os, time, sys, threading, pickle, fitz, signal, gradio as gr
from pathlib import Path

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


PDF_DIR = "pdfs"
CACHE_DIR = "cache"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Global variables to store state
vector_stores = []
memory = None
rag_chain = None
chat_history = []  # This will remain a list of tuples (user_msg, bot_msg)
ENABLE_MEMORY = False  # Toggle for memory feature

# Enable more verbose logging
def log_info(message):
    print(f"[INFO] {message}")

def log_error(message):
    print(f"[ERROR] {message}")

def log_debug(message):
    print(f"[DEBUG] {message}")

def extract_text(pdf_path, cache_path):
    if os.path.exists(cache_path):
        print(f"📄 Using cached text for: {os.path.basename(pdf_path)}")
        return open(cache_path, "r", encoding="utf-8").read()
    print(f"📄 Extracting text from {os.path.basename(pdf_path)}...")
    text = "\n".join([page.get_text() for page in fitz.open(pdf_path)])
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text

def build_vector_store(text, faiss_cache, metadata):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if os.path.exists(faiss_cache):
        print("🔁 Loading cached FAISS index...")
        # When loading from cache, we need to reconstruct the metadata
        vector_store = FAISS.load_local(faiss_cache, embeddings, allow_dangerous_deserialization=True)
        print(f"📊 Metadata for {metadata.get('title', 'Unknown')}: {metadata}")
        return vector_store

    print("🧠 Embedding text...")
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)
    
    # Add document title to each chunk's metadata
    docs_with_metadata = []
    for chunk in chunks:
        docs_with_metadata.append((chunk, {"source": metadata.get("title", "Unknown")}, ))
    
    # Add metadata as a special chunk
    metadata_chunk = "\n".join(f"{k}: {v}" for k, v in metadata.items())
    docs_with_metadata.insert(0, (f"[METADATA]\n{metadata_chunk}", {"source": metadata.get("title", "Unknown"), "is_metadata": True}, ))
    
    print(f"📊 Added metadata for {metadata.get('title', 'Unknown')}: {metadata}")
    
    vector_store = FAISS.from_texts([doc[0] for doc in docs_with_metadata], embeddings, metadatas=[doc[1] for doc in docs_with_metadata])
    vector_store.save_local(faiss_cache)
    return vector_store

def combine_retrievers(vector_stores):
    """Combine results from multiple retrievers."""
    llm = OllamaLLM(model="llama3", temperature=0)
    
    retrievers = []
    for vs in vector_stores:
        retrievers.append(vs["vector_store"].as_retriever(search_kwargs={"k": 3}))
    
    def retrieve_from_all(query):
        all_docs = []
        for i, retriever in enumerate(retrievers):
            # Use the newer 'invoke' method instead of the deprecated 'get_relevant_documents'
            docs = retriever.invoke(query)
            for doc in docs:
                if not hasattr(doc, "metadata") or doc.metadata is None:
                    doc.metadata = {}
                # Ensure source is set, using the vector store's metadata
                doc.metadata["source"] = vector_stores[i]["metadata"]["title"]
                
                # Check if this is a metadata document
                if "[METADATA]" in doc.page_content:
                    doc.metadata["is_metadata"] = True
                    
            all_docs.extend(docs)
            
            # Add an empty document with just the metadata if none was found
            found_metadata = any(getattr(doc.metadata, "is_metadata", False) for doc in docs)
            if not found_metadata:
                metadata_string = "\n".join(f"{k}: {v}" for k, v in vector_stores[i]["metadata"].items())
                metadata_doc = Document(
                    page_content=f"[METADATA]\n{metadata_string}",
                    metadata={"source": vector_stores[i]["metadata"]["title"], "is_metadata": True}
                )
                all_docs.append(metadata_doc)
        
        # Sort by relevance (if available) or any other criterion
        return all_docs
    
    return retrieve_from_all

def setup_conversation_memory():
    """Initialize conversation memory to store chat history."""
    if not ENABLE_MEMORY:
        return None
    return ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"
    )

def setup_rag_chain(vector_stores, memory):
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

    llm = OllamaLLM(model="llama3.2", temperature=0, stream=True)

    def format_docs(docs):
        # Handle metadata documents specially
        metadata_sections = []
        content_sections = []
        
        for doc in docs:
            if doc.page_content.startswith("[METADATA]"):
                # Parse and format metadata
                metadata_txt = doc.page_content.replace("[METADATA]\n", "")
                metadata_sections.append(f"Document: {doc.metadata.get('source', 'Unknown')}\nMetadata: {metadata_txt}")
            else:
                # Regular content
                content_sections.append(f"Document: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}")
        
        # First show metadata sections, then content sections
        all_sections = metadata_sections + content_sections
        return "\n\n".join(all_sections)
    
    # Format chat history for the prompt
    def get_chat_history(inputs):
        if not ENABLE_MEMORY or not memory:
            return ""
        memory_data = memory.load_memory_variables({})
        history = memory_data.get("chat_history", [])
        formatted_history = []
        for message in history:
            if isinstance(message, HumanMessage):
                formatted_history.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_history.append(f"AI: {message.content}")
        return "\n".join(formatted_history)

    return (
        {
            "context": lambda x: format_docs(retriever(x["question"])), 
            "question": lambda x: x["question"],
            "chat_history": get_chat_history,
         }
        | prompt
        | llm
    )

def handle_sigint(sig, frame):
    """Handle SIGINT (Ctrl+C) gracefully."""
    print("\n\n👋 Exiting gracefully. Goodbye!")
    sys.exit(0)

def load_documents():
    """Load all PDF documents and prepare the RAG chain."""
    global vector_stores, memory, rag_chain
    
    try:
        # Check for PDFs in the main PDF directory
        pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf") and os.path.isfile(os.path.join(PDF_DIR, f))]
        
        if not pdfs:
            log_info("⚠️  No PDFs found in 'pdfs/' folder.")
            return "No PDFs found. Please upload some PDFs to get started."
        
        # Always use conversation memory
        memory = setup_conversation_memory()
        
        # Always process all PDFs
        log_info("📚 Found PDFs:")
        for name in pdfs:
            log_info(f" - {name}")
        
        log_info("\n⏳ Loading all documents and setting up chat...")
        vector_stores = []
        
        for pdf in pdfs:
            pdf_path = os.path.join(PDF_DIR, pdf)
            txt_cache = os.path.join(CACHE_DIR, pdf.replace("/", "_").replace("\\", "_") + ".txt")
            faiss_cache = os.path.join(CACHE_DIR, pdf.replace("/", "_").replace("\\", "_").replace(".pdf", "_faiss"))
            
            try:
                doc = fitz.open(pdf_path)
                num_pages = len(doc)
                file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
                metadata = {
                    "title": pdf,
                    "num_pages": num_pages,
                    "file_size_mb": f"{file_size_mb:.2f} MB"
                }
                
                text = extract_text(pdf_path, txt_cache)
                vector_store = build_vector_store(text, faiss_cache, metadata)
                
                vector_stores.append({
                    "vector_store": vector_store,
                    "metadata": metadata
                })
            except Exception as doc_err:
                log_error(f"Error processing document {pdf}: {str(doc_err)}")
        
        # Add a special query to check metadata
        summary = "\n📊 Document Summary:\n"
        for store in vector_stores:
            meta = store["metadata"]
            summary += f" - {meta['title']}: {meta['num_pages']} pages, {meta['file_size_mb']} size\n"
            
        rag_chain = setup_rag_chain(vector_stores, memory)
        
        return summary + "\n✅ Ready! Chat with your documents."
    
    except Exception as e:
        log_error(f"Error loading documents: {str(e)}")
        if "Failed to connect to Ollama" in str(e):
            return f"⚠️ Error: Could not connect to Ollama. Please make sure Ollama is running.\n\nDetails: {str(e)}"
        return f"⚠️ Error loading documents: {str(e)}"

def process_question(question):
    """Process user questions for both CLI and web interfaces."""
    global memory, rag_chain, chat_history
    
    try:
        if not vector_stores:
            return "Please load documents first."
            
        if ENABLE_MEMORY and question.lower() in ["clear memory", "forget", "clear context"]:
            memory.clear()
            chat_history = []
            return "🧹 Memory cleared! Previous context has been forgotten."
        
        # Pre-process to extract any special directives
        remember_info = ""
        if ENABLE_MEMORY and question.lower().startswith(("remember ", "note that ")):
            remember_info = "I'll remember that. "
        
        # For memory-enabled chains, we pass a dict with the question
        response = rag_chain.invoke({"question": question})
        
        # Save the interaction to memory if enabled
        if ENABLE_MEMORY and memory:
            memory.save_context(
                {"question": question}, 
                {"answer": response.content if hasattr(response, "content") else str(response)}
            )
        
        # Add the remember prefix to the response if needed
        answer = response.content if hasattr(response, "content") else str(response)
        if remember_info:
            answer = remember_info + answer
        
        return answer
    
    except Exception as e:
        error_msg = f"⚠️ Error processing question: {str(e)}"
        print(error_msg)
        return error_msg

def console_chat():
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
            
            # Define spinner function
            def spinner_func():
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
            # This is a backup in case the signal handler doesn't catch it
            print("\n\n👋 Exiting gracefully. Goodbye!")
            break

def simple_upload_handler(files):
    """Safely handle uploaded files from both UploadButton and File components."""
    try:
        if files is None:
            return "No files uploaded", "\n".join([store["metadata"]["title"] for store in vector_stores]) if vector_stores else ""
        
        # Convert to list if it's not already
        if not isinstance(files, list):
            files = [files]
            
        log_info(f"Received {len(files)} files of type: {type(files[0]) if files else 'None'}")
        
        results = []
        for file_obj in files:
            if file_obj is None:
                continue
                
            try:
                # Get the file name - different file objects might provide it differently
                if hasattr(file_obj, "name"):
                    file_name = os.path.basename(file_obj.name)
                elif hasattr(file_obj, "orig_name"):
                    file_name = file_obj.orig_name
                else:
                    # Generate a unique name if we can't determine it
                    file_name = f"uploaded_{int(time.time())}.pdf"
                
                # Create output path - saving directly to PDF_DIR, not UPLOAD_DIR
                output_path = os.path.join(PDF_DIR, file_name)
                abs_output_path = os.path.abspath(output_path)
                log_info(f"Saving to: {abs_output_path}")
                
                # Handle file content based on the type of object we received
                success = False
                
                # For binary data handling
                if isinstance(file_obj, bytes) or isinstance(file_obj, bytearray):
                    with open(output_path, "wb") as f:
                        f.write(file_obj)
                    log_info(f"Wrote bytes content for {file_name} ({len(file_obj)} bytes)")
                    success = True
                    
                # For file-like objects with read method
                elif hasattr(file_obj, "read"):
                    try:
                        content = file_obj.read()
                        with open(output_path, "wb") as f:
                            f.write(content)
                        
                        # Reset file pointer if possible
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
                        # Try to close the file if it has a close method
                        if hasattr(file_obj, "close"):
                            try:
                                file_obj.close()
                            except:
                                pass
                
                # For string paths
                elif isinstance(file_obj, str):
                    # Path to a temporary file
                    try:
                        import shutil
                        shutil.copy2(file_obj, output_path)
                        log_info(f"Copied from {file_obj} to {output_path}")
                        success = True
                    except Exception as copy_err:
                        log_error(f"Error copying file: {str(copy_err)}")
                
                # For objects with path attribute
                elif hasattr(file_obj, "path") and file_obj.path:
                    try:
                        import shutil
                        shutil.copy2(file_obj.path, output_path)
                        log_info(f"Copied from {file_obj.path} to {output_path}")
                        success = True
                    except Exception as path_err:
                        log_error(f"Error copying from path: {str(path_err)}")
                
                # Try raw data as last resort
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
                    results.append(f"Uploaded: {file_name} ({file_size} bytes)")
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
            return "No valid files uploaded", "\n".join([store["metadata"]["title"] for store in vector_stores]) if vector_stores else ""
        
        # List all files in PDF_DIR after upload
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
        log_info(f"Files in {PDF_DIR} after upload: {pdf_files}")
        
        # Force flush any file writes to disk
        time.sleep(0.5)  # Short delay to ensure file operations complete
        
        # Reload documents after successful upload
        try:
            load_result = load_documents()
            doc_titles = "\n".join([store["metadata"]["title"] for store in vector_stores]) if vector_stores else "No documents loaded"
            log_info(f"Documents loaded after upload: {doc_titles}")
            return "\n".join(results), doc_titles
        except Exception as e:
            error_msg = f"Error reloading documents: {str(e)}"
            log_error(error_msg)
            return "\n".join(results + [error_msg]), "Error loading documents"
            
    except Exception as e:
        error_msg = f"Error in upload handler: {str(e)}"
        log_error(error_msg)
        return error_msg, "\n".join([store["metadata"]["title"] for store in vector_stores]) if vector_stores else ""

def web_ui():
    """Run the Gradio web interface."""
    global chat_history
    
    # Initial load of documents 
    load_result = load_documents()
    
    # List all files in PDF_DIR at startup
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    log_info(f"Files in {PDF_DIR} at startup: {pdf_files}")
    
    # Use a basic theme that works across Gradio versions
    with gr.Blocks(title="AI Doc Assist") as demo:
        gr.Markdown("# 🤖 AI Doc Assist")
        gr.Markdown("Chat with your PDF documents using AI. Upload PDFs and ask questions!")

        if "Error" in load_result:
            gr.Markdown(f"### ⚠️ Warning\n{load_result}")
        
        with gr.Row():
            with gr.Column(scale=1):
                # PDF management - on the left
                gr.Markdown("### 📄 Document Management")
                
                # Simple file upload component
                file_component = gr.File(
                    label="Drag & Drop PDFs here (or click to upload)",
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="binary"  # Ensure binary mode for PDF files
                )
                
                upload_output = gr.Textbox(label="Upload Results", lines=2)
                reload_btn = gr.Button("Reload PDFs")
                
                doc_list = gr.Textbox(
                    label="Available Documents", 
                    value="\n".join([store["metadata"]["title"] for store in vector_stores]) if vector_stores else "No documents loaded", 
                    lines=10
                )
            
            with gr.Column(scale=2):
                # Chat interface - on the right
                gr.Markdown("### 💬 Chat with Documents")
                
                # Use standard chat interface without specifying type
                chatbot = gr.Chatbot(height=450)
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask a question about your PDFs...",
                        container=False,
                        scale=7
                    )
                    submit = gr.Button("Send")
                
                with gr.Row():
                    clear = gr.Button("Clear Chat History")
        
        # Connect file upload component - using upload event rather than change
        file_component.upload(
            simple_upload_handler,
            inputs=[file_component],
            outputs=[upload_output, doc_list]
        )
        
        # Handle reload button
        def reload_docs():
            log_info("Manual reload requested")
            load_documents()
            pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
            log_info(f"Files in {PDF_DIR} after reload: {pdf_files}")
            return "\n".join([store["metadata"]["title"] for store in vector_stores]) if vector_stores else "No documents loaded"
            
        reload_btn.click(reload_docs, outputs=doc_list)

        # Add automatic refresh on page load
        demo.load(reload_docs, outputs=doc_list)
        
        # Handle chat interactions - simplified for compatibility
        def respond(message, chat_history):
            if not message.strip():
                return "", chat_history
                
            # Process the question and get the answer
            answer = process_question(message)
            
            # Format the chat history as tuples (user, bot)
            chat_history.append((message, answer))
            
            return "", chat_history
            
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        submit.click(respond, [msg, chatbot], [msg, chatbot])
        
        # Handle clear button
        def clear_chat():
            global chat_history
            memory.clear()
            chat_history = []
            return None
            
        clear.click(clear_chat, outputs=chatbot)
        
    # Configure Gradio for stateful operation
    demo.queue()
    return demo

def main():
    # Ensure the PDF and cache directories exist with proper permissions
    try:
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
        # Run the web interface
        app = web_ui()
        app.launch(share=True)
    else:
        # Run the console interface
        console_chat()

if __name__ == "__main__":
    main() 