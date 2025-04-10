import os, time, sys, threading, pickle, fitz, signal  # PyMuPDF

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
    embeddings = OllamaEmbeddings(model="llama3")

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
            docs = retriever.get_relevant_documents(query)
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
    return ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"
    )

def setup_rag_chain(vector_stores, memory):
    retriever = combine_retrievers(vector_stores)

    # Improved prompt template that handles personal information and PDF content
    prompt = PromptTemplate.from_template(
        "You are a helpful assistant answering questions about PDF files and maintaining a conversation with the user.\n"
        "If the question is about metadata such as number of pages or file size, "
        "answer with the information for all relevant documents.\n"
        "For questions about the documents, answer concisely based on the document content.\n"
        "For personal questions or information the user has shared with you previously, use the chat history to respond appropriately.\n"
        "Always include the document source name (filename) for any PDF information you provide.\n\n"
        "Document Context:\n{context}\n\n"
        + "Chat History:\n{chat_history}\n\n" +
        "Question: {question}\n\n"
        "Answer:"
    )

    llm = OllamaLLM(model="llama3", temperature=0, stream=True)

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

def loading_spinner(stop_event):
    spin = ['|', '/', '-', '\\']
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r⏳ Thinking... {spin[idx % 4]}")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)

def handle_sigint(sig, frame):
    """Handle SIGINT (Ctrl+C) gracefully."""
    print("\n\n👋 Exiting gracefully. Goodbye!")
    sys.exit(0)

def main():
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, handle_sigint)
    
    # Check for PDFs
    pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    if not pdfs:
        print("⚠️  No PDFs found in 'pdfs/' folder.")
        return

    # Always use conversation memory
    memory = setup_conversation_memory()
    
    # Always process all PDFs
    print("📚 Found PDFs:")
    for name in pdfs:
        print(f" - {name}")
    
    print("\n⏳ Loading all documents and setting up chat...")
    vector_stores = []
    
    for pdf in pdfs:
        pdf_path = os.path.join(PDF_DIR, pdf)
        txt_cache = os.path.join(CACHE_DIR, pdf + ".txt")
        faiss_cache = os.path.join(CACHE_DIR, pdf.replace(".pdf", "_faiss"))
        
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
    
    # Add a special query to check metadata
    print("\n📊 Document Summary:")
    for store in vector_stores:
        meta = store["metadata"]
        print(f" - {meta['title']}: {meta['num_pages']} pages, {meta['file_size_mb']} size")
        
    rag_chain = setup_rag_chain(vector_stores, memory)
    
    print("\n✅ Ready! Chat with your documents. Type questions or 'exit' to quit.")
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
            if question.lower() in ["clear memory", "forget", "clear context"]:
                memory.clear()
                print("🧹 Memory cleared! Previous context has been forgotten.")
                continue

            stop_event = threading.Event()
            thread = threading.Thread(target=loading_spinner, args=(stop_event,))
            thread.start()

            try:
                # Pre-process to extract any special directives for memory
                remember_info = ""
                if question.lower().startswith(("remember ", "note that ")):
                    remember_info = "I'll remember that. "
                
                # For memory-enabled chains, we pass a dict with the question
                response = rag_chain.invoke({"question": question})
                
                # Save the interaction to memory
                memory.save_context(
                    {"question": question}, 
                    {"answer": response.content if hasattr(response, "content") else str(response)}
                )
                
                # Add the remember prefix to the response if needed
                if remember_info:
                    if hasattr(response, "content"):
                        response.content = remember_info + response.content
                    else:
                        response = remember_info + str(response)
            finally:
                stop_event.set()
                thread.join()

            print("\r🤖 Bot:", flush=True)

            # Check if response is a string or has `.content`
            answer = response.content if hasattr(response, "content") else str(response)

            for word in answer.split():
                print(word, end=" ", flush=True)
                time.sleep(0.03)
            print()
        except KeyboardInterrupt:
            # This is a backup in case the signal handler doesn't catch it
            print("\n\n👋 Exiting gracefully. Goodbye!")
            break


if __name__ == "__main__":
    main()
