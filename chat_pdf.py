import os, time, sys, threading, pickle, fitz  # PyMuPDF

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


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

def setup_rag_chain(vector_stores):
    retriever = combine_retrievers(vector_stores)

    prompt = PromptTemplate.from_template(
        "You are a helpful assistant answering questions about multiple PDF files.\n"
        "If the question is about metadata such as number of pages or file size, "
        "answer with the information for all relevant documents.\n"
        "For all other questions, answer concisely based on the document content.\n"
        "Always include the document source name (filename) for any information you provide.\n\n"
        "Document Context:\n{context}\n\n"
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

    return (
        {
            "context": lambda x: format_docs(retriever(x)), 
            "question": RunnablePassthrough(),
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

def main():
    pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    if not pdfs:
        print("⚠️  No PDFs found in 'pdfs/' folder.")
        return

    print("📚 Available PDFs:")
    for i, name in enumerate(pdfs, 1):
        print(f"{i}: {name}")
    
    print("\nOptions:")
    print("1: Select single PDF")
    print("2: Search across all PDFs")
    
    mode = int(input("Choose an option: "))
    
    if mode == 1:
        # Single PDF mode - original functionality
        choice = int(input("Select a PDF to chat with: ")) - 1
        pdf = pdfs[choice]
        pdf_path = os.path.join(PDF_DIR, pdf)
        txt_cache = os.path.join(CACHE_DIR, pdf + ".txt")
        faiss_cache = os.path.join(CACHE_DIR, pdf.replace(".pdf", "_faiss"))

        # Show metadata
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
        
        # Original single-PDF functionality
        rag_chain = setup_rag_chain([{"vector_store": vector_store, "metadata": metadata}])
        
        print(f"✅ Ready to chat with {pdf}! Type your questions below ('exit' to quit).\n")

    elif mode == 2:
        # Multi-PDF mode - new functionality
        print("Loading all PDFs...")
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
            
            print(f"⏳ Processing {pdf}...")
            text = extract_text(pdf_path, txt_cache)
            vector_store = build_vector_store(text, faiss_cache, metadata)
            
            vector_stores.append({
                "vector_store": vector_store,
                "metadata": metadata
            })
        
        # Add a special query to check metadata
        print("\n📊 Document Metadata Summary:")
        for store in vector_stores:
            meta = store["metadata"]
            print(f" - {meta['title']}: {meta['num_pages']} pages, {meta['file_size_mb']} size")
            
        rag_chain = setup_rag_chain(vector_stores)
        
        print(f"\n✅ Ready to search across {len(pdfs)} PDFs! Type your questions below ('exit' to quit).\n")
    
    else:
        print("Invalid option. Exiting...")
        return

    while True:
        question = input("🧑 You: ")
        if question.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        stop_event = threading.Event()
        thread = threading.Thread(target=loading_spinner, args=(stop_event,))
        thread.start()

        try:
            response = rag_chain.invoke(question)
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


if __name__ == "__main__":
    main()
