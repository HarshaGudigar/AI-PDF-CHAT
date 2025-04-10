import os, time, sys, threading, pickle, fitz  # PyMuPDF

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough


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
        return FAISS.load_local(faiss_cache, embeddings, allow_dangerous_deserialization=True)

    print("🧠 Embedding text...")
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)
    metadata_chunk = "\n".join(f"{k}: {v}" for k, v in metadata.items())
    chunks.insert(0, f"[METADATA]\n{metadata_chunk}")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local(faiss_cache)
    return vector_store


def setup_rag_chain(vector_store, metadata):
    retriever = vector_store.as_retriever()

    prompt = PromptTemplate.from_template(
        "You are a helpful assistant answering questions about a PDF file.\n"
        "If the question is about metadata such as number of pages or file size, "
        "answer ONLY with the raw value (e.g., '0.09 MB' or '4 pages'). Do NOT include any explanations.\n"
        "For all other questions, answer concisely based on the document content.\n\n"
        "Metadata:\n{metadata}\n\n"
        "Document Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    llm = OllamaLLM(model="llama3", temperature=0, stream=True)

    return (
        {
            "context": retriever, 
            "question": RunnablePassthrough(),
            "metadata": lambda x: "\n".join(f"{k}: {v}" for k, v in metadata.items()),
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
    choice = int(input("Select a PDF to chat with: ")) - 1
    pdf = pdfs[choice]
    pdf_path = os.path.join(PDF_DIR, pdf)
    txt_cache = os.path.join(CACHE_DIR, pdf + ".txt")
    
    faiss_cache = os.path.join(CACHE_DIR, pdf.replace(".pdf", "_faiss"))

     # 🆕 Show metadata
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    metadata = {
     "num_pages": num_pages,
     "file_size_mb": f"{file_size_mb:.2f} MB"
    }

    text = extract_text(pdf_path, txt_cache)
    vector_store = build_vector_store(text, faiss_cache, metadata)
    rag_chain = setup_rag_chain(vector_store, metadata)

    print("✅ Ready! Type your questions below ('exit' to quit).\n")

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
