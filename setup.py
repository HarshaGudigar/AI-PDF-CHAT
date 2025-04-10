from setuptools import setup, find_packages

setup(
    name="pdf-chat",
    version="0.1.0",
    description="Chat with multiple PDF documents using LLMs with both console and web interface",
    author="PDF Chat Team",
    packages=find_packages(),
    install_requires=[
        "pymupdf",
        "langchain",
        "langchain-ollama",
        "langchain-community",
        "langchain-text-splitters",
        "langchain-core",
        "faiss-cpu",
        "gradio",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "pdf-chat=chat_pdf:main",
            "pdf-chat-web=chat_pdf_web:main",
        ],
    },
) 