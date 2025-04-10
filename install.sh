#!/bin/bash

# Create and activate virtual environment
echo "Creating Python virtual environment..."
python -m venv venv

# Activate environment (this varies by platform)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Unix/MacOS
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -e .

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Please install Ollama from https://ollama.com/download"
    echo "After installing, run: ollama pull llama3"
else
    echo "Ollama found. Pulling llama3 model..."
    ollama pull llama3
fi

# Create directories
mkdir -p pdfs
mkdir -p cache

echo ""
echo "Installation complete!"
echo ""
echo "To use the console interface:"
echo "  python chat_pdf.py"
echo ""
echo "To use the web interface:"
echo "  python chat_pdf_web.py --web"
echo ""
echo "Place your PDF files in the 'pdfs' directory or upload them via the web interface." 