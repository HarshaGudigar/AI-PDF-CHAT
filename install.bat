@echo off
echo Creating Python virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -e .

echo Checking for Ollama...
where ollama >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Ollama not found. Please install Ollama from https://ollama.com/download
    echo After installing, run: ollama pull llama3
) else (
    echo Ollama found. Pulling llama3 model...
    ollama pull llama3
)

echo Creating directories...
if not exist pdfs mkdir pdfs
if not exist cache mkdir cache

echo.
echo Installation complete!
echo.
echo To use the console interface:
echo   python chat_pdf.py
echo.
echo To use the web interface:
echo   python chat_pdf_web.py --web
echo.
echo Place your PDF files in the 'pdfs' directory or upload them via the web interface.
echo.
pause 