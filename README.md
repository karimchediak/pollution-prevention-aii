# 🌊 Pollution Prevention AI

An educational Retrieval-Augmented Generation (RAG) chatbot that answers
questions about pollution prevention, recycling, waste reduction, air and
water pollution, and environmental protection.

## Features
- RAG pipeline using OpenAI embeddings and ChromaDB
- Local answer generation with Ollama (llama3.2)
- Chat-style Streamlit interface with chat memory
- RAG ON / OFF comparison mode
- Retrieved source citations
- Voice input using OpenAI Whisper
- Guardrails that refuse out-of-domain questions

## Tech Stack
- Python
- Streamlit
- ChromaDB
- OpenAI (embeddings + Whisper)
- Ollama

## How It Works
1. `cleaned_text.txt` is split into chunks (`chunking.py`)
2. Each chunk is embedded with OpenAI (`embedding_helper.py`)
3. Chunks are stored in a ChromaDB collection
4. User questions are embedded and matched against stored chunks
5. Retrieved context + system prompt are sent to Ollama
6. The assistant answers using only retrieved evidence, with citations

## Setup
```bash
pip install -r requirements.txt
ollama pull llama3.2
streamlit run app.py