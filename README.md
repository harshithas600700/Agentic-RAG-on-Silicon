# Agentic-RAG-on-Silicon

A high-precision, privacy-first Document Intelligence system optimized for Apple Silicon (M1/M2/M3). This project implements an advanced "Retrieve-and-Rerank" architecture to provide accurate, context-aware answers from local PDF datasets without ever sending data to the cloud.



## 🌟 Key Features
* **100% Local Inference:** Powered by **Ollama** and **Llama 3.2**. Data never leaves your machine.
* **Two-Stage Retrieval:** Initial vector search followed by a **BGE-Reranker** (Cross-Encoder) stage to maximize accuracy and eliminate hallucinations.
* **Streamlit UI:** A clean, browser-based interface for PDF uploads and interactive chatting.
* **Hardware Optimized:** Leverages Apple's Unified Memory and Metal acceleration for fast local performance.

## 🛠️ Tech Stack
- **LLM:** Llama 3.2 (3B)
- **Orchestration:** LlamaIndex
- **Reranker:** BAAI/bge-reranker-base
- **Frontend:** Streamlit
- **Embeddings:** HuggingFace BGE-Small-v1.5

## 🚀 Quick Start

### 1. Prerequisites
Install [Ollama](https://ollama.com) and pull the model:
```bash
ollama pull llama3.2:3b
