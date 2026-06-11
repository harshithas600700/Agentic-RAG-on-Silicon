# 📚 Librarian Pro: Agentic RAG Framework

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Orchestration-LlamaIndex%20%7C%20CrewAI-orange)](https://github.com/lucatume/crewai-llama-index)
[![Local LLM](https://img.shields.io/badge/Inference-Ollama%20(Llama3.2)-red)](https://ollama.com/)

**Librarian Pro** is a high-precision, privacy-first Document Intelligence Framework. While standard Retrieval-Augmented Generation (RAG) applications suffer from context fragmentation and single-pass LLM hallucinations, Librarian Pro uses a **decoupled Agentic Architecture** that isolates semantic retrieval from autonomous logical validation.

---

## 🏗️ Architectural Blueprint

The framework separates the data acquisition layer from the cognitive reasoning layer to maximize factual accuracy while optimizing local compute performance.

### 1. Contextual Retrieval Phase (LlamaIndex)
Traditional RAG pipelines often lose critical nuances due to arbitrary text splitting. We mitigate this using a **"Small-to-Big"** retrieval strategy:
* **Sentence-Window Parsing:** Input PDFs are parsed into granular, single-sentence nodes to ensure high-density vector mapping.
* **Metadata Replacement:** At query time, the system targets the exact semantic sentence matched by the embedding model, but replaces it with an expanded, contiguous text window (neighboring sentences) before injection into the LLM. This provides structural narrative coherence without inflating memory overhead.

### 2. The Consensus Engine (CrewAI)
Raw LLM outputs are treated as speculative drafts. The system forces a deterministic, sequential multi-agent validation loop:
* **The Lead Data Analyst:** Responsible for targeted data extraction and synthesis. This agent is strictly grounded within the retrieved context text window.
* **The QA Auditor:** Acts as a technical skeptic. It cross-references the Analyst’s synthesis directly against the raw source documentation to trap hallucinations, missed technical constraints, or structural contradictions.

---

## 🚀 Engineering Features & Safeguards

* **Hardware-Aware Design:** Optimized for **Apple Silicon Unified Memory Architecture (UMA)**. By using lightweight, local models, the pipeline achieves zero-copy data efficiencies between the vector space and the GPU cores.
* **Low-Confidence Circuit Breaker:** Implements a strict cosine similarity safeguard threshold (`score < 0.3`). If the retrieval engine cannot find a mathematically relevant match in the document database, it aborts the agent execution phase entirely to eliminate hallucination vectors.
* **Auditability & Observability:** Surfaced via an interactive Streamlit UI expander, exposing the raw context data and the internal multi-agent debate log for complete pipeline transparency.

---

## 🛠️ Technical Challenges & Resolutions

Developing a production-grade local multi-agent system highlighted several framework friction points:

* **Context Window Overload:** Early experiments with heavy re-ranking models caused severe latency spikes on local hardware. 
  * *Resolution:* Shifted processing weight to a sentence-window architecture, maintaining precision while cutting processing overhead.
* **Pydantic Validation Failures:** Upstream deprecations within legacy LangChain wrappers triggered data type validation crashes when passed into CrewAI agents.
  * *Resolution:* Refactored the core configuration to utilize CrewAI’s native `LLM` loader, establishing direct socket routing to the local Ollama instance.
* **Semantic Drift:** Irrelevant queries caused basic RAG layers to pass "garbage" text data to the LLM, forcing a fabricated answer.
  * *Resolution:* Engineered an explicit conditional gate inspecting raw node retrieval scores prior to spinning up compute resources.

---

## ⚙️ Technology Stack

* **Core Pipeline Orchestration:** LlamaIndex & CrewAI
* **Local Inference Host:** Ollama (`llama3.2:3b`)
* **Vector Embeddings:** `BAAI/bge-small-en-v1.5` (384-dimensional vector space)
* **User Interface:** Streamlit (Reactive Session State management)

---

## 💻 Deployment & Execution

### Prerequisites
Ensure your local `Ollama` instance is running and has the model pulled:
```bash
ollama pull llama3.2:3b
