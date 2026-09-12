# AI Research Assistant

An autonomous, fully offline research copilot engineered to execute arbitrary web discovery, intelligent document ingestion, and mathematically guarded Retrieval-Augmented Generation (RAG). Designed for local consumer hardware, the system operates entirely without external API calls, cloud subscriptions, or telemetry leakage.

---

## Architecture Overview

```text
                        ┌─────────────────────────────────────┐
                        │        Gradio 6.x Reactive UI       │
                        │    (Emerald Green / Slate Theme)    │
                        └──────────────────┬──────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────┐
                        │    Ollama Engine (Local Daemon)     │
                        │        Model: qwen2.5:3b            │
                        └───────┬─────────────────────┬───────┘
                                │                     │
      Native Tool Calling Loop  │                     │  Vector Ingestion / Strict Query
    (max_iterations=3, ddgs)   │                     │  (RecursiveSplitter + nomic-embed)
                                ▼                     ▼
             ┌────────────────────────┐         ┌───────────────────────────────┐
             │   Web Tools Module     │         │       RAG Pipeline Core       │
             │  • DuckDuckGo Search   │         │  • LangChain Chunking (1000)  │
             │  • BeautifulSoup4 DOM  │         │  • ChromaDB Persistent Store  │
             │  • YouTube Transcripts │         │  • Distance Threshold (0.8)   │
             └───────────┬────────────┘         └───────────────┬───────────────┘
                         │                                      │
                         └──────────────────┬───────────────────┘
                                            ▼
                        ┌─────────────────────────────────────┐
                        │ Context Aggregator & Report Builder │
                        │  Grounded Markdown Synthesis Engine │
                        └─────────────────────────────────────┘

```

---

## Core Capabilities

* **100% Offline & Privacy-Preserving:** Executes completely on the local host with zero remote telemetry or proprietary API fees.


* **Autonomous Tool-Calling Loop:** Implements native JSON schema function calling powered by `qwen2.5:3b` with a hard limit of 3 iterations to prevent execution runaways.


* **Strict Semantic Distance Gating:** Retrieved vector chunks must satisfy a mathematical cutoff (`DISTANCE_THRESHOLD = 0.8`). Queries yielding hits outside this metric trigger an explicit `### ⚠️ Insufficient Context` response rather than allowing LLM hallucination.


* **Session Source Scoping:** RAG context retrieval restricts search scopes strictly to URLs registered in the active UI session list, ignoring leftover chunks from past investigations.


* **Fault-Tolerant Scraping & Ingestion:**
* Enforces a 10-second request timeout with custom browser headers.


* Native YouTube handling via `youtube-transcript-api` to extract subtitles and reject captionless videos cleanly with `gr.Warning` alerts.




* **One-Click Knowledge Base Flush:** A UI action drops the underlying collection and cleans session state instantly.



---

## Technical Specifications

| Layer | Technology | Function | Resource Footprint |
| --- | --- | --- | --- |
| **Inference Runtime** | Ollama (v0.3+) | Local model daemon & JSON orchestration

 | ~120 MB RAM

 |
| **Reasoning Agent** | `qwen2.5:3b` | Multi-turn reasoning, tool execution, report synthesis

 | ~2.2 GB VRAM

 |
| **Vector Embeddings** | `nomic-embed-text` | 768-dimensional dense semantic representations

 | ~580 MB VRAM

 |
| **Vector Database** | ChromaDB (`PersistentClient`) | Local non-volatile vector persistence (`./chroma_db`)

 | Disk-based

 |
| **Text Partitioning** | `langchain_text_splitters` | `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=150`)

 | Negligible CPU

 |
| **Web Scraping** | `ddgs`, `requests`, `bs4` | Live web crawling, sanitization, and text extraction

 | Network bounded

 |
| **Video Processing** | `youtube-transcript-api` | Extraction of speech subtitles for arbitrary video URLs

 | Network bounded

 |
| **UI Framework** | Gradio 6.x | High-contrast emerald green theme, reactive events

 | Local HTTP server

 |

---

## Repository Structure

```bash
AI-Research-Assistant/
├── app_ui_skeleton.py                 # Gradio 6.x interface, event listeners, and UI state
├── rag_manager.py                     # Chunking, vector persistence, and threshold gating
├── main.py                            # Agent orchestration, Ollama tool-calling loop, tools
├── integration_notes.py               # Interface bridge notes and RAG connection helpers
├── requirements.txt                   # Project runtime dependencies
├── AI_Research_Assistant_Presentation.pdf # 11-slide technical engineering deck
├── .gitignore                         # Exclusions for __pycache__, venv, and chroma_db/
└── README.md                          # Comprehensive project documentation

```

---

## Installation & Setup

### 1. System Requirements

* Operating System: Linux (Fedora/Ubuntu tested) or Windows with WSL2.
* Python: 3.10 to 3.14.
* Hardware: GPU with at least 4 GB VRAM or modern multi-threaded CPU.



### 2. Install & Start Ollama

Download and install Ollama from [ollama.com](https://ollama.com/), then pull the required models:

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text

```

Ensure the Ollama service is active:

```bash
curl http://localhost:11434/api/tags

```

### 3. Clone Repository & Setup Virtual Environment

```bash
git clone https://github.com/abdorizk61/AI-Research-Assistant.git
cd AI-Research-Assistant

python3 -m venv venv
source venv/bin/activate

```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

---

## Running the Application

Launch the research assistant interface:

```bash
python app_ui_skeleton.py

```

Open your browser and navigate to:

```text
http://127.0.0.1:7860

```

---

## Engineering Safeguards

* **Grounded Retrieval Cutoff:** If retrieved document chunks produce distances strictly greater than `0.8`, the query is flagged as irrelevant and generation is avoided.


* **Empty Collection Bypass:** Empty knowledge bases trigger early return flows, saving inference cycles.


* **HTML Stripping:** Ingested pages are stripped of `<script>`, `<style>`, header, and footer tags prior to chunking, preventing noisy markup from entering vector space.


* **Graceful Network Degradation:** Broken links, 403 Forbidden blockers, or disabled YouTube captions generate user-facing warnings (`gr.Warning`) and skip database writes.



---

## Team & Attribution

Developed by **Team 1**:

* **Abdelrahman Rizk**

* **Sabri Mohamed**
