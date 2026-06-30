<div align="center">

# 🚀 Advanced RAG Architectures
### *I Love RAG (Retrieval-Augmented Generation)*

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](#)
[![Gemini](https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)](#)
[![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)](#)
[![FAISS](https://img.shields.io/badge/FAISS-000000?style=for-the-badge&logo=meta&logoColor=white)](#)

> **Transforming raw text into intelligent, grounded, and context-aware AI systems.**
> This repository contains six distinct projects, each exploring a progressively complex RAG architecture.

</div>

---

## 🏗️ The 6 Architectures

As outlined in the project roadmap (reference: `image_742b9e.png`), this repository covers the evolution of RAG systems from basic semantic search to complex, multi-modal reasoning agents.

| Step | Architecture | Core Concept | Tech Stack Highlight |
| :---: | :--- | :--- | :--- |
| <kbd># 1</kbd> | **<kbd>Standard RAG</kbd>** | Direct semantic search and context stuffing. | FAISS, Gemini Flash, PyPDFLoader |
| <kbd># 2</kbd> | **<kbd>HyDE RAG</kbd>** | Generating hypothetical answers to improve retrieval accuracy in dense domains. | Groq (Llama 3), MPNet Embeddings |
| <kbd># 3</kbd> | **<kbd>Fusion RAG</kbd>** | Multi-query generation and Reciprocal Rank Fusion (RRF) for robust search. | *TBD* |
| <kbd># 4</kbd> | **<kbd>Agentic RAG</kbd>** | LLM-driven routing, tool use, and self-correction loops. | *TBD* |
| <kbd># 5</kbd> | **<kbd>Graph RAG</kbd>** | Knowledge graphs for entity relationship mapping and multi-hop reasoning. | *TBD* |
| <kbd># 6</kbd> | **<kbd>Multi-Modal RAG</kbd>** | Processing and retrieving across text, images, and tables simultaneously. | *TBD* |

---

## 📂 Project Spotlights

### 1. LexiQuery (Standard RAG)
A legal tech assistant designed to process massive legal contracts.
* Slices multi-page PDFs into overlapping contextual chunks.
* Embeds text using lightweight, local HuggingFace models.
* Enforces strict citation rules to prevent AI hallucination.

### 2. MedLit Search (HyDE RAG)
An advanced medical literature research tool.
* Bypasses the "vocabulary gap" between short user queries and dense medical abstracts.
* Uses Groq to hallucinate a scientifically accurate "fake" paper as a search vector.
* Retrieves highly specific biomedical pathways and clinical trial data.

### 3. Fusion RAG *(Coming Soon)*
*Documentation in progress.*

### 4. Agentic RAG *(Coming Soon)*
*Documentation in progress.*

### 5. Graph RAG *(Coming Soon)*
*Documentation in progress.*

### 6. Multi-Modal RAG *(Coming Soon)*
*Documentation in progress.*

---

## ⚙️ Quick Start Installation

Clone the repository and install the required dependencies to run these pipelines locally.

**Clone the repo:**
```bash
git clone [https://github.com/babaralimahar/RAG-Projects.git](https://github.com/babaralimahar/RAG-Projects.git)
cd RAG-Projects
