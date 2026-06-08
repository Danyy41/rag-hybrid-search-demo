# rag-hybrid-search-demo
Simple RAG pipeline using BM25 and vector search over internal documents.
# RAG Hybrid Search Demo

A simple Retrieval-Augmented Generation project that demonstrates:

- BM25 keyword search
- Vector semantic search
- Hybrid retrieval

The system searches internal documents and returns the most relevant answer.

## Technologies

- Python
- Sentence Transformers
- Rank-BM25
- NumPy
## Example Queries

- What should I do if an API key is exposed?
- What is the policy on password sharing?
- How quickly must incidents be reported?

## Architecture

Question
    ↓
BM25 Search
    ↓
Vector Search
    ↓
Hybrid Retrieval
    ↓
GPT-4o-mini
    ↓
Answer