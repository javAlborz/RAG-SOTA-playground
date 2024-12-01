# RAGATHON
This project implements a Retrieval Augmented Generation (RAG) system with multiple retrieval methods, allowing users to experiment with different approaches to document retrieval and question answering.

## Overview

The system provides three increasingly sophisticated approaches to document retrieval:

1. **Basic Embedding-based RAG**
   - Uses vector embeddings (Voyage AI in memory for now) for semantic search

2. **Hybrid Search RAG**
   - Combines vector embeddings with BM25 text search

3. **Reranked Hybrid Search RAG**
   - Builds on hybrid search by adding Cohere's reranking
  
4. **CONTEXTUAL Basic Embedding-based RAG**

5. **CONTEXTUAL Hybrid Search RAG**

6. **CONTEXTUAL Reranked Hybrid Search RAG**

   
## Setup

1. Create a virtual environment:

2. Install dependencies:
```bash
pip install anthropic voyageai cohere elasticsearch pandas numpy python-dotenv PyPDF2
```

3. Create a `.env` file with your API keys. I will send mine on teams on request:
```
ANTHROPIC_API_KEY=your_key_here
VOYAGE_API_KEY=your_key_here
COHERE_API_KEY=your_key_here

```

4. Start Elasticsearch (required for hybrid methods):
```bash
docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e "xpack.security.enabled=false" elasticsearch:8.8.0
```

## Usage

The system supports various document formats:
- CSV files
- Text files
- JSON files
- PDF documents


## Project Structure

- `rag1.py`: Basic embedding-based RAG
- `rag2.py`: Hybrid search RAG (embedding + BM25)
- `rag3.py`: Reranked hybrid RAG
- `data/`: Directory for source documents
- `DocumentLoader`: Handles various file formats and chunking
- `VectorDB`/`HybridVectorDB`/`RerankedHybridVectorDB`: Different retrieval implementations

## Future Improvements and ToDos

- Implementation of contextual embeddings
- User interface improvements
- Using our own endpoints. Requires small llama model for contextual embedding sidequest

- refactor:
  - common document loader class
