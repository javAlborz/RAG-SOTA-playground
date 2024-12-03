import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
import openai
from pathlib import Path
import PyPDF2
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from my_pgvector import PGVector

# Load environment variables
load_dotenv()

from my_docloader import DocumentLoader

class HybridVectorDB:
    """Vector database with hybrid search capabilities (PostgreSQL embeddings + Elasticsearch BM25)"""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        # Initialize OpenAI client for local embeddings
        if base_url is None:
            base_url = os.getenv("LOCAL_EMBEDDINGS_URL", "http://localhost:8000")
        if api_key is None:
            api_key = os.getenv("LOCAL_API_KEY", "dummy-key")
            
        self.embeddings_client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        # Initialize PostgreSQL vector store for embeddings
        self.vector_store = PGVector()
        
        # Initialize Elasticsearch client for BM25
        self.es_client = Elasticsearch("http://localhost:9200")
        self.index_name = "hybrid_search_index"
        self.create_es_index()
        
        self.query_cache = {}

    def create_es_index(self):
        """Create Elasticsearch index with appropriate settings"""
        if not self.es_client.indices.exists(index=self.index_name):
            index_settings = {
                "settings": {
                    "analysis": {"analyzer": {"default": {"type": "english"}}},
                    "similarity": {"default": {"type": "BM25"}},
                },
                "mappings": {
                    "properties": {
                        "content": {"type": "text", "analyzer": "english"},
                        "metadata": {"type": "object", "enabled": True}
                    }
                }
            }
            self.es_client.indices.create(index=self.index_name, body=index_settings)

    def add_documents(self, documents: List[Dict[str, str]]):
        """Add documents to both PostgreSQL and Elasticsearch"""
        print("Processing documents...")
        
        # Generate embeddings using local endpoint
        texts_to_embed = [doc['content'] for doc in documents]
        print("Generating embeddings...")
        batch_size = 128
        
        with tqdm(total=len(texts_to_embed), desc="Embedding documents") as pbar:
            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i:i + batch_size]
                batch_documents = documents[i:i + batch_size]
                
                # Generate embeddings
                response = self.embeddings_client.embeddings.create(
                    model="/e5",  # or your local model name
                    input=batch
                )
                
                # Prepare embeddings for PostgreSQL
                batch_embeddings = []
                for j, embedding_data in enumerate(response.data):
                    doc = batch_documents[j]
                    batch_embeddings.append((
                        doc['content'],
                        list(embedding_data.embedding),
                        doc['metadata']
                    ))
                
                # Insert batch into PostgreSQL
                self.vector_store.insert_embeddings(batch_embeddings)
                pbar.update(len(batch))
        
        # Add to Elasticsearch
        print("Adding documents to Elasticsearch...")
        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "content": doc['content'],
                    "metadata": doc['metadata']
                }
            }
            for doc in documents
        ]
        bulk(self.es_client, actions)
        self.es_client.indices.refresh(index=self.index_name)
        
        print(f"Added {len(documents)} documents to both databases")

    def search(self, query: str, k: int = 20, semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
        """Hybrid search combining PostgreSQL vector similarity and Elasticsearch BM25"""
        # Get semantic search results from PostgreSQL
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            response = self.embeddings_client.embeddings.create(
                model="/e5",  # or your local model name
                input=[query]
            )
            query_embedding = list(response.data[0].embedding)
            self.query_cache[query] = query_embedding

        semantic_results = self.vector_store.search_similar(
            query_embedding,
            limit=k*2,  # Get more results for fusion
            threshold=0.3
        )
        
        # Get BM25 results from Elasticsearch
        es_response = self.es_client.search(
            index=self.index_name,
            body={
                "query": {
                    "match": {
                        "content": query
                    }
                },
                "size": k*2  # Get more results for fusion
            }
        )
        
        # Prepare BM25 results
        bm25_results = [
            {
                'content': hit['_source']['content'],
                'metadata': hit['_source']['metadata'],
                'score': hit['_score'],
                'source': 'bm25'
            }
            for hit in es_response['hits']['hits']
        ]
        
        # Format semantic results
        semantic_results = [
            {
                'content': result['content'],
                'metadata': result['metadata'],
                'score': result['similarity'],
                'source': 'semantic'
            }
            for result in semantic_results
        ]
        
        # Combine and normalize scores
        all_results = semantic_results + bm25_results
        
        # Use content as key to remove duplicates and combine scores
        combined_results = {}
        for result in all_results:
            content = result['content']
            if content not in combined_results:
                combined_results[content] = {
                    'content': content,
                    'metadata': result['metadata'],
                    'semantic_score': 0.0,
                    'bm25_score': 0.0
                }
            
            if result['source'] == 'semantic':
                combined_results[content]['semantic_score'] = result['score']
            else:
                combined_results[content]['bm25_score'] = result['score']
        
        # Normalize scores and compute final score
        max_semantic = max(r['semantic_score'] for r in combined_results.values())
        max_bm25 = max(r['bm25_score'] for r in combined_results.values())
        
        for result in combined_results.values():
            norm_semantic = result['semantic_score'] / max_semantic if max_semantic > 0 else 0
            norm_bm25 = result['bm25_score'] / max_bm25 if max_bm25 > 0 else 0
            result['final_score'] = (semantic_weight * norm_semantic + 
                                   (1 - semantic_weight) * norm_bm25)
        
        # Sort by final score and return top k
        final_results = sorted(
            combined_results.values(),
            key=lambda x: x['final_score'],
            reverse=True
        )[:k]
        
        return final_results

class HybridRAGSystem:
    """RAG system using hybrid search"""
    
    def __init__(self, 
                 embeddings_url: str = None,
                 llm_url: str = None,
                 api_key: str = None):
        # Initialize vector database with local embeddings endpoint
        self.vector_db = HybridVectorDB(base_url=embeddings_url, api_key=api_key)
        
        # Initialize OpenAI client for local LLM endpoint
        if llm_url is None:
            llm_url = os.getenv("LOCAL_LLM_URL", "http://localhost:8001")
        if api_key is None:
            api_key = os.getenv("LOCAL_API_KEY", "dummy-key")
            
        self.llm = openai.OpenAI(
            base_url=llm_url,
            api_key=api_key
        )
        self.current_file = None

    def load_document(self, file_path: str):
        """Load a document into the RAG system"""
        print(f"Loading document: {file_path}")
        self.current_file = file_path
        chunks = DocumentLoader.load_file(file_path)
        self.vector_db.add_documents(chunks)
        print("Document loaded successfully")

    def query(self, question: str, k: int = 20) -> str:
        """Query the RAG system"""
        # Retrieve relevant chunks using hybrid search
        relevant_chunks = self.vector_db.search(question, k=k)
        
        # Prepare context for the LLM
        context = "\n\n".join([
            f"Content {i+1} (Score: {chunk['final_score']:.3f}):\n{chunk['content']}"
            for i, chunk in enumerate(relevant_chunks)
        ])
        
        # Create the prompt
        prompt = f"""Here is some context information to help answer a question:

{context}

Question: {question}

Please provide a clear and concise answer based on the context provided. If the context doesn't contain enough information to answer the question fully, please indicate that."""

        # Get response from local LLM using OpenAI-compatible endpoint
        response = self.llm.chat.completions.create(
            model="/llama-3.1-70b",  # Use your local model name
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0
        )
        
        return response.choices[0].message.content

def main():
    # Initialize RAG system with local endpoints
    rag = HybridRAGSystem(
        embeddings_url=os.getenv("LOCAL_EMBEDDINGS_URL", "http://localhost:8000"),
        llm_url=os.getenv("LOCAL_LLM_URL", "http://localhost:8001"),
        api_key=os.getenv("LOCAL_API_KEY", "dummy-key")
    )
    
    # Show available files
    data_dir = "data"
    available_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    
    print("\nAvailable files:")
    for i, file in enumerate(available_files, 1):
        print(f"{i}. {file}")
    
    # Get file selection
    while True:
        try:
            selection = int(input("\nSelect a file number to load (1-{}): ".format(len(available_files))))
            if 1 <= selection <= len(available_files):
                break
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Load selected file
    selected_file = os.path.join(data_dir, available_files[selection-1])
    rag.load_document(selected_file)
    
    # Interactive query loop
    print("\nDocument loaded! You can now ask questions (type 'exit' to quit)")
    while True:
        question = input("\nEnter your question: ")
        if question.lower() == 'exit':
            break
        
        response = rag.query(question)
        print("\nResponse:", response)

if __name__ == "__main__":
    main()