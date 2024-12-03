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

class DanishHybridVectorDB:
    """Vector database optimized for Danish language with hybrid search capabilities"""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        # Initialize OpenAI client for embeddings
        self.embeddings_client = openai.OpenAI(
            base_url=base_url or os.getenv("LOCAL_EMBEDDINGS_URL", "http://localhost:8000"),
            api_key=api_key or os.getenv("LOCAL_API_KEY", "dummy-key")
        )
        
        # Initialize PostgreSQL vector store
        try:
            self.vector_store = PGVector()
        except Exception as e:
            print(f"Error initializing PGVector: {e}")
            raise
        
        # Initialize Elasticsearch
        try:
            self.es_client = Elasticsearch("http://localhost:9200")
            self.index_name = "danish_hybrid_search_index"
            self.create_es_index()
        except Exception as e:
            print(f"Error initializing Elasticsearch: {e}")
            raise
            
        self.query_cache = {}

    def create_es_index(self):
        """Create Elasticsearch index with Danish-specific settings"""
        if not self.es_client.indices.exists(index=self.index_name):
            index_settings = {
                "settings": {
                    "analysis": {
                        "filter": {
                            "danish_stop": {
                                "type": "stop",
                                "stopwords": "_danish_"
                            },
                            "danish_stemmer": {
                                "type": "stemmer",
                                "language": "danish"
                            }
                        },
                        "analyzer": {
                            "danish": {
                                "tokenizer": "standard",
                                "filter": [
                                    "lowercase",
                                    "danish_stop",
                                    "danish_stemmer"
                                ]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "danish"
                        },
                        "metadata": {
                            "type": "object",
                            "enabled": True
                        }
                    }
                }
            }
            try:
                self.es_client.indices.create(index=self.index_name, body=index_settings)
            except Exception as e:
                print(f"Error creating Elasticsearch index: {e}")
                raise

    def add_documents(self, documents: List[Dict[str, str]]):
        """Add documents with Danish-optimized processing"""
        print("Processing documents...")
        
        texts_to_embed = [doc['content'] for doc in documents]
        print("Generating embeddings...")
        batch_size = 32  # Smaller batch size for better memory management
        
        with tqdm(total=len(texts_to_embed), desc="Embedding documents") as pbar:
            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i:i + batch_size]
                batch_documents = documents[i:i + batch_size]
                
                try:
                    # Generate embeddings
                    response = self.embeddings_client.embeddings.create(
                        model="/e5",  # Use multilingual model
                        input=batch
                    )
                    
                    # Prepare batch embeddings
                    batch_embeddings = []
                    for j, embedding_data in enumerate(response.data):
                        doc = batch_documents[j]
                        batch_embeddings.append((
                            doc['content'],
                            list(embedding_data.embedding),
                            doc['metadata']
                        ))
                    
                    # Insert into PostgreSQL
                    self.vector_store.insert_embeddings(batch_embeddings)
                except Exception as e:
                    print(f"Error processing batch {i//batch_size + 1}: {e}")
                    continue
                
                pbar.update(len(batch))
        
        # Add to Elasticsearch
        print("Adding documents to Elasticsearch...")
        try:
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
            print(f"Successfully added {len(documents)} documents to both databases")
        except Exception as e:
            print(f"Error adding documents to Elasticsearch: {e}")

    def search(self, query: str, k: int = 5, semantic_weight: float = 0.6) -> List[Dict[str, Any]]:
        """Hybrid search optimized for Danish language"""
        results = []
        
        # Get query embedding
        try:
            if query in self.query_cache:
                query_embedding = self.query_cache[query]
            else:
                response = self.embeddings_client.embeddings.create(
                    model="/e5",
                    input=[query]
                )
                query_embedding = list(response.data[0].embedding)
                self.query_cache[query] = query_embedding
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []

        # Get vector search results
        try:
            semantic_results = self.vector_store.search_similar(
                query_embedding,
                limit=k*2,
                threshold=0.25
            )
        except Exception as e:
            print(f"Error in vector search: {e}")
            semantic_results = []

        # Get Elasticsearch results
        try:
            es_response = self.es_client.search(
                index=self.index_name,
                body={
                    "query": {
                        "match": {
                            "content": {
                                "query": query,
                                "analyzer": "danish"
                            }
                        }
                    },
                    "size": k*2
                }
            )
            
            bm25_results = [
                {
                    'content': hit['_source']['content'],
                    'metadata': hit['_source']['metadata'],
                    'score': hit['_score'],
                    'source': 'bm25'
                }
                for hit in es_response['hits']['hits']
            ]
        except Exception as e:
            print(f"Error in Elasticsearch search: {e}")
            bm25_results = []

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

        # Combine results
        all_results = semantic_results + bm25_results
        if not all_results:
            return []

        # Combine and normalize scores
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

        # Normalize and combine scores
        if combined_results:
            max_semantic = max((r['semantic_score'] for r in combined_results.values()), default=1)
            max_bm25 = max((r['bm25_score'] for r in combined_results.values()), default=1)
            
            for result in combined_results.values():
                norm_semantic = result['semantic_score'] / max_semantic if max_semantic > 0 else 0
                norm_bm25 = result['bm25_score'] / max_bm25 if max_bm25 > 0 else 0
                result['final_score'] = (semantic_weight * norm_semantic + 
                                       (1 - semantic_weight) * norm_bm25)

            final_results = sorted(
                combined_results.values(),
                key=lambda x: x['final_score'],
                reverse=True
            )[:k]
            
            return final_results
        
        return []

    def __del__(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'vector_store'):
                self.vector_store.close_connection()
        except Exception as e:
            print(f"Error during cleanup: {e}")


class DanishRAGSystem:
    """RAG system optimized for Danish language"""
    
    def __init__(self, 
                 embeddings_url: str = None,
                 llm_url: str = None,
                 api_key: str = None):
        self.vector_db = DanishHybridVectorDB(base_url=embeddings_url, api_key=api_key)
        
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
        """Load a Danish document into the RAG system"""
        print(f"Loading document: {file_path}")
        self.current_file = file_path
        chunks = DocumentLoader.load_file(file_path)
        self.vector_db.add_documents(chunks)
        print("Document loaded successfully")

    def query(self, question: str, k: int = 5) -> str:
        """Query the RAG system in Danish"""
        relevant_chunks = self.vector_db.search(question, k=k)
        
        context = "\n\n".join([
            f"Indhold {i+1} (Score: {chunk['final_score']:.3f}):\n{chunk['content']}"
            for i, chunk in enumerate(relevant_chunks)
        ])
        
        prompt = f"""Her er noget kontekst information til at hjælpe med at besvare et spørgsmål:

{context}

Spørgsmål: {question}

Giv venligst et klart og præcist svar baseret på den givne kontekst. Hvis konteksten ikke indeholder nok information til at besvare spørgsmålet fuldt ud, skal du angive dette."""

        response = self.llm.chat.completions.create(
            model="/llama-3.1-70b",  # Use your local model
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0
        )
        
        return response.choices[0].message.content

def main():
    rag = DanishRAGSystem(
        embeddings_url=os.getenv("LOCAL_EMBEDDINGS_URL", "http://localhost:8000"),
        llm_url=os.getenv("LOCAL_LLM_URL", "http://localhost:8001"),
        api_key=os.getenv("LOCAL_API_KEY", "dummy-key")
    )
    
    data_dir = "data"
    available_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    
    print("\nTilgængelige filer:")
    for i, file in enumerate(available_files, 1):
        print(f"{i}. {file}")
    
    while True:
        try:
            selection = int(input("\nVælg et filnummer der skal indlæses (1-{}): ".format(len(available_files))))
            if 1 <= selection <= len(available_files):
                break
            print("Ugyldigt valg. Prøv igen.")
        except ValueError:
            print("Indtast venligst et gyldigt nummer.")
    
    selected_file = os.path.join(data_dir, available_files[selection-1])
    rag.load_document(selected_file)
    
    print("\nDokument indlæst! Du kan nu stille spørgsmål (skriv 'exit' for at afslutte)")
    while True:
        question = input("\nIndtast dit spørgsmål: ")
        if question.lower() == 'exit':
            break
        
        response = rag.query(question)
        print("\nSvar:", response)

if __name__ == "__main__":
    main()