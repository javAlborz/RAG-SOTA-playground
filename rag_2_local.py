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

# Load environment variables
load_dotenv()

# DocumentLoader class remains unchanged
class DocumentLoader:
    """Handles loading and preprocessing of different document types"""
    
    @staticmethod
    def load_file(file_path: str) -> List[Dict[str, str]]:
        """
        Load a file and return a list of chunks with metadata
        Each chunk is a dict with 'content' and 'metadata' fields
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.csv':
            return DocumentLoader._load_csv(file_path)
        elif file_extension == '.txt':
            return DocumentLoader._load_txt(file_path)
        elif file_extension == '.json':
            return DocumentLoader._load_json(file_path)
        elif file_extension == '.pdf':
            return DocumentLoader._load_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    @staticmethod
    def _load_csv(file_path: str, chunk_size: int = 5) -> List[Dict[str, str]]:
        """Load CSV file and return chunks"""
        df = pd.read_csv(file_path)
        chunks = []
        
        # Process DataFrame in chunks
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i + chunk_size]
            content = chunk_df.to_string()
            chunks.append({
                'content': content,
                'metadata': {
                    'source': file_path,
                    'chunk_id': f"chunk_{i//chunk_size}",
                    'rows': f"{i}-{min(i + chunk_size, len(df))}"
                }
            })
        return chunks

    @staticmethod
    def _load_txt(file_path: str, chunk_size: int = 1000) -> List[Dict[str, str]]:
        """Load text file and return chunks"""
        chunks = []
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            
        # Split into paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = ''
        chunk_id = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append({
                    'content': current_chunk.strip(),
                    'metadata': {
                        'source': file_path,
                        'chunk_id': f"chunk_{chunk_id}"
                    }
                })
                chunk_id += 1
                current_chunk = para
            else:
                current_chunk += '\n\n' + para if current_chunk else para
        
        if current_chunk:  # Add the last chunk
            chunks.append({
                'content': current_chunk.strip(),
                'metadata': {
                    'source': file_path,
                    'chunk_id': f"chunk_{chunk_id}"
                }
            })
        
        return chunks

    @staticmethod
    def _load_json(file_path: str) -> List[Dict[str, str]]:
        """Load JSON file and return chunks"""
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        chunks = []
        if isinstance(data, list):
            for i, item in enumerate(data):
                chunks.append({
                    'content': json.dumps(item, indent=2),
                    'metadata': {
                        'source': file_path,
                        'chunk_id': f"chunk_{i}"
                    }
                })
        else:
            chunks.append({
                'content': json.dumps(data, indent=2),
                'metadata': {
                    'source': file_path,
                    'chunk_id': 'chunk_0'
                }
            })
        return chunks

    @staticmethod
    def _load_pdf(file_path: str, chunk_size: int = 1000) -> List[Dict[str, str]]:
        """Load PDF file and return chunks"""
        chunks = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            current_chunk = ''
            chunk_id = 0
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                paragraphs = text.split('\n\n')
                for para in paragraphs:
                    if len(current_chunk) + len(para) > chunk_size and current_chunk:
                        chunks.append({
                            'content': current_chunk.strip(),
                            'metadata': {
                                'source': file_path,
                                'chunk_id': f"chunk_{chunk_id}",
                                'page': page_num
                            }
                        })
                        chunk_id += 1
                        current_chunk = para
                    else:
                        current_chunk += '\n\n' + para if current_chunk else para
            
            if current_chunk:  # Add the last chunk
                chunks.append({
                    'content': current_chunk.strip(),
                    'metadata': {
                        'source': file_path,
                        'chunk_id': f"chunk_{chunk_id}",
                        'page': len(pdf_reader.pages) - 1
                    }
                })
        
        return chunks

class HybridVectorDB:
    """Vector database with hybrid search capabilities (embeddings + BM25)"""
    
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
        
        # Initialize Elasticsearch client for BM25
        self.es_client = Elasticsearch("http://localhost:9200")
        self.index_name = "hybrid_search_index"
        self.create_es_index()
        
        self.embeddings = []
        self.chunks = []
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
        """Add documents to both vector store and Elasticsearch"""
        print("Processing documents...")
        
        # Generate embeddings using local endpoint
        texts_to_embed = [doc['content'] for doc in documents]
        print("Generating embeddings...")
        batch_size = 128
        with tqdm(total=len(texts_to_embed), desc="Embedding documents") as pbar:
            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i:i + batch_size]
                response = self.embeddings_client.embeddings.create(
                    model="/e5",  # or your local model name
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                self.embeddings.extend(batch_embeddings)
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
        
        self.chunks.extend(documents)
        print(f"Added {len(documents)} documents to both databases")

    def search(self, query: str, k: int = 5, semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
        """Hybrid search combining embedding similarity and BM25"""
        # Get embedding search results
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            response = self.embeddings_client.embeddings.create(
                model="/e5",  # or your local model name
                input=[query]
            )
            query_embedding = response.data[0].embedding
            self.query_cache[query] = query_embedding

        # Calculate embedding similarities
        similarities = np.dot(self.embeddings, query_embedding)
        top_semantic_indices = np.argsort(similarities)[::-1][:k*2]  # Get more results for fusion
        
        # Get BM25 search results
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
        
        # Prepare results for rank fusion
        semantic_results = [
            {
                'content': self.chunks[idx]['content'],
                'metadata': self.chunks[idx]['metadata'],
                'score': float(similarities[idx]),
                'source': 'semantic'
            }
            for idx in top_semantic_indices
        ]
        
        bm25_results = [
            {
                'content': hit['_source']['content'],
                'metadata': hit['_source']['metadata'],
                'score': hit['_score'],
                'source': 'bm25'
            }
            for hit in es_response['hits']['hits']
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

    def query(self, question: str, k: int = 5) -> str:
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