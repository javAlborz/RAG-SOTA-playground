import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
import voyageai
import anthropic
import cohere
from pathlib import Path
import PyPDF2
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import time

# Load environment variables
load_dotenv()

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


class ContextualRerankedHybridVectorDB:
    """Vector database with contextual hybrid search and reranking capabilities"""
    
    def __init__(self, api_key: str = None, anthropic_api_key: str = None):
        # Initialize clients
        if api_key is None:
            api_key = os.getenv("VOYAGE_API_KEY")
        if anthropic_api_key is None:
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            
        self.voyage_client = voyageai.Client(api_key=api_key)
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.co = cohere.Client(os.getenv("COHERE_API_KEY"))
        
        # Initialize Elasticsearch
        self.es_client = Elasticsearch("http://localhost:9200")
        self.index_name = "contextual_reranked_hybrid_search_index"
        self.create_es_index()
        
        self.embeddings = []
        self.chunks = []
        self.query_cache = {}
        
        # Token tracking
        self.token_counts = {
            'input': 0,
            'output': 0,
            'cache_read': 0,
            'cache_creation': 0
        }

    def create_es_index(self):
        """Create Elasticsearch index with support for contextual search"""
        if not self.es_client.indices.exists(index=self.index_name):
            index_settings = {
                "settings": {
                    "analysis": {"analyzer": {"default": {"type": "english"}}},
                    "similarity": {"default": {"type": "BM25"}},
                },
                "mappings": {
                    "properties": {
                        "content": {"type": "text", "analyzer": "english"},
                        "context": {"type": "text", "analyzer": "english"},
                        "metadata": {"type": "object", "enabled": True}
                    }
                }
            }
            self.es_client.indices.create(index=self.index_name, body=index_settings)

    def _generate_context(self, document: str, chunk: str) -> str:
        """Generate contextual description for a chunk using Claude"""
        DOCUMENT_CONTEXT_PROMPT = """
        <document>
        {doc_content}
        </document>
        """

        CHUNK_CONTEXT_PROMPT = """
        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk_content}
        </chunk>

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """

        response = self.anthropic_client.beta.prompt_caching.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            temperature=0.0,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=document),
                            "cache_control": {"type": "ephemeral"}
                        },
                        {
                            "type": "text",
                            "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                        }
                    ]
                }
            ],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )
        
        # Update token counts
        self.token_counts['input'] += response.usage.input_tokens
        self.token_counts['output'] += response.usage.output_tokens
        self.token_counts['cache_read'] += response.usage.cache_read_input_tokens
        self.token_counts['cache_creation'] += response.usage.cache_creation_input_tokens
        
        return response.content[0].text

    def add_documents(self, documents: List[Dict[str, str]], source_text: str):
        """Add documents with contextual information to all search components"""
        print("Processing documents and generating contextual embeddings...")
        texts_to_embed = []
        es_documents = []
        processed_chunks = []
        
        for doc in tqdm(documents, desc="Generating context"):
            # Generate contextual description
            context = self._generate_context(source_text, doc['content'])
            
            # Combine content with context for embedding
            contextualized_text = f"{doc['content']}\n\nContext: {context}"
            texts_to_embed.append(contextualized_text)
            
            # Prepare document for Elasticsearch
            es_doc = {
                "_index": self.index_name,
                "_source": {
                    "content": doc['content'],
                    "context": context,
                    "metadata": {
                        **doc['metadata'],
                        'context': context
                    }
                }
            }
            es_documents.append(es_doc)
            
            # Store enhanced chunk
            processed_chunks.append({
                'content': doc['content'],
                'metadata': {
                    **doc['metadata'],
                    'context': context
                },
                'contextualized_text': contextualized_text
            })
        
        # Generate embeddings
        print("Generating embeddings...")
        batch_size = 128
        with tqdm(total=len(texts_to_embed), desc="Embedding chunks") as pbar:
            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i:i + batch_size]
                batch_embeddings = self.voyage_client.embed(batch, model="voyage-2").embeddings
                self.embeddings.extend(batch_embeddings)
                pbar.update(len(batch))
        
        # Add to Elasticsearch
        print("Adding documents to Elasticsearch...")
        bulk(self.es_client, es_documents)
        self.es_client.indices.refresh(index=self.index_name)
        
        self.chunks = processed_chunks
        
        # Print token usage statistics
        print("\nToken Usage Statistics:")
        print(f"Total input tokens: {self.token_counts['input']}")
        print(f"Total output tokens: {self.token_counts['output']}")
        print(f"Cache creation tokens: {self.token_counts['cache_creation']}")
        print(f"Cache read tokens: {self.token_counts['cache_read']}")
        
        cache_savings = (self.token_counts['cache_read'] / 
                        (self.token_counts['input'] + self.token_counts['cache_read'] + self.token_counts['cache_creation'])) * 100
        print(f"Cache savings: {cache_savings:.2f}% (90% discount on cached tokens)")

    def search(self, query: str, k: int = 5, semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
        """Contextual hybrid search with reranking"""
        # Get more results initially for reranking
        initial_k = k * 3
        
        # Get embedding search results
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.voyage_client.embed([query], model="voyage-2").embeddings[0]
            self.query_cache[query] = query_embedding

        # Calculate embedding similarities
        similarities = np.dot(self.embeddings, query_embedding)
        top_semantic_indices = np.argsort(similarities)[::-1][:initial_k]
        
        # Get contextual BM25 search results
        es_response = self.es_client.search(
            index=self.index_name,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content^2", "context"],
                        "type": "best_fields"
                    }
                },
                "size": initial_k
            }
        )
        
        # Prepare results for rank fusion
        semantic_results = [
            {
                'content': self.chunks[idx]['content'],
                'metadata': self.chunks[idx]['metadata'],
                'contextualized_text': self.chunks[idx]['contextualized_text'],
                'score': float(similarities[idx]),
                'source': 'semantic'
            }
            for idx in top_semantic_indices
        ]
        
        bm25_results = [
            {
                'content': hit['_source']['content'],
                'metadata': hit['_source']['metadata'],
                'contextualized_text': f"{hit['_source']['content']}\n\nContext: {hit['_source']['metadata']['context']}",
                'score': hit['_score'],
                'source': 'bm25'
            }
            for hit in es_response['hits']['hits']
        ]
        
        # Combine and normalize scores
        all_results = semantic_results + bm25_results
        combined_results = {}
        
        for result in all_results:
            content = result['content']
            if content not in combined_results:
                combined_results[content] = {
                    'content': content,
                    'metadata': result['metadata'],
                    'contextualized_text': result['contextualized_text'],
                    'semantic_score': 0.0,
                    'bm25_score': 0.0
                }
            
            if result['source'] == 'semantic':
                combined_results[content]['semantic_score'] = result['score']
            else:
                combined_results[content]['bm25_score'] = result['score']
        
        # Normalize scores and compute initial ranking
        max_semantic = max(r['semantic_score'] for r in combined_results.values())
        max_bm25 = max(r['bm25_score'] for r in combined_results.values())
        
        for result in combined_results.values():
            norm_semantic = result['semantic_score'] / max_semantic if max_semantic > 0 else 0
            norm_bm25 = result['bm25_score'] / max_bm25 if max_bm25 > 0 else 0
            result['initial_score'] = (semantic_weight * norm_semantic + 
                                     (1 - semantic_weight) * norm_bm25)
        
        # Get top candidates for reranking
        candidates = sorted(
            combined_results.values(),
            key=lambda x: x['initial_score'],
            reverse=True
        )[:initial_k]
        
        # Prepare documents for reranking - use contextualized text
        rerank_docs = [c['contextualized_text'] for c in candidates]
        
        # Rerank using Cohere
        rerank_results = self.co.rerank(
            model="rerank-english-v2.0",
            query=query,
            documents=rerank_docs,
            top_n=k
        )
        
        # Add a small delay to avoid rate limiting
        time.sleep(0.1)
        
        # Prepare final results
        final_results = []
        for result in rerank_results.results:
            original_result = candidates[result.index]
            final_results.append({
                'content': original_result['content'],
                'metadata': original_result['metadata'],
                'initial_score': original_result['initial_score'],
                'rerank_score': result.relevance_score,
                'final_score': result.relevance_score
            })
        
        return final_results

class ContextualRerankedHybridRAGSystem:
    """RAG system using contextual hybrid search with reranking"""
    
    def __init__(self):
        self.vector_db = ContextualRerankedHybridVectorDB()
        self.llm = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.current_file = None
        self.source_text = None

    def load_document(self, file_path: str):
        """Load a document into the RAG system"""
        print(f"Loading document: {file_path}")
        self.current_file = file_path
        
        # Load the entire document first for context
        with open(file_path, 'r', encoding='utf-8') as f:
            self.source_text = f.read()
        
        chunks = DocumentLoader.load_file(file_path)
        self.vector_db.add_documents(chunks, self.source_text)
        print("Document loaded successfully")

    def query(self, question: str, k: int = 5) -> str:
        """Query the RAG system"""
        relevant_chunks = self.vector_db.search(question, k=k)
        
        # Include both content and context in the prompt
        context = "\n\n".join([
            f"Content {i+1} (Initial Score: {chunk['initial_score']:.3f}, Rerank Score: {chunk['rerank_score']:.3f}):\n{chunk['content']}\nContext: {chunk['metadata']['context']}"
            for i, chunk in enumerate(relevant_chunks)
        ])
        
        prompt = f"""Here is some context information to help answer a question:

{context}

Question: {question}

Please provide a clear and concise answer based on the context provided. If the context doesn't contain enough information to answer the question fully, please indicate that."""

        response = self.llm.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text

def main():
    # Initialize RAG system
    rag = ContextualRerankedHybridRAGSystem()
    
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
    print("Note: This system uses contextual hybrid search with reranking")
    print("(contextual embeddings + contextual BM25 + Cohere reranking)")
    
    while True:
        question = input("\nEnter your question: ")
        if question.lower() == 'exit':
            break
        
        try:
            response = rag.query(question)
            print("\nResponse:", response)
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            print("Please try again with a different question or check if the services (Elasticsearch, etc.) are running properly.")

if __name__ == "__main__":
    main()