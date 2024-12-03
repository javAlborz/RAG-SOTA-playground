import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
import voyageai
import anthropic
from pathlib import Path
import PyPDF2
from tqdm import tqdm

# Load environment variables
load_dotenv()

from my_docloader import DocumentLoader

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


class ContextualVectorDB:
    """Vector database with contextual embeddings"""
    
    def __init__(self, api_key: str = None, anthropic_api_key: str = None):
        if api_key is None:
            api_key = os.getenv("VOYAGE_API_KEY")
        if anthropic_api_key is None:
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            
        self.voyage_client = voyageai.Client(api_key=api_key)
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.embeddings = []
        self.chunks = []
        self.query_cache = {}
        
        # Track token usage for cost analysis
        self.token_counts = {
            'input': 0,
            'output': 0,
            'cache_read': 0,
            'cache_creation': 0
        }

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
        """Add documents with contextual embeddings"""
        print("Processing documents and generating contextual embeddings...")
        texts_to_embed = []
        processed_chunks = []
        
        for doc in tqdm(documents, desc="Generating context"):
            # Generate contextual description
            context = self._generate_context(source_text, doc['content'])
            
            # Combine original content with context
            contextualized_text = f"{doc['content']}\n\nContext: {context}"
            texts_to_embed.append(contextualized_text)
            
            # Store enhanced metadata
            enhanced_chunk = {
                'content': doc['content'],
                'metadata': {
                    **doc['metadata'],
                    'context': context
                }
            }
            processed_chunks.append(enhanced_chunk)
        
        print("Generating embeddings...")
        batch_size = 128
        with tqdm(total=len(texts_to_embed), desc="Embedding chunks") as pbar:
            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i:i + batch_size]
                batch_embeddings = self.voyage_client.embed(batch, model="voyage-2").embeddings
                self.embeddings.extend(batch_embeddings)
                pbar.update(len(batch))
        
        self.chunks.extend(processed_chunks)
        
        # Print token usage statistics
        print("\nToken Usage Statistics:")
        print(f"Total input tokens: {self.token_counts['input']}")
        print(f"Total output tokens: {self.token_counts['output']}")
        print(f"Cache creation tokens: {self.token_counts['cache_creation']}")
        print(f"Cache read tokens: {self.token_counts['cache_read']}")
        
        cache_savings = (self.token_counts['cache_read'] / 
                        (self.token_counts['input'] + self.token_counts['cache_read'] + self.token_counts['cache_creation'])) * 100
        print(f"Cache savings: {cache_savings:.2f}% (90% discount on cached tokens)")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.voyage_client.embed([query], model="voyage-2").embeddings[0]
            self.query_cache[query] = query_embedding

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append({
                'content': self.chunks[idx]['content'],
                'metadata': self.chunks[idx]['metadata'],
                'similarity': float(similarities[idx])
            })
        
        return results



class ContextualRAGSystem:
    """RAG system using contextual embeddings"""
    
    def __init__(self):
        self.vector_db = ContextualVectorDB()
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
        
        # Then load chunks
        chunks = DocumentLoader.load_file(file_path)
        self.vector_db.add_documents(chunks, self.source_text)
        print("Document loaded successfully")

    def query(self, question: str, k: int = 5) -> str:
        """Query the RAG system"""
        relevant_chunks = self.vector_db.search(question, k=k)
        
        # Include both content and context in the prompt
        context = "\n\n".join([
            f"Content {i+1}:\n{chunk['content']}\nContext: {chunk['metadata']['context']}"
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
    rag = ContextualRAGSystem()
    
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