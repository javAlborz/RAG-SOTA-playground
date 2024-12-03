#<PGVECTOR-SCRIPT>
import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
from pathlib import Path
import PyPDF2
from tqdm import tqdm
import openai
from my_pgvector import PGVector
import psycopg2

# Load environment variables
load_dotenv()

from my_docloader import DocumentLoader

class VectorDB:
    """Vector database using PostgreSQL for storing and searching embeddings"""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        if base_url is None:
            base_url = os.getenv("LOCAL_EMBEDDINGS_URL", "http://localhost:8000")
        if api_key is None:
            api_key = os.getenv("LOCAL_API_KEY", "dummy-key")
            
        # Configure OpenAI client for local endpoint
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        # Initialize PGVector for persistent storage
        self.vector_store = PGVector()
        self.query_cache = {}

    def add_documents(self, documents: List[Dict[str, str]]):
        """Add documents to the vector database"""
        print("Processing documents...")
        texts_to_embed = [doc['content'] for doc in documents]
        
        print("Generating embeddings...")
        batch_size = 128
        with tqdm(total=len(texts_to_embed)) as pbar:
            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i:i + batch_size]
                batch_documents = documents[i:i + batch_size]
                
                # Use OpenAI-compatible embeddings endpoint
                response = self.client.embeddings.create(
                    model="/e5",  # or your local model name
                    input=batch
                )
                
                # Prepare embeddings for database insertion
                batch_embeddings = []
                for j, embedding_data in enumerate(response.data):
                    doc = batch_documents[j]
                    batch_embeddings.append((
                        doc['content'],                # document text
                        list(embedding_data.embedding),  # convert numpy array to list
                        doc['metadata']                # metadata
                    ))
                
                # Debug print
                print(f"\nPrepared batch of {len(batch_embeddings)} embeddings")
                print(f"Sample embedding length: {len(batch_embeddings[0][1])}")
                print(f"Sample metadata: {batch_embeddings[0][2]}")
                
                # Insert into PostgreSQL
                self.vector_store.insert_embeddings(batch_embeddings)
                pbar.update(len(batch))
        
        print(f"Added {len(documents)} documents to the database")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using PostgreSQL vector similarity search"""
        try:
            if query in self.query_cache:
                query_embedding = self.query_cache[query]
            else:
                response = self.client.embeddings.create(
                    model="/e5",
                    input=[query]
                )
                query_embedding = list(response.data[0].embedding)  # convert to list
                self.query_cache[query] = query_embedding

            print(f"\nGenerated query embedding of length: {len(query_embedding)}")

            # Use PGVector's search functionality
            results = self.vector_store.search_similar(
                query_embedding,
                limit=k,
                threshold=0.3  # Lowered threshold for testing
            )
            
            print(f"Found {len(results)} matches in database")
            
            if not results:
                print("Warning: No similar documents found in the database")
                
            return results
            
        except Exception as e:
            print(f"Error in vector search: {str(e)}")
            raise  # Re-raise to see full traceback

class RAGSystem:
    """Main RAG system combining vector database and LLM"""
    
    def __init__(self, 
                 embeddings_url: str = None,
                 llm_url: str = None,
                 api_key: str = None):
        self.vector_db = VectorDB(base_url=embeddings_url, api_key=api_key)
        
        # Configure OpenAI client for local LLM endpoint
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
        # Retrieve relevant chunks
        relevant_chunks = self.vector_db.search(question, k=k)
        
        # Debug print to verify chunks are being retrieved
        print(f"\nFound {len(relevant_chunks)} relevant chunks")
        
        if not relevant_chunks:
            return "No relevant information found in the documents to answer this question."
        
        # Prepare context for the LLM
        context = "\n\n".join([
            f"Content {i+1}:\n{chunk['content']}"
            for i, chunk in enumerate(relevant_chunks)
        ])
        
        # Debug print to verify context length
        print(f"Context length: {len(context)} characters")
        
        # Create the prompt
        prompt = f"""Here is some context information to help answer a question:

{context}

Question: {question}

Please provide a clear and concise answer based on the context provided. If the context doesn't contain enough information to answer the question fully, please indicate that."""

        # Get response from local LLM using OpenAI-compatible endpoint
        try:
            response = self.llm.chat.completions.create(
                model="/llama-3.1-70b",  # Use your local model name
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting LLM response: {str(e)}")
            return f"Error: Unable to generate response - {str(e)}"
def main():
    # Initialize RAG system with local endpoints
    rag = RAGSystem(
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

#</PGVECTOR-SCRIPT>