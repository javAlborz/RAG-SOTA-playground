import json
from typing import List, Dict, Any, Callable
from tqdm import tqdm
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the RAG implementation
sys.path.append(str(Path(__file__).parent.parent))

# Change for script that will get evaluated. ToDo: generalize to accept different RAG implementations
from rag1 import DocumentLoader
# from rag1 import VectorDB
from rag2 import HybridVectorDB as VectorDB
from rag3 import RerankedHybridVectorDB as VectorDB

class RAGEvaluator:
    """Evaluates RAG system performance using Pass@k metric"""
    
    def __init__(self, chunks_path: str, eval_set_path: str):
        """
        Initialize evaluator with paths to evaluation data
        
        Args:
            chunks_path: Path to the codebase chunks JSON file
            eval_set_path: Path to the evaluation set JSONL file
        """
        self.chunks_path = chunks_path
        self.eval_set_path = eval_set_path
        self.chunks_data = self._load_chunks()
        self.eval_data = self._load_eval_set()

    def _load_chunks(self) -> List[Dict]:
        """Load the codebase chunks"""
        with open(self.chunks_path, 'r') as f:
            return json.load(f)

    def _load_eval_set(self) -> List[Dict]:
        """Load the evaluation dataset"""
        with open(self.eval_set_path, 'r') as f:
            return [json.loads(line) for line in f]

    def prepare_vector_db(self) -> VectorDB:
        """Prepare the vector database with the evaluation chunks"""
        db = VectorDB()
        
        # Transform chunks into the format expected by VectorDB
        documents = []
        for doc in self.chunks_data:
            for chunk in doc['chunks']:
                documents.append({
                    'content': chunk['content'],
                    'metadata': {
                        'doc_id': doc['doc_id'],
                        'original_uuid': doc['original_uuid'],
                        'chunk_id': chunk['chunk_id'],
                        'original_index': chunk['original_index']
                    }
                })
        
        # Add documents to the database
        db.add_documents(documents)
        return db

    def evaluate(self, vector_db: VectorDB, k_values: List[int] = [5, 10, 20]) -> Dict[int, float]:
        """
        Evaluate RAG system performance using multiple k values
        
        Args:
            vector_db: The vector database to evaluate
            k_values: List of k values to evaluate Pass@k
            
        Returns:
            Dictionary mapping k values to Pass@k scores
        """
        results = {}
        
        for k in k_values:
            print(f"\nEvaluating Pass@{k}")
            pass_at_k = self._evaluate_pass_at_k(vector_db, k)
            results[k] = pass_at_k
            
        return results

    def _evaluate_pass_at_k(self, vector_db: VectorDB, k: int) -> float:
        """Calculate Pass@k metric"""
        total_score = 0
        total_queries = len(self.eval_data)
        
        for query_item in tqdm(self.eval_data, desc=f"Calculating Pass@{k}"):
            query = query_item['query']
            golden_chunk_uuids = query_item['golden_chunk_uuids']
            
            # Find all golden chunk contents
            golden_contents = []
            for doc_uuid, chunk_index in golden_chunk_uuids:
                golden_doc = next(
                    (doc for doc in query_item['golden_documents'] if doc['uuid'] == doc_uuid),
                    None
                )
                if not golden_doc:
                    print(f"Warning: Golden document not found for UUID {doc_uuid}")
                    continue
                
                golden_chunk = next(
                    (chunk for chunk in golden_doc['chunks'] if chunk['index'] == chunk_index),
                    None
                )
                if not golden_chunk:
                    print(f"Warning: Golden chunk not found for index {chunk_index} in document {doc_uuid}")
                    continue
                
                golden_contents.append(golden_chunk['content'].strip())
            
            if not golden_contents:
                print(f"Warning: No golden contents found for query: {query}")
                continue
            
            # Get retrieved documents
            retrieved_docs = vector_db.search(query, k=k)
            
            # Count how many golden chunks are in the top k retrieved documents
            chunks_found = 0
            for golden_content in golden_contents:
                for doc in retrieved_docs[:k]:
                    retrieved_content = doc['content'].strip()
                    if retrieved_content == golden_content:
                        chunks_found += 1
                        break
            
            query_score = chunks_found / len(golden_contents)
            total_score += query_score
        
        average_score = total_score / total_queries
        pass_at_k = average_score * 100
        
        return pass_at_k

def main():
    # Initialize evaluator
    evaluator = RAGEvaluator(
        chunks_path='eval/codebase_chunks.json',
        eval_set_path='eval/evaluation_set.jsonl'
    )
    
    # Prepare vector database
    print("Preparing vector database...")
    vector_db = evaluator.prepare_vector_db()
    
    # Run evaluation
    k_values = [5, 10, 20]
    results = evaluator.evaluate(vector_db, k_values)
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 20)
    for k, score in results.items():
        print(f"Pass@{k}: {score:.2f}%")

if __name__ == "__main__":
    main()