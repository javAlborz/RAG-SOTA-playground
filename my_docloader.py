from typing import List, Dict
from pathlib import Path
import pandas as pd
import json
import PyPDF2

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
        elif file_extension in ['.txt', '.md']:  # Handle .md same as .txt
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