#<PGVECTOR-CLASS>
import os
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from typing import List, Tuple, Dict, Any
import json

load_dotenv()

class PGVector:
    """PostgreSQL vector database handler with pgvector extension"""
    
    def __init__(self):
        """Initialize PGVector with database setup"""
        self.conn = None
        self.setup_database()
        
    def setup_database(self):
        """Set up the database, extensions, and required tables"""
        # Connect to PostgreSQL server (without specific database)
        db_config = {
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
        }
        
        # Create database if it doesn't exist
        server_conn = psycopg2.connect(**db_config)
        server_conn.autocommit = True
        
        cursor = server_conn.cursor()
        db_name = os.getenv("DB_NAME")
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (db_name,))
        if not cursor.fetchone():
            cursor.execute(f"CREATE DATABASE {db_name};")
            print(f"Database '{db_name}' created.")
        
        cursor.close()
        server_conn.close()
        
        # Connect to the specific database and set up tables
        db_config["dbname"] = db_name
        conn = psycopg2.connect(**db_config)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Create pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Drop existing table if exists (during development)
        cursor.execute("DROP TABLE IF EXISTS embeddings;")
        
        # Create table with text-based metadata storage
        cursor.execute("""
            CREATE TABLE embeddings (
                id SERIAL PRIMARY KEY,
                doc_fragment TEXT NOT NULL,
                embeddings vector(4096),
                metadata TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create index for vector similarity search
        try:
            cursor.execute("""
                CREATE INDEX ON embeddings 
                USING ivfflat (embeddings vector_cosine_ops)
                WITH (lists = 100);
            """)
        except psycopg2.Error as e:
            print(f"Warning: Could not create vector index: {e}")
        
        cursor.close()
        conn.close()
        print("Database setup completed successfully.")

    def get_db_connection(self):
        """Establish database connection and register vector extension"""
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(
                dbname=os.getenv('DB_NAME'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD'),
                host=os.getenv('DB_HOST'),
                port=os.getenv('DB_PORT')
            )
            register_vector(self.conn)

    def insert_embeddings(self, embeddings_data: List[Tuple[str, list, dict]]):
        """
        Insert embeddings with their corresponding text and metadata into the database.
        
        Args:
            embeddings_data: List of tuples containing (text, embedding_vector, metadata)
        """
        self.get_db_connection()
        cursor = self.conn.cursor()
        
        try:
            for text, vector, metadata in embeddings_data:
                # Convert metadata dict to JSON string
                metadata_json = json.dumps(metadata)
                cursor.execute(
                    """
                    INSERT INTO embeddings (doc_fragment, embeddings, metadata)
                    VALUES (%s, %s, %s)
                    """,
                    (text, vector, metadata_json)
                )
            self.conn.commit()
            print(f"Successfully inserted {len(embeddings_data)} embeddings")
            
        except Exception as e:
            self.conn.rollback()
            print(f"Error during insertion: {str(e)}")
            raise e
        finally:
            cursor.close()

    def search_similar(self, query_embedding: list, limit: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query_embedding: The query embedding vector
            limit: Maximum number of results to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of dictionaries containing matched documents and their metadata
        """
        self.get_db_connection()
        cursor = self.conn.cursor()
        
        try:
            print(f"\nExecuting similarity search with threshold {threshold}")
            # Use CTE for better query organization
            cursor.execute(
                """
                WITH similarity_results AS (
                    SELECT 
                        doc_fragment,
                        metadata,
                        1 - (embeddings <=> %s::vector) as similarity
                    FROM embeddings
                )
                SELECT * FROM similarity_results
                WHERE similarity > %s
                ORDER BY similarity DESC
                LIMIT %s;
                """,
                (str(query_embedding), threshold, limit)
            )
            
            results = []
            for doc, metadata_str, similarity in cursor.fetchall():
                print(f"Found match with similarity: {similarity}")
                try:
                    # Parse the metadata string back into a dict
                    metadata = json.loads(metadata_str) if metadata_str else {}
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode metadata: {metadata_str}")
                    metadata = {}
                    
                results.append({
                    'content': doc,
                    'metadata': metadata,
                    'similarity': float(similarity)
                })
            
            return results
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            raise e
        finally:
            cursor.close()

    def get_total_embeddings(self) -> int:
        """Get total number of embeddings stored in the database"""
        self.get_db_connection()
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM embeddings;")
            return cursor.fetchone()[0]
        finally:
            cursor.close()

    def clear_embeddings(self):
        """Clear all embeddings from the database"""
        self.get_db_connection()
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("TRUNCATE TABLE embeddings;")
            self.conn.commit()
            print("All embeddings cleared from database")
        finally:
            cursor.close()

    def close_connection(self):
        """Close the database connection"""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close_connection()

    def __del__(self):
        """Cleanup on deletion"""
        self.close_connection()


if __name__== "__main__":
    pgvector = PGVector()
    pgvector.setup_database()