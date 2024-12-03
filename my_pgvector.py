import os
import psycopg2
from dotenv import load_dotenv
import requests
import numpy as np
from pgvector.psycopg2 import register_vector

load_dotenv()

class PGVector:
    def __init__(self):
        self.conn = None
        self.create_db()

    def create_db(self):
        db_config = {
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
        }

        conn = psycopg2.connect(**db_config)
        conn.autocommit = True  # Enable autocommit for creating the database

        cursor = conn.cursor()
        cursor.execute(
            f"SELECT 1 FROM pg_database WHERE datname = '{os.getenv('DB_NAME')}';"
        )
        database_exists = cursor.fetchone()
        cursor.close()

        if not database_exists:
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE {os.getenv('DB_NAME')};")
            cursor.close()
            print("Database created.")

        conn.close()
        db_config["dbname"] = os.getenv("DB_NAME")
        conn = psycopg2.connect(**db_config)
        conn.autocommit = True

        cursor = conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.close()

        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (id serial PRIMARY KEY, doc_fragment text, embeddings vector(4096));"
        )
        cursor.close()

        print("Database setup completed.")

    def get_db_connection(self):
        """
        Returns a connection object to the PostgreSQL database.

        The connection details are read from environment variables:
        - DB_NAME
        - DB_USER
        - DB_PASSWORD
        - DB_HOST
        - DB_PORT
        """
        self.conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )
        register_vector(self.conn)  # Register the vector type for the connection

    def insert_embeddings(self, embeddings):
        """
        Inserts a list of embeddings into the database.

        Args:
            embeddings (list): A list of tuples containing the text and embedding to insert.
        """
        if self.conn is None:
            self.get_db_connection()

        cur = self.conn.cursor()

        # Insert each embedding into the database
        for i, (text, vector) in enumerate(embeddings):
            cur.execute(
                "INSERT INTO embeddings (id, doc_fragment, embeddings) VALUES (%s, %s, %s)",
                (i, text, vector)
            )

        self.conn.commit()
        cur.close()

    def close_connection(self):
        if self.conn is not None:
            self.conn.close()

    def get_retrieval_condition(self, query_embedding, threshold=0.7):
        """
        Generate a SQL condition for retrieving relevant embeddings from a database.
        Args:
            query_embedding (numpy array): The query embedding.
            threshold (float, optional): The minimum cosine similarity required for an embedding to be considered relevant. Defaults to 0.7.
        Returns:
            str: The SQL condition for retrieving relevant embeddings.
        """
        # Convert query embedding to a string format for SQL query
        query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
        # SQL condition for cosine similarity
        condition = f"embeddings <=> '{query_embedding_str}' < {threshold} ORDER BY embeddings <=> '{query_embedding_str}'"
        return condition

    def rag_query(self, query):
        """
        Perform a RAG query to answer a question.
        Args:
            query (str): The question to answer.
        Returns:
            str: The answer to the question.
        """
        # Generate query embedding using the endpoint
        _, query_embedding = generate_embeddings(query)
        # Retrieve relevant embeddings from the database
        retrieval_condition = self.get_retrieval_condition(query_embedding)
        if self.conn is None:
            self.get_db_connection()
        cursor = self.conn.cursor()
        cursor.execute(
            f"SELECT doc_fragment FROM embeddings WHERE {retrieval_condition} LIMIT 5"
        )
        retrieved = cursor.fetchall()
        rag_query = ' '.join([row[0] for row in retrieved])
        query_template = self.template.format(context=rag_query, question=query)
        # Use the endpoint to generate the response
        llm_endpoint = os.getenv('LLM_ENDPOINT_URL')
        api_key = os.getenv('API_KEY')
        model_name = os.getenv('MODEL_NAME')  # Adjust this based on your LLM endpoint
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "messages": [
                {"role": "system", "content": "You are a friendly documentation search bot."},
                {"role": "user", "content": query_template}
            ],
            "model": model_name  # Adjust this based on your LLM endpoint
        }
        try:
            response = requests.post(llm_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            if 'output' in result:
                return result['output']
            elif 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                print(f"Unexpected response format: {result}")
                return "I'm sorry, but I couldn't generate a response. Please try again."
        except requests.exceptions.RequestException as e:
            print(f"Error generating response: {e}")
            print(f"Response content: {response.content}")
            return "I'm sorry, but there was an error processing your request. Please try again later."

