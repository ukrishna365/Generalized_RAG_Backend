import os
import sys
import json
import numpy as np
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Load environment variables
load_dotenv()

# --- CONFIGURABLE PATHS (edit these as needed) ---
DATABASE_URL = os.getenv("POSTGRES_URL", "postgresql+psycopg2://user:password@localhost/your_db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# --------------------------------------------------

# Database connection
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

class RAGQueryEngine:
    def __init__(self):
        self.session = Session()
        self.setup_embedding_model()
    
    def setup_embedding_model(self):
        """Initialize SPECTER2 embedding model"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
            self.model = AutoModel.from_pretrained("allenai/specter2_base")
            self.torch = torch  # Store torch reference
            print("SPECTER2 embedding model loaded successfully")
        except ImportError:
            print("Warning: transformers not available, using fallback embedding")
            self.model = None
            self.tokenizer = None
            self.torch = None
    
    def embed_query(self, query: str) -> List[float]:
        """Convert text query to embedding vector"""
        if self.model is None:
            # Fallback: simple hash-based embedding (not recommended for production)
            import hashlib
            hash_obj = hashlib.md5(query.encode())
            return [float(int(hash_obj.hexdigest()[i:i+2], 16)) / 255.0 for i in range(0, 32, 2)] * 24  # 768 dims
        
        # Use SPECTER2 for proper embedding
        inputs = self.tokenizer(query, return_tensors="pt", max_length=512, truncation=True, padding=True)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        return embedding.tolist()
    
    def retrieve_relevant_content(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant content from database using vector similarity"""
        # For now, we'll do a simple text-based search
        # In production, you'd use pgvector for proper vector similarity search
        
        query = """
        SELECT 
            f.file_name,
            tc.text_markdown,
            id.caption as image_caption,
            'text' as content_type
        FROM file f
        LEFT JOIN text_chunk tc ON f.file_id = tc.file_id
        LEFT JOIN image_data id ON f.file_id = id.file_id
        WHERE tc.text_markdown IS NOT NULL
        ORDER BY f.uploaded_at DESC
        LIMIT :top_k
        """
        
        try:
            result = self.session.execute(text(query), {"top_k": top_k})
            rows = []
            for row in result:
                # Convert SQLAlchemy Row to dictionary
                row_dict = {
                    'file_name': row.file_name,
                    'text_markdown': row.text_markdown,
                    'image_caption': row.image_caption,
                    'content_type': row.content_type
                }
                rows.append(row_dict)
            return rows
        except Exception as e:
            print(f"Database query error: {e}")
            return []
    
    def generate_answer(self, query: str, relevant_content: List[Dict[str, Any]]) -> str:
        """Generate answer using OpenAI API"""
        if not OPENAI_API_KEY:
            return "Error: OPENAI_API_KEY not set"
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Prepare context from retrieved content
            context_parts = []
            for content in relevant_content:
                if content.get('text_markdown'):
                    context_parts.append(f"Text: {content['text_markdown']}")
                if content.get('image_caption'):
                    context_parts.append(f"Image: {content['image_caption']}")
            
            context = "\n\n".join(context_parts)
            
            # Create prompt
            prompt = f"""Based on the following context, answer the user's question.

Context:
{context}

Question: {query}

Answer:"""
            
            # Call OpenAI API with new format
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def query(self, user_question: str) -> Dict[str, Any]:
        """Main query function - complete RAG pipeline"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Processing query: {user_question}")
        
        # Step 1: Embed the query
        query_embedding = self.embed_query(user_question)
        
        # Step 2: Retrieve relevant content
        relevant_content = self.retrieve_relevant_content(query_embedding)
        
        # Step 3: Generate answer
        answer = self.generate_answer(user_question, relevant_content)
        
        # Step 4: Prepare response
        response = {
            "query": user_question,
            "answer": answer,
            "sources": relevant_content,
            "timestamp": timestamp
        }
        
        final_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{final_timestamp}] Query completed")
        
        return response
    
    def close(self):
        """Close database session"""
        self.session.close()

# --- Example Usage ---
if __name__ == "__main__":
    # Test the query engine
    engine = RAGQueryEngine()
    
    test_questions = [
        "What is the main topic of the documents?",
        "What are the key findings?",
        "Can you summarize the content?"
    ]
    
    for question in test_questions:
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        result = engine.query(question)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {len(result['sources'])} items retrieved")
    
    engine.close() 