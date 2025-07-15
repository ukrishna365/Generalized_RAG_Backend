import os
import uuid
import json
import numpy as np
from sqlalchemy import create_engine, Column, String, Text, TIMESTAMP, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- ORM Models ---
Base = declarative_base()

class File(Base):
    __tablename__ = 'file'
    file_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_name = Column(Text, nullable=False)
    file_type = Column(Text, nullable=False)
    uploaded_at = Column(TIMESTAMP, default=datetime.utcnow)

class TextChunk(Base):
    __tablename__ = 'text_chunk'
    chunk_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_id = Column(UUID(as_uuid=True), ForeignKey('file.file_id', ondelete='CASCADE'))
    text_markdown = Column(Text, nullable=False)

class TableData(Base):
    __tablename__ = 'table_data'
    table_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_id = Column(UUID(as_uuid=True), ForeignKey('file.file_id', ondelete='CASCADE'))
    table_markdown = Column(Text, nullable=False)
    caption = Column(Text)

class ImageData(Base):
    __tablename__ = 'image_data'
    image_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_id = Column(UUID(as_uuid=True), ForeignKey('file.file_id', ondelete='CASCADE'))
    caption = Column(Text)

class VectorData(Base):
    __tablename__ = 'vector_data'
    vector_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_id = Column(UUID(as_uuid=True), ForeignKey('file.file_id', ondelete='CASCADE'))
    source_id = Column(UUID(as_uuid=True), nullable=False)
    modality = Column(String, nullable=False)
    embedder_name = Column(String, nullable=False)
    embedding = Column(Vector(768))

# --- Database Connection ---
DATABASE_URL = os.getenv("POSTGRES_URL", "postgresql+psycopg2://user:password@localhost/your_db")
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# --- Loader Function ---
def load_raganything_output(json_path, file_name, file_type):
    session = Session()
    # 1. Insert file record
    file_row = File(file_name=file_name, file_type=file_type)
    session.add(file_row)
    session.commit()  # To get file_id
    file_id = file_row.file_id

    # 2. Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 3. Example: Insert text chunks and their embeddings
    # (You may need to adjust this depending on your JSON structure)
    chunks = data.get('chunks', []) if isinstance(data, dict) else data
    for chunk in chunks:
        chunk_id = uuid.uuid4()
        text = chunk.get('text', '')
        embedding = chunk.get('embedding', None)  # Should be a list of 768 floats
        # Insert text chunk
        text_row = TextChunk(chunk_id=chunk_id, file_id=file_id, text_markdown=text)
        session.add(text_row)
        # Insert embedding if present
        if embedding:
            vector_row = VectorData(
                file_id=file_id,
                source_id=chunk_id,
                modality='text',
                embedder_name='specter2',
                embedding=embedding
            )
            session.add(vector_row)
    # 4. Commit all
    session.commit()
    session.close()
    print(f"Loaded data from {json_path} into database.")

# --- Example Usage ---
if __name__ == "__main__":
    # Example: path to a model.json file from RAG-Anything output
    json_path = r"C:\Users\kuppa\DS_Projects\Generalized_RAG_Backend3\RAG_Backend\outputs\2506.23338v1\auto\2506.23338v1_model.json"
    file_name = "2506.23338v1.pdf"
    file_type = "pdf"
    load_raganything_output(json_path, file_name, file_type) 