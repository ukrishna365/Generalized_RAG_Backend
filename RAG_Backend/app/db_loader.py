import os
import uuid
import json
import numpy as np
import glob
from sqlalchemy import create_engine, Column, String, Text, TIMESTAMP, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURABLE PATHS (edit these as needed) ---
OUTPUT_FOLDER = r"C:\Users\kuppa\DS_Projects\Generalized_RAG_Backend3\RAG_Backend\outputs"
DATABASE_URL = os.getenv("POSTGRES_URL", "postgresql+psycopg2://user:password@localhost/your_db")
# --------------------------------------------------

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
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# --- Loader Function ---
def load_raganything_output(json_path, file_name, file_type):
    session = Session()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Starting database load for {file_name}")
    
    # 1. Insert file record
    file_row = File(file_name=file_name, file_type=file_type)
    session.add(file_row)
    session.commit()  # To get file_id
    file_id = file_row.file_id

    # 2. Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 3. Process content_list.json structure
    text_chunks = []
    image_data = []
    
    if isinstance(data, list):
        # Handle content_list.json format
        for item in data:
            if item.get('type') == 'text':
                text_chunks.append({
                    'text': item.get('text', ''),
                    'page_idx': item.get('page_idx', 0),
                    'text_level': item.get('text_level', None)
                })
            elif item.get('type') == 'image':
                image_data.append({
                    'img_path': item.get('img_path', ''),
                    'caption': item.get('image_caption', []),
                    'page_idx': item.get('page_idx', 0)
                })
    
    # 4. Insert text chunks
    for chunk_data in text_chunks:
        chunk_id = uuid.uuid4()
        text_row = TextChunk(
            chunk_id=chunk_id, 
            file_id=file_id, 
            text_markdown=chunk_data['text']
        )
        session.add(text_row)
    
    # 5. Insert image data
    for img_data in image_data:
        image_id = uuid.uuid4()
        caption_text = ' '.join(img_data['caption']) if img_data['caption'] else ''
        image_row = ImageData(
            image_id=image_id,
            file_id=file_id,
            caption=caption_text
        )
        session.add(image_row)
    
    # 6. Commit all
    session.commit()
    session.close()
    
    final_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{final_timestamp}] Loaded {len(text_chunks)} text chunks and {len(image_data)} images from {json_path} into database.")

# --- Batch Processing Function ---
def process_all_outputs():
    """Process all content_list.json files in the outputs directory"""
    if not os.path.exists(OUTPUT_FOLDER):
        print(f"Output folder not found: {OUTPUT_FOLDER}")
        return
    
    # Find all content_list.json files
    content_list_pattern = os.path.join(OUTPUT_FOLDER, "**", "auto", "*_content_list.json")
    content_list_files = glob.glob(content_list_pattern, recursive=True)
    
    if not content_list_files:
        print(f"No content_list.json files found in {OUTPUT_FOLDER}")
        return
    
    print(f"Found {len(content_list_files)} files to process")
    
    for json_path in content_list_files:
        try:
            # Extract file name from path
            # Path: outputs/filename/auto/filename_content_list.json
            path_parts = json_path.split(os.sep)
            filename_with_ext = path_parts[-1].replace('_content_list.json', '')
            
            # Determine file type from original file extension
            file_type = "unknown"
            if filename_with_ext.endswith('.pdf'):
                file_type = "pdf"
            elif filename_with_ext.endswith('.docx'):
                file_type = "docx"
            elif filename_with_ext.endswith('.pptx'):
                file_type = "pptx"
            elif filename_with_ext.endswith('.md'):
                file_type = "md"
            
            load_raganything_output(json_path, filename_with_ext, file_type)
            
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            continue

# --- Example Usage ---
if __name__ == "__main__":
    # Process all files in outputs directory
    process_all_outputs() 