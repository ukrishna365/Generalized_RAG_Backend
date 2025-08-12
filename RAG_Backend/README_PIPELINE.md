# Complete RAG Pipeline Implementation

This document describes the complete RAG (Retrieval-Augmented Generation) pipeline that processes documents, stores them in PostgreSQL, and provides query capabilities.

## 🏗️ Pipeline Overview

```
User Upload → File Extraction → Database Storage → Query Interface → Answer Generation
```

### **Components:**

1. **File Extractor** (`file_extractor2.py`) - Processes PDF, DOCX, PPTX, MD files
2. **Database Loader** (`db_loader.py`) - Stores extracted content in PostgreSQL
3. **Query Engine** (`query_engine.py`) - Handles RAG queries and retrieval
4. **Web Interface** (`web_interface.py`) - Flask web app for queries
5. **CLI Interface** (`cli_interface.py`) - Command-line query interface

## 📋 Prerequisites

### **Required Software:**
- Python 3.8+
- PostgreSQL with pgvector extension
- LibreOffice (for Office document processing)

### **Environment Variables:**
Create a `.env` file in the project root:
```env
POSTGRES_URL=postgresql+psycopg2://username:password@localhost/database_name
OPENAI_API_KEY=your_openai_api_key_here
```

## 🚀 Quick Start

### **1. Install Dependencies**
```bash
# For file extraction
pip install -r requirements.txt

# For web interface
pip install -r requirements_web.txt
```

### **2. Set Up Database**
```bash
# Run the SQL setup script in pgAdmin or psql
psql -d your_database -f RAG_Backend/scripts/setup_db.sql
```

### **3. Process Documents**
```bash
# Extract content from files in inputs folder
python RAG_Backend/app/file_extractor2.py

# Load extracted content into database
python RAG_Backend/app/db_loader.py
```

### **4. Start Query Interface**

**Option A: Web Interface**
```bash
python RAG_Backend/app/web_interface.py
# Open http://localhost:5000 in your browser
```

**Option B: CLI Interface**
```bash
python RAG_Backend/app/cli_interface.py
```

## 📁 File Structure

```
RAG_Backend/
├── app/
│   ├── file_extractor2.py      # Document processing
│   ├── db_loader.py            # Database storage
│   ├── query_engine.py         # RAG query engine
│   ├── web_interface.py        # Flask web app
│   └── cli_interface.py        # Command-line interface
├── inputs/                     # Place documents here
├── outputs/                    # Extracted content
├── scripts/
│   └── setup_db.sql           # Database schema
└── requirements_web.txt        # Web dependencies
```

## 🔧 Configuration

### **File Extractor Settings** (`file_extractor2.py`)
```python
# --- CONFIGURABLE PATHS ---
INPUT_FOLDER = r"C:\path\to\inputs"
OUTPUT_FOLDER = r"C:\path\to\outputs"
RAG_STORAGE_DIR = r"C:\path\to\rag_storage"
LIBREOFFICE_PATH = r"C:\Program Files\LibreOffice\program"
MAX_WORKERS = 4  # Adjust based on CPU cores
# --------------------------------------------------
```

### **Database Loader Settings** (`db_loader.py`)
```python
# --- CONFIGURABLE PATHS ---
OUTPUT_FOLDER = r"C:\path\to\outputs"
DATABASE_URL = os.getenv("POSTGRES_URL", "postgresql://...")
# --------------------------------------------------
```

### **Query Engine Settings** (`query_engine.py`)
```python
# --- CONFIGURABLE PATHS ---
DATABASE_URL = os.getenv("POSTGRES_URL", "postgresql://...")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# --------------------------------------------------
```

## 🔍 Usage Examples

### **Processing Documents**
```bash
# 1. Place documents in inputs folder
# 2. Run extraction
python RAG_Backend/app/file_extractor2.py

# 3. Load to database
python RAG_Backend/app/db_loader.py
```

### **Querying via Web Interface**
1. Start the web server: `python RAG_Backend/app/web_interface.py`
2. Open http://localhost:5000
3. Enter your question in the text area
4. Click "Submit Query"

### **Querying via CLI**
```bash
python RAG_Backend/app/cli_interface.py
# Then type questions interactively
```

## 🗄️ Database Schema

The pipeline creates these tables:

- **`file`** - Document metadata
- **`text_chunk`** - Extracted text content
- **`image_data`** - Image captions
- **`table_data`** - Table content
- **`vector_data`** - Embeddings (for future vector search)

## 🔧 Advanced Configuration

### **Vector Search (Future Enhancement)**
To enable proper vector similarity search:

1. Install pgvector extension in PostgreSQL
2. Update `query_engine.py` to use vector similarity queries
3. Generate embeddings for all content during extraction

### **Custom Embedding Models**
Replace SPECTER2 with other models:
```python
# In query_engine.py
self.tokenizer = AutoTokenizer.from_pretrained("your-model")
self.model = AutoModel.from_pretrained("your-model")
```

### **Alternative LLM Providers**
Replace OpenAI with other providers:
```python
# In query_engine.py, modify generate_answer() method
# Use different API calls for other providers
```

## 🐛 Troubleshooting

### **Common Issues:**

1. **LibreOffice not found**
   - Install LibreOffice and add to PATH
   - Update `LIBREOFFICE_PATH` in config

2. **Database connection errors**
   - Check `POSTGRES_URL` in `.env`
   - Ensure PostgreSQL is running
   - Verify database exists

3. **OpenAI API errors**
   - Check `OPENAI_API_KEY` in `.env`
   - Verify API key is valid
   - Check API quota

4. **Memory issues during processing**
   - Reduce `MAX_WORKERS` in file extractor
   - Process smaller batches
   - Increase system RAM

## 📊 Performance Tips

1. **File Processing**: Use 4-6 workers for 8-core systems
2. **Database**: Use SSD storage for PostgreSQL
3. **Embeddings**: Use GPU for SPECTER2 model
4. **Web Interface**: Use production WSGI server for deployment

## 🔄 Complete Pipeline Workflow

1. **Upload** documents to `inputs/` folder
2. **Extract** content using `file_extractor2.py`
3. **Store** in PostgreSQL using `db_loader.py`
4. **Query** via web interface or CLI
5. **Generate** answers using OpenAI API

## 📈 Monitoring

Check database for processed content:
```sql
-- View all processed files
SELECT * FROM file ORDER BY uploaded_at DESC;

-- Count text chunks per file
SELECT f.file_name, COUNT(tc.chunk_id) as chunk_count
FROM file f LEFT JOIN text_chunk tc ON f.file_id = tc.file_id
GROUP BY f.file_id, f.file_name;
``` 