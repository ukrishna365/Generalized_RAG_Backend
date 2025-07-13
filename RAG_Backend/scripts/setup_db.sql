-- Create the `file` table to track uploads
CREATE TABLE IF NOT EXISTS file (
    file_id UUID PRIMARY KEY,
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    uploaded_at TIMESTAMP DEFAULT now()
);

-- Store text chunks parsed from documents
CREATE TABLE IF NOT EXISTS text_chunk (
    chunk_id UUID PRIMARY KEY,
    file_id UUID REFERENCES file(file_id) ON DELETE CASCADE,
    text_markdown TEXT NOT NULL
);

-- Store table data extracted from documents (markdown or csv-style text)
CREATE TABLE IF NOT EXISTS table_data (
    table_id UUID PRIMARY KEY,
    file_id UUID REFERENCES file(file_id) ON DELETE CASCADE,
    table_markdown TEXT NOT NULL,
    caption TEXT
);

-- Store images extracted from documents, indexed by caption
CREATE TABLE IF NOT EXISTS image_data (
    image_id UUID PRIMARY KEY,
    file_id UUID REFERENCES file(file_id) ON DELETE CASCADE,
    caption TEXT
);

-- Store embeddings (vectors) tied to chunks, tables, or image captions
CREATE TABLE IF NOT EXISTS vector_data (
    vector_id UUID PRIMARY KEY,
    file_id UUID REFERENCES file(file_id) ON DELETE CASCADE,
    source_id UUID NOT NULL,
    modality TEXT NOT NULL CHECK (modality IN ('text', 'table', 'image')),
    embedder_name TEXT NOT NULL,
    embedding vector(768)
);
