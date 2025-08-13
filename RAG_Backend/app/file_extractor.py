# --- CONFIGURABLE PATHS (edit these as needed) ---
INPUT_FOLDER = r"C:\Users\kuppa\DS_Projects\Generalized_RAG_Backend3\RAG_Backend\inputs"
OUTPUT_FOLDER = r"C:\Users\kuppa\DS_Projects\Generalized_RAG_Backend3\RAG_Backend\outputs"
RAG_STORAGE_DIR = r"C:\Users\kuppa\DS_Projects\Generalized_RAG_Backend3\rag_storage"
LIBREOFFICE_PATH = r"C:\Program Files\LibreOffice\program"
USE_GPU = True  # Set to False to force CPU mode

# Performance tuning parameters
PROCESSING_TIMEOUT = 600  # 10 minutes timeout for document processing
MAX_RETRIES = 1  # Number of retries for failed processing
SKIP_EQUATION_ANALYSIS = True  # Skip equation analysis for PPTX files to avoid JSON errors
EMBEDDING_BATCH_SIZE = 2  # Reduced batch size to avoid OpenAI rate limits (was 16)

# OpenAI configuration
OPENAI_MODEL = "gpt-4o"  # Primary model
# --------------------------------------------------

import os
import json
from typing import Dict, Any, Optional
import asyncio
import glob
import argparse
import time
from datetime import datetime
from contextlib import contextmanager
import torch
import signal

# Lightweight logging helpers
def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(message: str) -> None:
    print(f"[{_now_str()}] {message}")

@contextmanager
def stage(stage_name: str):
    start = time.perf_counter()
    log(f"START {stage_name}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log(f"END   {stage_name} (took {elapsed:.2f}s)")

try:
    from raganything import RAGAnything, RAGAnythingConfig
except ImportError:
    raise ImportError("RAGAnything must be installed. Run 'pip install raganything'.")

from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from transformers.models.auto.tokenization_auto import AutoTokenizer
from adapters import AutoAdapterModel

SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.md'}

# SPECTER2 embedding and LLM setup
specter2_model_name = "allenai/specter2_base"
specter2_adapter = "allenai/specter2"
with stage("Load tokenizer"):
    tokenizer = AutoTokenizer.from_pretrained(specter2_model_name)
with stage("Load base model"):
    model = AutoAdapterModel.from_pretrained(specter2_model_name)
with stage("Load & activate adapter"):
    model.load_adapter(specter2_adapter, source="hf", load_as="proximity", set_active=True)
    model.set_active_adapters("proximity")

# Set device based on USE_GPU
with stage("Move model to device"):
    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        log(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Clear GPU cache and set memory management
        torch.cuda.empty_cache()
        log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB total")
    else:
        device = torch.device("cpu")
        model.to(device)
        log("Using CPU")

# Print configuration info
log(f"Using rag_storage at: {RAG_STORAGE_DIR}")

# Set LibreOffice path for subprocesses
if os.path.exists(LIBREOFFICE_PATH):
    # Add LibreOffice to environment variables for subprocesses
    os.environ['PATH'] = LIBREOFFICE_PATH + os.pathsep + os.environ.get('PATH', '')
    log(f"Added LibreOffice to PATH: {LIBREOFFICE_PATH}")
else:
    log(f"Warning: LibreOffice not found at {LIBREOFFICE_PATH}")

def gpt4o_llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment.")
    
    # Add rate limit handling
    max_retries = 3
    base_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            return openai_complete_if_cache(
                OPENAI_MODEL,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                **kwargs
            )
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                log(f"OpenAI quota exceeded (attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    log("Max retries reached for OpenAI API. Check your billing and quota limits.")
                    raise
                # Exponential backoff
                delay = base_delay * (2 ** attempt)
                log(f"Waiting {delay}s before retry...")
                time.sleep(delay)
            else:
                raise

def safe_equation(equation: str) -> str:
    """Escape backslashes and control characters for JSON safety."""
    if not isinstance(equation, str):
        return equation
    return equation.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")

def specter2_embed(texts):
    # Dynamic batch sizing to avoid OpenAI rate limits
    batch_size = EMBEDDING_BATCH_SIZE  # Start with configured batch size
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch = [{"title": t, "abstract": ""} for t in batch_texts]
        text_batch = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in batch]
        
        # Estimate token count for this batch
        estimated_tokens = sum(len(t.split()) * 1.3 for t in batch_texts)  # Rough estimate
        log(f"Processing batch {i//batch_size + 1}: {len(batch_texts)} texts, ~{estimated_tokens:.0f} tokens")
        
        inputs = tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Monitor GPU memory before processing
        if device.type == 'cuda':
            gpu_memory_before = torch.cuda.memory_allocated() / 1024**2
            log(f"GPU Memory before batch {i//batch_size + 1}: {gpu_memory_before:.1f}MB")
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.extend(embeddings)
        
        # Clear GPU cache after each batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gpu_memory_after = torch.cuda.memory_allocated() / 1024**2
            log(f"GPU Memory after batch {i//batch_size + 1}: {gpu_memory_after:.1f}MB")
        
        # Add small delay between batches to avoid rate limits
        time.sleep(0.1)  # 100ms delay between batches
    
    return all_embeddings

async def specter2_embed_async(texts):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, specter2_embed, texts)

embedding_func = EmbeddingFunc(
    embedding_dim=768,
    max_token_size=512,
    func=specter2_embed_async
)

async def extract_structured_content(filepath: str, output_dir: str = OUTPUT_FOLDER) -> Optional[Dict[str, Any]]:
    log(f"Begin extract for: {filepath}")
    if not os.path.exists(filepath):
        log(f"File not found: {filepath}")
        return None
    
    # Check if output already exists to avoid reprocessing
    base = os.path.splitext(os.path.basename(filepath))[0]
    output_dir_path = os.path.join(output_dir, base)
    auto_dir_path = os.path.join(output_dir_path, 'auto')
    
    if os.path.exists(auto_dir_path):
        existing_json_files = glob.glob(os.path.join(auto_dir_path, '*.json'))
        if existing_json_files:
            log(f"Output already exists for {filepath}, skipping processing")
            # Still try to parse existing output
            pass
    
    os.makedirs(output_dir, exist_ok=True)
    with stage("RAGAnything init"):
        rag = RAGAnything(
            config=RAGAnythingConfig(working_dir=RAG_STORAGE_DIR),
            llm_model_func=gpt4o_llm_model_func,
            embedding_func=embedding_func
        )
    try:
        with stage("Process document (parse/convert/extract)"):
            # Add retry logic with timeout
            for attempt in range(MAX_RETRIES + 1):
                try:
                    log(f"Processing attempt {attempt + 1}/{MAX_RETRIES + 1} for {filepath}")
                    
                    # Set a reasonable timeout for document processing
                    await asyncio.wait_for(
                        rag.process_document_complete(
                            file_path=filepath,
                            output_dir=output_dir,
                            parse_method="auto"
                        ),
                        timeout=PROCESSING_TIMEOUT
                    )
                    log(f"Processing completed successfully on attempt {attempt + 1}")
                    break  # Success, exit retry loop
                    
                except asyncio.TimeoutError:
                    log(f"Processing timed out after {PROCESSING_TIMEOUT} seconds for {filepath} (attempt {attempt + 1})")
                    if attempt == MAX_RETRIES:
                        log(f"Max retries reached for {filepath}")
                        return None
                    continue
                    
                except Exception as e:
                    log(f"Error during document processing (attempt {attempt + 1}): {e}")
                    if attempt == MAX_RETRIES:
                        log(f"Max retries reached for {filepath}")
                        return None
                    continue
        log(f"Files in output directory after processing: {os.listdir(output_dir)}")
        base = os.path.splitext(os.path.basename(filepath))[0]
        output_dir_path = os.path.join(output_dir, base)
        
        # Check for auto/ subdirectory specifically
        auto_dir_path = os.path.join(output_dir_path, 'auto')
        log(f"Looking for JSON files in: {output_dir_path}")
        log(f"Also checking auto subdirectory: {auto_dir_path}")
        
        # Recursively search for any .json file in the output directory
        with stage("Discover & parse JSON output"):
            # First try the auto/ subdirectory specifically
            if os.path.exists(auto_dir_path):
                json_files = glob.glob(os.path.join(auto_dir_path, '*.json'))
                log(f"Found JSON files in auto/ directory: {json_files}")
            else:
                # Fallback to recursive search
                json_files = glob.glob(os.path.join(output_dir_path, '**', '*.json'), recursive=True)
                log(f"Found JSON files (recursive search): {json_files}")
            
            if json_files:
                # Prefer content_list.json if available
                content_list_files = [f for f in json_files if 'content_list.json' in f]
                if content_list_files:
                    json_file = content_list_files[0]
                else:
                    json_file = json_files[0]
                
                log(f"Using JSON file: {json_file}")
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        raw_content = f.read()
                        try:
                            result = json.loads(raw_content)
                        except json.JSONDecodeError as e:
                            log(f"Error parsing output JSON: {e}")
                            log(f"Raw JSON content (first 1000 chars): {raw_content[:1000]}")
                            return None
                    return result
                except Exception as e:
                    log(f"Error reading/parsing JSON output for {filepath}: {e}")
                    return None
            else:
                log(f"No JSON files found in {output_dir_path} or {auto_dir_path}")
                return None
    except Exception as e:
        log(f"Error extracting structured content from {filepath}: {e}")
        return None

def extract_structured_content_sync(filepath: str, output_dir: str = OUTPUT_FOLDER) -> Optional[Dict[str, Any]]:
    return asyncio.run(extract_structured_content(filepath, output_dir))

def find_supported_files(input_folder):
    files = []
    for root, dirs, filenames in os.walk(input_folder):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(root, filename))
    return files

async def batch_extract(input_folder: str, output_folder: str):
    files = find_supported_files(input_folder)
    if not files:
        print(f"No supported files found in {input_folder}.")
        return
    print(f"Found {len(files)} files to process.")
    for file_path in files:
        print(f"Processing: {file_path}")
        result = await extract_structured_content(file_path, output_folder)
        if result:
            print(f"Extraction complete for: {file_path}")
        else:
            print(f"Extraction failed for: {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch extract files from a folder using RAG-Anything.")
    parser.add_argument('--input-folder', type=str, default=INPUT_FOLDER, help='Folder containing files to extract')
    parser.add_argument('--output-folder', type=str, default=OUTPUT_FOLDER, help='Folder to save outputs')
    args = parser.parse_args()
    asyncio.run(batch_extract(args.input_folder, args.output_folder)) 