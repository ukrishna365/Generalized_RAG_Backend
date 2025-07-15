# --- CONFIGURABLE PATHS (edit these as needed) ---
INPUT_FOLDER = r"C:\Users\kuppa\DS_Projects\Generalized_RAG_Backend3\RAG_Backend\inputs"
OUTPUT_FOLDER = r"C:\Users\kuppa\DS_Projects\Generalized_RAG_Backend3\RAG_Backend\outputs"
USE_GPU = True  # Set to False to force CPU mode
# --------------------------------------------------

import os
import json
from typing import Dict, Any, Optional
import asyncio
import glob
import argparse
import torch

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
tokenizer = AutoTokenizer.from_pretrained(specter2_model_name)
model = AutoAdapterModel.from_pretrained(specter2_model_name)
model.load_adapter(specter2_adapter, source="hf", load_as="proximity", set_active=True)
model.set_active_adapters("proximity")

# Set device based on USE_GPU
if USE_GPU and torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    model.to(device)
    print("Using CPU")

def gpt4o_llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment.")
    return openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=api_key,
        **kwargs
    )

def specter2_embed(texts):
    batch = [{"title": t, "abstract": ""} for t in texts]
    text_batch = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in batch]
    inputs = tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embeddings

async def specter2_embed_async(texts):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, specter2_embed, texts)

embedding_func = EmbeddingFunc(
    embedding_dim=768,
    max_token_size=512,
    func=specter2_embed_async
)

async def extract_structured_content(filepath: str, output_dir: str = OUTPUT_FOLDER) -> Optional[Dict[str, Any]]:
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    os.makedirs(output_dir, exist_ok=True)
    rag = RAGAnything(
        config=RAGAnythingConfig(working_dir="./rag_storage"),
        llm_model_func=gpt4o_llm_model_func,
        embedding_func=embedding_func
    )
    try:
        await rag.process_document_complete(
            file_path=filepath,
            output_dir=output_dir,
            parse_method="auto"
        )
        print("Files in output directory after processing:", os.listdir(output_dir))
        base = os.path.splitext(os.path.basename(filepath))[0]
        output_dir_path = os.path.join(output_dir, base)
        # Recursively search for any .json file in the output directory
        json_files = glob.glob(os.path.join(output_dir_path, '**', '*.json'), recursive=True)
        print("Found JSON files:", json_files)
        if json_files:
            with open(json_files[0], "r", encoding="utf-8") as f:
                result = json.load(f)
            return result
        else:
            print(f"No JSON files found in {output_dir_path}")
            return None
    except Exception as e:
        print(f"Error extracting structured content from {filepath}: {e}")
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