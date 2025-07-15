import os
import json
from typing import Dict, Any, Optional
import asyncio
import glob

try:
    from raganything import RAGAnything, RAGAnythingConfig
except ImportError:
    raise ImportError("RAGAnything must be installed. Run 'pip install raganything'.")

# Import the OpenAI LLM function from LightRAG
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc

# SPECTER2 imports
from transformers.models.auto.tokenization_auto import AutoTokenizer
from adapters import AutoAdapterModel
import torch

DEFAULT_OUTPUT_DIR = r"C:\Users\kuppa\DS_Projects\Generalized_RAG_Backend3\RAG_Backend\outputs"

# Define the LLM function using GPT-4o and the API key from the environment
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

# SPECTER2 embedding setup (load once)
specter2_model_name = "allenai/specter2_base"
specter2_adapter = "allenai/specter2"
tokenizer = AutoTokenizer.from_pretrained(specter2_model_name)
model = AutoAdapterModel.from_pretrained(specter2_model_name)
model.load_adapter(specter2_adapter, source="hf", load_as="proximity", set_active=True)
model.set_active_adapters("proximity")
model.eval()

def specter2_embed(texts):
    batch = [{"title": t, "abstract": ""} for t in texts]
    text_batch = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in batch]
    inputs = tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
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

async def extract_structured_content(filepath: str, output_dir: str = DEFAULT_OUTPUT_DIR) -> Optional[Dict[str, Any]]:
    """
    Extract structured content from a file using RAG-Anything.
    The function processes the file and loads the resulting structured content from the output directory.
    Returns a dictionary with keys like 'text', 'images', 'tables', 'equations', etc.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    os.makedirs(output_dir, exist_ok=True)

    # Initialize RAGAnything with OpenAI GPT-4o as the LLM function and SPECTER2 as the embedding function
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
        # Look for a directory named after the file base name
        base = os.path.splitext(os.path.basename(filepath))[0]
        output_dir_path = os.path.join(output_dir, base)
        if os.path.isdir(output_dir_path):
            json_files = glob.glob(os.path.join(output_dir_path, '*.json'))
            if json_files:
                with open(json_files[0], "r", encoding="utf-8") as f:
                    result = json.load(f)
                return result
            else:
                print(f"No JSON files found in {output_dir_path}")
                return None
        else:
            print(f"Output directory not found: {output_dir_path}")
            return None
    except Exception as e:
        print(f"Error extracting structured content from {filepath}: {e}")
        return None

# Synchronous wrapper for convenience
def extract_structured_content_sync(filepath: str, output_dir: str = DEFAULT_OUTPUT_DIR) -> Optional[Dict[str, Any]]:
    """
    Synchronous wrapper for extract_structured_content.
    """
    return asyncio.run(extract_structured_content(filepath, output_dir)) 