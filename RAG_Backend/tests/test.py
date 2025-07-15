from raganything import RAGAnything, RAGAnythingConfig
import os
import glob
import asyncio
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from transformers.models.auto.tokenization_auto import AutoTokenizer
from adapters import AutoAdapterModel
import torch

PDF_PATH = r"C:\Users\kuppa\DS_Projects\Generalized_RAG_Backend3\RAG_Backend\inputs\test123.pdf"
OUTPUT_DIR = r"C:\Users\kuppa\DS_Projects\Generalized_RAG_Backend3\RAG_Backend\outputs\test123_pdf"

# SPECTER2 embedding and LLM setup
specter2_model_name = "allenai/specter2_base"
specter2_adapter = "allenai/specter2"
tokenizer = AutoTokenizer.from_pretrained(specter2_model_name)
model = AutoAdapterModel.from_pretrained(specter2_model_name)
model.load_adapter(specter2_adapter, source="hf", load_as="proximity", set_active=True)
model.set_active_adapters("proximity")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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

async def main():
    rag = RAGAnything(
        config=RAGAnythingConfig(working_dir="./rag_storage"),
        llm_model_func=gpt4o_llm_model_func,
        embedding_func=embedding_func
    )
    try:
        await rag.process_document_complete(
            file_path=PDF_PATH,
            output_dir=OUTPUT_DIR,
            parse_method="auto"
        )
        print("MinerU/RAG-Anything extraction completed.")
        # List all files in the output directory
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                print("Output file:", os.path.join(root, file))
        # Print found JSON files
        json_files = glob.glob(os.path.join(OUTPUT_DIR, '**', '*.json'), recursive=True)
        print("Found JSON files:", json_files)
    except Exception as e:
        print("Error during extraction:", e)

if __name__ == "__main__":
    asyncio.run(main())
