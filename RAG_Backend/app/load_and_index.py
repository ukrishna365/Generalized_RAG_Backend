import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status(rag)  # <-- pass rag!
    return rag

async def main():
    rag = None
    try:
        rag = await initialize_rag()
        await rag.ainsert("The most popular AI agent framework of all time is probably Langchain.")
        await rag.ainsert("Under the Langchain hood we also have LangGraph, LangServe, and LangSmith.")
        await rag.ainsert("Many people prefer using other frameworks like Agno or Pydantic AI instead of Langchain.")
        await rag.ainsert("It is very easy to use Python with all of these AI agent frameworks.")

        # hybrid search
        mode = "hybrid"
        result = await rag.aquery(
            "What programming language should I use for coding AI agents?",
            param=QueryParam(mode=mode)
        )
        print(result)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())
