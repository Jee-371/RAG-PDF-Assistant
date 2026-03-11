import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf, embed_texts
from vector_DB import QdrantStorage
from custom_types import RAGChunkSrc, RAGQuery, RAGSearch, RAGUpsert
from transformers import pipeline

llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

inngest_client = inngest.Inngest(
    app_id="rag_app", 
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id='RAG: Ingest PDF',
    trigger=inngest.TriggerEvent(event='rag/ingest_pdf')
)

async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id",pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkSrc) -> RAGUpsert:
        chunks = chunks_and_src.chunks
        source_id= chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsert(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load and chunk", lambda: _load(ctx), output_type=RAGChunkSrc)
    ingested = await ctx.step.run("embed and upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsert)
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id=  "RAG: Query PDF",
    trigger= inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearch:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearch(contexts=found['contexts'], sources=found['sources'])
    
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearch)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )
    response = llm(
    f"""You answer questions using only the provided context.

    {user_content}
    """,
        max_new_tokens=1024,
        temperature=0.2
    )

    answer = response[0]["generated_text"].strip()

    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts)
    }

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])