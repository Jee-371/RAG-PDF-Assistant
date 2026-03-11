import pydantic

class RAGChunkSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: str = None

class RAGUpsert(pydantic.BaseModel):
    ingested: int

class RAGSearch(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]

class RAGQuery(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int