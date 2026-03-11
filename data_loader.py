from llama_index.core import VectorStoreIndex
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline

EMBED_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
EMBED_DIM = 384

llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=200
)

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=0)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    nodes = splitter.get_nodes_from_documents(docs)
    chunks = [node.get_content() for node in nodes]
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:

    embeddings = EMBED_MODEL.encode(texts)

    return embeddings.tolist()