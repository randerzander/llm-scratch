import lancedb
from lancedb.rerankers import RRFReranker
from sentence_transformers import SentenceTransformer
import numpy as np

embedding_model = SentenceTransformer("BAAI/BGE-Base-EN-v1.5", device='cuda')

def query_embed(q):
    embedding = embedding_model.encode([q], batch_size=1)[0]
    return embedding.astype(np.float32)

def query(q, user="1=1", channel="1=1", limit=25):
    if user != "1=1":
        user = f"(user = '{user}')"
    if channel != "1=1":
        channel = f"(channel = '{channel}')"
    results = (
        table.search(query_type="hybrid")
        .text(q)
        .vector(query_embed(q))
        .rerank(reranker=RRFReranker())  # Default k value
        .where(f"{user} and {channel}", prefilter=False)
        .limit(limit)
        .to_pandas()
    )
    return results

db = lancedb.connect("./vector_db")
table = db.open_table("test")

# Perform hybrid search with default reranker
res = query("What kind of laptops is randerzander interested in?")
print(res)
