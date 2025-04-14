import lancedb
from lancedb.rerankers import RRFReranker
from sentence_transformers import SentenceTransformer
import numpy as np
import time
from datetime import datetime, timedelta
import pandas as pd

from llm_scratch import gpt_4o_mini as gpt4
from llm_scratch import gemini as llm

model_name = "sentence-transformers/static-retrieval-mrl-en-v1"
embedding_model = SentenceTransformer(model_name)

def embed(q):
    embedding = embedding_model.encode([q], batch_size=1)[0]
    return embedding.astype(np.float32)

def base_query(q, user="1=1", channel="1=1", limit=25):
    if user != "1=1":
        user = f"(user = '{user}')"
    if channel != "1=1":
        channel = f"(channel = '{channel}')"
    results = (
        table.search(query_type="hybrid")
        .text(q)
        .vector(embed(q))
        .rerank(reranker=RRFReranker())  # Default k value
        .where(f"{user} and {channel}", prefilter=False)
        .limit(limit)
        .select(["row_id", "post", "user", "date", "channel"])
        .to_pandas()
    )

    return results

def get_post_window(table, result_row, post_window_size=25, hour_cutoff=6):
    # Extract row_id and channel from the result row
    row_id = result_row["row_id"]
    channel = result_row["channel"]
    original_date = datetime.fromisoformat(str(result_row["date"]))

    # Compute the range of row_ids to query
    min_row_id = max(0, row_id - post_window_size)
    max_row_id = row_id + post_window_size

    # Query LanceDB for rows in the specified range and channel
    results = (
        table.search()  # No query parameter needed for standard filtering
        .where(f"channel == '{channel}' AND row_id BETWEEN {min_row_id} AND {max_row_id}")
        .select(["row_id", "post", "user", "date", "channel"])
        .to_pandas()
    )
    start_time = original_date - timedelta(hours=hour_cutoff)
    end_time = original_date + timedelta(hours=hour_cutoff)
    
    filtered_results = results[
        (results["date"].apply(lambda x: datetime.fromisoformat(str(x))) >= start_time) &
        (results["date"].apply(lambda x: datetime.fromisoformat(str(x))) <= end_time)
    ]

    return filtered_results.sort_values("row_id")

def query(q, limit=5):
    t0 = time.time()
    base_results = base_query("self review", limit=limit)
    t1 = time.time()
    print(f"Base query: {t1-t0}")
    windows = []
    times = []
    for i in range(len(base_results)):
        t0 = time.time()
        window = get_post_window(table, base_results.iloc[i])
        windows.append(window)
        t1 = time.time()
        times.append(t1-t0)
    print(times)
    print(sum(times))
    return windows

def render_thread(thread):
    return thread[["user", "post"]].to_csv(index=False)

def rerank(q, threads):
    thread_strs = [render_thread(thread) for thread in threads]
    embeddings = embedding_model.encode(thread_strs, normalize_embeddings=True)
    df = pd.DataFrame({
        "thread": thread_strs,
        "vector": embeddings.tolist()
        })
    tmp = db.create_table(
        "tmp",
        data=df.to_dict(orient="records"),
        mode="overwrite"  # Overwrite if table exists
    )
    tmp.create_fts_index("thread", replace=True)
    results = (
        tmp.search(query_type="hybrid")
        .text(q)
        .vector(embed(q))
        .rerank(reranker=RRFReranker())  # Default k value
        .to_pandas()
    )
    return results

def hyde(q):
    return llm(f"Construct a short, concise hypothetical document responsive to the user's query. No more than four sentences.\n\nUser query:\n{q}")

def summarize(res):
    return llm("Summarize: " + res)

db = lancedb.connect("./vector_db")
table = db.open_table("test")
q = "What does relevance score mean compared to vector similarity score"
hyde_doc = hyde(q)
print(hyde_doc)
threads = query(hyde_doc)

reranked = rerank(q, threads)
summaries = [summarize(row["thread"]) for _, row in reranked.iterrows()]
