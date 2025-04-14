import lancedb
from lancedb.rerankers import RRFReranker
from sentence_transformers import SentenceTransformer

import pyarrow as pa
import pandas as pd
import numpy as np
import time, json, datetime, os
from dateutil.parser import parse

def log(key, val):
    dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{dt}] {key}: {val:.2f}")
    open("log.txt", "a").write(json.dumps({key: val}) + "\n")

# parse log
base_dir = "channel_logs/"
t0 = time.time()
def parse_log(fn):
    text = open(fn, "r").read()
    raw_posts = text.split("_|_\n")[:-1]
    posts, dts, users = [], [], []
    skipped = []
    for post in raw_posts:
        if len(post.split("|_|")) < 3:
            #print(f"Skipping invalid post:\n{post}")
            skipped.append(post)
            continue
        dt = post.split("|_|")[0].split(".")[0].strip()
        try:
            dt = parse(dt)
            dts.append(dt)
        except:
            print(f"Skipping invalid date:\n{dt}")
            skipped.append(post)
            continue
        user = post.split("|_|")[1]
        users.append(user.split("|")[1])
        posts.append(post.split("|_|")[2])
        
    print(f"{fn}: {len(posts)} posts, skipped {len(skipped)}")
    with open("skipped.txt", "a") as f:
        f.write("\n".join(skipped))
    return dts, users, posts

all_dts, all_users, all_posts = [], [], []

#model_name = "mixedbread-ai/mxbai-embed-large-v1"
model_name = "sentence-transformers/static-retrieval-mrl-en-v1"
embedding_model = SentenceTransformer(model_name)

#for fn in ["bot-stuff_1308637302658175006.txt"]:
dfs = []
embedding_time = 0
for fn in os.listdir(base_dir):
    dts, users, posts = parse_log(os.path.join(base_dir, fn))
    df = pd.DataFrame({
        "post": posts,
        "user": users,
        "date": dts
    })
    df = df.assign(row_id=range(len(df)))
    t0 = time.time()
    embeddings = embedding_model.encode(posts, normalize_embeddings=True)
    df["vector"] = embeddings.tolist()
    t1 = time.time()
    embedding_time = embedding_time + (t1-t0)
    df["channel"] = fn.split("_")[0]
    dfs.append(df)

    #all_dts.extend(dts)
    #all_users.extend(users)
    #all_posts.extend(posts)

print(f"Total posts: {len(all_posts)}")

#df = pd.DataFrame({
#    "post": all_posts,
#    "user": all_users,
#    "date": all_dts
#})
t1 = time.time()
log("parse_time", t1 - t0)

log("embedding_time", embedding_time)

t0 = time.time()
db = lancedb.connect("./vector_db")
df = pd.concat(dfs)
print("Dfs concatted..")
table = db.create_table(
    "test",
    data=df.to_dict(orient="records"),
    mode="overwrite"  # Overwrite if table exists
)
print("Table created..")
#table.create_index(metric="cosine")
#print("Index created..")
table.create_fts_index("post", replace=True)
t1 = time.time()
log("db_creation_time", t1 - t0)

def query_embed(q):
    embedding = embedding_model.encode([query], normalize_embeddings=True, batch_size=1)[0]
    return embedding.astype(np.float32)

# Perform hybrid search with default reranker
query = "What kind of laptops is randerzander interested in?"
vec = query_embed(query)
results = (
    table.search(query_type="hybrid")
    .text(query)
    .vector(vec)
    .rerank(reranker=RRFReranker())  # Default k value
    .limit(5)
    .to_pandas()
)

print(results)
