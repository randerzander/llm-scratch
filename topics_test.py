from bertopic.representation import KeyBERTInspired, LlamaCPP
from llama_cpp import Llama

from sentence_transformers import SentenceTransformer
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN

from bertopic import BERTopic

from datasets import load_dataset
import re, json, os
from itertools import chain

all_docs = []
for article_dir in ["/home/dev/projects/reader/3-7-24", "/home/dev/projects/reader/3-8-24"]:
    article_dir = "/home/dev/projects/reader/3-7-24/"
    docs = list(chain.from_iterable([json.loads(open(article_dir+fn, "r").read()) for fn in os.listdir(article_dir)]))
    docs = [doc["text"].replace("< Back to 68k.news US front page\n\n", "")[0:3500] for doc in docs]
    docs = [doc.replace("\n", "") for doc in docs if "Invalid HTML" not in doc and "Failed to get the article" not in doc]
    all_docs += docs

docs = all_docs

model_dir = "/home/dev/projects/models/"
llm = Llama(model_path=model_dir+"nous-hermes-2-mixtral-8x7b-dpo.Q4_K_M.gguf", n_gpu_layers=-1, n_ctx=4096, stop=["Q:", "\\n"])

prompt = """ Q:
[DOCUMENTS]

Text above illustrates a topic defined by keywords: '[KEYWORDS]'

Provide a concise, highly specific label (include names, places, etc.) for the topic
A:
"""
representation_model = {
   "KeyBERT": KeyBERTInspired(),
   "LLM": LlamaCPP(llm, prompt=prompt),
}

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.encode(docs, show_progress_bar=True)# Pre-reduce embeddings for visualization purposes

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

topic_model = BERTopic(
 embedding_model=embedding_model,
 umap_model=umap_model,
 hdbscan_model=hdbscan_model,
 representation_model=representation_model, # Hyperparameters
 top_n_words=15,
 verbose=True
)

topics, probs = topic_model.fit_transform(docs, embeddings)
print(topic_model.get_topic_info())
topic_model.save("best2", serialization="pickle")
