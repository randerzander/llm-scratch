from ragatouille import RAGPretrainedModel
import kagglehub, os
import pandas as pd
import requests
import shutil
import time


RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

def load_kaggle_docs():
    # Download latest version
    path = kagglehub.dataset_download("tgdivy/poetry-foundation-poems")
    df = pd.read_csv(path+"/"+os.listdir(path)[0])
    print(len(df))
    documents = [f"{row['Title'].strip()} by {row['Poet'].strip()}\n{row['Poem']}" for _, row in df.iterrows()]

    df = pd.read_csv("hf://datasets/merve/poetry/poetry.csv")
    print(len(df))
    documents = documents + [f"{row['poem name']} by {row['author']}\n{row['content']}" for _, row in df.iterrows()]
    return documents

def load_other_docs():
    urls = [
        "https://www.gutenberg.org/cache/epub/574/pg574.txt", #blake
        "https://www.gutenberg.org/cache/epub/4800/pg4800.txt", #shelley
        "https://www.gutenberg.org/cache/epub/10219/pg10219.txt", #wordsworth
        "https://www.gutenberg.org/cache/epub/12145/pg12145.txt",
        "https://www.gutenberg.org/cache/epub/12383/pg12383.txt",
        "https://www.gutenberg.org/cache/epub/52836/pg52836.txt",
        "https://www.gutenberg.org/cache/epub/47143/pg47143.txt",
        "https://www.gutenberg.org/cache/epub/32459/pg32459.txt",
        "https://www.gutenberg.org/cache/epub/47651/pg47651.txt",

    ]
    docs = [requests.get(url).content for url in urls]
    return docs

def index(index_name, documents):
    try:
        shutil.rmtree(f".ragatouille/colbert/indexes/{index_name}")
    except:
        pass
    t0 = time.time()
    index_path = RAG.index(index_name=index_name, collection=documents)
    t1 = time.time()
    print(f"Index building on {len(documents)} took {t1-t0} s")
    return index_path


def load_rag(index_path):
    return RAGPretrainedModel.from_index(index_path)

colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
def query(q, k=1):
    t0 = time.time()
    res = RAG.search(query=q, k=k)
    t1 = time.time()
    print(f"Found top {k} in {t1-t0}")
    return res

t0 = time.time()
t1 = time.time()
print(f"Colbert took {t1-t0} s to init")
if __name__ == "__main__":
    documents = load_kaggle_docs()
    print(len(documents))

    index_path = index("my_index", documents)
    RAG = load_rag(index_path)

    q = "What can I use to open a window?"
    res = query(q, 3)
    print(res)

